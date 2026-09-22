import os
import sys
import json
import asyncio
import threading
import queue
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from fastmcp import Client
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

MCP_SERVERS = {
    "datapilot": {
        "command": sys.executable,
        "args": ["-m", "server.server"],
        "cwd": str(PROJECT_ROOT),
        "env": {
            "FASTMCP_SHOW_SERVER_BANNER": "false",
            "FASTMCP_LOG_LEVEL": "WARNING",
        },
    },
}

SYSTEM_PROMPT = """You are DataPilot, a data analysis assistant with tools for loading, profiling and modelling datasets.

Workflow:
- A dataset must be loaded with load_dataset before any other tool can use it. Paths are relative to the project root; sample data lives in data/.
- load_dataset returns a dataset_id (the file name without its extension). Pass that id to every other tool. Use list_datasets to see what is already loaded instead of reloading.
- Charts are saved as PNG files under reports/. Tell the user the returned file path.
- train_random_forest is a classification model; target must be an existing column name.
- Base every number you report on tool output. If a tool returns an error, fix the call (for example, load the dataset first) and retry before asking the user.
- Keep answers concise. Summarise large results instead of dumping them."""


def build_model() -> ChatOpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY is not set. Add it to the .env file in the project root.")
        st.stop()

    return ChatOpenAI(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        temperature=0,
    )


def result_text(result) -> str:
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


def to_langchain_tool(client: Client, tool) -> StructuredTool:
    async def call(**arguments) -> str:
        result = await client.call_tool(tool.name, arguments, raise_on_error=False)

        if result.is_error:
            return f"Error: {result_text(result)}"

        if result.structured_content is not None:
            return json.dumps(result.structured_content, default=str)

        return result_text(result) or "The tool ran successfully and returned an empty result."

    return StructuredTool(
        name=tool.name,
        description=tool.description or tool.name,
        args_schema=tool.input_schema,
        coroutine=call,
    )


class AgentRunner:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

        self.client = None
        self.agent = None

        future = asyncio.run_coroutine_threadsafe(self.setup(), self.loop)
        future.result()

    async def setup(self):
        model = build_model()
        self.client = Client({"mcpServers": MCP_SERVERS})
        await self.client.__aenter__()
        tools = [to_langchain_tool(self.client, tool) for tool in await self.client.list_tools()]
        self.agent = create_agent(model, tools, system_prompt=SYSTEM_PROMPT)

    def load_dataset(self, file_path: str) -> dict:
        async def _load():
            result = await self.client.call_tool(
                "load_dataset", {"file_path": file_path}, raise_on_error=False
            )
            if result.is_error:
                raise RuntimeError(result_text(result))
            return result.structured_content

        return asyncio.run_coroutine_threadsafe(_load(), self.loop).result()

    def ask_stream(self, messages):
        q = queue.Queue()

        async def _run():
            try:
                seen = len(messages)
                async for state in self.agent.astream({"messages": messages}, stream_mode="values"):
                    for message in state["messages"][seen:]:
                        if isinstance(message, AIMessage):
                            q.put(("message", message))
                    seen = len(state["messages"])
                q.put(("done", state["messages"]))
            except Exception as e:
                import traceback
                traceback.print_exc()
                q.put(("error", e))

        asyncio.run_coroutine_threadsafe(_run(), self.loop)

        while True:
            msg_type, data = q.get()
            if msg_type == "done":
                yield ("done", data)
                break
            elif msg_type == "message":
                yield ("message", data)
            elif msg_type == "error":
                raise data


@st.cache_resource
def get_agent_runner():
    return AgentRunner()


def render_message(msg):
    text = getattr(msg, "content", getattr(msg, "text", ""))
    if text:
        st.write(text)
        # Try to find paths to images and render them
        image_paths = re.findall(r"(reports[\\/][a-zA-Z0-9_\-\.]+\.png)", text)
        for img_path in image_paths:
            full_path = PROJECT_ROOT / img_path
            if full_path.exists():
                st.image(str(full_path))

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for call in msg.tool_calls:
            with st.expander(f"🛠️ Tool Call: {call['name']}"):
                st.json(call['args'])


def ingest_upload(runner, uploaded_file):
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / Path(uploaded_file.name).name
    file_path.write_bytes(uploaded_file.getvalue())

    try:
        info = runner.load_dataset(str(file_path))
    except Exception as e:
        return {"error": str(e)}

    columns = ", ".join(info["column_names"][:30])
    st.session_state.messages.append(
        HumanMessage(
            content=(
                f"I uploaded '{uploaded_file.name}'. It is already loaded as dataset_id "
                f"'{info['dataset_id']}' ({info['rows']} rows, {info['columns']} columns: {columns}). "
                "Use this dataset for my questions and do not ask me for a file path."
            ),
            additional_kwargs={"note": True},
        )
    )

    return info


def main():
    st.set_page_config(page_title="DataPilot", page_icon="🚀", layout="wide")
    st.title("DataPilot 🚀")
    st.markdown("Upload a dataset and ask questions about it.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploads" not in st.session_state:
        st.session_state.uploads = {}

    try:
        runner = get_agent_runner()
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        st.stop()

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV/Excel)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.file_id not in st.session_state.uploads:
            st.session_state.uploads[uploaded_file.file_id] = ingest_upload(runner, uploaded_file)

        upload = st.session_state.uploads[uploaded_file.file_id]
        if "error" in upload:
            st.sidebar.error(f"Could not load {uploaded_file.name}: {upload['error']}")
        else:
            st.sidebar.success(
                f"Loaded '{upload['dataset_id']}': {upload['rows']} rows, {upload['columns']} columns"
            )

    # Display previous messages
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            if msg.additional_kwargs.get("note"):
                continue
            with st.chat_message("user"):
                st.write(getattr(msg, "content", getattr(msg, "text", "")))
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                render_message(msg)

    if prompt := st.chat_input("Ask a question about the dataset"):
        with st.chat_message("user"):
            st.write(prompt)

        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.chat_message("assistant"):
            st_placeholder = st.empty()

            try:
                for msg_type, data in runner.ask_stream(st.session_state.messages):
                    if msg_type == "message":
                        render_message(data)
                    elif msg_type == "done":
                        st.session_state.messages = data
            except Exception as e:
                st.error(f"Error during agent execution: {e}")


if __name__ == "__main__":
    main()
