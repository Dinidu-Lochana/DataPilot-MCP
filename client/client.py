import os
import sys
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
- train_model predicts a target column, auto-detecting classification vs regression; target must be an existing column name. It compares a few models by cross-validation and saves the best one under models/. Report the best model, its test metrics, and the top few feature importances. Missing values in feature columns are imputed automatically, so this does not need cleaning first.
- If the user asks to clean, drop, or remove missing values (or rows with them), use drop_missing_values on the loaded dataset_id rather than asking them to edit the file themselves. It updates the dataset in place, so later tool calls see the cleaned data. Report how many rows were removed.
- Base every number you report on tool output. If a tool returns an error, fix the call (for example, load the dataset first) and retry before asking the user.
- Keep answers concise. Summarise large results instead of dumping them."""


def build_model() -> ChatOpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY is not set. Add it to the .env file in the project root.")

    return ChatOpenAI(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        temperature=0,
    )


def result_text(result) -> str:
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


# langchain-mcp-adapters caps mcp<2, which conflicts with fastmcp (mcp>=2), so tools are bridged by hand.
def to_langchain_tool(client: Client, tool) -> StructuredTool:
    async def call(**arguments) -> str:
        result = await client.call_tool(tool.name, arguments, raise_on_error=False)

        if result.is_error:
            return f"Error: {result_text(result)}"

        if result.structured_content is not None:
            return json.dumps(result.structured_content, default=str)

        # An empty list comes back with no content blocks at all; a blank reply makes the model retry.
        return result_text(result) or "The tool ran successfully and returned an empty result."

    return StructuredTool(
        name=tool.name,
        description=tool.description or tool.name,
        args_schema=tool.input_schema,
        coroutine=call,
    )


async def ask(agent, messages: list) -> list:
    seen = len(messages)

    async for state in agent.astream({"messages": messages}, stream_mode="values"):
        for message in state["messages"][seen:]:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    print(f"  [tool] {call['name']}({json.dumps(call['args'])})")
        seen = len(state["messages"])

    print(f"\nagent> {state['messages'][-1].text}")

    return state["messages"]


async def chat(agent, tool_count: int) -> None:
    print(f"DataPilot client ready ({tool_count} tools). Type 'exit' to quit.")

    history: list = []

    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            history = await ask(agent, history + [HumanMessage(question)])
        except Exception as error:
            print(f"\n[error] {type(error).__name__}: {error}")


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model = build_model()

    try:
        # One open session keeps a single server process alive, so loaded datasets persist between turns.
        async with Client({"mcpServers": MCP_SERVERS}) as client:
            tools = [to_langchain_tool(client, tool) for tool in await client.list_tools()]
            agent = create_agent(model, tools, system_prompt=SYSTEM_PROMPT)

            await chat(agent, len(tools))
    finally:
        await model.root_async_client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
