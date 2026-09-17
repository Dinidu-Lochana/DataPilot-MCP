from fastmcp import FastMCP

from server.services.dataset_service import DatasetService


mcp = FastMCP("DataPilot")

dataset_service = DatasetService()


@mcp.tool
def hello(name: str) -> str:
    """
    Say hello to a user.
    """
    return f"Hello {name}! DataPilot MCP is working."


@mcp.tool
def load_dataset(file_path: str) -> dict:
    """
    Load a CSV, Excel, JSON, or Parquet dataset.
    """

    return dataset_service.load_dataset(file_path)


@mcp.tool
def list_datasets() -> list:
    """
    List all datasets currently loaded.
    """

    return dataset_service.list_datasets()


if __name__ == "__main__":
    mcp.run()