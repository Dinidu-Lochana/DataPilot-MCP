from fastmcp import FastMCP

from server.services.dataset_service import DatasetService
from server.services.profiling_service import ProfilingService

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


@mcp.tool
def profile_dataset(dataset_id: str) -> dict:
    """
    Generate a statistical profile of a loaded dataset.
    """

    dataframe = dataset_service.get_dataset(dataset_id)

    return ProfilingService.profile(dataframe)


@mcp.tool
def descriptive_statistics(dataset_id: str) -> dict:
    """
    Calculate descriptive statistics for numerical columns.
    """

    dataframe = dataset_service.get_dataset(dataset_id)

    return ProfilingService.descriptive_statistics(
        dataframe
    )


@mcp.tool
def missing_value_analysis(dataset_id: str) -> dict:
    """
    Analyze missing values in a dataset.
    """

    dataframe = dataset_service.get_dataset(dataset_id)

    missing = dataframe.isnull().sum()

    percentage = (
        dataframe.isnull().mean() * 100
    )

    return {
        "missing_count": missing.to_dict(),
        "missing_percentage": percentage.round(2).to_dict(),
    }


@mcp.tool
def duplicate_analysis(dataset_id: str) -> dict:
    """
    Analyze duplicate rows in a dataset.
    """

    dataframe = dataset_service.get_dataset(dataset_id)

    duplicate_count = int(
        dataframe.duplicated().sum()
    )

    return {
        "duplicate_rows": duplicate_count,
        "percentage": round(
            duplicate_count / len(dataframe) * 100,
            2
        )
        if len(dataframe) > 0
        else 0,
    }


@mcp.tool
def correlation_analysis(dataset_id: str) -> dict:
    """
    Calculate correlations between numerical variables.
    """

    dataframe = dataset_service.get_dataset(dataset_id)

    numeric_data = dataframe.select_dtypes(
        include="number"
    )

    if numeric_data.empty:
        return {
            "message": "No numerical columns found."
        }

    return numeric_data.corr().round(4).to_dict()


@mcp.tool
def detect_outliers(dataset_id: str) -> dict:
    """
    Detect numerical outliers using the IQR method.
    """

    dataframe = dataset_service.get_dataset(dataset_id)

    numeric_data = dataframe.select_dtypes(
        include="number"
    )

    results = {}

    for column in numeric_data.columns:

        q1 = numeric_data[column].quantile(0.25)
        q3 = numeric_data[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = numeric_data[
            (numeric_data[column] < lower_bound)
            | (numeric_data[column] > upper_bound)
        ]

        results[column] = {
            "outlier_count": len(outliers),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }

    return results


@mcp.tool
def create_histogram(dataset_id: str,column: str,) -> str:
    """
    Create a histogram for a numerical column.
    """

    dataframe = dataset_service.get_dataset(dataset_id)

    return VisualizationService.histogram(
        dataframe,
        column,
        dataset_id,
    )


if __name__ == "__main__":
    mcp.run()