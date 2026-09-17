from pathlib import Path
import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".json",
    ".parquet",
}


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a supported dataset into a Pandas DataFrame.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}"
        )

    extension = path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(path)

    if extension == ".xlsx":
        return pd.read_excel(path)

    if extension == ".json":
        return pd.read_json(path)

    if extension == ".parquet":
        return pd.read_parquet(path)

    raise ValueError("Unsupported dataset format")