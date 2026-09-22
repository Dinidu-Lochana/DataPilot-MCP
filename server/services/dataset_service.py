from pathlib import Path
from server.tools.dataset import load_dataframe


class DatasetService:

    def __init__(self):
        self.datasets = {}

    def load_dataset(self, file_path: str) -> dict:

        dataframe = load_dataframe(file_path)

        dataset_id = Path(file_path).stem

        self.datasets[dataset_id] = dataframe

        return {
            "dataset_id": dataset_id,
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "column_names": dataframe.columns.tolist(),
        }

    def get_dataset(self, dataset_id: str):

        if dataset_id not in self.datasets:
            raise ValueError(
                f"Dataset '{dataset_id}' is not loaded."
            )

        return self.datasets[dataset_id]

    def list_datasets(self):

        return [
            {
                "dataset_id": dataset_id,
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
            }
            for dataset_id, dataframe in self.datasets.items()
        ]

    def drop_missing_values(self, dataset_id: str, columns: list[str] | None = None) -> dict:

        dataframe = self.get_dataset(dataset_id)

        if columns:
            unknown_columns = [c for c in columns if c not in dataframe.columns]

            if unknown_columns:
                raise ValueError(
                    f"Column(s) not found: {', '.join(unknown_columns)}"
                )

        rows_before = len(dataframe)

        cleaned = dataframe.dropna(subset=columns)

        self.datasets[dataset_id] = cleaned

        return {
            "dataset_id": dataset_id,
            "columns_checked": columns or dataframe.columns.tolist(),
            "rows_before": rows_before,
            "rows_removed": rows_before - len(cleaned),
            "rows_after": len(cleaned),
        }