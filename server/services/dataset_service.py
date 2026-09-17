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