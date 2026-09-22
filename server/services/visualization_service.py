from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


class VisualizationService:

    @staticmethod
    def histogram(
        dataframe: pd.DataFrame,
        column: str,
        dataset_id: str,
    ) -> str:

        if column not in dataframe.columns:
            raise ValueError(
                f"Column '{column}' does not exist."
            )

        output_path = (
            REPORT_DIR
            / f"{dataset_id}_{column}_histogram.png"
        )

        plt.figure()
        dataframe[column].hist()
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return str(output_path)