import pandas as pd


class ProfilingService:

    @staticmethod
    def profile(dataframe: pd.DataFrame) -> dict:

        numeric_columns = dataframe.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = dataframe.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        missing_values = (
            dataframe.isnull()
            .sum()
            .to_dict()
        )

        unique_values = (
            dataframe.nunique()
            .to_dict()
        )

        return {
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "column_names": dataframe.columns.tolist(),
            "data_types": {
                column: str(dtype)
                for column, dtype
                in dataframe.dtypes.items()
            },
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "missing_values": missing_values,
            "duplicate_rows": int(
                dataframe.duplicated().sum()
            ),
            "unique_values": unique_values,
        }

    @staticmethod
    def descriptive_statistics(dataframe: pd.DataFrame) -> dict:

        numeric_data = dataframe.select_dtypes(
            include="number"
        )

        if numeric_data.empty:
            return {}

        statistics = numeric_data.describe()

        return statistics.to_dict()