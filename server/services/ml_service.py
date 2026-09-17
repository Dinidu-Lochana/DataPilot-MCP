import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class MLService:

    @staticmethod
    def train_random_forest(
        dataframe: pd.DataFrame,
        target: str,
    ):

        X = dataframe.drop(columns=[target])
        y = dataframe[target]

        categorical_columns = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        numerical_columns = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                    categorical_columns,
                ),
                (
                    "numerical",
                    "passthrough",
                    numerical_columns,
                ),
            ]
        )

        model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                    ),
                ),
            ]
        )

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        return {
            "accuracy": accuracy_score(
                y_test,
                predictions,
            ),
            "precision": precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
        }