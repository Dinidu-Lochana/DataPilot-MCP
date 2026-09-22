from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# A target is treated as regression when it is numeric and has more distinct
# as a classification label instead.
REGRESSION_UNIQUE_THRESHOLD = 15

CLASSIFICATION_MODELS = {
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
}

REGRESSION_MODELS = {
    "random_forest": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    "gradient_boosting": lambda: GradientBoostingRegressor(random_state=42),
    "linear_regression": lambda: LinearRegression(),
}


class MLService:

    @staticmethod
    def _detect_task(y: pd.Series) -> str:

        if not pd.api.types.is_numeric_dtype(y):
            return "classification"

        if y.nunique() <= REGRESSION_UNIQUE_THRESHOLD:
            return "classification"

        return "regression"

    @staticmethod
    def _build_pipeline(model, categorical_columns, numerical_columns) -> Pipeline:

        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        numerical_pipeline = Pipeline(
            steps=[("impute", SimpleImputer(strategy="median"))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", categorical_pipeline, categorical_columns),
                ("numerical", numerical_pipeline, numerical_columns),
            ]
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    @staticmethod
    def _feature_importances(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, scoring: str) -> dict:

        # Permutation importance is computed per original column (not per one-hot
        # dummy) and, unlike raw model coefficients, is comparable across features
        # on different scales - raw linear-model coefficients are not, since a
        # coefficient on a 0/1 dummy and a coefficient on raw income aren't on the
        # same footing.
        result = permutation_importance(
            pipeline, X_test, y_test, scoring=scoring, n_repeats=10, random_state=42,
        )

        importances = dict(zip(X_test.columns, result.importances_mean.tolist()))

        return dict(
            sorted(importances.items(), key=lambda item: item[1], reverse=True)[:15]
        )

    @staticmethod
    def train_model(
        dataframe: pd.DataFrame,
        target: str,
        dataset_id: str,
    ) -> dict:

        if target not in dataframe.columns:
            raise ValueError(f"Column '{target}' does not exist.")

        dataframe = dataframe.dropna(subset=[target])

        X = dataframe.drop(columns=[target])
        y = dataframe[target]

        task = MLService._detect_task(y)

        categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()

        candidates = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
        scoring = "f1_weighted" if task == "classification" else "r2"

        can_stratify = task == "classification" and y.value_counts().min() >= 2
        stratify = y if can_stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify,
        )

        cv_folds = min(5, len(X_train), y_train.value_counts().min()) if task == "classification" else min(5, len(X_train))

        leaderboard = {}
        fitted_pipelines = {}

        for name, factory in candidates.items():
            pipeline = MLService._build_pipeline(factory(), categorical_columns, numerical_columns)

            cv_scores = (
                cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring=scoring)
                if cv_folds >= 2
                else [float("nan")]
            )

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            if task == "classification":
                test_metrics = {
                    "accuracy": accuracy_score(y_test, predictions),
                    "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
                    "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
                    "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
                }
            else:
                test_metrics = {
                    "r2": r2_score(y_test, predictions),
                    "mae": mean_absolute_error(y_test, predictions),
                    "rmse": root_mean_squared_error(y_test, predictions),
                }

            leaderboard[name] = {
                "cv_score_mean": float(pd.Series(cv_scores).mean()),
                "cv_score_std": float(pd.Series(cv_scores).std(ddof=0)),
                "test_metrics": test_metrics,
            }
            fitted_pipelines[name] = pipeline

        best_name = max(leaderboard, key=lambda name: leaderboard[name]["cv_score_mean"])
        best_pipeline = fitted_pipelines[best_name]

        model_path = MODEL_DIR / f"{dataset_id}_{target}_{best_name}.joblib"
        joblib.dump(best_pipeline, model_path)

        return {
            "task": task,
            "cv_scoring_metric": scoring,
            "best_model": best_name,
            "leaderboard": leaderboard,
            "feature_importances": MLService._feature_importances(
                best_pipeline, X_test, y_test, scoring
            ),
            "model_path": str(model_path),
        }
