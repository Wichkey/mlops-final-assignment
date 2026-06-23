from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import feature_columns
from src.utils import (
    ROOT_DIR,
    ensure_dir,
    get_target_column,
    load_config,
    setup_logging,
)

logger = logging.getLogger(__name__)


def year_based_split(
    frame: pd.DataFrame,
    test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["fiscal_year"] < test_year].copy()
    test = frame[frame["fiscal_year"] == test_year].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    directional_accuracy = float(
        np.mean(np.sign(y_true_arr) == np.sign(y_pred_arr))
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
    }


def _models_dir(config: dict[str, Any]) -> Path:
    return ROOT_DIR / config["paths"]["models"]


def _processed_dir(config: dict[str, Any]) -> Path:
    return ROOT_DIR / config["paths"]["processed"]


def _build_pipeline(model_name: str, random_seed: int) -> Pipeline:
    estimators: dict[str, Any] = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(random_state=random_seed),
        "random_forest": RandomForestRegressor(random_state=random_seed),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=random_seed,
        ),
    }

    if model_name not in estimators:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", estimators[model_name]),
        ]
    )


def _setup_mlflow(config: dict[str, Any]) -> None:
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


def train_models(config: dict[str, Any] | None = None) -> dict[str, Path]:
    setup_logging()
    config = config or load_config()
    _setup_mlflow(config)

    features_path = _processed_dir(config) / "features.csv"
    frame = pd.read_csv(features_path)

    test_year = int(config["training"]["test_year"])
    target_column = get_target_column(config)
    feature_cols = feature_columns(frame)
    model_names = list(config["training"]["models"])
    random_seed = int(config["random_seed"])

    train_frame, test_frame = year_based_split(frame, test_year)

    x_train = train_frame[feature_cols]
    y_train = train_frame[target_column]
    x_test = test_frame[feature_cols]
    y_test = test_frame[target_column]

    n_train = len(train_frame)
    n_test = len(test_frame)

    model_results: dict[str, dict[str, Any]] = {}
    best_model_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_test_rmse = float("inf")

    for model_name in model_names:
        pipeline = _build_pipeline(model_name, random_seed)
        pipeline.fit(x_train, y_train)

        train_metrics = compute_metrics(y_train, pipeline.predict(x_train))
        test_metrics = compute_metrics(y_test, pipeline.predict(x_test))

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            mlflow.log_param("target", target_column)
            mlflow.log_param("test_year", test_year)
            mlflow.log_param("n_train", n_train)
            mlflow.log_param("n_test", n_test)
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(metric_name, metric_value)

        model_results[model_name] = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }

        if test_metrics["rmse"] < best_test_rmse:
            best_test_rmse = test_metrics["rmse"]
            best_model_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None or best_model_name is None:
        raise RuntimeError("No models were trained")

    models_dir = ensure_dir(_models_dir(config))
    model_path = models_dir / "model.joblib"
    metadata_path = models_dir / "metadata.json"

    joblib.dump(best_pipeline, model_path)

    best_results = model_results[best_model_name]
    metadata = {
        "model_name": best_model_name,
        "target_column": target_column,
        "test_year": test_year,
        "feature_columns": feature_cols,
        "train_rows": n_train,
        "test_rows": n_test,
        "models": model_results,
        "train_metrics": best_results["train_metrics"],
        "test_metrics": best_results["test_metrics"],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    logger.info("Wrote %s (best model: %s)", model_path, best_model_name)
    logger.info("Wrote %s", metadata_path)
    for model_name in model_names:
        metrics = model_results[model_name]["test_metrics"]
        logger.info(
            "%s — MAE: %.4f, RMSE: %.4f, R²: %.4f, directional: %.4f",
            model_name,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            metrics["directional_accuracy"],
        )

    return {
        "model": model_path,
        "metadata": metadata_path,
    }
