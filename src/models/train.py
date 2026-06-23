"""Model training with year-based chronological split."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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
    """Split on fiscal_year: train < test_year, test == test_year."""
    train = frame[frame["fiscal_year"] < test_year].copy()
    test = frame[frame["fiscal_year"] == test_year].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Return MAE, RMSE, R², and directional accuracy."""
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


def _build_pipeline(model_name: str) -> Pipeline:
    if model_name != "linear_regression":
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def train_models(config: dict[str, Any] | None = None) -> dict[str, Path]:
    """Train models from features.csv and write model artifacts."""
    setup_logging()
    config = config or load_config()

    features_path = _processed_dir(config) / "features.csv"
    frame = pd.read_csv(features_path)

    test_year = int(config["training"]["test_year"])
    target_column = get_target_column(config)
    feature_cols = feature_columns(frame)
    model_name = config["training"]["models"][0]

    train_frame, test_frame = year_based_split(frame, test_year)

    x_train = train_frame[feature_cols]
    y_train = train_frame[target_column]
    x_test = test_frame[feature_cols]
    y_test = test_frame[target_column]

    pipeline = _build_pipeline(model_name)
    pipeline.fit(x_train, y_train)

    train_metrics = compute_metrics(y_train, pipeline.predict(x_train))
    test_metrics = compute_metrics(y_test, pipeline.predict(x_test))

    models_dir = ensure_dir(_models_dir(config))
    model_path = models_dir / "model.joblib"
    metadata_path = models_dir / "metadata.json"

    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": model_name,
        "target_column": target_column,
        "test_year": test_year,
        "feature_columns": feature_cols,
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    logger.info("Wrote %s", model_path)
    logger.info("Wrote %s", metadata_path)
    logger.info(
        "Test metrics — MAE: %.4f, RMSE: %.4f, R²: %.4f, directional: %.4f",
        test_metrics["mae"],
        test_metrics["rmse"],
        test_metrics["r2"],
        test_metrics["directional_accuracy"],
    )

    return {
        "model": model_path,
        "metadata": metadata_path,
    }
