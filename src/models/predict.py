from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.utils import ROOT_DIR, ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

PREDICTION_COLUMN = "predicted_revenue_growth"


def _models_dir(config: dict[str, Any]) -> Path:
    return ROOT_DIR / config["paths"]["models"]


def _batch_dir(config: dict[str, Any]) -> Path:
    return ROOT_DIR / config["paths"]["batch"]


def load_artifacts(
    config: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load the persisted model.joblib and its companion metadata.json."""
    config = config or load_config()
    models_dir = _models_dir(config)
    model_path = models_dir / "model.joblib"
    metadata_path = models_dir / "metadata.json"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run 'python main.py train' first."
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found at {metadata_path}. "
            "Run 'python main.py train' first."
        )

    model = joblib.load(model_path)
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return model, metadata


def load_on_demand_dataset(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load the forward-scoring dataset written by build_features()."""
    config = config or load_config()
    path = _batch_dir(config) / "on_demand_dataset.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"On-demand dataset not found at {path}. "
            "Run 'python main.py features' first."
        )
    return pd.read_csv(path)


def run_predict(config: dict[str, Any] | None = None) -> dict[str, Path]:
    """Score the on-demand dataset and write predictions to the batch directory."""
    setup_logging()
    config = config or load_config()

    model, metadata = load_artifacts(config)
    feature_cols = list(metadata["feature_columns"])

    frame = load_on_demand_dataset(config)
    missing = [column for column in feature_cols if column not in frame.columns]
    if missing:
        raise KeyError(
            "On-demand dataset is missing required feature columns: "
            f"{missing}"
        )

    predictions = model.predict(frame[feature_cols])

    feature_year = int(config["prediction"]["feature_year"])
    target_year = int(config["prediction"]["target_year"])

    output = frame.copy()
    output[PREDICTION_COLUMN] = predictions
    output["feature_year"] = feature_year
    output["predicted_for_year"] = target_year

    batch_dir = ensure_dir(_batch_dir(config))
    output_path = batch_dir / "on_demand_predictions.csv"
    output.to_csv(output_path, index=False)

    logger.info(
        "Wrote %d predictions to %s (model=%s, %d -> %d)",
        len(output),
        output_path,
        metadata.get("model_name", "unknown"),
        feature_year,
        target_year,
    )
    return {"on_demand_predictions": output_path}


if __name__ == "__main__":
    run_predict()
