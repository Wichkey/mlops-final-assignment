from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config, ROOT_DIR, get_target_column  # noqa: E402
from src.data.fetch import ensure_committed_snapshots  # noqa: E402
from src.features.build_features import ( # noqa: E402
    TARGET_COLUMNS,
    build_features, 
    feature_columns,
  )  
from src.models.train import train_models  # noqa: E402

REQUIRED_METRIC_KEYS = {"mae", "rmse", "r2", "directional_accuracy"}
EXPECTED_MODELS = {
    "linear_regression",
    "ridge",
    "random_forest",
    "gradient_boosting",
}

@pytest.fixture(scope="session")
def config() -> dict:
    """Load pipeline config and force the offline fetch path for tests."""
    cfg = load_config()
    cfg["fetch"]["use_live_apis"] = False  # never hit yfinance / FRED in CI
    return cfg

@pytest.fixture(scope="session")
def feature_paths(config) -> dict:
    """Materialize committed snapshots, then build feature table"""
    ensure_committed_snapshots(config)
    return build_features(config)


@pytest.fixture(scope="session")
def features_df(feature_paths) -> pd.DataFrame:
    return pd.read_csv(feature_paths["features"])


@pytest.fixture(scope="session")
def on_demand_df(feature_paths) -> pd.DataFrame:
    return pd.read_csv(feature_paths["on_demand_dataset"])


@pytest.fixture(scope="session")
def trained_output(feature_paths, config) -> dict:
    """Run train_models() once and expose its return value."""
    result = train_models(config)
    assert result is not None, "train_models() returned None"
    return result


@pytest.fixture(scope="session")
def metadata(trained_output, config) -> dict:
    """Load models/metadata.json produced by train_models()."""
    meta_path = ROOT_DIR / config["paths"]["models"] / "metadata.json"
    assert meta_path.is_file(), f"metadata.json not found at {meta_path}"
    return json.loads(meta_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_features_table_non_empty(features_df, config) -> None:
    """~n_companies x (synthetic_years - 1) rows; last year has no t+1 target."""
    n_companies = int(config["fetch"]["n_companies"])
    years = [int(y) for y in config["fetch"]["synthetic_years"]]
    expected_rows = n_companies * (len(years) - 1)  # 500 * 8 = 4000 by default
    assert len(features_df) > 0, "feature table is empty"
    assert abs(len(features_df) - expected_rows) <= n_companies, (
        f"expected ~{expected_rows} rows, got {len(features_df)}"
    )
    assert {"company_id", "ticker", "fiscal_year"}.issubset(features_df.columns)


def test_no_nans_in_feature_columns(features_df) -> None:
    """Every modelling feature must be fully populated."""
    cols = feature_columns(features_df)
    assert cols, "feature_columns() returned an empty list"
    nan_count = int(features_df[cols].isna().sum().sum())
    assert nan_count == 0, f"feature columns contain {nan_count} NaN values"


def test_targets_excluded_from_feature_columns(features_df) -> None:
    """Target columns must never leak into the model input set."""
    cols = feature_columns(features_df)
    target_col = get_target_column()
    assert target_col == "target_revenue_growth"
    assert target_col not in cols
    for tgt in TARGET_COLUMNS:
        assert tgt not in cols
    assert not any(c.startswith("target_") for c in cols)


def test_chronological_split(features_df, config) -> None:
    """Train year < test_year == 2023; test partition holds only 2023 rows."""
    test_year = int(config["training"]["test_year"])
    assert test_year == 2023
    train = features_df[features_df["fiscal_year"] < test_year]
    test = features_df[features_df["fiscal_year"] == test_year]
    assert not train.empty, "training partition is empty"
    assert not test.empty, "test partition is empty"
    assert train["fiscal_year"].max() < test_year
    assert (test["fiscal_year"] == test_year).all()


def test_on_demand_dataset_uses_feature_year(on_demand_df, config) -> None:
    """The scoring batch must contain only rows for prediction.feature_year (2025)."""
    feature_year = int(config["prediction"]["feature_year"])
    assert feature_year == 2025
    assert "fiscal_year" in on_demand_df.columns
    years = sorted(on_demand_df["fiscal_year"].unique().tolist())
    assert years == [feature_year], (
        f"on_demand_dataset should contain only year {feature_year}; got {years}"
    )
    for tgt in TARGET_COLUMNS:
        assert tgt not in on_demand_df.columns, (
            f"target column '{tgt}' leaked into on_demand_dataset"
        )


def test_train_models_returns_required_metrics(trained_output) -> None:
    """train_models() must surface mae, rmse, r2, directional_accuracy."""
    assert isinstance(trained_output, dict), "train_models() must return a dict"

    # Accept either a flat metrics dict or one wrapped under "test_metrics"
    # (matches the structure persisted in models/metadata.json).
    if REQUIRED_METRIC_KEYS.issubset(trained_output.keys()):
        flat = trained_output
    elif isinstance(trained_output.get("test_metrics"), dict):
        flat = trained_output["test_metrics"]
    else:
        pytest.fail(
            "train_models() output missing required metric keys; "
            f"got keys={list(trained_output)}"
        )

    for key in REQUIRED_METRIC_KEYS:
        assert key in flat, f"missing metric '{key}'"
        assert isinstance(flat[key], (int, float)), (
            f"metric '{key}' must be numeric, got {type(flat[key])._name_}"
        )


def test_metadata_lists_four_models(metadata, config) -> None:
    """models/metadata.json must enumerate the four configured models."""
    configured = set(config["training"]["models"])
    assert configured == EXPECTED_MODELS
    assert "models" in metadata, "metadata.json missing 'models' section"
    assert set(metadata["models"].keys()) == EXPECTED_MODELS

    for name, entry in metadata["models"].items():
        assert "train_metrics" in entry, f"{name} missing train_metrics block"
        assert "test_metrics" in entry, f"{name} missing test_metrics block"
        assert REQUIRED_METRIC_KEYS.issubset(entry["test_metrics"]), (
            f"{name} test_metrics missing required metric keys"
        )

    assert int(metadata["test_year"]) == int(config["training"]["test_year"]) == 2023
    assert metadata.get("model_name") in EXPECTED_MODELS