from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import ROOT_DIR, ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

RAW_FINANCIAL_COLUMNS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "total_assets",
    "total_equity",
    "total_debt",
    "current_assets",
    "current_liabilities",
    "free_cash_flow",
]

ENGINEERED_FEATURE_COLUMNS = [
    "gross_margin",
    "operating_margin",
    "net_margin",
    "roa",
    "roe",
    "debt_to_equity",
    "current_ratio",
    "fcf_margin",
    "asset_turnover",
    "prior_revenue_growth",
    "annual_return",
    "annual_volatility",
    "log_market_cap",
    "gdp_growth",
    "dgs10",
    "cpi",
    "unrate",
]

TARGET_COLUMNS = [
    "target_revenue_growth",
    "target_net_income_growth",
    "target_operating_margin",
]

ID_COLUMNS = ["company_id", "ticker", "fiscal_year"]


def _raw_dir(config: dict[str, Any]) -> Path:
    return ROOT_DIR / config["paths"]["raw"]


def _processed_dir(config: dict[str, Any]) -> Path:
    return ROOT_DIR / config["paths"]["processed"]


def _batch_dir(config: dict[str, Any]) -> Path:
    return ROOT_DIR / config["paths"]["batch"]


def load_raw_snapshots(config: dict[str, Any] | None = None) -> pd.DataFrame:
    config = config or load_config()
    raw_dir = _raw_dir(config)

    fundamentals = pd.read_csv(raw_dir / "fundamentals_snapshot.csv")
    stock = pd.read_csv(raw_dir / "stock_snapshot.csv")
    macro = pd.read_csv(raw_dir / "macro_snapshot.csv")

    panel = fundamentals.merge(
        stock,
        on=["company_id", "ticker", "fiscal_year"],
        how="left",
    )
    panel = panel.merge(macro, on="fiscal_year", how="left")
    panel = panel.sort_values(["company_id", "fiscal_year"])
    return panel.reset_index(drop=True)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )
    return pd.Series(result, index=numerator.index)


def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()

    frame["gross_margin"] = _safe_ratio(
        frame["gross_profit"],
        frame["revenue"],
    )
    frame["operating_margin"] = _safe_ratio(
        frame["operating_income"],
        frame["revenue"],
    )
    frame["net_margin"] = _safe_ratio(frame["net_income"], frame["revenue"])
    frame["roa"] = _safe_ratio(frame["net_income"], frame["total_assets"])
    frame["roe"] = _safe_ratio(frame["net_income"], frame["total_equity"])
    frame["debt_to_equity"] = _safe_ratio(
        frame["total_debt"],
        frame["total_equity"],
    )
    frame["current_ratio"] = _safe_ratio(
        frame["current_assets"],
        frame["current_liabilities"],
    )
    frame["fcf_margin"] = _safe_ratio(
        frame["free_cash_flow"],
        frame["revenue"],
    )
    frame["asset_turnover"] = _safe_ratio(
        frame["revenue"],
        frame["total_assets"],
    )
    frame["log_market_cap"] = np.log1p(frame["market_cap"].clip(lower=0))

    frame["prior_revenue_growth"] = frame.groupby("company_id")[
        "revenue"
    ].transform(lambda values: values / values.shift(1) - 1)
    frame["prior_revenue_growth"] = frame["prior_revenue_growth"].fillna(0.0)

    frame["target_revenue_growth"] = frame.groupby("company_id")[
        "revenue"
    ].transform(lambda values: values.shift(-1) / values - 1)
    frame["target_net_income_growth"] = frame.groupby("company_id")[
        "net_income"
    ].transform(lambda values: values.shift(-1) / values - 1)
    frame["target_operating_margin"] = _safe_ratio(
        frame.groupby("company_id")["operating_income"].shift(-1),
        frame.groupby("company_id")["revenue"].shift(-1),
    )

    return frame


def feature_columns(frame: pd.DataFrame | None = None) -> list[str]:
    excluded = set(ID_COLUMNS + RAW_FINANCIAL_COLUMNS + TARGET_COLUMNS)
    excluded.add("market_cap")

    if frame is None:
        return ENGINEERED_FEATURE_COLUMNS.copy()

    return [
        column
        for column in frame.columns
        if column not in excluded
    ]


def _labeled_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.dropna(subset=TARGET_COLUMNS).copy()
    labeled = labeled.replace([np.inf, -np.inf], np.nan)
    labeled = labeled.dropna(subset=feature_columns(labeled))
    return labeled.reset_index(drop=True)


def build_on_demand_dataset(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    feature_year = int(config["prediction"]["feature_year"])
    on_demand = frame[frame["fiscal_year"] == feature_year].copy()
    columns = ID_COLUMNS + feature_columns(on_demand)
    return on_demand[columns].reset_index(drop=True)


def build_features(config: dict[str, Any] | None = None) -> dict[str, Path]:
    setup_logging()
    config = config or load_config()

    panel = load_raw_snapshots(config)
    engineered = engineer_features(panel)
    features = _labeled_feature_table(engineered)
    on_demand = build_on_demand_dataset(engineered, config)

    processed_dir = ensure_dir(_processed_dir(config))
    batch_dir = ensure_dir(_batch_dir(config))

    features_path = processed_dir / "features.csv"
    on_demand_path = batch_dir / "on_demand_dataset.csv"

    features.to_csv(features_path, index=False)
    on_demand.to_csv(on_demand_path, index=False)

    test_year = int(config["training"]["test_year"])
    test_rows = int((features["fiscal_year"] == test_year).sum())
    feature_cols = feature_columns(features)
    nan_count = int(features[feature_cols].isna().sum().sum())

    logger.info("Wrote %s (%d rows)", features_path, len(features))
    logger.info("Wrote %s (%d rows)", on_demand_path, len(on_demand))
    logger.info(
        "Feature table: %d feature columns, %d NaNs, %d test-year rows",
        len(feature_cols),
        nan_count,
        test_rows,
    )

    return {
        "features": features_path,
        "on_demand_dataset": on_demand_path,
    }
