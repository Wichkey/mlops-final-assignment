"""Data ingestion from yfinance and FRED."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import ROOT_DIR, ensure_dir, load_config, setup_logging

logger = logging.getLogger(__name__)

SNAPSHOT_FILES = (
    "fundamentals_snapshot.csv",
    "stock_snapshot.csv",
    "macro_snapshot.csv",
)

FUNDAMENTAL_COLUMNS = [
    "company_id",
    "ticker",
    "fiscal_year",
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

STOCK_COLUMNS = [
    "company_id",
    "ticker",
    "fiscal_year",
    "annual_return",
    "annual_volatility",
    "market_cap",
]

MACRO_COLUMNS = [
    "fiscal_year",
    "gdp_growth",
    "dgs10",
    "cpi",
    "unrate",
]


def _raw_dir(config: dict[str, Any]) -> Path:
    return ROOT_DIR / config["paths"]["raw"]


def _snapshot_paths(raw_dir: Path) -> dict[str, Path]:
    return {name: raw_dir / name for name in SNAPSHOT_FILES}


def snapshots_exist(config: dict[str, Any] | None = None) -> bool:
    """Return True when all raw snapshot CSVs are present."""
    config = config or load_config()
    paths = _snapshot_paths(_raw_dir(config))
    return all(path.is_file() for path in paths.values())


def ensure_snapshots(config: dict[str, Any] | None = None) -> dict[str, Path]:
    """Use existing raw snapshots when present; otherwise fetch live data."""
    config = config or load_config()
    paths = _snapshot_paths(_raw_dir(config))
    if snapshots_exist(config):
        logger.info("Raw snapshots already present in %s", _raw_dir(config))
        return paths
    logger.info("Raw snapshots missing; fetching live data")
    return fetch_snapshots(config)


def _assign_tickers(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"company_id": f"COMP_{index + 1:04d}", "ticker": ticker}
            for index, ticker in enumerate(tickers)
        ]
    )


def write_snapshots(
    fundamentals: pd.DataFrame,
    stock: pd.DataFrame,
    macro: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist snapshot CSVs to the configured raw data directory."""
    config = config or load_config()
    raw_dir = ensure_dir(_raw_dir(config))
    outputs = _snapshot_paths(raw_dir)

    fundamentals.to_csv(outputs["fundamentals_snapshot.csv"], index=False)
    stock.to_csv(outputs["stock_snapshot.csv"], index=False)
    macro.to_csv(outputs["macro_snapshot.csv"], index=False)

    logger.info(
        "Wrote raw snapshots to %s (%d company-year rows)",
        raw_dir,
        len(fundamentals),
    )
    return outputs


def _fetch_fred_series(
    series_id: str,
    start: datetime,
    end: datetime,
) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    frame = pd.read_csv(
        url,
        parse_dates=["observation_date"],
        index_col="observation_date",
    )
    value_column = frame.columns[0]
    series = pd.to_numeric(frame[value_column], errors="coerce")
    return series.loc[start:end]


def _annual_macro_from_fred(
    years: list[int],
    fred_cfg: dict[str, str],
) -> pd.DataFrame:
    start = datetime(min(years), 1, 1)
    end = datetime(max(years), 12, 31)
    series_frames: dict[str, pd.Series] = {}

    for column, series_id in fred_cfg.items():
        logger.info("Fetching FRED series %s (%s)", column, series_id)
        series_frames[column] = _fetch_fred_series(series_id, start, end)

    macro_rows: list[dict[str, Any]] = []
    for year in years:
        row: dict[str, Any] = {"fiscal_year": year}
        year_slice = slice(f"{year}-01-01", f"{year}-12-31")

        gdp = series_frames["gdp"].loc[year_slice].dropna()
        if len(gdp) >= 2:
            growth = (gdp.iloc[-1] / gdp.iloc[0] - 1) * 100
            row["gdp_growth"] = round(float(growth), 4)
        elif not gdp.empty:
            row["gdp_growth"] = 0.0
        else:
            row["gdp_growth"] = np.nan

        dgs10 = series_frames["dgs10"].loc[year_slice].dropna()
        if not dgs10.empty:
            row["dgs10"] = round(float(dgs10.mean()), 4)
        else:
            row["dgs10"] = np.nan

        cpi = series_frames["cpi"].loc[year_slice].dropna()
        row["cpi"] = round(float(cpi.iloc[-1]), 4) if not cpi.empty else np.nan

        unrate = series_frames["unrate"].loc[year_slice].dropna()
        if not unrate.empty:
            row["unrate"] = round(float(unrate.mean()), 4)
        else:
            row["unrate"] = np.nan

        macro_rows.append(row)

    macro = pd.DataFrame(macro_rows, columns=MACRO_COLUMNS)
    macro = macro.sort_values("fiscal_year")
    for column in ["gdp_growth", "dgs10", "cpi", "unrate"]:
        macro[column] = macro[column].ffill().bfill()

    if macro[["gdp_growth", "dgs10", "cpi", "unrate"]].isna().any().any():
        raise RuntimeError("FRED macro series contain unfillable gaps")

    return macro.reset_index(drop=True)


def _live_stock_metrics(ticker: str, years: list[int]) -> pd.DataFrame:
    import yfinance as yf

    stock = yf.Ticker(ticker)
    history = stock.history(
        start=f"{min(years) - 1}-01-01",
        end=f"{max(years) + 1}-01-01",
        auto_adjust=True,
    )
    if history.empty:
        raise ValueError(f"No price history for {ticker}")

    shares_outstanding = stock.fast_info.get("shares")
    annual_rows: list[dict[str, Any]] = []
    for year in years:
        year_prices = history.loc[f"{year}-01-01":f"{year}-12-31"]["Close"]
        prev_prices = history.loc[
            f"{year - 1}-01-01":f"{year - 1}-12-31"
        ]["Close"]
        if year_prices.empty or prev_prices.empty:
            continue

        annual_return = float(year_prices.iloc[-1] / prev_prices.iloc[-1] - 1)
        daily_returns = year_prices.pct_change().dropna()
        annual_volatility = float(daily_returns.std() * np.sqrt(252))
        year_end_price = float(year_prices.iloc[-1])
        if shares_outstanding:
            market_cap = year_end_price * float(shares_outstanding)
        else:
            market_cap = np.nan

        annual_rows.append(
            {
                "ticker": ticker,
                "fiscal_year": year,
                "annual_return": round(annual_return, 6),
                "annual_volatility": round(annual_volatility, 6),
                "market_cap": round(market_cap, 2) if pd.notna(market_cap) else np.nan,
            }
        )

    if not annual_rows:
        raise ValueError(f"Could not derive annual stock metrics for {ticker}")

    return pd.DataFrame(annual_rows)


def _fiscal_year_from_column(column: Any) -> int:
    if hasattr(column, "year"):
        return int(column.year)
    return int(pd.Timestamp(column).year)


def _statement_value(
    frame: pd.DataFrame,
    label: str,
    column: Any,
) -> float:
    if label in frame.index:
        value = frame.loc[label, column]
        if pd.notna(value):
            return float(value)
    return np.nan


def _live_fundamentals(ticker: str, years: list[int]) -> pd.DataFrame:
    import yfinance as yf

    stock = yf.Ticker(ticker)
    income = stock.financials
    balance = stock.balance_sheet
    cashflow = stock.cashflow

    if income.empty or balance.empty:
        raise ValueError(f"No financial statements for {ticker}")

    rows: list[dict[str, Any]] = []
    for column in income.columns:
        fiscal_year = _fiscal_year_from_column(column)
        if fiscal_year not in years:
            continue

        revenue = _statement_value(income, "Total Revenue", column)
        gross_profit = _statement_value(income, "Gross Profit", column)
        operating_income = _statement_value(income, "Operating Income", column)
        net_income = _statement_value(income, "Net Income", column)
        total_assets = _statement_value(balance, "Total Assets", column)
        total_equity = _statement_value(balance, "Stockholders Equity", column)
        total_debt = _statement_value(balance, "Total Debt", column)
        current_assets = _statement_value(balance, "Current Assets", column)
        current_liabilities = _statement_value(
            balance,
            "Current Liabilities",
            column,
        )
        free_cash_flow = _statement_value(cashflow, "Free Cash Flow", column)

        if pd.isna(revenue):
            continue

        rows.append(
            {
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "revenue": round(float(revenue), 2),
                "gross_profit": round(float(gross_profit), 2)
                if pd.notna(gross_profit)
                else np.nan,
                "operating_income": round(float(operating_income), 2)
                if pd.notna(operating_income)
                else np.nan,
                "net_income": round(float(net_income), 2)
                if pd.notna(net_income)
                else np.nan,
                "total_assets": round(float(total_assets), 2)
                if pd.notna(total_assets)
                else np.nan,
                "total_equity": round(float(total_equity), 2)
                if pd.notna(total_equity)
                else np.nan,
                "total_debt": round(float(total_debt), 2)
                if pd.notna(total_debt)
                else np.nan,
                "current_assets": round(float(current_assets), 2)
                if pd.notna(current_assets)
                else np.nan,
                "current_liabilities": round(float(current_liabilities), 2)
                if pd.notna(current_liabilities)
                else np.nan,
                "free_cash_flow": round(float(free_cash_flow), 2)
                if pd.notna(free_cash_flow)
                else np.nan,
            }
        )

    if not rows:
        raise ValueError(f"No overlapping fiscal years for {ticker}")

    return pd.DataFrame(rows)


def fetch_snapshots(
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Fetch company and macro snapshots from yfinance and FRED."""
    config = config or load_config()
    fetch_cfg = config["fetch"]
    years = [int(year) for year in fetch_cfg["fiscal_years"]]
    tickers = config["tickers"]

    companies = _assign_tickers(tickers)
    macro = _annual_macro_from_fred(years, config["fred"])

    fundamentals_frames: list[pd.DataFrame] = []
    stock_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        company_id = companies.loc[
            companies["ticker"] == ticker,
            "company_id",
        ].iloc[0]
        try:
            live_fundamentals = _live_fundamentals(ticker, years)
            live_stock = _live_stock_metrics(ticker, years)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping live fetch for %s: %s", ticker, exc)
            continue

        live_fundamentals["company_id"] = company_id
        live_stock["company_id"] = company_id
        fundamentals_frames.append(live_fundamentals)
        stock_frames.append(live_stock)
        logger.info("Fetched live data for %s (%s)", ticker, company_id)

    if not fundamentals_frames:
        raise RuntimeError("No live company data could be fetched")

    fundamentals = pd.concat(fundamentals_frames, ignore_index=True)
    stock = pd.concat(stock_frames, ignore_index=True)
    fundamentals = fundamentals[FUNDAMENTAL_COLUMNS]
    stock = stock[STOCK_COLUMNS]

    return write_snapshots(
        fundamentals=fundamentals,
        stock=stock,
        macro=macro,
        config=config,
    )


def run_fetch(config: dict[str, Any] | None = None) -> dict[str, Path]:
    """Run the fetch stage using live yfinance and FRED APIs."""
    setup_logging()
    config = config or load_config()
    logger.info("Running live fetch via yfinance and FRED")
    return fetch_snapshots(config)
