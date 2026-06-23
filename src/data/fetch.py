"""Data ingestion: synthetic panel generation and optional live API refresh."""

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
    "synthetic_company_year_panel.csv",
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
    """Return True when all committed raw snapshot CSVs are present."""
    config = config or load_config()
    paths = _snapshot_paths(_raw_dir(config))
    return all(path.is_file() for path in paths.values())


def _assign_tickers(n_companies: int, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for index in range(n_companies):
        company_id = f"COMP_{index + 1:04d}"
        if index < len(tickers):
            ticker = tickers[index]
        else:
            ticker = f"SYN_{index + 1:04d}"
        rows.append({"company_id": company_id, "ticker": ticker})
    return pd.DataFrame(rows)


def _generate_synthetic_macro(
    years: list[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    gdp_growth = rng.normal(2.4, 0.8, size=len(years))
    dgs10 = np.clip(rng.normal(2.8, 0.9, size=len(years)), 0.5, None)
    cpi_base = 240.0
    cpi_levels = []
    for growth in rng.normal(2.5, 1.0, size=len(years)):
        cpi_base *= 1 + growth / 100
        cpi_levels.append(cpi_base)
    unrate = np.clip(rng.normal(4.5, 1.2, size=len(years)), 2.5, None)

    return pd.DataFrame(
        {
            "fiscal_year": years,
            "gdp_growth": np.round(gdp_growth, 4),
            "dgs10": np.round(dgs10, 4),
            "cpi": np.round(cpi_levels, 4),
            "unrate": np.round(unrate, 4),
        }
    )


def _generate_synthetic_fundamentals(
    companies: pd.DataFrame,
    years: list[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, company in companies.iterrows():
        revenue = float(rng.uniform(5e8, 2e11))
        gross_margin = float(rng.uniform(0.25, 0.55))
        op_margin = float(gross_margin * rng.uniform(0.35, 0.75))
        net_margin = float(op_margin * rng.uniform(0.55, 0.9))
        asset_turnover = float(rng.uniform(0.4, 1.6))
        equity_ratio = float(rng.uniform(0.35, 0.65))
        liability_share = float(rng.uniform(0.15, 0.35))
        current_ratio = float(rng.uniform(0.9, 2.4))

        for year in years:
            growth = float(rng.normal(0.06, 0.12))
            revenue = max(revenue * (1 + growth), 1e7)

            total_assets = revenue / asset_turnover
            total_equity = total_assets * equity_ratio
            total_debt = max(total_assets - total_equity, 0.0)
            current_liabilities = total_assets * liability_share
            current_assets = current_liabilities * current_ratio

            rows.append(
                {
                    "company_id": company["company_id"],
                    "ticker": company["ticker"],
                    "fiscal_year": year,
                    "revenue": round(revenue, 2),
                    "gross_profit": round(revenue * gross_margin, 2),
                    "operating_income": round(revenue * op_margin, 2),
                    "net_income": round(revenue * net_margin, 2),
                    "total_assets": round(total_assets, 2),
                    "total_equity": round(total_equity, 2),
                    "total_debt": round(total_debt, 2),
                    "current_assets": round(current_assets, 2),
                    "current_liabilities": round(current_liabilities, 2),
                    "free_cash_flow": round(
                        revenue * net_margin * rng.uniform(0.6, 1.1),
                        2,
                    ),
                }
            )
    return pd.DataFrame(rows, columns=FUNDAMENTAL_COLUMNS)


def _generate_synthetic_stock(
    fundamentals: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in fundamentals.iterrows():
        rows.append(
            {
                "company_id": row["company_id"],
                "ticker": row["ticker"],
                "fiscal_year": row["fiscal_year"],
                "annual_return": round(float(rng.normal(0.1, 0.22)), 6),
                "annual_volatility": round(float(rng.uniform(0.18, 0.45)), 6),
                "market_cap": round(
                    float(row["revenue"] * rng.uniform(1.5, 7.0)),
                    2,
                ),
            }
        )
    return pd.DataFrame(rows, columns=STOCK_COLUMNS)


def _build_panel(
    fundamentals: pd.DataFrame,
    stock: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    panel = fundamentals.merge(
        stock,
        on=["company_id", "ticker", "fiscal_year"],
        how="left",
    )
    return panel.merge(macro, on="fiscal_year", how="left")


def generate_synthetic_snapshots(
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Generate and write reproducible synthetic raw snapshots."""
    config = config or load_config()
    fetch_cfg = config["fetch"]
    n_companies = int(fetch_cfg["n_companies"])
    years = [int(year) for year in fetch_cfg["synthetic_years"]]
    seed = int(config["random_seed"])

    rng = np.random.default_rng(seed)
    companies = _assign_tickers(n_companies, config["tickers"])
    macro = _generate_synthetic_macro(years, rng)
    fundamentals = _generate_synthetic_fundamentals(companies, years, rng)
    stock = _generate_synthetic_stock(fundamentals, rng)
    panel = _build_panel(fundamentals, stock, macro)

    return write_snapshots(
        fundamentals=fundamentals,
        stock=stock,
        macro=macro,
        panel=panel,
        config=config,
    )


def write_snapshots(
    fundamentals: pd.DataFrame,
    stock: pd.DataFrame,
    macro: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist snapshot CSVs to the configured raw data directory."""
    config = config or load_config()
    raw_dir = ensure_dir(_raw_dir(config))
    outputs = _snapshot_paths(raw_dir)

    fundamentals.to_csv(outputs["fundamentals_snapshot.csv"], index=False)
    stock.to_csv(outputs["stock_snapshot.csv"], index=False)
    macro.to_csv(outputs["macro_snapshot.csv"], index=False)
    panel.to_csv(outputs["synthetic_company_year_panel.csv"], index=False)

    logger.info(
        "Wrote raw snapshots to %s (%d company-year rows)",
        raw_dir,
        len(panel),
    )
    return outputs


def _annual_macro_from_fred(
    years: list[int],
    fred_cfg: dict[str, str],
) -> pd.DataFrame:
    import pandas_datareader.data as web

    start = datetime(min(years), 1, 1)
    end = datetime(max(years), 12, 31)
    series_frames: dict[str, pd.Series] = {}

    for column, series_id in fred_cfg.items():
        logger.info("Fetching FRED series %s (%s)", column, series_id)
        frame = web.DataReader(series_id, "fred", start, end)
        series_frames[column] = frame[series_id]

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
    if macro[["gdp_growth", "dgs10", "cpi", "unrate"]].isna().any().any():
        logger.warning(
            "FRED macro series contain gaps; filling from synthetic macro",
        )
        synthetic_macro = _generate_synthetic_macro(
            years,
            np.random.default_rng(42),
        )
        macro = macro.set_index("fiscal_year")
        synthetic_macro = synthetic_macro.set_index("fiscal_year")
        macro = macro.combine_first(synthetic_macro).reset_index()
    return macro


def _live_stock_metrics(ticker: str, years: list[int]) -> pd.DataFrame:
    import yfinance as yf

    history = yf.Ticker(ticker).history(
        start=f"{min(years) - 1}-01-01",
        end=f"{max(years) + 1}-01-01",
        auto_adjust=True,
    )
    if history.empty:
        raise ValueError(f"No price history for {ticker}")

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
        market_cap = float(year_prices.iloc[-1] * 1e9)

        annual_rows.append(
            {
                "ticker": ticker,
                "fiscal_year": year,
                "annual_return": round(annual_return, 6),
                "annual_volatility": round(annual_volatility, 6),
                "market_cap": round(market_cap, 2),
            }
        )

    if not annual_rows:
        raise ValueError(f"Could not derive annual stock metrics for {ticker}")

    return pd.DataFrame(annual_rows)


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
        fiscal_year = int(getattr(column, "year", column))
        if fiscal_year not in years:
            continue

        revenue = _statement_value(income, "Total Revenue", column)
        gross_profit = _statement_value(
            income,
            "Gross Profit",
            column,
            revenue * 0.4,
        )
        operating_income = _statement_value(
            income,
            "Operating Income",
            column,
            revenue * 0.15,
        )
        net_income = _statement_value(
            income,
            "Net Income",
            column,
            operating_income * 0.7,
        )
        total_assets = _statement_value(
            balance,
            "Total Assets",
            column,
            revenue * 1.2,
        )
        total_equity = _statement_value(
            balance,
            "Stockholders Equity",
            column,
            total_assets * 0.5,
        )
        total_debt = _statement_value(
            balance,
            "Total Debt",
            column,
            total_assets * 0.3,
        )
        current_assets = _statement_value(
            balance,
            "Current Assets",
            column,
            total_assets * 0.25,
        )
        current_liabilities = _statement_value(
            balance,
            "Current Liabilities",
            column,
            total_assets * 0.15,
        )
        free_cash_flow = _statement_value(
            cashflow,
            "Free Cash Flow",
            column,
            net_income * 0.8,
        )

        rows.append(
            {
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "revenue": round(float(revenue), 2),
                "gross_profit": round(float(gross_profit), 2),
                "operating_income": round(float(operating_income), 2),
                "net_income": round(float(net_income), 2),
                "total_assets": round(float(total_assets), 2),
                "total_equity": round(float(total_equity), 2),
                "total_debt": round(float(total_debt), 2),
                "current_assets": round(float(current_assets), 2),
                "current_liabilities": round(float(current_liabilities), 2),
                "free_cash_flow": round(float(free_cash_flow), 2),
            }
        )

    if not rows:
        raise ValueError(f"No overlapping fiscal years for {ticker}")

    return pd.DataFrame(rows)


def _statement_value(
    frame: pd.DataFrame,
    label: str,
    column: Any,
    default: float,
) -> float:
    if label in frame.index:
        value = frame.loc[label, column]
        if pd.notna(value):
            return float(value)
    return float(default)


def fetch_live_snapshots(
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Refresh snapshots from yfinance and FRED with synthetic fallback."""
    config = config or load_config()
    fetch_cfg = config["fetch"]
    years = [int(year) for year in fetch_cfg["synthetic_years"]]
    n_companies = int(fetch_cfg["n_companies"])
    tickers = config["tickers"]

    companies = _assign_tickers(n_companies, tickers)
    macro = _annual_macro_from_fred(years, config["fred"])

    synthetic = generate_synthetic_snapshots(config)
    synthetic_panel = pd.read_csv(
        synthetic["synthetic_company_year_panel.csv"],
    )
    fundamentals = synthetic_panel[FUNDAMENTAL_COLUMNS].copy()
    stock = synthetic_panel[STOCK_COLUMNS].copy()

    live_tickers = tickers[: min(len(tickers), n_companies)]
    for ticker in live_tickers:
        company_rows = companies.loc[
            companies["ticker"] == ticker,
            "company_id",
        ]
        company_id = company_rows.iloc[0]
        try:
            live_fundamentals = _live_fundamentals(ticker, years)
            live_stock = _live_stock_metrics(ticker, years)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping live fetch for %s: %s", ticker, exc)
            continue

        live_fundamentals["company_id"] = company_id
        live_stock["company_id"] = company_id

        fundamentals = _replace_company_rows(
            fundamentals,
            company_id,
            live_fundamentals,
        )
        stock = _replace_company_rows(stock, company_id, live_stock)
        logger.info("Updated live data for %s (%s)", ticker, company_id)

    panel = _build_panel(fundamentals, stock, macro)
    return write_snapshots(
        fundamentals=fundamentals,
        stock=stock,
        macro=macro,
        panel=panel,
        config=config,
    )


def _replace_company_rows(
    frame: pd.DataFrame,
    company_id: str,
    replacement: pd.DataFrame,
) -> pd.DataFrame:
    remaining = frame[frame["company_id"] != company_id]
    ordered_columns = frame.columns.tolist()
    replacement = replacement[ordered_columns]
    return pd.concat([remaining, replacement], ignore_index=True)


def ensure_committed_snapshots(
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Ensure offline snapshots exist, generating synthetic data if needed."""
    config = config or load_config()
    raw_dir = ensure_dir(_raw_dir(config))
    paths = _snapshot_paths(raw_dir)

    if snapshots_exist(config):
        logger.info("Raw snapshots already present in %s", raw_dir)
        return paths

    logger.info("Raw snapshots missing; generating synthetic panel")
    return generate_synthetic_snapshots(config)


def run_fetch(config: dict[str, Any] | None = None) -> dict[str, Path]:
    """Run the fetch stage using live APIs or offline synthetic snapshots."""
    setup_logging()
    config = config or load_config()

    if config["fetch"]["use_live_apis"]:
        logger.info("Running live fetch via yfinance and FRED")
        return fetch_live_snapshots(config)

    logger.info("Running offline fetch (synthetic snapshots when missing)")
    return ensure_committed_snapshots(config)
