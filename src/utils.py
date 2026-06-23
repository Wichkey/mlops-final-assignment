"""Shared configuration and filesystem helpers for the pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]

TARGET_COLUMN_MAP = {
    "revenue_growth": "target_revenue_growth",
    "net_income_growth": "target_net_income_growth",
    "operating_margin": "target_operating_margin",
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load pipeline configuration from YAML."""
    config_path = Path(path) if path else ROOT_DIR / "config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging for CLI stages."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def get_target_column(config: dict[str, Any] | None = None) -> str:
    """Return the feature-table column name for the configured target."""
    if config is None:
        config = load_config()

    target_type = config["target"]["type"]
    try:
        return TARGET_COLUMN_MAP[target_type]
    except KeyError as exc:
        known = ", ".join(sorted(TARGET_COLUMN_MAP))
        raise ValueError(
            f"Unknown target type '{target_type}'. Expected one of: {known}"
        ) from exc
