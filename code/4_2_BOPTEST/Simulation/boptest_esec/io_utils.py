"""Portable tabular I/O for experiment artifacts.

CSV is always written for auditability and portability. Parquet is written as
an additional representation when a parquet engine is available.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_frame(frame: pd.DataFrame, base_path: Path) -> list[str]:
    """Write ``base_path.csv`` and, when supported, ``base_path.parquet``."""
    if base_path.suffix:
        base_path = base_path.with_suffix("")
    csv_path = base_path.with_suffix(".csv")
    frame.to_csv(csv_path, index=False)
    formats = ["csv"]
    try:
        frame.to_parquet(base_path.with_suffix(".parquet"), index=False)
        formats.append("parquet")
    except (ImportError, ModuleNotFoundError):
        pass
    return formats


def read_frame(base_path: Path) -> pd.DataFrame:
    """Read parquet when available/readable, otherwise fall back to CSV."""
    if base_path.suffix:
        base_path = base_path.with_suffix("")
    parquet_path = base_path.with_suffix(".parquet")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except (ImportError, ModuleNotFoundError):
            pass
    csv_path = base_path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path, dtype={"mechanism_config": str})
    raise FileNotFoundError(f"no table found for {base_path} (.parquet or .csv)")


def table_exists(base_path: Path) -> bool:
    if base_path.suffix:
        base_path = base_path.with_suffix("")
    return base_path.with_suffix(".parquet").exists() or base_path.with_suffix(".csv").exists()
