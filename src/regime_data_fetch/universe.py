from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def load_symbols_from_pit_constituents_parquet(parquet_path: str | os.PathLike[str]) -> list[str]:
    """Load the current workflow stock universe from PIT constituent intervals."""
    path = Path(parquet_path)
    frame = pd.read_parquet(path)
    if "ticker" not in frame.columns:
        raise ValueError(f"PIT constituents parquet missing ticker column: {path}")
    symbols = sorted({str(value).strip() for value in frame["ticker"].dropna().tolist() if str(value).strip()})
    if not symbols:
        raise ValueError(f"PIT constituents parquet has no ticker values: {path}")
    return symbols
