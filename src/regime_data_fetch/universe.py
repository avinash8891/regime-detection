from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


FIXED_UNIVERSE_SYMBOL_COUNT = 762
FIXED_UNIVERSE_TREE_NAME = "daily_ohlcv_762"
FIXED_UNIVERSE_LOCAL_PATH = f"data/raw/{FIXED_UNIVERSE_TREE_NAME}"


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


def load_symbols_from_daily_ohlcv_tree(tree_root: str | os.PathLike[str]) -> list[str]:
    """Load the fixed profile stock universe from a partitioned OHLCV tree."""
    root = Path(tree_root)
    symbols = sorted(
        {
            child.name.split("=", 1)[1].strip()
            for child in root.iterdir()
            if child.is_dir() and child.name.startswith("symbol=") and child.name.split("=", 1)[1].strip()
        }
    )
    if not symbols:
        raise ValueError(f"Daily OHLCV tree has no symbol partitions: {root}")
    return symbols
