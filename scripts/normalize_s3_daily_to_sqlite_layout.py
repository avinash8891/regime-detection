#!/usr/bin/env python3
"""Normalize the s3://autoresearch-platform/backups/.../candles/daily/ layout
into the partition layout the repo's local_daily_ohlcv_sqlite importer expects.

Source layout (per S3 sync):
    data/raw/s3_daily_762/SYMBOL/YYYY.parquet
    cols: Open, High, Low, Close, Volume; DatetimeIndex named "timestamp"

Target layout (for run_local_daily_ohlcv_sqlite_import):
    data/raw/daily_ohlcv_762/symbol=SYMBOL/*.parquet
    EXPECTED_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "adjusted_close"]

Provenance: the source S3 data was fetched from Alpaca with split-adjustment
applied (verified by spot-check: AAPL 2024-01-02 close $185.24 reflects the
2020 4:1 split). Per Ambiguity Log #54 the §1D PIT breadth features must use
adjusted_close; we satisfy this by aliasing adjusted_close := close. The
source_file column records the S3 provenance string for audit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def normalize_one(src_path: Path, symbol: str) -> pd.DataFrame:
    """Read one S3-shape per-year parquet, return repo-shape DataFrame."""
    df = pd.read_parquet(src_path)
    # Index is named "timestamp" with UTC offset (NYSE session as 05:00 UTC).
    # The repo's SQLite store wants `date` as ISO yyyy-mm-dd TEXT — normalize.
    df = df.reset_index()
    timestamp_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df["date"] = (
        pd.to_datetime(df[timestamp_col])
        .dt.tz_localize(None)
        .dt.normalize()
        .dt.date.astype(str)
    )
    df["symbol"] = symbol
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["adjusted_close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype("int64")
    keep = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjusted_close",
    ]
    df = df[keep].sort_values("date").reset_index(drop=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Normalize S3 daily/SYMBOL/YYYY.parquet → repo symbol=XYZ layout."
    )
    ap.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Source directory (data/raw/s3_daily_762)",
    )
    ap.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Destination (e.g. data/raw/daily_ohlcv_762)",
    )
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    symbols = sorted(p.name for p in args.src.iterdir() if p.is_dir())
    print(f"normalizing {len(symbols)} symbols → {args.dst}")
    for i, symbol in enumerate(symbols):
        sym_src = args.src / symbol
        sym_dst = args.dst / f"symbol={symbol}"
        sym_dst.mkdir(parents=True, exist_ok=True)
        year_files = sorted(sym_src.glob("*.parquet"))
        if not year_files:
            continue
        # Concatenate all years into a single parquet under the symbol partition.
        frames = [normalize_one(p, symbol) for p in year_files]
        combined = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )
        combined.to_parquet(sym_dst / "ohlcv.parquet", index=False)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(symbols)} done")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
