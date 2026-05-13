"""Shared loader helpers for V2 calibration / walk-forward / shadow A/B scripts.

Extracted from ``scripts/run_v2_calibration.py`` so the V2 walk-forward gate
(§9.1) and 60-session shadow A/B (§9.3) runners can reuse the same data-prep
plumbing instead of duplicating the per-input ``_load_*`` blocks.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


def load_market_data(daily_ohlcv_dir: Path) -> pd.DataFrame:
    """Load v1-shape (SPY/RSP/VIXY) long-format market DataFrame.

    Mirrors ``scripts/run_v2_calibration.py::_load_market_data``.
    """
    df = pd.read_parquet(daily_ohlcv_dir)
    keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
    out = df[df["symbol"].isin(["SPY", "RSP", "VIXY"])][keep].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_close_dict(
    daily_ohlcv_dir: Path,
    symbols: list[str],
    spy_index: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    """Pivot daily OHLCV parquet into close-series keyed by symbol, reindexed
    to ``spy_index``. Mirrors ``run_v2_calibration._load_close_dict``.
    """
    df = pd.read_parquet(daily_ohlcv_dir)
    df["date"] = pd.to_datetime(df["date"])
    out: dict[str, pd.Series] = {}
    for sym in symbols:
        sub = df[df["symbol"] == sym].sort_values("date").set_index("date")
        if sub.empty:
            continue
        out[sym] = sub["close"].astype(float).reindex(spy_index).rename(sym)
    return out


def load_macro_series(
    macro_parquet: Path,
    pmi_path: Path | None,
) -> dict[str, pd.Series]:
    """Load FRED macro + manual PMI into a name-keyed dict.

    Mirrors ``run_v2_calibration._load_macro_series``.
    """
    macro = pd.read_parquet(macro_parquet)
    macro["date"] = pd.to_datetime(macro["date"])
    series_dict: dict[str, pd.Series] = {}
    for name, group in macro.groupby("logical_name"):
        s = group.set_index("date")["value"].astype(float).sort_index()
        series_dict[name] = s.rename(name)
    for sid, group in macro.groupby("series_id"):
        s = group.set_index("date")["value"].astype(float).sort_index()
        series_dict.setdefault(sid, s.rename(sid))
    if pmi_path and pmi_path.exists():
        pmi_df = pd.read_csv(pmi_path, sep="\t")
        if "release_date_local" in pmi_df.columns and "actual" in pmi_df.columns:
            pmi_df["release_date_local"] = pd.to_datetime(
                pmi_df["release_date_local"], format="%d-%m-%Y"
            )
            pmi = (
                pmi_df.set_index("release_date_local")["actual"]
                .astype(float)
                .sort_index()
            )
            series_dict["pmi_manufacturing"] = pmi.rename("pmi_manufacturing")
    if "DGS10" in series_dict and "dgs10" not in series_dict:
        series_dict["dgs10"] = series_dict["DGS10"].rename("dgs10")
    if "DGS2" in series_dict and "dgs2" not in series_dict:
        series_dict["dgs2"] = series_dict["DGS2"].rename("dgs2")
    return series_dict


# Cross-asset symbols pulled by V2 §2B / §2C / §3 axes. Mirrors the
# ``cross_asset_symbols`` list in ``scripts/run_v2_calibration.py::main``.
CROSS_ASSET_SYMBOLS: list[str] = [
    "QQQ", "IWM", "EFA", "EEM", "TLT", "HYG", "LQD", "GLD",
    "USO", "UUP", "DBC", "KRE",
    "XLY", "XLI", "XLP", "XLU",
]
