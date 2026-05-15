"""Shared loader helpers for V2 calibration / walk-forward / shadow A/B scripts.

Extracted from ``scripts/run_v2_calibration.py`` so the V2 walk-forward gate
(§9.1) and 60-session shadow A/B (§9.3) runners can reuse the same data-prep
plumbing instead of duplicating the per-input ``_load_*`` blocks.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from regime_detection.loaders import (
    load_aggregate_forward_eps_revision_series,
    load_cpi_nowcast_series,
    load_macro_series as load_fred_macro_series,
)


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
    *,
    cpi_nowcast_parquet: Path | None = None,
    eps_weekly_history_parquet: Path | None = None,
) -> dict[str, pd.Series]:
    """Load FRED macro + manual PMI + the §2B nowcast / EPS-revision seams
    into a name-keyed dict.

    ``cpi_nowcast_parquet`` and ``eps_weekly_history_parquet`` default to
    their canonical locations under ``data/raw/`` — siblings of
    ``macro_parquet`` (which lives at ``data/raw/macro/...``). When a file
    is absent the series is simply omitted: the §2B `inflation_shock`
    single-signal limb / `earnings_*` labels stay dark, exactly as before
    the fetchers ran. ``run_cleveland_fed_nowcast_fetch`` and the
    ``aggregate_eps`` weekly accumulator produce these parquets.
    """
    series_dict = load_fred_macro_series(macro_parquet)
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
    # §2B nowcast / EPS-revision seams (ADR 0006 / Ambiguity Log #48). Both
    # parquets live as siblings of macro_parquet under data/raw/; load them
    # when present so the §2B `inflation_shock` single-signal limb and the
    # `earnings_*` labels light up. Absent file -> series omitted, labels
    # stay dark (graceful, same as pre-fetcher behaviour).
    data_root = macro_parquet.parent.parent
    if cpi_nowcast_parquet is None:
        cpi_nowcast_parquet = (
            data_root / "cleveland_fed_nowcast" / "cpi_nowcast.parquet"
        )
    if eps_weekly_history_parquet is None:
        eps_weekly_history_parquet = (
            data_root / "aggregate_forward_eps" / "sp500_eps_weekly_history.parquet"
        )
    if cpi_nowcast_parquet.exists():
        series_dict["cpi_nowcast"] = load_cpi_nowcast_series(cpi_nowcast_parquet)
    else:
        logger.info("cpi_nowcast parquet not found at %s — skipping", cpi_nowcast_parquet)
    if eps_weekly_history_parquet.exists():
        series_dict["aggregate_forward_eps_revision"] = (
            load_aggregate_forward_eps_revision_series(eps_weekly_history_parquet)
        )
    else:
        logger.info(
            "EPS weekly-history parquet not found at %s — skipping",
            eps_weekly_history_parquet,
        )
    return series_dict


# Cross-asset symbols pulled by V2 §2B / §2C / §3 axes. Mirrors the
# ``cross_asset_symbols`` list in ``scripts/run_v2_calibration.py::main``.
CROSS_ASSET_SYMBOLS: list[str] = [
    "QQQ", "IWM", "EFA", "EEM", "TLT", "HYG", "LQD", "GLD",
    "USO", "UUP", "DBC", "KRE",
    "XLY", "XLI", "XLP", "XLU",
]
