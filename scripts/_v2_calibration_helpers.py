"""Shared loader helpers for V2 calibration / walk-forward / shadow A/B scripts.

Extracted from ``scripts/run_v2_calibration.py`` so the V2 walk-forward gate
(§9.1) and 60-session shadow A/B (§9.3) runners can reuse the same data-prep
plumbing instead of duplicating the per-input ``_load_*`` blocks.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from regime_detection.loaders import (
    load_aggregate_forward_eps_revision_series,
    load_cpi_nowcast_series,
    load_macro_series as load_fred_macro_series,
)


logger = logging.getLogger(__name__)


def default_pmi_path(data_root: Path) -> Path:
    return data_root / "pmi" / "us_ism_pmi_history.parquet"


# TODO(simplify, owner=regime-maintainers, ticket=TD-CALIBRATION-REPORTING): hoist `_reporting_label` (4 near-identical copies in
# run_v2_walkforward_gate.py, run_v2_shadow_ab_gate.py, profile_engine_30d.py,
# audit_layer2_30d.py) into a single `axis_reporting_label(output, *, default=None)`
# helper here. Each caller's fallback (None vs "not_wired" vs str(active_label))
# becomes a `default` argument. Skipped during 2026-05-16 simplify pass because
# semantics diverge subtly across callers and a regression here would silently
# corrupt gate metrics.
#
# TODO(simplify, owner=regime-maintainers, ticket=TD-CALIBRATION-MANIFEST-ARGS): add `add_manifest_args(parser)` / `materialize_from_args(args, *,
# repo_root, required_for)` helpers to remove the 4-runner copy-paste of
# --manifest/--artifact-store/--data-root wiring. Also fixes the ordering bug in
# run_v2_calibration.py where materialize_if_requested runs BEFORE daily_dir /
# macro_parquet are derived from args.data_root (other runners derive first).
def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def load_market_data(daily_ohlcv_dir: Path) -> pd.DataFrame:
    """Load v1-shape (SPY/RSP/VIXY) long-format market DataFrame.

    Mirrors ``scripts/run_v2_calibration.py::_load_market_data``.
    """
    df = _read_daily_ohlcv(daily_ohlcv_dir, symbols=["SPY", "RSP", "VIXY"])
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
    df = _read_daily_ohlcv(daily_ohlcv_dir, symbols=symbols)
    df["date"] = pd.to_datetime(df["date"])
    out: dict[str, pd.Series] = {}
    for sym in symbols:
        sub = df[df["symbol"] == sym].sort_values("date").set_index("date")
        if sub.empty:
            continue
        out[sym] = sub["close"].astype(float).reindex(spy_index).rename(sym)
    return out


def _read_daily_ohlcv(
    daily_ohlcv_dir: Path, *, symbols: list[str] | None = None
) -> pd.DataFrame:
    if daily_ohlcv_dir.is_file():
        return pd.read_parquet(daily_ohlcv_dir)
    if not daily_ohlcv_dir.exists():
        raise FileNotFoundError(daily_ohlcv_dir)
    frames: list[pd.DataFrame] = []
    if symbols is not None:
        for symbol in symbols:
            symbol_dir = daily_ohlcv_dir / f"symbol={symbol}"
            candidates = [symbol_dir / "ohlcv.parquet"]
            if symbol_dir.exists():
                candidates.extend(sorted(symbol_dir.glob("*.parquet")))
            symbol_file = next((path for path in candidates if path.exists()), None)
            if symbol_file is None:
                continue
            frame = pd.read_parquet(symbol_file)
            if "symbol" not in frame.columns:
                frame = frame.assign(symbol=symbol)
            frames.append(frame)
    else:
        for parquet_file in sorted(daily_ohlcv_dir.rglob("*.parquet")):
            frame = pd.read_parquet(parquet_file)
            if "symbol" not in frame.columns:
                parent = parquet_file.parent.name
                if parent.startswith("symbol="):
                    frame = frame.assign(symbol=parent.removeprefix("symbol="))
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no parquet OHLCV files found under {daily_ohlcv_dir}")
    return pd.concat(frames, ignore_index=True)


def load_macro_series(
    macro_parquet: Path,
    pmi_path: Path | None,
    *,
    cpi_nowcast_parquet: Path | None = None,
    eps_weekly_history_parquet: Path | None = None,
) -> dict[str, pd.Series]:
    """Load FRED macro + PMI + the §2B nowcast / EPS-revision seams
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
        pmi = _load_pmi_manufacturing_series(pmi_path)
        if pmi is not None:
            series_dict["pmi_manufacturing"] = pmi
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
        logger.info(
            "cpi_nowcast parquet not found at %s — skipping", cpi_nowcast_parquet
        )
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


def _load_pmi_manufacturing_series(pmi_path: Path) -> pd.Series | None:
    if pmi_path.suffix.lower() == ".parquet":
        history_path = pmi_path.with_name("us_ism_pmi_history.parquet")
        latest_path = pmi_path.with_name("us_ism_pmi.parquet")
        candidates = [path for path in (history_path, latest_path) if path.exists()]
        if pmi_path.exists() and pmi_path not in candidates:
            candidates.append(pmi_path)
        if not candidates:
            return None
        pmi_df = pd.concat(
            [pd.read_parquet(path) for path in candidates],
            ignore_index=True,
        )
        required = {"series_name", "value", "release_timestamp"}
        if not required.issubset(pmi_df.columns):
            return None
        pmi_df = pmi_df[pmi_df["series_name"] == "manufacturing"].copy()
        if pmi_df.empty:
            return None
        release_timestamp = pd.to_datetime(
            pmi_df["release_timestamp"],
            utc=True,
        )
        pmi_df["release_date_local"] = (
            release_timestamp.dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        pmi_df = pmi_df.drop_duplicates(
            subset=["release_date_local"], keep="last"
        )
        return (
            pmi_df.set_index("release_date_local")["value"]
            .astype(float)
            .sort_index()
            .rename("pmi_manufacturing")
        )

    pmi_df = pd.read_csv(pmi_path, sep="\t")
    if "release_date_local" not in pmi_df.columns or "actual" not in pmi_df.columns:
        return None
    pmi_df["release_date_local"] = pd.to_datetime(
        pmi_df["release_date_local"], format="%d-%m-%Y"
    )
    return (
        pmi_df.set_index("release_date_local")["actual"]
        .astype(float)
        .sort_index()
        .rename("pmi_manufacturing")
    )


# Cross-asset symbols pulled by V2 §2B / §2C / §3 axes. Mirrors the
# ``cross_asset_symbols`` list in ``scripts/run_v2_calibration.py::main``.
CROSS_ASSET_SYMBOLS: list[str] = [
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "HYG",
    "LQD",
    "GLD",
    "USO",
    "UUP",
    "DBC",
    "KRE",
    "XLY",
    "XLI",
    "XLP",
    "XLU",
]
