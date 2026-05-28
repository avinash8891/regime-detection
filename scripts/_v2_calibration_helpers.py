"""Shared loader helpers for V2 calibration / walk-forward / shadow A/B scripts.

Extracted from ``scripts/run_v2_calibration.py`` so the V2 walk-forward gate
(§9.1) and 60-session shadow A/B (§9.3) runners can reuse the same data-prep
plumbing instead of duplicating the per-input ``_load_*`` blocks.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from regime_data_fetch.daily_ohlcv_contract import require_symbol_partition_frame
from regime_data_fetch.materialization import materialize_if_requested
from regime_data_fetch.manifest_inputs import (
    MANIFEST_INPUT_FLAGS,
    MANIFEST_INPUT_SPECS,
    get_manifest_input_spec,
    resolve_runner_input_paths,
)
from regime_detection.credit_funding import (
    REQUIRED_CROSS_ASSET_KEYS as CREDIT_FUNDING_CROSS_ASSET_KEYS,
)
from regime_detection.comparison import axis_reporting_label
from regime_detection.fragility_universe import (
    CROSS_ASSET_SYMBOLS,
)
from regime_detection.inflation_growth import (
    REQUIRED_CROSS_ASSET_KEYS as INFLATION_GROWTH_CROSS_ASSET_KEYS,
)
from regime_detection.loaders import (
    load_aggregate_forward_eps_revision_series,
    load_cpi_nowcast_series,
    load_macro_series as load_fred_macro_series,
)
from regime_shared.pandas_compat import cow_safe_assign

logger = logging.getLogger(__name__)


def default_pmi_path(data_root: Path) -> Path:
    spec = get_manifest_input_spec("pmi_path")
    assert spec.default_relpath is not None
    return data_root.joinpath(*spec.default_relpath)


def synthetic_pit_intervals_from_sector_closes(
    sector_etf_closes: dict[str, pd.Series],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": list(sector_etf_closes),
            "start_date": [
                series.index.min().date() for series in sector_etf_closes.values()
            ],
            "end_date": [None] * len(sector_etf_closes),
        }
    )


def constituent_ohlcv_from_sector_closes(
    sector_etf_closes: dict[str, pd.Series],
) -> dict[str, pd.DataFrame]:
    return {
        symbol: pd.DataFrame(
            {
                "open": series.astype(float),
                "high": series.astype(float),
                "low": series.astype(float),
                "close": series.astype(float),
                "volume": pd.Series(1_000_000, index=series.index, dtype="int64"),
                "adjusted_close": series.astype(float),
            }
        )
        for symbol, series in sector_etf_closes.items()
    }


def register_manifest_input_args(
    parser: argparse.ArgumentParser,
    *,
    include_required_paths: bool = True,
) -> None:
    """The single source of truth is the registry in ``manifest_inputs.py``;
    adding a spec there automatically adds the corresponding CLI flag to
    every runner that calls this helper, so the historic drift between
    ``ARTIFACT_BY_FIELD`` and per-runner argparse blocks can no longer
    silently lose a manifest field. Set ``include_required_paths=False`` for
    runners that need to register only optional paths and supply
    ``daily_dir``/``constituent_tree`` themselves.
    """
    for spec in MANIFEST_INPUT_SPECS:
        if spec.is_required and not include_required_paths:
            continue
        parser.add_argument(spec.cli_flag, dest=spec.field, type=Path, default=None)


def apply_manifest_input_defaults(
    args: argparse.Namespace,
    data_root: Path,
    *,
    fields: frozenset[str] | None = None,
) -> None:
    """For every spec with a canonical ``default_relpath``, set
    ``args.<field>`` from ``data_root`` when the runner did not already
    populate it (via CLI override or manifest resolution).

    Replaces the per-runner ``if args.X is None: args.X = data_root / ...``
    blocks. Pass ``fields`` to restrict defaulting to a subset.
    """
    for spec in MANIFEST_INPUT_SPECS:
        if spec.default_relpath is None:
            continue
        if fields is not None and spec.field not in fields:
            continue
        if getattr(args, spec.field, None) is None:
            setattr(args, spec.field, data_root.joinpath(*spec.default_relpath))


def axis_reporting_label_not_wired(output: Any | None) -> str:
    label = axis_reporting_label(output, default="not_wired")
    assert label is not None
    return label


def add_manifest_args(
    parser: argparse.ArgumentParser,
    *,
    data_root_default: Path,
    action: str,
) -> None:
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=f"Optional artifact manifest to materialize before {action}.",
    )
    parser.add_argument(
        "--artifact-store",
        default=None,
        help="Optional artifact-store root override for --manifest.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=data_root_default,
        help="Local data/raw root used for manifest materialization.",
    )


def manifest_input_overrides(argv: list[str]) -> frozenset[str]:
    overrides: set[str] = set()
    for field, flag in MANIFEST_INPUT_FLAGS.items():
        if any(item == flag or item.startswith(f"{flag}=") for item in argv):
            overrides.add(field)
    return frozenset(overrides)


def apply_manifest_input_paths(
    args: argparse.Namespace,
    *,
    runner_name: str,
    repo_root: Path,
    required_fields: frozenset[str] | None = None,
) -> None:
    if args.manifest is None:
        return
    resolved = resolve_runner_input_paths(
        manifest_path=args.manifest,
        data_root=args.data_root,
        runner_name=runner_name,
        cli_values={
            field: getattr(args, field, None) for field in MANIFEST_INPUT_FLAGS
        },
        cli_overrides=args.manifest_input_overrides,
        repo_root=repo_root,
        **({"required_fields": required_fields} if required_fields is not None else {}),
    )
    for field in MANIFEST_INPUT_FLAGS:
        setattr(args, field, getattr(resolved, field))
    args.manifest_resolved_inputs = resolved.resolved_from_manifest
    args.manifest_cli_overrides = resolved.cli_overrides


def materialize_manifest_from_args(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    required_for: str,
) -> None:
    materialize_if_requested(
        manifest_path=args.manifest,
        local_root=args.data_root,
        repo_root=repo_root,
        store_root=args.artifact_store,
        required_for=required_for,
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def load_market_data(daily_ohlcv_dir: Path) -> pd.DataFrame:
    """Load v1-shape (SPY/RSP/VIX) long-format market DataFrame.

    Mirrors ``scripts/run_v2_calibration.py::_load_market_data``.
    """
    required_symbols = ["SPY", "RSP", "VIX"]
    df = _read_daily_ohlcv(daily_ohlcv_dir, symbols=required_symbols)
    keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
    out = cow_safe_assign(
        df,
        {"date": pd.to_datetime(df["date"])},
        columns=keep,
    )
    max_dates = out.groupby("symbol")["date"].max()
    missing = sorted(set(required_symbols) - set(max_dates.index))
    if missing:
        raise FileNotFoundError(
            f"daily OHLCV missing required market symbols: {missing}"
        )
    common_end = max_dates.loc[required_symbols].min()
    out = out[out["date"] <= common_end].copy()
    _require_daily_ohlcv_calendar_coverage(
        out,
        symbols=required_symbols,
        expected_index=pd.DatetimeIndex(
            out.loc[out["symbol"] == "SPY", "date"].sort_values().unique()
        ),
    )
    out = cow_safe_assign(out, {"date": out["date"].dt.date}, columns=keep)
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
    df = cow_safe_assign(df, {"date": pd.to_datetime(df["date"])})
    out: dict[str, pd.Series] = {}
    for sym in symbols:
        sub = df[df["symbol"] == sym].sort_values("date").set_index("date")
        if sub.empty:
            continue
        aligned = sub["close"].astype(float).reindex(spy_index).rename(sym)
        _require_close_series_calendar_coverage(aligned)
        out[sym] = aligned
    return out


def _require_daily_ohlcv_calendar_coverage(
    frame: pd.DataFrame,
    *,
    symbols: list[str],
    expected_index: pd.DatetimeIndex,
) -> None:
    for symbol in symbols:
        observed = pd.DatetimeIndex(
            frame.loc[frame["symbol"] == symbol, "date"].sort_values().unique()
        )
        _require_calendar_gap_free(
            label=symbol, observed=observed, expected=expected_index
        )


def _require_close_series_calendar_coverage(series: pd.Series) -> None:
    observed = pd.DatetimeIndex(series.index[series.notna()])
    if observed.empty:
        return
    expected = pd.DatetimeIndex(series.index)
    _require_calendar_gap_free(
        label=str(series.name) if series.name is not None else "<unnamed>",
        observed=observed,
        expected=expected,
    )


def _require_calendar_gap_free(
    *,
    label: str,
    observed: pd.DatetimeIndex,
    expected: pd.DatetimeIndex,
) -> None:
    """Raise ``ValueError`` if ``expected``, restricted to ``observed``'s
    min..max range, contains any date missing from ``observed``.
    """
    if len(observed) == 0:
        return
    in_range = expected[(expected >= observed.min()) & (expected <= observed.max())]
    missing = in_range.difference(observed)
    if missing.empty:
        return
    examples = ", ".join(ts.strftime("%Y-%m-%d") for ts in missing[:5])
    raise ValueError(
        "daily OHLCV calendar coverage gap: "
        f"symbol={label} missing {len(missing)} session row(s); examples: {examples}"
    )


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
            require_symbol_partition_frame(
                frame, expected_symbol=symbol, source=symbol_file
            )
            frames.append(frame)
    else:
        for parquet_file in sorted(daily_ohlcv_dir.rglob("*.parquet")):
            frame = pd.read_parquet(parquet_file)
            parent = parquet_file.parent.name
            if parent.startswith("symbol="):
                partition_symbol = parent.removeprefix("symbol=")
                require_symbol_partition_frame(
                    frame, expected_symbol=partition_symbol, source=parquet_file
                )
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no parquet OHLCV files found under {daily_ohlcv_dir}")
    return pd.concat(frames, ignore_index=True)


def load_macro_series(
    macro_parquet: Path,
    pmi_path: Path | None,
    *,
    cpi_nowcast_parquet: Path | None,
    eps_weekly_history_parquet: Path | None,
) -> dict[str, pd.Series]:
    """Load FRED macro + PMI + the §2B nowcast / EPS-revision seams
    into a name-keyed dict.

    All four paths are passed explicitly by the caller. The historic
    sibling-path discovery fallback (``macro_parquet.parent.parent``) was
    removed because it silently masked manifest-router gaps — a missing
    ``cpi_nowcast_parquet`` would be filled by an empty default path, the
    §2B inflation-shock limb would degrade to NaN, and no error would
    surface. Callers must now route the path explicitly: either via the
    manifest router (``apply_manifest_input_paths``) or via
    ``apply_manifest_input_defaults`` for non-manifest invocations.

    When ``cpi_nowcast_parquet`` or ``eps_weekly_history_parquet`` is
    ``None`` or its file does not exist, the corresponding series is
    omitted and the dependent §2B labels stay dark — same end state as
    before, but the missing-file warning now points at the explicit path
    the caller chose, which is debuggable.
    """
    series_dict = load_fred_macro_series(macro_parquet)
    if pmi_path and pmi_path.exists():
        pmi = _load_pmi_manufacturing_series(pmi_path)
        if pmi is not None:
            series_dict["pmi_manufacturing"] = pmi
    if cpi_nowcast_parquet is not None and cpi_nowcast_parquet.exists():
        series_dict["cpi_nowcast"] = load_cpi_nowcast_series(cpi_nowcast_parquet)
    else:
        logger.warning(
            "cpi_nowcast parquet not found at %s — Layer 2 inflation surprise "
            "input is unwired; re-fetch with "
            "scripts/fetch_regime_engine_v1_data.py --fetch macro",
            cpi_nowcast_parquet,
        )
    if eps_weekly_history_parquet is not None and eps_weekly_history_parquet.exists():
        series_dict["aggregate_forward_eps_revision"] = (
            load_aggregate_forward_eps_revision_series(eps_weekly_history_parquet)
        )
    else:
        logger.warning(
            "EPS weekly-history parquet not found at %s — Layer 2 earnings "
            "revision input is unwired; refresh with "
            "scripts/fetch_regime_engine_v1_data.py --fetch eps "
            "(operator-assisted; requires an S&P workbook).",
            eps_weekly_history_parquet,
        )
    return series_dict


def _load_pmi_manufacturing_series(pmi_path: Path) -> pd.Series | None:
    history_path = pmi_path.with_name("us_ism_pmi_history.parquet")
    latest_path = pmi_path.with_name("us_ism_pmi.parquet")
    candidates = [path for path in (history_path, latest_path) if path.exists()]
    if pmi_path.exists() and pmi_path not in candidates:
        if pmi_path.suffix in {".parquet", ".pq"}:
            candidates.append(pmi_path)
        else:
            logger.warning("pmi_path %s is not a parquet file; ignoring.", pmi_path)
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
    pmi_df = cow_safe_assign(
        pmi_df,
        {
            "release_date_local": release_timestamp.dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize()
        },
    )
    pmi_df = pmi_df.drop_duplicates(subset=["release_date_local"], keep="last")
    return (
        pmi_df.set_index("release_date_local")["value"]
        .astype(float)
        .sort_index()
        .rename("pmi_manufacturing")
    )


# Symbols loaded into MarketContext.cross_asset_closes by V2 runners.
# This is broader than the network-fragility universe because §2B and §2C
# also read their required ETF inputs from cross_asset_closes.
RUNNER_CROSS_ASSET_SYMBOLS: list[str] = list(
    dict.fromkeys(
        [
            *CROSS_ASSET_SYMBOLS,
            *CREDIT_FUNDING_CROSS_ASSET_KEYS,
            *INFLATION_GROWTH_CROSS_ASSET_KEYS,
        ]
    )
)
