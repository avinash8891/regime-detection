#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from regime_detection.calendar import require_nyse_trading_day
from regime_detection.config import RegimeConfig, default_config_text
from regime_detection.dependency_payload_contracts import (
    dependency_payload_contracts_report as _v2_dependency_payload_contracts,
)
from regime_detection.engine import RegimeEngine
from regime_detection.fragility_universe import SECTOR_ETFS
from regime_detection.loaders import load_event_calendar
from regime_detection.rule_provenance import rule_provenance_payload
from regime_detection.shadow_storage import (
    ensure_shadow_layout,
    insert_incident,
    insert_run_row,
    latest_config_sha256,
    load_archived_event_calendar,
    load_archived_macro_series,
    load_archived_market_data,
    open_shadow_db,
    update_run_row_failure,
    update_run_row_success,
    utc_iso_now,
    write_archived_inputs,
)
from regime_detection.versioning import engine_version as resolved_engine_version
from regime_data_fetch.artifact_store import sha256_bytes
from regime_data_fetch.materialization import materialize_if_requested
from regime_shared.pandas_compat import cow_safe_assign
from scripts._v2_calibration_helpers import (
    RUNNER_CROSS_ASSET_SYMBOLS,
    apply_manifest_input_defaults,
    load_macro_series,
    register_manifest_input_args,
)


def _normalize_market_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"market_data missing required columns: {missing}")
    out = cow_safe_assign(
        df,
        {"date": pd.to_datetime(df["date"]).dt.date},
    )
    keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
    return out[keep].sort_values(["date", "symbol"]).reset_index(drop=True)


def _normalize_event_calendar(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        raise ValueError(
            "event_calendar_path is required for shadow runs; materialize the "
            "manifest event_calendar artifact before running."
        )
    if not path.exists():
        raise FileNotFoundError(
            f"event_calendar_path does not exist: {path}. "
            "Materialize the manifest event_calendar artifact before running."
        )
    return load_event_calendar(path)


def _load_v2_daily_ohlcv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return _normalize_market_data(path)


def _close_series_by_symbol(
    frame: pd.DataFrame, symbols: tuple[str, ...]
) -> dict[str, pd.Series]:
    present = set(frame["symbol"].unique())
    missing = sorted(set(symbols) - present)
    if missing:
        raise ValueError(f"v2_daily_ohlcv missing required symbols: {missing}")
    out: dict[str, pd.Series] = {}
    for symbol in symbols:
        sub = frame[frame["symbol"] == symbol].sort_values("date")
        out[symbol] = pd.Series(
            sub["close"].astype(float).to_numpy(),
            index=pd.to_datetime(sub["date"]),
            name=symbol,
        )
    return out


def _load_pit_intervals(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    required = {"ticker", "start_date", "end_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"pit_constituent_intervals missing required columns: {missing}"
        )
    parsed_end_date = pd.to_datetime(frame["end_date"], errors="coerce").dt.date
    out = cow_safe_assign(
        frame,
        {
            "start_date": pd.to_datetime(frame["start_date"]).dt.date,
            "end_date": parsed_end_date.where(parsed_end_date.notna(), None),
        },
    )
    return out


def _constituent_ohlcv_from_daily(
    daily_ohlcv: pd.DataFrame,
    pit_intervals: pd.DataFrame | None,
) -> dict[str, pd.DataFrame] | None:
    if pit_intervals is None:
        return None
    result: dict[str, pd.DataFrame] = {}
    for ticker in sorted(set(pit_intervals["ticker"].astype(str))):
        sub = daily_ohlcv[daily_ohlcv["symbol"] == ticker].sort_values("date")
        if sub.empty:
            continue
        idx = pd.to_datetime(sub["date"])
        result[ticker] = pd.DataFrame(
            {
                "open": sub["open"].astype(float).to_numpy(),
                "high": sub["high"].astype(float).to_numpy(),
                "low": sub["low"].astype(float).to_numpy(),
                "close": sub["close"].astype(float).to_numpy(),
                "volume": sub["volume"].astype("int64").to_numpy(),
                "adjusted_close": sub["close"].astype(float).to_numpy(),
            },
            index=idx,
        )
    return result


def _default_pit_intervals_from_daily(daily_ohlcv: pd.DataFrame) -> pd.DataFrame:
    start_dates = (
        daily_ohlcv[daily_ohlcv["symbol"].isin(SECTOR_ETFS)]
        .groupby("symbol")["date"]
        .min()
    )
    return pd.DataFrame(
        {
            "ticker": list(start_dates.index),
            "start_date": list(start_dates.values),
            "end_date": [None] * len(start_dates),
        }
    )


def _load_v2_macro_series(
    *,
    macro_parquet_path: Path | None,
    pmi_path: Path | None,
    cpi_nowcast_parquet_path: Path | None,
    aggregate_forward_eps_weekly_history_parquet_path: Path | None,
) -> dict[str, pd.Series] | None:
    if macro_parquet_path is None:
        return None
    return load_macro_series(
        macro_parquet_path,
        pmi_path,
        cpi_nowcast_parquet=cpi_nowcast_parquet_path,
        eps_weekly_history_parquet=aggregate_forward_eps_weekly_history_parquet_path,
    )


def _write_output_json(output_path: Path, payload_json: str) -> None:
    output_path.write_text(payload_json + "\n", encoding="utf-8")


def _shadow_output_json(output: object, *, config: RegimeConfig | None = None) -> str:
    """Serialize a shadow output with diagnostic dependency contracts."""

    if not hasattr(output, "model_dump"):
        raise TypeError("shadow output must provide model_dump")
    payload = output.model_dump(mode="json")
    payload["v2_dependency_payload_contracts"] = _v2_dependency_payload_contracts()
    payload["rule_provenance"] = rule_provenance_payload(config=config)
    return json.dumps(payload, indent=2, sort_keys=True)


def run_shadow(
    *,
    as_of_date: date,
    market_data_path: Path,
    output_root: Path,
    event_calendar_path: Path | None = None,
    config_path: Path | None = None,
    v2_daily_ohlcv_path: Path | None = None,
    pit_constituent_intervals_path: Path | None = None,
    macro_parquet_path: Path | None = None,
    pmi_path: Path | None = None,
    cpi_nowcast_parquet_path: Path | None = None,
    aggregate_forward_eps_weekly_history_parquet_path: Path | None = None,
) -> dict[str, Any]:
    require_nyse_trading_day(as_of_date)
    event_df = _normalize_event_calendar(event_calendar_path)
    paths = ensure_shadow_layout(output_root)
    conn = open_shadow_db(paths["db"])
    try:
        market_data = _normalize_market_data(market_data_path)
        market_slice = (
            market_data[market_data["date"] <= as_of_date].copy().reset_index(drop=True)
        )
        v2_daily = _load_v2_daily_ohlcv(v2_daily_ohlcv_path)
        v2_slice = (
            None
            if v2_daily is None
            else v2_daily[v2_daily["date"] <= as_of_date].copy().reset_index(drop=True)
        )
        pit_intervals = _load_pit_intervals(pit_constituent_intervals_path)
        v2_kwargs: dict[str, Any] = {}
        if v2_slice is not None:
            macro_series = _load_v2_macro_series(
                macro_parquet_path=macro_parquet_path,
                pmi_path=pmi_path,
                cpi_nowcast_parquet_path=cpi_nowcast_parquet_path,
                aggregate_forward_eps_weekly_history_parquet_path=(
                    aggregate_forward_eps_weekly_history_parquet_path
                ),
            )
            if pit_intervals is None:
                pit_intervals = _default_pit_intervals_from_daily(v2_slice)
            v2_kwargs["sector_etf_closes"] = _close_series_by_symbol(
                v2_slice, SECTOR_ETFS
            )
            v2_kwargs["cross_asset_closes"] = _close_series_by_symbol(
                v2_slice, RUNNER_CROSS_ASSET_SYMBOLS
            )
            v2_kwargs["pit_constituent_intervals"] = pit_intervals
            v2_kwargs["constituent_ohlcv"] = _constituent_ohlcv_from_daily(
                v2_slice,
                pit_intervals,
            )
            if macro_series is not None:
                v2_kwargs["macro_series"] = macro_series

        engine = RegimeEngine(config_path=config_path)
        engine_version = resolved_engine_version()
        config_version = engine.config.config_version
        # F-018: hash the resolved config *content*, not just the coarse config_version
        # Literal. Two different configs can share one Literal, so a same-Literal
        # content change would otherwise span a single qualification window. Capture
        # the hash on every run and, when it differs from the window's prior hash,
        # insert a breaking incident that resets qualification from this session.
        config_text = (
            config_path.read_text(encoding="utf-8")
            if config_path is not None
            else default_config_text()
        )
        config_sha256 = sha256_bytes(config_text.encode("utf-8"))
        prior_config_sha256 = latest_config_sha256(
            conn=conn,
            engine_version=engine_version,
            config_version=config_version,
        )
        if prior_config_sha256 is not None and prior_config_sha256 != config_sha256:
            insert_incident(
                conn=conn,
                incident_date=as_of_date,
                description=(
                    "Config content changed mid-window for "
                    f"{engine_version}/{config_version}: "
                    f"{prior_config_sha256[:12]} -> {config_sha256[:12]}; "
                    "qualification window reset"
                ),
                resolution=None,
                breaks_qualification=True,
            )
        run_timestamp = utc_iso_now()
        archive_dir = paths["input_archives"] / as_of_date.isoformat()
        archived_market_path, archived_events_path, _ = write_archived_inputs(
            archive_dir=archive_dir,
            market_slice=market_slice,
            event_df=event_df,
            macro_series=v2_kwargs.get("macro_series"),
        )
        insert_run_row(
            conn=conn,
            run_timestamp=run_timestamp,
            as_of_date=as_of_date,
            engine_version=engine_version,
            config_version=config_version,
            archive_dir=archive_dir,
            config_sha256=config_sha256,
        )

        try:
            archived_market = load_archived_market_data(archived_market_path)
            archived_events = load_archived_event_calendar(archived_events_path)
            archived_macro = load_archived_macro_series(
                archive_dir / "macro_series.parquet"
            )
            if archived_macro is not None:
                v2_kwargs["macro_series"] = archived_macro
            output = engine.classify(
                as_of_date=as_of_date,
                market_data=archived_market,
                event_calendar=archived_events,
                **v2_kwargs,
            )
            output_path = paths["outputs"] / f"{as_of_date.isoformat()}.json"
            _write_output_json(
                output_path, _shadow_output_json(output, config=engine.config)
            )
            update_run_row_success(
                conn=conn,
                as_of_date=as_of_date,
                engine_version=engine_version,
                config_version=config_version,
                output_path=output_path,
            )
            return {
                "status": "success",
                "as_of_date": as_of_date.isoformat(),
                "db_path": str(paths["db"]),
                "output_path": str(output_path),
                "input_archive_path": str(archive_dir),
                "engine_version": output.engine_version,
                "config_version": output.config_version,
            }
        except Exception as exc:
            failure_reason = str(exc)
            update_run_row_failure(
                conn=conn,
                as_of_date=as_of_date,
                engine_version=engine_version,
                config_version=config_version,
                failure_reason=failure_reason,
            )
            return {
                "status": "failure",
                "as_of_date": as_of_date.isoformat(),
                "db_path": str(paths["db"]),
                "input_archive_path": str(archive_dir),
                "engine_version": engine_version,
                "config_version": config_version,
                "failure_reason": failure_reason,
            }
    finally:
        conn.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the V1 forward shadow classification for one NYSE session."
    )
    parser.add_argument("--as-of-date", required=True, type=date.fromisoformat)
    parser.add_argument("--market-data", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--v2-daily-ohlcv", type=Path, default=None)
    parser.add_argument("--pit-constituent-intervals", type=Path, default=None)
    parser.add_argument("--macro-parquet", type=Path, default=None)
    register_manifest_input_args(parser, include_required_paths=False)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional artifact manifest to materialize before running.",
    )
    parser.add_argument(
        "--artifact-store",
        default=None,
        help="Optional artifact-store root override for --manifest.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="Local data/raw root used for manifest materialization.",
    )
    args = parser.parse_args()
    args.event_calendar = None
    apply_manifest_input_defaults(
        args,
        args.data_root,
        fields=frozenset(
            {
                "event_calendar",
                "macro_parquet",
                "pmi_path",
                "cpi_nowcast_parquet",
                "aggregate_forward_eps_weekly_history_parquet",
            }
        ),
    )
    return args


def main() -> int:
    args = _parse_args()
    materialize_if_requested(
        manifest_path=args.manifest,
        local_root=args.data_root,
        repo_root=REPO_ROOT,
        store_root=args.artifact_store,
        required_for="shadow_regime",
    )
    result = run_shadow(
        as_of_date=args.as_of_date,
        market_data_path=args.market_data,
        output_root=args.output_root,
        event_calendar_path=args.event_calendar,
        config_path=args.config_path,
        v2_daily_ohlcv_path=args.v2_daily_ohlcv,
        pit_constituent_intervals_path=args.pit_constituent_intervals,
        macro_parquet_path=args.macro_parquet,
        pmi_path=args.pmi_path,
        cpi_nowcast_parquet_path=args.cpi_nowcast_parquet,
        aggregate_forward_eps_weekly_history_parquet_path=(
            args.aggregate_forward_eps_weekly_history_parquet
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
