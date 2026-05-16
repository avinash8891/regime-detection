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
sys.path.insert(0, str(REPO_ROOT / "src"))

from regime_detection.calendar import require_nyse_trading_day  # noqa: E402
from regime_detection.engine import RegimeEngine  # noqa: E402
from regime_detection.loaders import load_event_calendar  # noqa: E402
from regime_detection.shadow_storage import (  # noqa: E402
    ensure_shadow_layout,
    insert_run_row,
    load_archived_event_calendar,
    load_archived_market_data,
    open_shadow_db,
    update_run_row_failure,
    update_run_row_success,
    utc_iso_now,
    write_archived_inputs,
)
from regime_detection.versioning import engine_version as resolved_engine_version  # noqa: E402


def _normalize_market_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"market_data missing required columns: {missing}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
    return out[keep].sort_values(["date", "symbol"]).reset_index(drop=True)


def _normalize_event_calendar(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return load_event_calendar(path)


def _write_output_json(output_path: Path, payload_json: str) -> None:
    output_path.write_text(payload_json + "\n", encoding="utf-8")


def run_shadow(
    *,
    as_of_date: date,
    market_data_path: Path,
    output_root: Path,
    event_calendar_path: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    require_nyse_trading_day(as_of_date)
    paths = ensure_shadow_layout(output_root)
    conn = open_shadow_db(paths["db"])
    try:
        market_data = _normalize_market_data(market_data_path)
        event_df = _normalize_event_calendar(event_calendar_path)
        market_slice = market_data[market_data["date"] <= as_of_date].copy().reset_index(drop=True)

        engine = RegimeEngine(config_path=config_path)
        engine_version = resolved_engine_version()
        config_version = engine.config.config_version
        run_timestamp = utc_iso_now()
        archive_dir = paths["input_archives"] / as_of_date.isoformat()
        archived_market_path, archived_events_path, _ = write_archived_inputs(
            archive_dir=archive_dir,
            market_slice=market_slice,
            event_df=event_df,
        )
        insert_run_row(
            conn=conn,
            run_timestamp=run_timestamp,
            as_of_date=as_of_date,
            engine_version=engine_version,
            config_version=config_version,
            archive_dir=archive_dir,
        )

        try:
            archived_market = load_archived_market_data(archived_market_path)
            archived_events = load_archived_event_calendar(archived_events_path)
            output = engine.classify(
                as_of_date=as_of_date,
                market_data=archived_market,
                event_calendar=archived_events,
            )
            output_path = paths["outputs"] / f"{as_of_date.isoformat()}.json"
            _write_output_json(output_path, output.model_dump_json(indent=2))
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
    parser = argparse.ArgumentParser(description="Run the V1 forward shadow classification for one NYSE session.")
    parser.add_argument("--as-of-date", required=True, type=date.fromisoformat)
    parser.add_argument("--market-data", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--event-calendar", type=Path, default=None)
    parser.add_argument("--config-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_shadow(
        as_of_date=args.as_of_date,
        market_data_path=args.market_data,
        output_root=args.output_root,
        event_calendar_path=args.event_calendar,
        config_path=args.config_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
