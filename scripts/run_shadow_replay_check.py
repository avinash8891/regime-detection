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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from regime_detection.engine import RegimeEngine
from regime_detection.fragility_universe import CROSS_ASSET_SYMBOLS, SECTOR_ETFS
from regime_detection.rule_provenance import rule_provenance_payload
from regime_detection.shadow_storage import (
    fetch_run_row,
    insert_replay_check,
    load_archived_event_calendar,
    load_archived_market_data,
    open_shadow_db,
    utc_iso_now,
)
from _v2_calibration_helpers import (
    constituent_ohlcv_from_sector_closes,
    synthetic_pit_intervals_from_sector_closes,
)
from run_shadow_regime import _v2_dependency_payload_contracts


def _close_series_by_symbol(
    frame: pd.DataFrame, symbols: tuple[str, ...]
) -> dict[str, pd.Series]:
    present = set(frame["symbol"].unique())
    if not set(symbols).issubset(present):
        return {}
    return {
        symbol: pd.Series(
            frame[frame["symbol"] == symbol]
            .sort_values("date")["close"]
            .astype(float)
            .to_numpy(),
            index=pd.to_datetime(
                frame[frame["symbol"] == symbol].sort_values("date")["date"]
            ),
            name=symbol,
        )
        for symbol in symbols
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _diff_values(replayed: Any, stored: Any) -> Any:
    if replayed == stored:
        return None
    if isinstance(replayed, dict) and isinstance(stored, dict):
        keys = sorted(set(replayed) | set(stored))
        nested = {}
        for key in keys:
            child = _diff_values(replayed.get(key), stored.get(key))
            if child is not None:
                nested[key] = child
        return nested or None
    if isinstance(replayed, list) and isinstance(stored, list):
        if replayed == stored:
            return None
    return {"replayed": _jsonable(replayed), "stored": _jsonable(stored)}


def run_replay_check(
    *,
    output_root: Path,
    as_of_date: date,
    config_path: Path | None = None,
) -> dict[str, Any]:
    conn = open_shadow_db(output_root / "regime_shadow.db")
    try:
        run_row = fetch_run_row(conn=conn, as_of_date=as_of_date)
        if run_row is None:
            raise ValueError(
                f"No shadow run found for as_of_date={as_of_date.isoformat()}"
            )
        if run_row["status"] != "success":
            raise ValueError(
                f"Shadow run for {as_of_date.isoformat()} is not successful: {run_row['status']}"
            )
        if run_row["output_path"] is None:
            raise ValueError(
                f"Shadow run for {as_of_date.isoformat()} has no output_path"
            )

        archive_dir = Path(run_row["input_archive_path"])
        market_path = archive_dir / "market_data.parquet"
        events_path = archive_dir / "events.yaml"

        market_data = load_archived_market_data(market_path)
        if events_path.exists():
            archived_events = load_archived_event_calendar(events_path)
        else:
            archived_events = None

        engine = RegimeEngine(config_path=config_path)
        sector_etf_closes = _close_series_by_symbol(market_data, SECTOR_ETFS)
        cross_asset_closes = _close_series_by_symbol(market_data, CROSS_ASSET_SYMBOLS)
        v2_kwargs: dict[str, Any] = {}
        if sector_etf_closes and cross_asset_closes:
            v2_kwargs["sector_etf_closes"] = sector_etf_closes
            v2_kwargs["cross_asset_closes"] = cross_asset_closes
            v2_kwargs["pit_constituent_intervals"] = (
                synthetic_pit_intervals_from_sector_closes(sector_etf_closes)
            )
            v2_kwargs["constituent_ohlcv"] = constituent_ohlcv_from_sector_closes(
                sector_etf_closes
            )
        replayed_output = engine.classify(
            as_of_date=as_of_date,
            market_data=market_data,
            event_calendar=archived_events,
            **v2_kwargs,
        )

        replayed_payload = json.loads(replayed_output.model_dump_json(indent=2))
        # Compare against the stored artifact contract as well as the model
        # fields. A replay is not exact if payload semantics drift silently.
        replayed_payload["v2_dependency_payload_contracts"] = (
            _v2_dependency_payload_contracts()
        )
        # Build provenance from the replay engine's active config so non-default
        # --config-path runs match the artifact written by run_shadow_regime.py
        # (which now also threads the active config through). Without this,
        # exact replay diffs would deterministically flag rule_provenance even
        # for otherwise byte-identical outputs.
        replayed_payload["rule_provenance"] = rule_provenance_payload(engine.config)
        stored_payload = json.loads(
            Path(run_row["output_path"]).read_text(encoding="utf-8")
        )
        diff = _diff_values(replayed_payload, stored_payload)
        matches = diff is None

        insert_replay_check(
            conn=conn,
            check_timestamp=utc_iso_now(),
            original_run_id=int(run_row["run_id"]),
            matches=matches,
            diff=diff,
        )
        return {
            "as_of_date": as_of_date.isoformat(),
            "run_id": int(run_row["run_id"]),
            "matches": matches,
            "diff": diff,
        }
    finally:
        conn.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a stored shadow run from archived inputs and record the result."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--as-of-date", required=True, type=date.fromisoformat)
    parser.add_argument("--config-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_replay_check(
        output_root=args.output_root,
        as_of_date=args.as_of_date,
        config_path=args.config_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
