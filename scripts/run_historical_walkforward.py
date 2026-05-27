#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from regime_detection.calendar import nyse_calendar  # noqa: E402
from regime_detection.engine import RegimeEngine  # noqa: E402
from regime_detection.loaders import load_event_calendar  # noqa: E402
from regime_detection.shadow_storage import event_rows_for_yaml  # noqa: E402
from regime_detection.versioning import (
    engine_version as resolved_engine_version,
)  # noqa: E402
from regime_data_fetch.materialization import materialize_if_requested  # noqa: E402
from scripts import run_shadow_regime as _run_shadow_regime  # noqa: E402
from regime_detection.fragility_universe import (
    CROSS_ASSET_SYMBOLS,
    SECTOR_ETFS,
)  # noqa: E402
from scripts._v2_calibration_helpers import (  # noqa: E402
    apply_manifest_input_defaults,
)

RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY,
    run_timestamp TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    config_version TEXT NOT NULL,
    status TEXT NOT NULL,
    failure_reason TEXT,
    input_archive_path TEXT NOT NULL,
    output_path TEXT,
    output_sha256 TEXT,
    UNIQUE (as_of_date, engine_version, config_version)
)
"""


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_v2_daily_ohlcv(path: Path | None) -> pd.DataFrame | None:
    shadow_module: Any = _run_shadow_regime
    return cast(pd.DataFrame | None, shadow_module._load_v2_daily_ohlcv(path))


def _load_pit_intervals(path: Path | None) -> pd.DataFrame | None:
    shadow_module: Any = _run_shadow_regime
    return cast(pd.DataFrame | None, shadow_module._load_pit_intervals(path))


def _default_pit_intervals_from_daily(daily_ohlcv: pd.DataFrame) -> pd.DataFrame:
    shadow_module: Any = _run_shadow_regime
    return cast(
        pd.DataFrame,
        shadow_module._default_pit_intervals_from_daily(daily_ohlcv),
    )


def _close_series_by_symbol(
    frame: pd.DataFrame, symbols: tuple[str, ...]
) -> dict[str, pd.Series]:
    shadow_module: Any = _run_shadow_regime
    return cast(
        dict[str, pd.Series], shadow_module._close_series_by_symbol(frame, symbols)
    )


def _constituent_ohlcv_from_daily(
    daily_ohlcv: pd.DataFrame,
    pit_intervals: pd.DataFrame | None,
) -> dict[str, pd.DataFrame] | None:
    shadow_module: Any = _run_shadow_regime
    return cast(
        dict[str, pd.DataFrame] | None,
        shadow_module._constituent_ohlcv_from_daily(daily_ohlcv, pit_intervals),
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_market_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df: pd.DataFrame = pd.read_parquet(path)
    else:
        pandas_module: Any = pd
        df = cast(pd.DataFrame, pandas_module.read_csv(path))
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"market_data missing required columns: {missing}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
    return out[keep].sort_values(["date", "symbol"]).reset_index(drop=True)


def _normalize_event_calendar(
    path: Path | None, *, allow_missing_event_calendar: bool = False
) -> pd.DataFrame | None:
    if path is None:
        if not allow_missing_event_calendar:
            raise ValueError(
                "event_calendar_path is required for historical walk-forward runs; "
                "materialize the manifest event_calendar artifact or use "
                "--allow-missing-event-calendar for debug-only runs."
            )
        return None
    if not path.exists():
        if not allow_missing_event_calendar:
            raise FileNotFoundError(
                f"event_calendar_path does not exist: {path}. "
                "Materialize the manifest event_calendar artifact or use "
                "--allow-missing-event-calendar for debug-only runs."
            )
        return None
    return load_event_calendar(path)


def _ensure_layout(output_root: Path) -> dict[str, Path]:
    paths = {
        "db": output_root / "regime_walkforward.db",
        "outputs": output_root / "outputs",
        "input_archives": output_root / "input_archives",
        "reports": output_root / "reports",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    paths["outputs"].mkdir(parents=True, exist_ok=True)
    paths["input_archives"].mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)
    return paths


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(RUNS_SCHEMA)
    conn.commit()
    return conn


def _sessions_between(start_date: date, end_date: date) -> list[date]:
    calendar: Any = nyse_calendar()
    schedule = cast(
        pd.DataFrame,
        calendar.schedule(start_date=start_date, end_date=end_date),
    )
    schedule_index = pd.DatetimeIndex(pd.Index(schedule.index))
    return [pd.Timestamp(ts).date() for ts in schedule_index.tolist()]


def _write_archived_inputs(
    *,
    archive_dir: Path,
    market_slice: pd.DataFrame,
    event_df: pd.DataFrame | None,
) -> tuple[Path, Path, Path]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    market_path = archive_dir / "market_data.parquet"
    events_path = archive_dir / "events.yaml"
    checksums_path = archive_dir / "checksums.json"

    market_slice.to_parquet(market_path, index=False)
    events_path.write_text(
        yaml.safe_dump({"events": event_rows_for_yaml(event_df)}, sort_keys=False),
        encoding="utf-8",
    )
    checksums_path.write_text(
        json.dumps(
            {
                "market_data.parquet": _sha256_file(market_path),
                "events.yaml": _sha256_file(events_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return market_path, events_path, checksums_path


def _insert_run_row(
    *,
    conn: sqlite3.Connection,
    run_timestamp: str,
    as_of_date: date,
    engine_version: str,
    config_version: str,
    archive_dir: Path,
) -> None:
    conn.execute(
        """
        INSERT INTO runs (
            run_timestamp, as_of_date, engine_version, config_version,
            status, failure_reason, input_archive_path, output_path, output_sha256
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_timestamp,
            as_of_date.isoformat(),
            engine_version,
            config_version,
            "in_progress",
            None,
            str(archive_dir),
            None,
            None,
        ),
    )
    conn.commit()


def _update_run_row_success(
    *,
    conn: sqlite3.Connection,
    as_of_date: date,
    engine_version: str,
    config_version: str,
    output_path: Path,
) -> None:
    conn.execute(
        """
        UPDATE runs
        SET status = ?, output_path = ?, output_sha256 = ?, failure_reason = NULL
        WHERE as_of_date = ? AND engine_version = ? AND config_version = ?
        """,
        (
            "success",
            str(output_path),
            _sha256_file(output_path),
            as_of_date.isoformat(),
            engine_version,
            config_version,
        ),
    )
    conn.commit()


def _update_run_row_failure(
    *,
    conn: sqlite3.Connection,
    as_of_date: date,
    engine_version: str,
    config_version: str,
    failure_reason: str,
) -> None:
    conn.execute(
        """
        UPDATE runs
        SET status = ?, failure_reason = ?
        WHERE as_of_date = ? AND engine_version = ? AND config_version = ?
        """,
        (
            "failure",
            failure_reason,
            as_of_date.isoformat(),
            engine_version,
            config_version,
        ),
    )
    conn.commit()


def _write_output_json(output_path: Path, payload_json: str) -> None:
    output_path.write_text(payload_json + "\n", encoding="utf-8")


def _transition_data_quality_status(transition_risk: Any) -> str | None:
    data_quality = getattr(transition_risk, "data_quality", None)
    if data_quality is None:
        return None
    if isinstance(data_quality, Mapping):
        mapping = cast(Mapping[str, Any], data_quality)
        status = mapping.get("status")
    else:
        status = getattr(data_quality, "status", None)
    return None if status is None else str(status)


def _transition_evidence_value(transition_risk: Any, key: str) -> Any:
    evidence = getattr(transition_risk, "evidence", None)
    if evidence is None:
        return None
    if isinstance(evidence, Mapping):
        mapping = cast(Mapping[str, Any], evidence)
        return mapping.get(key)
    evidence_getter = getattr(evidence, "get", None)
    if callable(evidence_getter):
        return evidence_getter(key)
    return getattr(evidence, key, None)


def _json_cell(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _event_calendar_summary_cells(output: Any) -> dict[str, Any]:
    event_calendar = output.structural_causal_state.event_calendar
    return {
        "event_calendar_primary_label": event_calendar.primary_label,
        "event_calendar_matching_labels": _json_cell(
            list(event_calendar.matching_labels)
        ),
    }


def _build_report_markdown(
    summary_df: pd.DataFrame, *, start_date: date, end_date: date
) -> str:
    lines = [
        "# Historical Walk-Forward Report",
        "",
        f"- start_date: `{start_date.isoformat()}`",
        f"- end_date: `{end_date.isoformat()}`",
        f"- session_count: `{len(summary_df)}`",
        f"- success_count: `{int(summary_df['status'].eq('success').sum())}`",
        f"- failure_count: `{int(summary_df['status'].eq('failure').sum())}`",
        "",
        "## Notes",
        "",
        "- This runner produces the archival and classification artifacts for the historical walk-forward gate.",
        "- Strategy-vs-baseline metrics are not computed in this runner; those belong to the later walk-forward report/comparison layer.",
    ]
    success_df = summary_df[summary_df["status"] == "success"].copy()
    if not success_df.empty:
        for col in [
            "trend_direction_active",
            "trend_character_active",
            "volatility_state_active",
            "breadth_state_active",
            "transition_risk_state",
        ]:
            counts = cast(
                dict[str, int],
                cast(Any, success_df[col].value_counts()).to_dict(),
            )
            lines.extend(
                [
                    "",
                    f"## {col}",
                    "",
                    "```json",
                    json.dumps(counts, indent=2, sort_keys=True),
                    "```",
                ]
            )
    return "\n".join(lines) + "\n"


def run_walkforward(
    *,
    market_data_path: Path,
    output_root: Path,
    start_date: date,
    end_date: date,
    event_calendar_path: Path | None = None,
    config_path: Path | None = None,
    v2_daily_ohlcv_path: Path | None = None,
    pit_constituent_intervals_path: Path | None = None,
    allow_missing_event_calendar: bool = False,
) -> dict[str, Any]:
    market_data = _normalize_market_data(market_data_path)
    event_df = _normalize_event_calendar(
        event_calendar_path,
        allow_missing_event_calendar=allow_missing_event_calendar,
    )
    v2_daily = _load_v2_daily_ohlcv(v2_daily_ohlcv_path)
    pit_intervals = _load_pit_intervals(pit_constituent_intervals_path)
    sessions = _sessions_between(start_date, end_date)
    engine = RegimeEngine(config_path=config_path)
    if not sessions:
        raise ValueError(
            f"No NYSE trading sessions found in requested window: {start_date.isoformat()}..{end_date.isoformat()}"
        )
    engine_version = resolved_engine_version()
    config_version = engine.config.config_version

    paths = _ensure_layout(output_root)
    conn = _open_db(paths["db"])
    try:
        summary_rows: list[dict[str, Any]] = []
        for as_of_date in sessions:
            run_timestamp = _utc_iso_now()
            archive_dir = paths["input_archives"] / as_of_date.isoformat()
            market_slice = (
                market_data[market_data["date"] <= as_of_date]
                .copy()
                .reset_index(drop=True)
            )
            _write_archived_inputs(
                archive_dir=archive_dir,
                market_slice=market_slice,
                event_df=event_df,
            )
            _insert_run_row(
                conn=conn,
                run_timestamp=run_timestamp,
                as_of_date=as_of_date,
                engine_version=engine_version,
                config_version=config_version,
                archive_dir=archive_dir,
            )

            try:
                v2_kwargs: dict[str, Any] = {}
                if v2_daily is not None:
                    v2_slice = (
                        v2_daily[v2_daily["date"] <= as_of_date]
                        .copy()
                        .reset_index(drop=True)
                    )
                    session_pit_intervals = pit_intervals
                    if session_pit_intervals is None:
                        session_pit_intervals = _default_pit_intervals_from_daily(
                            v2_slice
                        )
                    v2_kwargs["sector_etf_closes"] = _close_series_by_symbol(
                        v2_slice, SECTOR_ETFS
                    )
                    v2_kwargs["cross_asset_closes"] = _close_series_by_symbol(
                        v2_slice, CROSS_ASSET_SYMBOLS
                    )
                    v2_kwargs["pit_constituent_intervals"] = session_pit_intervals
                    v2_kwargs["constituent_ohlcv"] = _constituent_ohlcv_from_daily(
                        v2_slice,
                        session_pit_intervals,
                    )
                output = engine.classify(
                    as_of_date=as_of_date,
                    market_data=market_slice,
                    event_calendar=event_df,
                    **v2_kwargs,
                )
                output_path = paths["outputs"] / f"{as_of_date.isoformat()}.json"
                _write_output_json(output_path, output.model_dump_json(indent=2))
                _update_run_row_success(
                    conn=conn,
                    as_of_date=as_of_date,
                    engine_version=engine_version,
                    config_version=config_version,
                    output_path=output_path,
                )
                summary_rows.append(
                    {
                        "as_of_date": as_of_date.isoformat(),
                        "status": "success",
                        "failure_reason": None,
                        "engine_version": output.engine_version,
                        "config_version": output.config_version,
                        "output_path": str(output_path),
                        "input_archive_path": str(archive_dir),
                        "trend_direction_active": output.trend_direction.active_label,
                        "trend_character_active": output.trend_character.active_label,
                        "volatility_state_active": output.volatility_state.active_label,
                        "breadth_state_active": output.breadth_state.active_label,
                        **_event_calendar_summary_cells(output),
                        "transition_risk_state": output.transition_risk.state,
                        "transition_risk_score": output.transition_risk.score,
                        "transition_risk_primary_drivers": _json_cell(
                            output.transition_risk.primary_drivers
                        ),
                        "transition_risk_triggered_rules": _json_cell(
                            output.transition_risk.triggered_rules
                        ),
                        "transition_risk_data_quality_status": _transition_data_quality_status(
                            output.transition_risk
                        ),
                        "transition_risk_axis_switch_count": _transition_evidence_value(
                            output.transition_risk, "axis_switch_count"
                        ),
                        "transition_risk_recent_axis_switch_count": _transition_evidence_value(
                            output.transition_risk, "recent_axis_switch_count"
                        ),
                    }
                )
            except Exception as exc:
                failure_reason = str(exc)
                _update_run_row_failure(
                    conn=conn,
                    as_of_date=as_of_date,
                    engine_version=engine_version,
                    config_version=config_version,
                    failure_reason=failure_reason,
                )
                summary_rows.append(
                    {
                        "as_of_date": as_of_date.isoformat(),
                        "status": "failure",
                        "failure_reason": failure_reason,
                        "engine_version": engine_version,
                        "config_version": config_version,
                        "output_path": None,
                        "input_archive_path": str(archive_dir),
                        "trend_direction_active": None,
                        "trend_character_active": None,
                        "volatility_state_active": None,
                        "breadth_state_active": None,
                        "event_calendar_primary_label": None,
                        "event_calendar_matching_labels": None,
                        "transition_risk_state": None,
                        "transition_risk_score": None,
                        "transition_risk_primary_drivers": None,
                        "transition_risk_triggered_rules": None,
                        "transition_risk_data_quality_status": None,
                        "transition_risk_axis_switch_count": None,
                        "transition_risk_recent_axis_switch_count": None,
                    }
                )

        summary_df = pd.DataFrame(summary_rows)
        summary_path = paths["reports"] / "walkforward_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        report_path = paths["reports"] / "walkforward_report.md"
        report_path.write_text(
            _build_report_markdown(
                summary_df, start_date=start_date, end_date=end_date
            ),
            encoding="utf-8",
        )

        return {
            "session_count": len(sessions),
            "success_count": int(summary_df["status"].eq("success").sum()),
            "failure_count": int(summary_df["status"].eq("failure").sum()),
            "db_path": str(paths["db"]),
            "summary_path": str(summary_path),
            "report_path": str(report_path),
            "engine_version": engine_version,
            "config_version": config_version,
        }
    finally:
        conn.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run historical walk-forward classification."
    )
    parser.add_argument(
        "--market-data",
        required=True,
        type=Path,
        help="Path to parquet/csv market data.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory for walk-forward artifacts.",
    )
    parser.add_argument("--start-date", required=True, type=date.fromisoformat)
    parser.add_argument("--end-date", required=True, type=date.fromisoformat)
    parser.add_argument(
        "--allow-missing-event-calendar",
        action="store_true",
        help=(
            "Debug-only: run without scheduled event-calendar rows. "
            "Deterministic expiry/earnings labels still compute."
        ),
    )
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--v2-daily-ohlcv", type=Path, default=None)
    parser.add_argument("--pit-constituent-intervals", type=Path, default=None)
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
        args, args.data_root, fields=frozenset({"event_calendar"})
    )
    return args


def main() -> int:
    args = _parse_args()
    materialize_if_requested(
        manifest_path=args.manifest,
        local_root=args.data_root,
        repo_root=REPO_ROOT,
        store_root=args.artifact_store,
        required_for="historical_walkforward",
    )
    result = run_walkforward(
        market_data_path=args.market_data,
        output_root=args.output_root,
        start_date=args.start_date,
        end_date=args.end_date,
        event_calendar_path=args.event_calendar,
        config_path=args.config_path,
        v2_daily_ohlcv_path=args.v2_daily_ohlcv,
        pit_constituent_intervals_path=args.pit_constituent_intervals,
        allow_missing_event_calendar=args.allow_missing_event_calendar,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
