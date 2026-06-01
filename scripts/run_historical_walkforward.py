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

from regime_detection.calendar import nyse_calendar  # noqa: E402
from regime_detection.dependency_payload_contracts import (  # noqa: E402
    dependency_payload_contracts_report,
)
from regime_detection.engine import RegimeEngine  # noqa: E402
from regime_detection.loaders import load_event_calendar  # noqa: E402
from regime_detection.rule_provenance import rule_provenance_payload  # noqa: E402
from regime_detection.shadow_storage import (  # noqa: E402
    ensure_shadow_layout,
    insert_run_row,
    open_shadow_db,
    update_run_row_failure,
    update_run_row_success,
    utc_iso_now,
    write_archived_inputs,
)
from regime_detection.versioning import (
    engine_version as resolved_engine_version,
)  # noqa: E402
from regime_data_fetch.cli_common import parse_date  # noqa: E402
from regime_data_fetch.materialization import materialize_if_requested  # noqa: E402
from scripts.run_shadow_regime import (  # noqa: E402
    _close_series_by_symbol,
    _constituent_ohlcv_from_daily,
    _default_pit_intervals_from_daily,
    _load_pit_intervals,
    _load_v2_macro_series,
    _load_v2_daily_ohlcv,
)
from regime_detection.fragility_universe import SECTOR_ETFS  # noqa: E402
from scripts._v2_calibration_helpers import (  # noqa: E402
    RUNNER_CROSS_ASSET_SYMBOLS,
    apply_manifest_input_defaults,
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
    out = df.assign(date=pd.to_datetime(df["date"]).dt.date)
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


def _sessions_between(start_date: date, end_date: date) -> list[date]:
    schedule = nyse_calendar().schedule(start_date=start_date, end_date=end_date)
    return list(schedule.index.date)


def _write_output_json(output_path: Path, payload_json: str) -> None:
    output_path.write_text(payload_json + "\n", encoding="utf-8")


def _transition_data_quality_status(transition_risk: Any) -> str | None:
    data_quality = getattr(transition_risk, "data_quality", None)
    if data_quality is None:
        return None
    if isinstance(data_quality, dict):
        status = data_quality.get("status")
    else:
        status = getattr(data_quality, "status", None)
    return None if status is None else str(status)


def _transition_evidence_value(transition_risk: Any, key: str) -> Any:
    evidence = getattr(transition_risk, "evidence", None)
    if evidence is None:
        return None
    if isinstance(evidence, dict):
        return evidence.get(key)
    if hasattr(evidence, "get"):
        return evidence.get(key)
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


def _v2_dependency_payload_contracts_summary_cell() -> str:
    """JSON cell documenting which payload shapes crossed V2 axis edges."""

    return json.dumps(dependency_payload_contracts_report(), sort_keys=True)


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
            counts = success_df[col].value_counts().to_dict()
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


def build_v2_classify_kwargs(
    *,
    v2_slice: pd.DataFrame | None,
    pit_intervals: pd.DataFrame | None,
    macro_series: dict[str, pd.Series] | None,
) -> dict[str, Any]:
    """Build the V2 classify kwargs from an as-of v2 daily-OHLCV slice.

    Shared by the walk-forward runner and ``run_walkforward_replay_check`` so a
    replay reconstructs the V2 inputs (sector/cross-asset closes, PIT membership,
    constituent OHLCV, macro) byte-identically from the archived ``v2_daily`` slice
    (F-001). Returns an empty dict on the V1-only path (``v2_slice is None`` or empty).
    """
    # CR-009: an empty (non-None) slice means the as-of precedes the first v2_daily row.
    # Guarding only ``is None`` would build full V2 kwargs over an empty frame, raising in
    # _close_series_by_symbol (status=failure) instead of degrading to the V1 path.
    if v2_slice is None or v2_slice.empty:
        return {}
    session_pit_intervals = pit_intervals
    if session_pit_intervals is None:
        session_pit_intervals = _default_pit_intervals_from_daily(v2_slice)
    kwargs: dict[str, Any] = {
        "sector_etf_closes": _close_series_by_symbol(v2_slice, SECTOR_ETFS),
        "cross_asset_closes": _close_series_by_symbol(
            v2_slice, RUNNER_CROSS_ASSET_SYMBOLS
        ),
        "pit_constituent_intervals": session_pit_intervals,
        "constituent_ohlcv": _constituent_ohlcv_from_daily(
            v2_slice, session_pit_intervals
        ),
    }
    if macro_series is not None:
        kwargs["macro_series"] = macro_series
    return kwargs


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
    macro_parquet_path: Path | None = None,
    pmi_path: Path | None = None,
    cpi_nowcast_parquet_path: Path | None = None,
    aggregate_forward_eps_weekly_history_parquet_path: Path | None = None,
    allow_missing_event_calendar: bool = False,
) -> dict[str, Any]:
    market_data = _normalize_market_data(market_data_path)
    event_df = _normalize_event_calendar(
        event_calendar_path,
        allow_missing_event_calendar=allow_missing_event_calendar,
    )
    v2_daily = _load_v2_daily_ohlcv(v2_daily_ohlcv_path)
    pit_intervals = _load_pit_intervals(pit_constituent_intervals_path)
    macro_series = _load_v2_macro_series(
        macro_parquet_path=macro_parquet_path,
        pmi_path=pmi_path,
        cpi_nowcast_parquet_path=cpi_nowcast_parquet_path,
        aggregate_forward_eps_weekly_history_parquet_path=(
            aggregate_forward_eps_weekly_history_parquet_path
        ),
    )
    sessions = _sessions_between(start_date, end_date)
    engine = RegimeEngine(config_path=config_path)
    if not sessions:
        raise ValueError(
            f"No NYSE trading sessions found in requested window: {start_date.isoformat()}..{end_date.isoformat()}"
        )
    engine_version = resolved_engine_version()
    config_version = engine.config.config_version

    paths = ensure_shadow_layout(
        output_root, db_name="regime_walkforward.db", include_reports=True
    )
    conn = open_shadow_db(paths["db"])
    try:
        summary_rows: list[dict[str, Any]] = []
        for as_of_date in sessions:
            run_timestamp = utc_iso_now()
            archive_dir = paths["input_archives"] / as_of_date.isoformat()
            market_slice = (
                market_data[market_data["date"] <= as_of_date]
                .copy()
                .reset_index(drop=True)
            )
            v2_slice: pd.DataFrame | None = None
            if v2_daily is not None:
                v2_slice = (
                    v2_daily[v2_daily["date"] <= as_of_date]
                    .copy()
                    .reset_index(drop=True)
                )
            write_archived_inputs(
                archive_dir=archive_dir,
                market_slice=market_slice,
                event_df=event_df,
                # F-003: archive the same macro_series passed to classify()
                # (consumed only on the V2 path), so V2 macro-dependent labels
                # are reproducible from the archive. Mirrors run_shadow_regime.py.
                macro_series=(macro_series if v2_slice is not None else None),
                # F-001: archive the as-of v2 daily-OHLCV slice the runner derives
                # its V2 inputs (sector/cross-asset/PIT/constituent) from, so a
                # walk-forward replay can recompute the V2 axes byte-identically.
                v2_daily_slice=v2_slice,
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
                v2_kwargs = build_v2_classify_kwargs(
                    v2_slice=v2_slice,
                    pit_intervals=pit_intervals,
                    macro_series=macro_series,
                )
                output = engine.classify(
                    as_of_date=as_of_date,
                    market_data=market_slice,
                    event_calendar=event_df,
                    **v2_kwargs,
                )
                output_path = paths["outputs"] / f"{as_of_date.isoformat()}.json"
                _write_output_json(output_path, output.model_dump_json(indent=2))
                # F-019 / §5: the immutable per-date output JSON is a pure RegimeOutput
                # (engine_version/config_version/as_of_date only). Write a sibling
                # provenance sidecar so the per-date output artifact preserves all five
                # mandated fields (adds run_timestamp + input_archive_path) without
                # polluting the classification payload.
                _write_output_json(
                    paths["outputs"] / f"{as_of_date.isoformat()}.provenance.json",
                    json.dumps(
                        {
                            "as_of_date": as_of_date.isoformat(),
                            "engine_version": output.engine_version,
                            "config_version": output.config_version,
                            "run_timestamp": run_timestamp,
                            "input_archive_path": str(archive_dir),
                            "output_path": str(output_path),
                        },
                        indent=2,
                    ),
                )
                update_run_row_success(
                    conn=conn,
                    as_of_date=as_of_date,
                    engine_version=engine_version,
                    config_version=config_version,
                    output_path=output_path,
                )
                summary_rows.append(
                    {
                        "as_of_date": as_of_date.isoformat(),
                        "run_timestamp": run_timestamp,  # F-019 / §5
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
                        "v2_dependency_payload_contracts": _v2_dependency_payload_contracts_summary_cell(),
                        "classification_coverage": _json_cell(
                            output.classification_coverage.model_dump(mode="json")
                            if output.classification_coverage is not None
                            else None
                        ),
                        "rule_provenance": _json_cell(
                            rule_provenance_payload(config=engine.config)
                        ),
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
                update_run_row_failure(
                    conn=conn,
                    as_of_date=as_of_date,
                    engine_version=engine_version,
                    config_version=config_version,
                    failure_reason=failure_reason,
                )
                summary_rows.append(
                    {
                        "as_of_date": as_of_date.isoformat(),
                        "run_timestamp": run_timestamp,  # F-019 / §5
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
                        "v2_dependency_payload_contracts": None,
                        "classification_coverage": None,
                        "rule_provenance": None,
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
    parser.add_argument("--start-date", required=True, type=parse_date)
    parser.add_argument("--end-date", required=True, type=parse_date)
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
        macro_parquet_path=args.macro_parquet,
        pmi_path=args.pmi_path,
        cpi_nowcast_parquet_path=args.cpi_nowcast_parquet,
        aggregate_forward_eps_weekly_history_parquet_path=(
            args.aggregate_forward_eps_weekly_history_parquet
        ),
        allow_missing_event_calendar=args.allow_missing_event_calendar,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
