#!/usr/bin/env python3
"""Replay verification for a historical walk-forward batch (F-001).

Recomputes every successful walk-forward date from its ARCHIVED inputs and compares
the result to the stored output, then writes a single ``replay_verification.json``
bound to the frozen engine/config pair. This is the producer the §6 replay gate in
``build_walkforward_report.py`` consumes via ``--replay-results`` — it replaces the
prior hole where the gate trusted an operator-supplied ``all_passed`` JSON that no
repo script produced, and where the only replay implementation pointed at the shadow
DB (``regime_shadow.db``) instead of the walk-forward DB (``regime_walkforward.db``).

Inputs are reconstructed with the SAME ``build_v2_classify_kwargs`` helper the runner
uses, from the archived ``v2_daily`` slice (F-001a), so the V2 axes recompute
byte-identically. A walk-forward run that used explicit (non-default) PIT membership
intervals is out of scope for now — the runner's default is to derive PIT from the
daily frame, which replay reconstructs faithfully.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from regime_detection.engine import RegimeEngine
from regime_detection.shadow_storage import (
    ensure_shadow_layout,
    load_archived_event_calendar,
    load_archived_macro_series,
    load_archived_market_data,
    load_archived_pit_intervals,
    load_archived_v2_daily,
)
from regime_detection.versioning import engine_version as resolved_engine_version
from run_historical_walkforward import build_v2_classify_kwargs
from run_shadow_replay_check import _diff_touches_classification_fields, _diff_values


def _success_runs(db_path: Path) -> list[dict[str, Any]]:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT as_of_date, status, engine_version, config_version, "
            "input_archive_path, output_path FROM runs "
            "WHERE status = 'success' ORDER BY as_of_date"
        ).fetchall()
    return [dict(row) for row in rows]


def _replay_one(
    *, run: dict[str, Any], engine: RegimeEngine, paths: dict[str, Path]
) -> dict[str, Any]:
    as_of = date.fromisoformat(str(run["as_of_date"]))
    # CR-007 / CR-011: a success row must record its archive + output paths. A null is a
    # clean per-run ValueError naming the date, not an opaque TypeError from Path(None)
    # that aborts the whole replay batch before any verdict is written.
    for field in ("input_archive_path", "output_path"):
        if run[field] is None:
            raise ValueError(f"walk-forward run for {as_of.isoformat()} has no {field}")
    # CR-006: resolve the archive dir + stored output from output_root's layout (not the
    # absolute paths the producing host recorded in the DB) so a relocated/copied batch
    # — e.g. produced on one host and verified on CI — still replays.
    archive_dir = paths["input_archives"] / as_of.isoformat()
    output_path = paths["outputs"] / f"{as_of.isoformat()}.json"
    market_slice = load_archived_market_data(archive_dir / "market_data.parquet")
    events_path = archive_dir / "events.yaml"
    archived_events = (
        load_archived_event_calendar(events_path) if events_path.exists() else None
    )
    v2_slice = load_archived_v2_daily(archive_dir / "v2_daily.parquet")
    archived_macro = load_archived_macro_series(archive_dir / "macro_series.parquet")
    archived_pit = load_archived_pit_intervals(
        archive_dir / "pit_constituent_intervals.parquet"
    )

    v2_kwargs = build_v2_classify_kwargs(
        v2_slice=v2_slice,
        # CR-004: replay with the run's archived explicit PIT membership when present;
        # else None, so build_v2_classify_kwargs derives the default-from-daily universe
        # (the same path the run took when no explicit PIT was supplied).
        pit_intervals=archived_pit,
        macro_series=archived_macro,
    )
    replayed = engine.classify(
        as_of_date=as_of,
        market_data=market_slice,
        event_calendar=archived_events,
        **v2_kwargs,
    )
    replayed_payload = json.loads(replayed.model_dump_json(indent=2))
    stored_payload = json.loads(output_path.read_text(encoding="utf-8"))
    diff = _diff_values(replayed_payload, stored_payload)
    matches = diff is None
    return {
        "as_of_date": as_of.isoformat(),
        "matches": matches,
        "breaks_qualification": (
            False if matches else _diff_touches_classification_fields(diff)
        ),
        "diff": diff,
    }


def run_walkforward_replay_check(
    *, output_root: Path, config_path: Path | None = None
) -> dict[str, Any]:
    """Replay every successful walk-forward run and return the §6 verdict payload."""
    paths = ensure_shadow_layout(
        output_root, db_name="regime_walkforward.db", include_reports=True
    )
    if not paths["db"].exists():
        raise FileNotFoundError(f"walk-forward DB not found: {paths['db']}")
    runs = _success_runs(paths["db"])
    # CR-003: drive the engine from the batch's archived FROZEN config so the replay is
    # self-contained and reproduces the run's actual config — not an operator-supplied
    # --config-path that can default to the wrong config and false-fail the gate.
    # --config-path remains a fallback only for older batches that predate the frozen file.
    frozen_config = output_root / "frozen_config.yaml"
    engine = RegimeEngine(
        config_path=frozen_config if frozen_config.exists() else config_path
    )
    engine_version = resolved_engine_version()
    config_version = engine.config.config_version

    results = [_replay_one(run=run, engine=engine, paths=paths) for run in runs]
    all_passed = bool(results) and all(r["matches"] for r in results)
    return {
        "engine_version": engine_version,
        "config_version": config_version,
        "all_passed": all_passed,
        "results": results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a historical walk-forward batch from archived inputs and write "
            "replay_verification.json for the build_walkforward_report §6 gate."
        )
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="destination JSON (default: <output-root>/reports/replay_verification.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    verdict = run_walkforward_replay_check(
        output_root=args.output_root, config_path=args.config_path
    )
    out_path = args.out or (args.output_root / "reports" / "replay_verification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"replay_verification": str(out_path), "all_passed": verdict["all_passed"]},
            indent=2,
        )
    )
    return 0 if verdict["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
