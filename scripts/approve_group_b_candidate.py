#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from regime_data_fetch.event_sources.approvals import append_approval_record  # noqa: E402


def _utc_today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def main() -> int:
    parser = argparse.ArgumentParser(description="Approve a pending Group B event candidate.")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--approver", required=True)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--candidates", default=str(REPO_ROOT / "data/raw/event_calendar/candidates/event_candidates.parquet"))
    parser.add_argument("--overlay", default=str(REPO_ROOT / "configs/events/group_b_approvals.yaml"))
    args = parser.parse_args()

    candidates = pd.read_parquet(Path(args.candidates))
    matches = candidates[candidates["candidate_id"] == args.candidate_id]
    if matches.empty:
        raise SystemExit(f"candidate_id not found: {args.candidate_id}")
    row = matches.iloc[0]
    if row["event_type"] not in {"geopolitical_event", "budget"}:
        raise SystemExit(f"candidate_id is not a Group B candidate: {args.candidate_id}")
    if str(row.get("promotion_outcome", "")) != "withhold" or bool(row.get("requires_manual_review")) is not True:
        raise SystemExit(f"candidate_id is not pending manual review: {args.candidate_id}")
    source_count = int(row["source_count"]) if "source_count" in row and pd.notna(row["source_count"]) else 1
    append_approval_record(
        Path(args.overlay),
        event_type=str(row["event_type"]),
        event_date=dt.date.fromisoformat(str(row["date"])),
        candidate_id=args.candidate_id,
        source_count=source_count,
        approver=args.approver,
        approved_at=_utc_today(),
        notes=args.notes,
        importance=str(row["importance"]) if pd.notna(row["importance"]) else None,
    )
    print(str(args.overlay))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
