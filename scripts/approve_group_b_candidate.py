#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys
from typing import Any, cast

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from regime_data_fetch.event_sources.approvals import append_approval_record


def _utc_today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Approve a pending Group B event candidate."
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--approver", required=True)
    parser.add_argument("--notes", default=None)
    parser.add_argument(
        "--candidates",
        default=str(
            REPO_ROOT / "data/raw/event_calendar/candidates/event_candidates.parquet"
        ),
    )
    parser.add_argument(
        "--overlay", default=str(REPO_ROOT / "configs/events/group_b_approvals.yaml")
    )
    args = parser.parse_args()

    candidates: pd.DataFrame = pd.read_parquet(Path(args.candidates))
    candidate_rows = cast(
        list[dict[str, Any]],
        cast(Any, candidates).to_dict(orient="records"),
    )
    row = next(
        (
            item
            for item in candidate_rows
            if str(item.get("candidate_id")) == args.candidate_id
        ),
        None,
    )
    if row is None:
        raise SystemExit(f"candidate_id not found: {args.candidate_id}")
    event_type = str(row["event_type"])
    if event_type not in {"geopolitical_event", "budget"}:
        raise SystemExit(
            f"candidate_id is not a Group B candidate: {args.candidate_id}"
        )
    if (
        str(row.get("promotion_outcome", "")) != "withhold"
        or bool(row.get("requires_manual_review")) is not True
    ):
        raise SystemExit(
            f"candidate_id is not pending manual review: {args.candidate_id}"
        )
    source_count_value = row.get("source_count")
    source_count = (
        int(source_count_value)
        if source_count_value is not None and pd.notna(source_count_value)
        else 1
    )
    importance_value = row.get("importance")
    append_approval_record(
        Path(args.overlay),
        event_type=event_type,
        event_date=dt.date.fromisoformat(str(row["date"])),
        candidate_id=args.candidate_id,
        source_count=source_count,
        approver=args.approver,
        approved_at=_utc_today(),
        notes=args.notes,
        importance=str(importance_value) if pd.notna(importance_value) else None,
    )
    print(str(args.overlay))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
