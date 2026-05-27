from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false

import datetime as dt
from pathlib import Path
from typing import Any

import yaml

from regime_data_fetch.event_sources.models import ApprovalRecord

GROUP_B_EVENT_TYPES = {"geopolitical_event", "budget"}


def load_approval_overlay(path: Path) -> list[ApprovalRecord]:
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"approval overlay {path} must contain a mapping")
    raw_approvals = payload.get("approvals", [])
    if raw_approvals is None:
        raw_approvals = []
    if not isinstance(raw_approvals, list):
        raise ValueError(f"approval overlay {path} field approvals must be a list")

    approvals: list[ApprovalRecord] = []
    seen: set[tuple[str, dt.date]] = set()
    for idx, raw in enumerate(raw_approvals, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"approval overlay entry {idx} must be a mapping")
        approval = _parse_approval(raw, idx)
        key = (approval.event_type, approval.date)
        if key in seen:
            raise ValueError(
                f"duplicate approval for {approval.event_type} on {approval.date.isoformat()}"
            )
        seen.add(key)
        approvals.append(approval)
    return approvals


def append_approval_record(
    path: Path,
    *,
    event_type: str,
    event_date: dt.date,
    candidate_id: str,
    source_count: int,
    approver: str,
    approved_at: dt.date,
    notes: str | None = None,
    importance: str | None = None,
    window_days: tuple[int, int] | None = None,
) -> None:
    approvals = load_approval_overlay(path)
    key = (event_type, event_date)
    if any((approval.event_type, approval.date) == key for approval in approvals):
        raise ValueError(
            f"duplicate approval for {event_type} on {event_date.isoformat()}"
        )
    approval = ApprovalRecord(
        event_type=event_type,
        date=event_date,
        approved_label=event_type,
        approver=approver,
        approved_at=approved_at,
        evidence_candidate_id=candidate_id,
        evidence_source_count=source_count,
        importance=importance,
        window_days=window_days,
        notes=notes,
    )
    approvals.append(approval)
    _write_approval_overlay(path, approvals)
    load_approval_overlay(path)


def _parse_approval(raw: dict[str, Any], idx: int) -> ApprovalRecord:
    missing = [
        field
        for field in (
            "event_type",
            "date",
            "approved_label",
            "approver",
            "approved_at",
            "evidence_candidate_id",
            "evidence_source_count",
        )
        if field not in raw
    ]
    if missing:
        raise ValueError(
            f"approval overlay entry {idx} missing required fields: {', '.join(missing)}"
        )

    event_type = str(raw["event_type"])
    if event_type not in GROUP_B_EVENT_TYPES:
        raise ValueError(
            f"approval overlay entry {idx} has unknown event_type: {event_type}"
        )
    approved_label = str(raw["approved_label"])
    if approved_label != event_type:
        raise ValueError(
            f"approval overlay entry {idx} approved_label must match event_type"
        )

    date = _parse_date(raw["date"], f"approval overlay entry {idx} date")
    approved_at = _parse_date(
        raw["approved_at"], f"approval overlay entry {idx} approved_at"
    )
    source_count = int(raw["evidence_source_count"])
    if source_count < 1:
        raise ValueError(
            f"approval overlay entry {idx} evidence_source_count must be positive"
        )

    return ApprovalRecord(
        event_type=event_type,
        date=date,
        approved_label=approved_label,
        approver=str(raw["approver"]),
        approved_at=approved_at,
        evidence_candidate_id=str(raw["evidence_candidate_id"]),
        evidence_source_count=source_count,
        importance=(
            str(raw["importance"]) if raw.get("importance") is not None else None
        ),
        window_days=_parse_window_days(raw.get("window_days"), idx),
        notes=str(raw["notes"]) if raw.get("notes") is not None else None,
    )


def _parse_date(value: object, label: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD") from exc


def _parse_window_days(value: object, idx: int) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(
            f"approval overlay entry {idx} window_days must be a two-item list"
        )
    return (int(value[0]), int(value[1]))


def _write_approval_overlay(path: Path, approvals: list[ApprovalRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"approvals": [_approval_to_dict(approval) for approval in approvals]}
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _approval_to_dict(approval: ApprovalRecord) -> dict[str, object]:
    record: dict[str, object] = {
        "event_type": approval.event_type,
        "date": approval.date.isoformat(),
        "approved_label": approval.approved_label,
        "approver": approval.approver,
        "approved_at": approval.approved_at.isoformat(),
        "evidence_candidate_id": approval.evidence_candidate_id,
        "evidence_source_count": approval.evidence_source_count,
    }
    if approval.importance is not None:
        record["importance"] = approval.importance
    if approval.window_days is not None:
        record["window_days"] = list(approval.window_days)
    if approval.notes is not None:
        record["notes"] = approval.notes
    return record
