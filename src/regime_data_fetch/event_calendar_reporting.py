from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false

import datetime as dt
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CandidateArtifactRecords:
    candidates: list[dict[str, object | None]]
    validations: list[dict[str, object | None]]
    quarantine: list[dict[str, object | None]]


def build_candidate_artifact_records(
    *,
    candidates: list[object],
    validations: list[object],
    decisions: list[object],
) -> CandidateArtifactRecords:
    candidate_records = [
        _candidate_record(candidate, decisions) for candidate in candidates
    ]
    validation_records = [_validation_record(validation) for validation in validations]
    quarantined_keys = {
        getattr(decision, "candidate_key")
        for decision in decisions
        if getattr(decision, "outcome") == "quarantine"
    }
    quarantine_records = [
        record
        for record in candidate_records
        if (record["event_type"], dt.date.fromisoformat(str(record["date"])))
        in quarantined_keys
    ]
    return CandidateArtifactRecords(
        candidates=candidate_records,
        validations=validation_records,
        quarantine=quarantine_records,
    )


def build_group_a_report(
    *,
    candidates: list[object],
    decisions: list[object],
    output_paths: dict[str, Path],
    repo_root: Path,
) -> dict[str, object]:
    group_a_types = {"ECB_decision", "BOE_decision", "BOJ_decision", "election"}
    group_a_candidates = [
        candidate
        for candidate in candidates
        if getattr(candidate, "event_type") in group_a_types
    ]
    group_a_decisions = [
        decision
        for decision in decisions
        if getattr(decision, "candidate_key")[0] in group_a_types
    ]
    candidate_counts = Counter(
        getattr(candidate, "event_type") for candidate in group_a_candidates
    )
    promoted_counts = Counter(
        getattr(decision, "candidate_key")[0]
        for decision in group_a_decisions
        if getattr(decision, "outcome") == "promote"
    )
    quarantined_counts = Counter(
        getattr(decision, "candidate_key")[0]
        for decision in group_a_decisions
        if getattr(decision, "outcome") == "quarantine"
    )
    source_ids = sorted(
        {getattr(candidate, "source_id") for candidate in group_a_candidates}
    )
    return {
        "candidates": {key: candidate_counts[key] for key in sorted(candidate_counts)},
        "promoted": {key: promoted_counts[key] for key in sorted(promoted_counts)},
        "quarantined": {
            key: quarantined_counts[key] for key in sorted(quarantined_counts)
        },
        "source_ids": source_ids,
        "paths": {
            key: report_path(value, repo_root=repo_root)
            for key, value in output_paths.items()
        },
    }


def build_group_b_report(
    *,
    candidates: list[object],
    decisions: list[object],
    approval_overlay: list[object] | None,
) -> dict[str, object]:
    group_b_types = {"geopolitical_event", "budget"}
    group_b_candidates = [
        candidate
        for candidate in candidates
        if getattr(candidate, "event_type") in group_b_types
    ]
    group_b_decisions = [
        decision
        for decision in decisions
        if getattr(decision, "candidate_key")[0] in group_b_types
    ]
    candidate_counts = Counter(
        getattr(candidate, "event_type") for candidate in group_b_candidates
    )
    promoted_counts = Counter(
        getattr(decision, "candidate_key")[0]
        for decision in group_b_decisions
        if getattr(decision, "outcome") == "promote"
    )
    manual_review_counts = Counter(
        getattr(decision, "candidate_key")[0]
        for decision in group_b_decisions
        if getattr(decision, "outcome") == "withhold"
    )
    candidates_by_key = {
        (getattr(candidate, "event_type"), getattr(candidate, "date")): candidate
        for candidate in group_b_candidates
    }
    decisions_by_key = {
        getattr(decision, "candidate_key"): decision for decision in group_b_decisions
    }
    stale_approvals = []
    stale_evidence = []
    contradicted_approvals = []
    for approval in approval_overlay or []:
        key = (getattr(approval, "event_type"), getattr(approval, "date"))
        if key[0] not in group_b_types:
            continue
        candidate = candidates_by_key.get(key)
        decision = decisions_by_key.get(key)
        rendered_key = {"event_type": key[0], "date": key[1].isoformat()}
        if candidate is None:
            stale_approvals.append(rendered_key)
        elif decision is not None and getattr(decision, "outcome") == "quarantine":
            contradicted_approvals.append(rendered_key)
        elif getattr(candidate, "candidate_id", "") != getattr(
            approval, "evidence_candidate_id"
        ):
            stale_evidence.append(rendered_key)
    return {
        "candidates": {key: candidate_counts[key] for key in sorted(candidate_counts)},
        "promoted": {key: promoted_counts[key] for key in sorted(promoted_counts)},
        "manual_review_pending": {
            key: manual_review_counts[key] for key in sorted(manual_review_counts)
        },
        "stale_approvals": stale_approvals,
        "stale_evidence": stale_evidence,
        "contradicted_approvals": contradicted_approvals,
    }


def report_path(path: Path, *, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _candidate_record(
    candidate: object, decisions: list[object]
) -> dict[str, object | None]:
    decision = next(
        (
            item
            for item in decisions
            if getattr(item, "candidate_key")
            == (getattr(candidate, "event_type"), getattr(candidate, "date"))
        ),
        None,
    )
    release_timestamp = getattr(candidate, "release_timestamp_et")
    return {
        "date": getattr(candidate, "date").isoformat(),
        "event_type": getattr(candidate, "event_type"),
        "market": getattr(candidate, "market"),
        "importance": getattr(candidate, "importance"),
        "source_id": getattr(candidate, "source_id"),
        "candidate_id": getattr(candidate, "candidate_id", ""),
        "event_subtype": getattr(candidate, "event_subtype", None),
        "source_url": getattr(candidate, "source_url"),
        "raw_title": getattr(candidate, "raw_title"),
        "raw_snippet": getattr(candidate, "raw_snippet"),
        "is_future_scheduled": getattr(candidate, "is_future_scheduled"),
        "confidence": (
            getattr(decision, "final_confidence")
            if decision is not None
            else getattr(candidate, "confidence")
        ),
        "source_count": (
            getattr(decision, "source_count") if decision is not None else 1
        ),
        "requires_manual_review": (
            getattr(decision, "requires_manual_review")
            if decision is not None
            else getattr(candidate, "requires_manual_review")
        ),
        "promotion_outcome": (
            getattr(decision, "outcome") if decision is not None else None
        ),
        "promotion_reason": (
            getattr(decision, "reason") if decision is not None else None
        ),
        "release_timestamp_et": (
            release_timestamp.isoformat() if release_timestamp is not None else None
        ),
        "window_days": (
            list(getattr(candidate, "window_days"))
            if getattr(candidate, "window_days") is not None
            else None
        ),
    }


def _validation_record(validation: object) -> dict[str, object | None]:
    event_type, event_date = getattr(validation, "candidate_key")
    return {
        "event_type": event_type,
        "date": event_date.isoformat(),
        "validator_id": getattr(validation, "validator_id"),
        "verdict": getattr(validation, "verdict"),
        "evidence_url": getattr(validation, "evidence_url"),
        "evidence_snippet": getattr(validation, "evidence_snippet"),
    }
