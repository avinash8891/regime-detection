from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from regime_data_fetch.event_sources.approvals import load_approval_overlay
from regime_data_fetch.event_sources.deterministic_budget import DeterministicBudgetAdapter


def test_deterministic_budget_emits_exact_fy_deadline_rows() -> None:
    candidates = DeterministicBudgetAdapter(as_of_date=dt.date(2026, 5, 14)).fetch(
        start_year=2016,
        end_year=2018,
        store=None,
        run_id=None,
    )

    assert [(candidate.date, candidate.event_type, candidate.event_subtype, candidate.source_id) for candidate in candidates] == [
        (dt.date(2016, 9, 30), "budget", "fy_deadline", "usa.gov:federal-budget-process"),
        (dt.date(2017, 9, 30), "budget", "fy_deadline", "usa.gov:federal-budget-process"),
        (dt.date(2018, 9, 30), "budget", "fy_deadline", "usa.gov:federal-budget-process"),
    ]
    assert [candidate.requires_manual_review for candidate in candidates] == [False, False, False]
    assert [candidate.confidence for candidate in candidates] == ["high", "high", "high"]


def test_load_approval_overlay_parses_valid_records(tmp_path: Path) -> None:
    overlay_path = tmp_path / "group_b_approvals.yaml"
    overlay_path.write_text(
        """
approvals:
  - event_type: geopolitical_event
    date: "2022-02-24"
    approved_label: geopolitical_event
    approver: avinash
    approved_at: "2026-05-14"
    evidence_candidate_id: "abc123"
    evidence_source_count: 3
    importance: high
    window_days: [0, 0]
    notes: "Russia invasion of Ukraine."
"""
    )

    approvals = load_approval_overlay(overlay_path)

    assert len(approvals) == 1
    assert approvals[0].event_type == "geopolitical_event"
    assert approvals[0].date == dt.date(2022, 2, 24)
    assert approvals[0].evidence_candidate_id == "abc123"
    assert approvals[0].window_days == (0, 0)


def test_load_approval_overlay_rejects_duplicate_keys(tmp_path: Path) -> None:
    overlay_path = tmp_path / "group_b_approvals.yaml"
    overlay_path.write_text(
        """
approvals:
  - event_type: geopolitical_event
    date: "2022-02-24"
    approved_label: geopolitical_event
    approver: avinash
    approved_at: "2026-05-14"
    evidence_candidate_id: "abc123"
    evidence_source_count: 3
  - event_type: geopolitical_event
    date: "2022-02-24"
    approved_label: geopolitical_event
    approver: avinash
    approved_at: "2026-05-14"
    evidence_candidate_id: "def456"
    evidence_source_count: 2
"""
    )

    with pytest.raises(ValueError, match="duplicate approval"):
        load_approval_overlay(overlay_path)
