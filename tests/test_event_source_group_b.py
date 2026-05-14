from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from regime_data_fetch.event_sources.approvals import load_approval_overlay
from regime_data_fetch.event_sources.deterministic_budget import DeterministicBudgetAdapter
from regime_data_fetch.event_sources.validators_gpr_gdelt import GPRGDELTSignalGenerator


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


def test_gpr_gdelt_generator_flags_real_geopolitical_spike_date() -> None:
    gpr_csv = """date,gpr
2022-02-20,100
2022-02-21,101
2022-02-22,99
2022-02-23,101
2022-02-24,500
2022-02-25,120
"""
    gdelt_csv = """date,event_count,dominant_theme,source_url
2022-02-24,1200,Russia invasion of Ukraine,https://example.test/gdelt/20220224
"""
    generator = GPRGDELTSignalGenerator(
        gpr_fetcher=lambda: gpr_csv,
        gdelt_fetcher=lambda: gdelt_csv,
        min_history_days=3,
        stddev_threshold=2.0,
    )

    candidates = generator.generate(start_year=2022, end_year=2022, store=None, run_id=None)
    validations = generator.validate(candidates, store=None, run_id=None)

    assert [(candidate.date, candidate.event_type, candidate.source_id) for candidate in candidates] == [
        (dt.date(2022, 2, 24), "geopolitical_event", "gdelt:events-v2"),
        (dt.date(2022, 2, 24), "geopolitical_event", "gpr:caldara-iacoviello"),
    ]
    assert all(candidate.requires_manual_review for candidate in candidates)
    assert all(candidate.confidence == "medium" for candidate in candidates)
    assert {(validation.candidate_key, validation.validator_id, validation.verdict) for validation in validations} == {
        (("geopolitical_event", dt.date(2022, 2, 24)), "gdelt:events-v2", "confirm"),
        (("geopolitical_event", dt.date(2022, 2, 24)), "gpr:caldara-iacoviello", "confirm"),
    }
