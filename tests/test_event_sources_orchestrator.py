from __future__ import annotations

import datetime as dt

import pytest

from regime_data_fetch.event_calendar import _render_events_yaml
from regime_data_fetch.event_sources.models import ApprovalRecord, EventCandidate, ValidationResult
from regime_data_fetch.event_sources.orchestrator import (
    EventSourceOrchestrator,
    build_candidate_id,
    render_events_from_candidates,
)


def _candidate(
    event_date: dt.date,
    event_type: str = "ECB_decision",
    source_id: str = "ecb.europa.eu:monetary-policy-decisions",
) -> EventCandidate:
    return EventCandidate(
        date=event_date,
        event_type=event_type,
        market="GLOBAL" if event_type != "election" else "US",
        importance="high",
        source_id=source_id,
        source_url="https://example.test/source",
        raw_title="Monetary policy decision",
        raw_snippet="decision fixture",
        is_future_scheduled=False,
        confidence="medium",
        requires_manual_review=False,
        release_timestamp_et=None,
        window_days=(-5, 10) if event_type == "election" else None,
    )


class _Generator:
    source_id = "gpr:caldara-iacoviello"

    def __init__(self, candidates: list[EventCandidate]) -> None:
        self.candidates = candidates

    def generate(self, *, start_year: int, end_year: int, store, run_id) -> list[EventCandidate]:
        del start_year, end_year, store, run_id
        return self.candidates


class _Adapter:
    source_id = "usa.gov:federal-budget-process"

    def __init__(self, candidates: list[EventCandidate]) -> None:
        self.candidates = candidates

    def fetch(self, *, start_year: int, end_year: int, store, run_id) -> list[EventCandidate]:
        del start_year, end_year, store, run_id
        return self.candidates


def test_triangulation_promotes_official_primary_with_unknown_validator_at_medium() -> None:
    candidate = _candidate(dt.date(2026, 6, 11))
    orchestrator = EventSourceOrchestrator(primary_adapters=[], validators=[])

    decisions = orchestrator.triangulate([candidate], [])
    rendered = render_events_from_candidates([candidate], decisions)

    assert len(decisions) == 1
    assert decisions[0].outcome == "promote"
    assert decisions[0].final_confidence == "medium"
    assert decisions[0].source_count == 1
    assert decisions[0].requires_manual_review is False
    assert _render_events_yaml(rendered) == (
        'events:\n'
        '  - date: "2026-06-11"\n'
        '    release_timestamp_et: "2026-06-11T00:00:00-05:00"\n'
        '    market: "GLOBAL"\n'
        '    type: "ECB_decision"\n'
        '    importance: "high"\n'
        '    source: "ecb.europa.eu:monetary-policy-decisions"\n'
    )


def test_triangulation_quarantines_contradicted_candidate() -> None:
    candidate = _candidate(dt.date(2026, 6, 11))
    validation = ValidationResult(
        candidate_key=("ECB_decision", dt.date(2026, 6, 11)),
        validator_id="hf:aufklarer-central-bank-communications",
        verdict="contradict",
        evidence_url="https://example.test/hf",
        evidence_snippet="decision dated 2026-06-10",
    )
    orchestrator = EventSourceOrchestrator(primary_adapters=[], validators=[])

    decisions = orchestrator.triangulate([candidate], [validation])

    assert decisions[0].outcome == "quarantine"
    assert decisions[0].final_confidence == "low"
    assert decisions[0].requires_manual_review is True
    assert render_events_from_candidates([candidate], decisions) == []


def test_quarantine_rate_above_one_percent_stops_run() -> None:
    candidates = [
        _candidate(dt.date(2026, 6, 11)),
        _candidate(dt.date(2026, 7, 23)),
    ]
    validations = [
        ValidationResult(
            candidate_key=("ECB_decision", dt.date(2026, 6, 11)),
            validator_id="hf:aufklarer-central-bank-communications",
            verdict="contradict",
            evidence_url=None,
            evidence_snippet="conflicting decision fixture",
        )
    ]
    orchestrator = EventSourceOrchestrator(primary_adapters=[], validators=[])

    with pytest.raises(RuntimeError, match="quarantine rate"):
        orchestrator.enforce_quarantine_threshold(candidates, orchestrator.triangulate(candidates, validations))


def test_group_b_geopolitical_candidate_is_stamped_but_withheld_without_overlay() -> None:
    candidate = _candidate(
        dt.date(2022, 2, 24),
        event_type="geopolitical_event",
        source_id="gpr:caldara-iacoviello",
    )
    candidate = EventCandidate(
        **{**candidate.__dict__, "market": "GLOBAL", "requires_manual_review": True, "raw_title": "Russia invasion of Ukraine"}
    )
    orchestrator = EventSourceOrchestrator(primary_adapters=[], candidate_generators=[_Generator([candidate])], validators=[])

    candidates, validations, decisions, rendered = orchestrator.run(start_year=2022, end_year=2022, store=None, run_id=None)

    assert validations == []
    assert len(candidates) == 1
    assert candidates[0].candidate_id == build_candidate_id("geopolitical_event", dt.date(2022, 2, 24), ["gpr:caldara-iacoviello"])
    assert decisions[0].outcome == "withhold"
    assert decisions[0].requires_manual_review is True
    assert rendered == []


def test_group_b_overlay_approved_geopolitical_candidate_renders() -> None:
    event_date = dt.date(2022, 2, 24)
    candidate = _candidate(event_date, event_type="geopolitical_event", source_id="gpr:caldara-iacoviello")
    candidate = EventCandidate(
        **{**candidate.__dict__, "market": "GLOBAL", "requires_manual_review": True, "raw_title": "Russia invasion of Ukraine"}
    )
    approval = ApprovalRecord(
        event_type="geopolitical_event",
        date=event_date,
        approved_label="geopolitical_event",
        approver="avinash",
        approved_at=dt.date(2026, 5, 14),
        evidence_candidate_id=build_candidate_id("geopolitical_event", event_date, ["gpr:caldara-iacoviello"]),
        evidence_source_count=1,
        importance="high",
        window_days=(0, 0),
        notes="Russia invasion of Ukraine.",
    )
    orchestrator = EventSourceOrchestrator(
        primary_adapters=[],
        candidate_generators=[_Generator([candidate])],
        validators=[],
        approval_overlay=[approval],
    )

    candidates, _, decisions, rendered = orchestrator.run(start_year=2022, end_year=2022, store=None, run_id=None)

    assert candidates[0].candidate_id == approval.evidence_candidate_id
    assert decisions[0].outcome == "promote"
    assert decisions[0].requires_manual_review is False
    assert [(event.date, event.type, event.importance, event.window_days) for event in rendered] == [
        (event_date, "geopolitical_event", "high", (0, 0))
    ]


def test_group_b_deterministic_budget_candidate_auto_promotes() -> None:
    candidate = _candidate(dt.date(2026, 9, 30), event_type="budget", source_id="usa.gov:federal-budget-process")
    candidate = EventCandidate(
        **{
            **candidate.__dict__,
            "market": "US",
            "importance": "medium",
            "event_subtype": "fy_deadline",
            "requires_manual_review": False,
            "raw_title": "US federal fiscal year deadline",
        }
    )
    orchestrator = EventSourceOrchestrator(primary_adapters=[_Adapter([candidate])], validators=[])

    candidates, _, decisions, rendered = orchestrator.run(start_year=2026, end_year=2026, store=None, run_id=None)

    assert candidates[0].candidate_id == build_candidate_id("budget", dt.date(2026, 9, 30), ["usa.gov:federal-budget-process"])
    assert decisions[0].outcome == "promote"
    assert decisions[0].final_confidence == "high"
    assert [(event.date, event.type, event.source) for event in rendered] == [
        (dt.date(2026, 9, 30), "budget", "usa.gov:federal-budget-process")
    ]
