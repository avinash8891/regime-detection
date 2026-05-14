from __future__ import annotations

import datetime as dt

import pytest

from regime_data_fetch.event_calendar import _render_events_yaml
from regime_data_fetch.event_sources.models import EventCandidate, ValidationResult
from regime_data_fetch.event_sources.orchestrator import (
    EventSourceOrchestrator,
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
