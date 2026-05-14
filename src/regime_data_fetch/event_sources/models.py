from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal, Protocol

from regime_data_fetch.acquisition_store import AcquisitionStore

EventConfidence = Literal["low", "medium", "high"]
Verdict = Literal["confirm", "contradict", "unknown"]
PromotionOutcome = Literal["promote", "quarantine"]


@dataclass(frozen=True)
class EventCandidate:
    date: dt.date
    event_type: str
    market: str
    importance: str
    source_id: str
    source_url: str | None
    raw_title: str | None
    raw_snippet: str | None
    is_future_scheduled: bool
    confidence: EventConfidence
    requires_manual_review: bool
    release_timestamp_et: dt.datetime | None = None
    window_days: tuple[int, int] | None = None


@dataclass(frozen=True)
class ValidationResult:
    candidate_key: tuple[str, dt.date]
    validator_id: str
    verdict: Verdict
    evidence_url: str | None
    evidence_snippet: str | None


@dataclass(frozen=True)
class PromotionDecision:
    candidate_key: tuple[str, dt.date]
    outcome: PromotionOutcome
    final_confidence: EventConfidence
    source_count: int
    requires_manual_review: bool
    reason: str


class PrimaryAdapter(Protocol):
    source_id: str

    def fetch(
        self,
        *,
        start_year: int,
        end_year: int,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[EventCandidate]:
        ...


class SecondaryValidator(Protocol):
    validator_id: str

    def validate(
        self,
        candidates: list[EventCandidate],
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[ValidationResult]:
        ...


class AmbiguityResolver(Protocol):
    def resolve(
        self,
        candidate_key: tuple[str, dt.date],
        conflicting: list[EventCandidate],
    ) -> EventCandidate | None:
        ...
