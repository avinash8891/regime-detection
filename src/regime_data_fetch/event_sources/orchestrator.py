from __future__ import annotations

import datetime as dt
from collections import defaultdict

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_calendar import ScheduledEvent, US_EASTERN
from regime_data_fetch.event_sources.models import (
    AmbiguityResolver,
    EventCandidate,
    PrimaryAdapter,
    PromotionDecision,
    SecondaryValidator,
    ValidationResult,
)

DETERMINISTIC_SOURCE_IDS = {"fec.gov:election-dates"}


class EventSourceOrchestrator:
    def __init__(
        self,
        *,
        primary_adapters: list[PrimaryAdapter],
        validators: list[SecondaryValidator],
        resolver: AmbiguityResolver | None = None,
    ) -> None:
        self.primary_adapters = primary_adapters
        self.validators = validators
        self.resolver = resolver

    def run(
        self,
        *,
        start_year: int,
        end_year: int,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> tuple[list[EventCandidate], list[ValidationResult], list[PromotionDecision], list[ScheduledEvent]]:
        candidates: list[EventCandidate] = []
        for adapter in self.primary_adapters:
            candidates.extend(adapter.fetch(start_year=start_year, end_year=end_year, store=store, run_id=run_id))

        validations: list[ValidationResult] = []
        for validator in self.validators:
            validations.extend(validator.validate(candidates, store=store, run_id=run_id))

        decisions = self.triangulate(candidates, validations)
        self.enforce_quarantine_threshold(candidates, decisions)
        return candidates, validations, decisions, render_events_from_candidates(candidates, decisions)

    def triangulate(
        self,
        candidates: list[EventCandidate],
        validations: list[ValidationResult],
    ) -> list[PromotionDecision]:
        validations_by_key: dict[tuple[str, dt.date], list[ValidationResult]] = defaultdict(list)
        for validation in validations:
            validations_by_key[validation.candidate_key].append(validation)

        candidates_by_key: dict[tuple[str, dt.date], list[EventCandidate]] = defaultdict(list)
        for candidate in candidates:
            candidates_by_key[(candidate.event_type, candidate.date)].append(candidate)

        decisions: list[PromotionDecision] = []
        for key in sorted(candidates_by_key, key=lambda item: (item[1], item[0])):
            keyed_candidates = candidates_by_key[key]
            keyed_validations = validations_by_key.get(key, [])
            if any(validation.verdict == "contradict" for validation in keyed_validations):
                decisions.append(
                    PromotionDecision(
                        candidate_key=key,
                        outcome="quarantine",
                        final_confidence="low",
                        source_count=1 + sum(validation.verdict == "confirm" for validation in keyed_validations),
                        requires_manual_review=True,
                        reason="secondary validator contradicted official primary date",
                    )
                )
                continue

            confirms = sum(validation.verdict == "confirm" for validation in keyed_validations)
            candidate = keyed_candidates[0]
            if candidate.source_id in DETERMINISTIC_SOURCE_IDS:
                final_confidence = "high"
                reason = "deterministic primary without contradiction"
            elif confirms:
                final_confidence = "high"
                reason = "official primary confirmed by secondary validator"
            else:
                final_confidence = "medium"
                reason = "official primary with validators unknown or unavailable"

            decisions.append(
                PromotionDecision(
                    candidate_key=key,
                    outcome="promote",
                    final_confidence=final_confidence,
                    source_count=1 + confirms,
                    requires_manual_review=False,
                    reason=reason,
                )
            )
        return decisions

    def enforce_quarantine_threshold(
        self,
        candidates: list[EventCandidate],
        decisions: list[PromotionDecision],
    ) -> None:
        if not candidates:
            return
        quarantined = sum(decision.outcome == "quarantine" for decision in decisions)
        quarantine_rate = quarantined / len(candidates)
        if quarantine_rate > 0.01:
            raise RuntimeError(
                f"event-source quarantine rate {quarantine_rate:.2%} exceeded 1.00% threshold "
                f"({quarantined}/{len(candidates)} candidates)"
            )


def render_events_from_candidates(
    candidates: list[EventCandidate],
    decisions: list[PromotionDecision],
) -> list[ScheduledEvent]:
    candidates_by_key = {(candidate.event_type, candidate.date): candidate for candidate in candidates}
    rendered: list[ScheduledEvent] = []
    for decision in decisions:
        if decision.outcome != "promote":
            continue
        candidate = candidates_by_key[decision.candidate_key]
        release_timestamp_et = candidate.release_timestamp_et or dt.datetime(
            candidate.date.year,
            candidate.date.month,
            candidate.date.day,
            0,
            0,
            tzinfo=US_EASTERN,
        )
        rendered.append(
            ScheduledEvent(
                date=candidate.date,
                release_timestamp_et=release_timestamp_et,
                market=candidate.market,
                type=candidate.event_type,
                importance=candidate.importance,
                source=candidate.source_id,
                window_days=candidate.window_days,
            )
        )
    return sorted(rendered, key=lambda event: (event.release_timestamp_et, event.type))
