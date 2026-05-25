from __future__ import annotations

import datetime as dt
import hashlib
from collections import defaultdict
from dataclasses import replace

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_calendar import ScheduledEvent, US_EASTERN
from regime_data_fetch.event_sources.models import (
    ApprovalRecord,
    AmbiguityResolver,
    CandidateGenerator,
    EventCandidate,
    PrimaryAdapter,
    PromotionDecision,
    SecondaryValidator,
    ValidationResult,
)

DETERMINISTIC_SOURCE_IDS = {"fec.gov:election-dates"}
GROUP_B_EVENT_TYPES = {"geopolitical_event", "budget"}


class EventSourceOrchestrator:
    def __init__(
        self,
        *,
        primary_adapters: list[PrimaryAdapter],
        candidate_generators: list[CandidateGenerator] | None = None,
        validators: list[SecondaryValidator],
        approval_overlay: list[ApprovalRecord] | None = None,
        resolver: AmbiguityResolver | None = None,
    ) -> None:
        self.primary_adapters = primary_adapters
        self.candidate_generators = candidate_generators or []
        self.validators = validators
        self.approval_overlay = approval_overlay or []
        self.resolver = resolver

    def run(
        self,
        *,
        start_year: int,
        end_year: int,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> tuple[
        list[EventCandidate],
        list[ValidationResult],
        list[PromotionDecision],
        list[ScheduledEvent],
    ]:
        candidates: list[EventCandidate] = []
        for adapter in self.primary_adapters:
            candidates.extend(
                adapter.fetch(
                    start_year=start_year, end_year=end_year, store=store, run_id=run_id
                )
            )
        for generator in self.candidate_generators:
            candidates.extend(
                generator.generate(
                    start_year=start_year, end_year=end_year, store=store, run_id=run_id
                )
            )

        validations: list[ValidationResult] = []
        for validator in self.validators:
            validations.extend(
                validator.validate(candidates, store=store, run_id=run_id)
            )

        candidates = stamp_candidate_ids(candidates, validations)
        decisions = self.triangulate(candidates, validations)
        self.enforce_quarantine_threshold(candidates, decisions)
        return (
            candidates,
            validations,
            decisions,
            render_events_from_candidates(candidates, decisions, self.approval_overlay),
        )

    def triangulate(
        self,
        candidates: list[EventCandidate],
        validations: list[ValidationResult],
    ) -> list[PromotionDecision]:
        validations_by_key: dict[tuple[str, dt.date], list[ValidationResult]] = (
            defaultdict(list)
        )
        for validation in validations:
            validations_by_key[validation.candidate_key].append(validation)

        candidates_by_key: dict[tuple[str, dt.date], list[EventCandidate]] = (
            defaultdict(list)
        )
        for candidate in candidates:
            candidates_by_key[(candidate.event_type, candidate.date)].append(candidate)

        approvals_by_key = {
            (approval.event_type, approval.date): approval
            for approval in self.approval_overlay
        }
        decisions: list[PromotionDecision] = []
        for key in sorted(candidates_by_key, key=lambda item: (item[1], item[0])):
            keyed_candidates = candidates_by_key[key]
            keyed_validations = validations_by_key.get(key, [])
            confirms = sum(
                validation.verdict == "confirm" for validation in keyed_validations
            )
            candidate = keyed_candidates[0]
            source_count = len({item.source_id for item in keyed_candidates}) + confirms
            if any(
                validation.verdict == "contradict" for validation in keyed_validations
            ):
                decisions.append(
                    PromotionDecision(
                        candidate_key=key,
                        outcome="quarantine",
                        final_confidence="low",
                        source_count=source_count,
                        requires_manual_review=True,
                        reason="secondary validator contradicted official primary date",
                    )
                )
                continue

            if candidate.event_type in GROUP_B_EVENT_TYPES:
                decisions.append(
                    self._group_b_decision(
                        key, candidate, source_count, approvals_by_key
                    )
                )
                continue

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
                    source_count=source_count,
                    requires_manual_review=False,
                    reason=reason,
                )
            )
        return decisions

    def _group_b_decision(
        self,
        key: tuple[str, dt.date],
        candidate: EventCandidate,
        source_count: int,
        approvals_by_key: dict[tuple[str, dt.date], ApprovalRecord],
    ) -> PromotionDecision:
        if (
            candidate.event_type == "budget"
            and candidate.event_subtype == "fy_deadline"
        ):
            return PromotionDecision(
                candidate_key=key,
                outcome="promote",
                final_confidence="high",
                source_count=source_count,
                requires_manual_review=False,
                reason="deterministic fiscal-year budget deadline",
            )
        if candidate.event_type == "budget" and source_count >= 2:
            return PromotionDecision(
                candidate_key=key,
                outcome="promote",
                final_confidence="high",
                source_count=source_count,
                requires_manual_review=False,
                reason="budget event confirmed by at least two independent official sources",
            )
        if key in approvals_by_key:
            return PromotionDecision(
                candidate_key=key,
                outcome="promote",
                final_confidence=candidate.confidence,
                source_count=source_count,
                requires_manual_review=False,
                reason="group B approval overlay matched regenerated candidate",
            )
        return PromotionDecision(
            candidate_key=key,
            outcome="withhold",
            final_confidence=candidate.confidence,
            source_count=source_count,
            requires_manual_review=True,
            reason="group B candidate requires approval overlay before rendering",
        )

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


def build_candidate_id(
    event_type: str, event_date: dt.date, source_ids: list[str]
) -> str:
    source_part = "|".join(sorted(set(source_ids)))
    return hashlib.sha256(
        f"{event_type}|{event_date.isoformat()}|{source_part}".encode("utf-8")
    ).hexdigest()[:16]


def stamp_candidate_ids(
    candidates: list[EventCandidate], validations: list[ValidationResult]
) -> list[EventCandidate]:
    confirming_validators_by_key: dict[tuple[str, dt.date], list[str]] = defaultdict(
        list
    )
    for validation in validations:
        if validation.verdict == "confirm":
            confirming_validators_by_key[validation.candidate_key].append(
                validation.validator_id
            )

    candidates_by_key: dict[tuple[str, dt.date], list[EventCandidate]] = defaultdict(
        list
    )
    for candidate in candidates:
        candidates_by_key[(candidate.event_type, candidate.date)].append(candidate)

    candidate_ids = {
        key: build_candidate_id(
            key[0],
            key[1],
            [candidate.source_id for candidate in keyed_candidates]
            + confirming_validators_by_key.get(key, []),
        )
        for key, keyed_candidates in candidates_by_key.items()
    }
    return [
        replace(
            candidate,
            candidate_id=candidate_ids[(candidate.event_type, candidate.date)],
        )
        for candidate in candidates
    ]


def render_events_from_candidates(
    candidates: list[EventCandidate],
    decisions: list[PromotionDecision],
    approval_overlay: list[ApprovalRecord] | None = None,
) -> list[ScheduledEvent]:
    candidates_by_key = {
        (candidate.event_type, candidate.date): candidate for candidate in candidates
    }
    approvals_by_key = {
        (approval.event_type, approval.date): approval
        for approval in approval_overlay or []
    }
    rendered: list[ScheduledEvent] = []
    for decision in decisions:
        if decision.outcome != "promote":
            continue
        candidate = candidates_by_key[decision.candidate_key]
        approval = approvals_by_key.get(decision.candidate_key)
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
                importance=(
                    approval.importance
                    if approval and approval.importance
                    else candidate.importance
                ),
                source=candidate.source_id,
                window_days=(
                    approval.window_days
                    if approval and approval.window_days is not None
                    else candidate.window_days
                ),
                approved_label=(
                    approval.approved_label if approval is not None else None
                ),
            )
        )
    return sorted(rendered, key=lambda event: (event.release_timestamp_et, event.type))
