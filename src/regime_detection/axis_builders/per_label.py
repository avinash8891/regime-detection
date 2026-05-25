from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Protocol, TypeVar

from regime_detection.data_quality import quality_forces_unknown
from regime_detection.hysteresis import apply_data_quality_aware_hysteresis
from regime_detection.models import DataQuality


AxisOutputT = TypeVar("AxisOutputT")


class AxisOutputFactory(Protocol[AxisOutputT]):
    def __call__(
        self,
        *,
        raw_label: str,
        stable_label: str,
        active_label: str,
        evidence: dict[str, object],
        data_quality: DataQuality,
    ) -> AxisOutputT: ...


def build_per_label_axis_outputs(
    *,
    sessions: Sequence[date],
    raw_labels: Sequence[str],
    risk_rank: Mapping[str, int],
    deescalation_days_by_label: Mapping[str, int],
    default_deescalation_days: int,
    max_unknown_freeze_days: int = 0,
    data_quality: Sequence[DataQuality],
    evidence: Sequence[dict[str, object]],
    output_factory: AxisOutputFactory[AxisOutputT],
) -> dict[date, AxisOutputT]:
    stable_labels, active_labels, frozen_labels = apply_data_quality_aware_hysteresis(
        raw_labels=list(raw_labels),
        risk_rank=dict(risk_rank),
        deescalation_days_by_label=dict(deescalation_days_by_label),
        data_quality=data_quality,
        default_deescalation_days=default_deescalation_days,
        max_unknown_freeze_days=max_unknown_freeze_days,
    )

    outputs: dict[date, AxisOutputT] = {}
    for day, raw, stable, active, is_frozen, dq, day_evidence in zip(
        sessions,
        raw_labels,
        stable_labels,
        active_labels,
        frozen_labels,
        data_quality,
        evidence,
        strict=True,
    ):
        day_evidence = dict(day_evidence)
        if is_frozen:
            day_evidence["data_quality_freeze"] = True
        elif quality_forces_unknown(dq):
            # Force the canonical sentinel rather than trusting that the upstream
            # classifier already emitted raw=="unknown" — keeps this builder the
            # single point of truth for the DQ→unknown contract instead of a
            # silent dependency on every classifier doing the same check.
            stable = "unknown"
            active = "unknown"
        outputs[day] = output_factory(
            raw_label=raw,
            stable_label=stable,
            active_label=active,
            evidence=day_evidence,
            data_quality=dq,
        )
    return outputs
