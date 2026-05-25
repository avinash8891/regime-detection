from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from regime_detection.data_quality import quality_forces_unknown
from regime_detection.models import DataQuality


TLabel = TypeVar("TLabel", bound=str)


def apply_per_label_asymmetric_hysteresis(
    *,
    raw_labels: list[TLabel],
    risk_rank: dict[TLabel, int],
    deescalation_days_by_label: dict[TLabel, int],
    default_deescalation_days: int = 0,
) -> tuple[list[TLabel], list[TLabel]]:
    """
    Per-label asymmetric hysteresis (spec v2 §3.7):
    - Escalation (higher risk_rank) updates stable immediately, identical to V1.
    - De-escalation threshold depends on the label being LEFT (the current stable label):
      `deescalation_days_by_label.get(stable_label, default_deescalation_days)`.
    - active_label uses the same fast-path as V1: if raw is riskier than stable, active=raw else active=stable.
    """
    if default_deescalation_days < 0:
        raise ValueError("default_deescalation_days must be >= 0")
    for label, days in deescalation_days_by_label.items():
        if days < 0:
            raise ValueError(f"deescalation_days_by_label[{label!r}] must be >= 0")
    if not raw_labels:
        raise ValueError("raw_labels must be non-empty")

    stable: list[TLabel] = []
    active: list[TLabel] = []

    stable_label: TLabel = raw_labels[0]
    pending_label: TLabel | None = None
    pending_count = 0

    for raw in raw_labels:
        raw_rank = risk_rank[raw]
        stable_rank = risk_rank[stable_label]
        threshold = deescalation_days_by_label.get(stable_label, default_deescalation_days)

        if raw_rank > stable_rank:
            stable_label = raw
            pending_label = None
            pending_count = 0
        elif raw_rank < stable_rank or raw != stable_label:
            if threshold == 0:
                stable_label = raw
                pending_label = None
                pending_count = 0
            else:
                if pending_label != raw:
                    pending_label = raw
                    pending_count = 1
                else:
                    pending_count += 1
                if pending_count >= threshold:
                    stable_label = raw
                    pending_label = None
                    pending_count = 0
        else:
            pending_label = None
            pending_count = 0

        stable.append(stable_label)
        if risk_rank[raw] > risk_rank[stable_label]:
            active.append(raw)
        else:
            active.append(stable_label)

    return stable, active


def apply_data_quality_aware_hysteresis(
    *,
    raw_labels: list[TLabel],
    risk_rank: dict[TLabel, int],
    deescalation_days_by_label: dict[TLabel, int],
    data_quality: Sequence[DataQuality],
    default_deescalation_days: int = 0,
    max_unknown_freeze_days: int = 0,
) -> tuple[list[TLabel], list[TLabel], list[bool]]:
    """Apply hysteresis while freezing regime memory across short data gaps.

    ``apply_per_label_asymmetric_hysteresis`` treats every label, including
    ``unknown``, as a real regime state. This wrapper preserves that pure
    behavior for classified days, but when data quality forces an ``unknown``
    output it holds the last known stable/active label for up to
    ``max_unknown_freeze_days`` sessions. Once the freeze window is exhausted,
    the output becomes the explicit ``unknown`` sentinel and the internal
    hysteresis state resets to ``unknown``.
    """
    if max_unknown_freeze_days < 0:
        raise ValueError("max_unknown_freeze_days must be >= 0")
    if len(raw_labels) != len(data_quality):
        raise ValueError("raw_labels and data_quality must have the same length")
    if max_unknown_freeze_days == 0:
        stable, active = apply_per_label_asymmetric_hysteresis(
            raw_labels=raw_labels,
            risk_rank=risk_rank,
            deescalation_days_by_label=deescalation_days_by_label,
            default_deescalation_days=default_deescalation_days,
        )
        return stable, active, [False] * len(raw_labels)

    if not raw_labels:
        raise ValueError("raw_labels must be non-empty")
    if "unknown" not in risk_rank:
        raise ValueError("risk_rank must include 'unknown' for data-quality freezes")

    stable: list[TLabel] = []
    active: list[TLabel] = []
    frozen: list[bool] = []

    unknown_label = "unknown"
    data_gap_count = 0
    stable_label: TLabel | None = None
    pending_label: TLabel | None = None
    pending_count = 0
    last_known_stable: TLabel | None = None
    last_known_active: TLabel | None = None

    for raw, dq in zip(raw_labels, data_quality, strict=True):
        if quality_forces_unknown(dq):
            data_gap_count += 1
            if (
                last_known_stable is not None
                and last_known_active is not None
                and data_gap_count <= max_unknown_freeze_days
            ):
                stable.append(last_known_stable)
                active.append(last_known_active)
                frozen.append(True)
                continue
            stable_label = unknown_label  # type: ignore[assignment]
            pending_label = None
            pending_count = 0
            stable.append(stable_label)
            active.append(stable_label)
            frozen.append(False)
            last_known_stable = None
            last_known_active = None
            continue

        data_gap_count = 0
        if stable_label is None:
            stable_label = raw
        else:
            raw_rank = risk_rank[raw]
            stable_rank = risk_rank[stable_label]
            threshold = deescalation_days_by_label.get(
                stable_label, default_deescalation_days
            )
            if raw_rank > stable_rank:
                stable_label = raw
                pending_label = None
                pending_count = 0
            elif raw_rank < stable_rank or raw != stable_label:
                if threshold == 0:
                    stable_label = raw
                    pending_label = None
                    pending_count = 0
                else:
                    if pending_label != raw:
                        pending_label = raw
                        pending_count = 1
                    else:
                        pending_count += 1
                    if pending_count >= threshold:
                        stable_label = raw
                        pending_label = None
                        pending_count = 0
            else:
                pending_label = None
                pending_count = 0

        stable.append(stable_label)
        if risk_rank[raw] > risk_rank[stable_label]:
            current_active = raw
        else:
            current_active = stable_label
        active.append(current_active)
        frozen.append(False)
        if raw != unknown_label and stable_label != unknown_label:
            last_known_stable = stable_label
            last_known_active = current_active

    return stable, active, frozen
