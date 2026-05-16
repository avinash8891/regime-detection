from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar


TLabel = TypeVar("TLabel", bound=str)


def apply_asymmetric_hysteresis(
    *,
    raw_labels: list[TLabel],
    risk_rank: Mapping[str, int],
    escalation_days: int = 1,
    deescalation_days: int,
) -> tuple[list[TLabel], list[TLabel]]:
    """
    Generic asymmetric hysteresis (spec §2.10):
    - Escalation (higher risk_rank) requires `escalation_days` consecutive
      days on the candidate label. Default 1 preserves immediate escalation.
    - De-escalation requires `deescalation_days` consecutive days on the candidate label.
    - active_label uses fast-path: if raw is riskier than stable, active=raw else active=stable.
    """
    if escalation_days < 1:
        raise ValueError("escalation_days must be >= 1")
    if deescalation_days < 0:
        raise ValueError("deescalation_days must be >= 0")
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

        if raw_rank > stable_rank:
            if escalation_days == 1:
                stable_label = raw
                pending_label = None
                pending_count = 0
            else:
                if pending_label != raw:
                    pending_label = raw
                    pending_count = 1
                else:
                    pending_count += 1
                if pending_count >= escalation_days:
                    stable_label = raw
                    pending_label = None
                    pending_count = 0
        elif raw_rank < stable_rank or raw != stable_label:
            if deescalation_days == 0:
                stable_label = raw
                pending_label = None
                pending_count = 0
            else:
                if pending_label != raw:
                    pending_label = raw
                    pending_count = 1
                else:
                    pending_count += 1
                if pending_count >= deescalation_days:
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


def apply_per_label_asymmetric_hysteresis(
    *,
    raw_labels: list[TLabel],
    risk_rank: Mapping[str, int],
    deescalation_days_by_label: Mapping[str, int],
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
