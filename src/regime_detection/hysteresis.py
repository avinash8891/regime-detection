from __future__ import annotations

from typing import TypeVar


TLabel = TypeVar("TLabel", bound=str)


def apply_asymmetric_hysteresis(
    *,
    raw_labels: list[TLabel],
    risk_rank: dict[TLabel, int],
    deescalation_days: int,
) -> tuple[list[TLabel], list[TLabel]]:
    """
    Generic asymmetric hysteresis (spec §2.10):
    - Escalation (higher risk_rank) updates stable immediately.
    - De-escalation requires `deescalation_days` consecutive days on the candidate label.
    - active_label uses fast-path: if raw is riskier than stable, active=raw else active=stable.
    """
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
