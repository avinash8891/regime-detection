from __future__ import annotations

import pytest

from regime_detection.hysteresis import apply_per_label_asymmetric_hysteresis
from regime_detection.network_fragility_rules import NETWORK_FRAGILITY_RISK_RANK


# Use the canonical risk-rank constant (v2 §3.6) — never re-state it locally.
# A prior local fixture used `systemic_stress: 4` which silently diverged from
# the spec (systemic_stress: 3, tied with correlation_to_one); see
# Implementation Ambiguity Log entry #7.
NETWORK_FRAGILITY_RANK = NETWORK_FRAGILITY_RISK_RANK

NETWORK_FRAGILITY_DEESCALATION_DAYS: dict[str, int] = {
    "rising_fragility": 3,
    "correlation_concentration": 3,
    "correlation_to_one": 5,
    "systemic_stress": 5,
}


def test_escalation_is_immediate_regardless_of_label_threshold() -> None:
    raws = ["diversified_normal", "rising_fragility"]

    stable, active = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=NETWORK_FRAGILITY_RANK,
        deescalation_days_by_label=NETWORK_FRAGILITY_DEESCALATION_DAYS,
    )

    assert stable == ["diversified_normal", "rising_fragility"]
    assert active == ["diversified_normal", "rising_fragility"]


def test_deescalation_uses_threshold_keyed_on_label_being_left() -> None:
    # Start escalated to correlation_to_one (rank 3, threshold 5).
    # Then raw drops to diversified_normal (rank 0) for 6 days.
    raws = ["correlation_to_one"] + ["diversified_normal"] * 6

    stable, active = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=NETWORK_FRAGILITY_RANK,
        deescalation_days_by_label=NETWORK_FRAGILITY_DEESCALATION_DAYS,
    )

    # Day 0: stable = correlation_to_one (seeded from raw_labels[0]).
    # Day 1..5: 5 consecutive days of diversified_normal — pending_count reaches 5,
    #           which equals the correlation_to_one threshold, so stable commits to
    #           diversified_normal on day 5 (index 5 in zero-indexed listing).
    # Day 6: stable already diversified_normal.
    assert stable == ["correlation_to_one"] * 5 + ["diversified_normal"] * 2
    # active fast-path: when raw is lower-risk than stable, active = stable.
    # Once stable commits to diversified_normal, raw==stable so active = diversified_normal.
    assert active == ["correlation_to_one"] * 5 + ["diversified_normal"] * 2


def test_systemic_stress_requires_five_days_to_deescalate() -> None:
    raws = ["systemic_stress"] + ["diversified_normal"] * 5

    stable, active = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=NETWORK_FRAGILITY_RANK,
        deescalation_days_by_label=NETWORK_FRAGILITY_DEESCALATION_DAYS,
    )

    # systemic_stress threshold is 5. Day 1..4: count 1..4 (no commit).
    # Day 5: count = 5 reaches threshold, commits on day 5 (index 5).
    assert stable == ["systemic_stress"] * 5 + ["diversified_normal"]


def test_unmapped_label_falls_back_to_default_threshold() -> None:
    # diversified_normal is not in NETWORK_FRAGILITY_DEESCALATION_DAYS — default = 0 → immediate.
    # Start at diversified_normal (no escalation pending), escalate to rising_fragility
    # (threshold 3), then drop raw to diversified_normal for 2 days.
    raws = [
        "diversified_normal",
        "rising_fragility",
        "diversified_normal",
        "diversified_normal",
    ]

    stable, active = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=NETWORK_FRAGILITY_RANK,
        deescalation_days_by_label=NETWORK_FRAGILITY_DEESCALATION_DAYS,
        default_deescalation_days=0,
    )

    # Day 0: stable = diversified_normal.
    # Day 1: escalation to rising_fragility — immediate.
    # Day 2: pending=1, threshold(rising_fragility)=3 — not committed.
    # Day 3: pending=2 — not committed.
    assert stable == [
        "diversified_normal",
        "rising_fragility",
        "rising_fragility",
        "rising_fragility",
    ]


def test_threshold_resets_when_raw_changes_mid_deescalation() -> None:
    # Mid-pending switch of raw label resets the pending counter.
    raws = [
        "correlation_to_one",
        "diversified_normal",
        "diversified_normal",
        "correlation_concentration",  # different raw, resets pending
        "diversified_normal",
        "diversified_normal",
    ]

    stable, active = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=NETWORK_FRAGILITY_RANK,
        deescalation_days_by_label=NETWORK_FRAGILITY_DEESCALATION_DAYS,
    )

    # stable stays correlation_to_one throughout — no pending reaches 5.
    assert stable == ["correlation_to_one"] * 6


def test_same_label_streak_resets_pending() -> None:
    raws = [
        "systemic_stress",
        "diversified_normal",
        "systemic_stress",  # raw == stable — pending resets
        "diversified_normal",
        "diversified_normal",
        "diversified_normal",
        "diversified_normal",
        "diversified_normal",
    ]

    stable, active = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=NETWORK_FRAGILITY_RANK,
        deescalation_days_by_label=NETWORK_FRAGILITY_DEESCALATION_DAYS,
    )

    # Day 1: pending diversified_normal, count=1.
    # Day 2: raw == stable systemic_stress, pending resets.
    # Day 3..7: count 1..5; threshold 5 commits on day 7.
    assert stable == ["systemic_stress"] * 7 + ["diversified_normal"]


def test_active_fast_path_emits_raw_when_escalating_within_stable() -> None:
    # Escalation that hasn't been promoted to stable yet must still appear in active.
    raws = ["diversified_normal", "rising_fragility", "diversified_normal"]

    stable, active = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=NETWORK_FRAGILITY_RANK,
        deescalation_days_by_label=NETWORK_FRAGILITY_DEESCALATION_DAYS,
    )

    # Day 0: stable=diversified_normal, raw=diversified_normal, active=diversified_normal.
    # Day 1: escalation commits stable=rising_fragility; active=rising_fragility.
    # Day 2: raw=diversified_normal (rank 0) < stable=rising_fragility (rank 2);
    #        active uses stable.
    assert active == ["diversified_normal", "rising_fragility", "rising_fragility"]


def test_raises_on_negative_default_deescalation_days() -> None:
    with pytest.raises(ValueError, match="default_deescalation_days must be >= 0"):
        apply_per_label_asymmetric_hysteresis(
            raw_labels=["diversified_normal"],
            risk_rank=NETWORK_FRAGILITY_RANK,
            deescalation_days_by_label={},
            default_deescalation_days=-1,
        )


def test_raises_on_negative_per_label_value() -> None:
    with pytest.raises(ValueError, match="deescalation_days_by_label"):
        apply_per_label_asymmetric_hysteresis(
            raw_labels=["correlation_to_one"],
            risk_rank=NETWORK_FRAGILITY_RANK,
            deescalation_days_by_label={"correlation_to_one": -1},
        )


def test_raises_on_empty_raw_labels() -> None:
    with pytest.raises(ValueError, match="raw_labels must be non-empty"):
        apply_per_label_asymmetric_hysteresis(
            raw_labels=[],
            risk_rank=NETWORK_FRAGILITY_RANK,
            deescalation_days_by_label=NETWORK_FRAGILITY_DEESCALATION_DAYS,
        )
