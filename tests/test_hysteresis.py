from __future__ import annotations

import pytest

from regime_detection.hysteresis import apply_asymmetric_hysteresis


_RISK_RANK = {"low": 0, "medium": 1, "high": 2}


def test_apply_asymmetric_hysteresis_rejects_empty_raw_labels() -> None:
    with pytest.raises(ValueError, match="raw_labels must be non-empty"):
        apply_asymmetric_hysteresis(
            raw_labels=[],
            risk_rank=_RISK_RANK,
            deescalation_days=2,
        )


def test_apply_asymmetric_hysteresis_keeps_identical_labels_stable() -> None:
    stable, active = apply_asymmetric_hysteresis(
        raw_labels=["medium", "medium", "medium"],
        risk_rank=_RISK_RANK,
        deescalation_days=2,
    )

    assert stable == ["medium", "medium", "medium"]
    assert active == ["medium", "medium", "medium"]


def test_apply_asymmetric_hysteresis_suppresses_fast_deescalation_oscillation() -> None:
    stable, active = apply_asymmetric_hysteresis(
        raw_labels=["high", "low", "high", "low", "high"],
        risk_rank=_RISK_RANK,
        deescalation_days=2,
    )

    assert stable == ["high", "high", "high", "high", "high"]
    assert active == ["high", "high", "high", "high", "high"]


def test_apply_asymmetric_hysteresis_escalates_immediately_and_deescalates_after_window() -> None:
    stable, active = apply_asymmetric_hysteresis(
        raw_labels=["low", "high", "medium", "medium"],
        risk_rank=_RISK_RANK,
        deescalation_days=2,
    )

    assert stable == ["low", "high", "high", "medium"]
    assert active == ["low", "high", "high", "medium"]
