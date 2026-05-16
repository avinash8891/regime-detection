from __future__ import annotations

from datetime import date

import pandas as pd

from regime_detection.data_quality import (
    assess_series_input_quality,
    quality_forces_unknown,
)
from regime_detection.models import DataQuality


def _series(values: list[float | None]) -> pd.Series:
    return pd.Series(
        values,
        index=pd.bdate_range("2024-01-02", periods=len(values)),
        dtype="float64",
    )


def test_assess_series_input_quality_returns_stale_data_status() -> None:
    series = _series([1.0, 2.0, None, None, None])

    dq = assess_series_input_quality(
        as_of_date=date(2024, 1, 8),
        required_inputs=[series],
        required_trading_days=5,
        raw_label="bull",
        max_freshness_days=3,
        min_completeness=0.95,
    )

    assert dq.status == "stale_data"
    assert dq.freshness_days == 5
    assert dq.completeness == 0.4
    assert dq.reason == "stale_data"


def test_assess_series_input_quality_returns_degraded_above_insufficient_floor() -> None:
    series = _series([1.0, 2.0, 3.0, 4.0, None])

    dq = assess_series_input_quality(
        as_of_date=date(2024, 1, 8),
        required_inputs=[series],
        required_trading_days=5,
        raw_label="bull",
        max_freshness_days=3,
        min_completeness=0.95,
    )

    assert dq.status == "degraded"
    assert dq.freshness_days == 3
    assert dq.completeness == 0.8
    assert dq.reason == "incomplete_data"


def test_assess_series_input_quality_multiple_inputs_uses_worst_quality() -> None:
    complete = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    sparse = _series([1.0, None, None, None, 5.0])

    dq = assess_series_input_quality(
        as_of_date=date(2024, 1, 8),
        required_inputs=[complete, sparse],
        required_trading_days=5,
        raw_label="bull",
        max_freshness_days=3,
        min_completeness=0.95,
    )

    assert dq.status == "insufficient_data"
    assert dq.freshness_days == 0
    assert dq.completeness == 0.4
    assert dq.reason == "insufficient_data"


def test_quality_forces_unknown_only_for_terminal_bad_quality_statuses() -> None:
    assert quality_forces_unknown(
        DataQuality(status="insufficient_history", freshness_days=None, completeness=None)
    )
    assert quality_forces_unknown(
        DataQuality(status="insufficient_data", freshness_days=0, completeness=0.5)
    )
    assert quality_forces_unknown(
        DataQuality(status="stale_data", freshness_days=5, completeness=1.0)
    )
    assert not quality_forces_unknown(
        DataQuality(status="degraded", freshness_days=0, completeness=0.8)
    )
    assert not quality_forces_unknown(
        DataQuality(status="ok", freshness_days=0, completeness=1.0)
    )
