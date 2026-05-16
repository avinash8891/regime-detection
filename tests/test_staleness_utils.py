"""Unit tests for regime_detection._staleness_utils."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from regime_detection._staleness_utils import (
    _STALENESS_SENTINEL,
    calendar_staleness_days_series,
    safe_float,
    trading_staleness_series,
)

# Business-day session index used across tests
_SESSION_INDEX = pd.bdate_range("2024-01-02", periods=10)


# ---------------------------------------------------------------------------
# calendar_staleness_days_series()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_calendar_staleness_days_series_none_returns_all_sentinel() -> None:
    result = calendar_staleness_days_series(None, _SESSION_INDEX)

    assert len(result) == len(_SESSION_INDEX)
    assert (result == _STALENESS_SENTINEL).all()


@pytest.mark.unit
def test_calendar_staleness_days_series_real_sparse_series() -> None:
    # Build a sparse series: valid on day 0 (2024-01-02) and day 5 (2024-01-09).
    # All other sessions are NaN.
    idx = _SESSION_INDEX
    values = [float("nan")] * len(idx)
    values[0] = 100.0  # 2024-01-02
    values[5] = 200.0  # 2024-01-09
    series = pd.Series(values, index=idx)

    result = calendar_staleness_days_series(series, idx)

    # On day 0 itself: 0 calendar days of staleness.
    assert result.iloc[0] == 0

    # On day 1 (2024-01-03): 1 calendar day since 2024-01-02.
    assert result.iloc[1] == 1

    # On day 5 (2024-01-09): valid observation resets to 0.
    assert result.iloc[5] == 0

    # On day 6 (2024-01-10): 1 calendar day since 2024-01-09.
    assert result.iloc[6] == 1

    # On day 4 (2024-01-08): last valid was 2024-01-02 (Mon→Mon = 6 cal days).
    # 2024-01-08 is a Monday; 2024-01-02 is a Tuesday → 6 calendar days apart.
    expected_day4 = (idx[4] - idx[0]).days
    assert result.iloc[4] == expected_day4


@pytest.mark.unit
def test_calendar_staleness_days_series_no_valid_values() -> None:
    # Series with all NaN — should return sentinel everywhere except it
    # cannot forward-fill from a valid date, so sentinel fills in.
    series = pd.Series([float("nan")] * len(_SESSION_INDEX), index=_SESSION_INDEX)

    result = calendar_staleness_days_series(series, _SESSION_INDEX)

    assert (result == _STALENESS_SENTINEL).all()


# ---------------------------------------------------------------------------
# trading_staleness_series()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trading_staleness_series_none_returns_all_sentinel() -> None:
    result = trading_staleness_series(None, _SESSION_INDEX)

    assert len(result) == len(_SESSION_INDEX)
    assert (result == _STALENESS_SENTINEL).all()


@pytest.mark.unit
def test_trading_staleness_series_real_sparse_series() -> None:
    # Valid on session 0 and session 5; rest NaN.
    idx = _SESSION_INDEX
    values = [float("nan")] * len(idx)
    values[0] = 100.0
    values[5] = 200.0
    series = pd.Series(values, index=idx)

    result = trading_staleness_series(series, idx)

    # Session 0: last valid is session 0 → 0 sessions stale.
    assert result.iloc[0] == 0

    # Session 1: last valid was session 0 → 1 trading session stale.
    assert result.iloc[1] == 1

    # Session 4: last valid was session 0 → 4 trading sessions stale.
    assert result.iloc[4] == 4

    # Session 5: valid observation → 0 sessions stale.
    assert result.iloc[5] == 0

    # Session 7: last valid was session 5 → 2 trading sessions stale.
    assert result.iloc[7] == 2


@pytest.mark.unit
def test_trading_staleness_series_counts_sessions_not_calendar_days() -> None:
    # Use an index that spans a weekend to confirm we count sessions, not days.
    # 2024-01-05 (Fri) and 2024-01-08 (Mon) are consecutive trading sessions.
    idx = pd.bdate_range("2024-01-05", periods=4)  # Fri, Mon, Tue, Wed
    values = [100.0, float("nan"), float("nan"), float("nan")]
    series = pd.Series(values, index=idx)

    result = trading_staleness_series(series, idx)

    # Fri: 0 sessions stale.
    assert result.iloc[0] == 0
    # Mon: 1 trading session stale (not 3 calendar days).
    assert result.iloc[1] == 1
    # Tue: 2 trading sessions stale.
    assert result.iloc[2] == 2


# ---------------------------------------------------------------------------
# safe_float()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_safe_float_returns_float_when_dt_present_and_real() -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    series = pd.Series([1.5, 2.5, 3.5], index=idx)
    dt = idx[1]

    result = safe_float(series, dt)

    assert result == pytest.approx(2.5)
    assert isinstance(result, float)


@pytest.mark.unit
def test_safe_float_returns_nan_when_dt_present_and_nan() -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    series = pd.Series([1.5, float("nan"), 3.5], index=idx)
    dt = idx[1]

    result = safe_float(series, dt)

    assert math.isnan(result)


@pytest.mark.unit
def test_safe_float_returns_nan_when_dt_not_in_index() -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    series = pd.Series([1.5, 2.5, 3.5], index=idx)
    dt = pd.Timestamp("2025-06-15")  # Not in index

    result = safe_float(series, dt)

    assert math.isnan(result)
