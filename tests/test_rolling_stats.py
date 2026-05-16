from __future__ import annotations

import math

import pandas as pd
import pytest

from regime_detection._rolling_stats import period_return, rolling_change_zscore, sma


def test_rolling_change_zscore_returns_nan_until_full_change_and_normalizer_windows() -> None:
    series = pd.Series([1.0, 2.0, 4.0, 7.0], index=pd.bdate_range("2024-01-02", periods=4))

    out = rolling_change_zscore(
        series,
        change_window=2,
        normalizer_window=3,
        output_name="change_z",
    )

    assert out.name == "change_z"
    assert out.isna().all()


def test_rolling_change_zscore_masks_zero_variance_windows_to_nan() -> None:
    series = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 14.0],
        index=pd.bdate_range("2024-01-02", periods=5),
    )

    out = rolling_change_zscore(
        series,
        change_window=1,
        normalizer_window=3,
    )

    assert out.isna().all()


def test_rolling_change_zscore_computes_change_zscore_against_rolling_change_distribution() -> None:
    series = pd.Series(
        [10.0, 11.0, 13.0, 16.0, 20.0],
        index=pd.bdate_range("2024-01-02", periods=5),
    )

    out = rolling_change_zscore(
        series,
        change_window=1,
        normalizer_window=3,
    )

    assert out.iloc[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# sma()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sma_returns_nan_for_insufficient_window() -> None:
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx)

    result = sma(series, 3)

    assert math.isnan(result.iloc[0])
    assert math.isnan(result.iloc[1])
    assert not math.isnan(result.iloc[2])
    assert not math.isnan(result.iloc[3])
    assert not math.isnan(result.iloc[4])


@pytest.mark.unit
def test_sma_hand_computed() -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    series = pd.Series([100.0, 110.0, 120.0], index=idx)

    result = sma(series, 2)

    assert math.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(105.0)
    assert result.iloc[2] == pytest.approx(115.0)


@pytest.mark.unit
def test_sma_window_1_equals_series() -> None:
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)

    result = sma(series, 1)

    for i in range(len(series)):
        assert result.iloc[i] == pytest.approx(series.iloc[i])


# ---------------------------------------------------------------------------
# period_return()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_period_return_hand_computed() -> None:
    idx = pd.date_range("2024-01-02", periods=2, freq="B")
    series = pd.Series([100.0, 110.0], index=idx)

    result = period_return(series, 1)

    assert math.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.10)


@pytest.mark.unit
def test_period_return_periods_2() -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    series = pd.Series([100.0, 110.0, 121.0], index=idx)

    result = period_return(series, 2)

    assert math.isnan(result.iloc[0])
    assert math.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(0.21)


@pytest.mark.unit
def test_period_return_first_periods_values_are_nan() -> None:
    periods = 3
    n = 8
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    series = pd.Series([float(i * 10) for i in range(1, n + 1)], index=idx)

    result = period_return(series, periods)

    for i in range(periods):
        assert math.isnan(result.iloc[i]), f"Expected NaN at position {i}"
    for i in range(periods, n):
        assert not math.isnan(result.iloc[i]), f"Expected non-NaN at position {i}"


# ---------------------------------------------------------------------------
# rolling_change_zscore()
# ---------------------------------------------------------------------------


def test_rolling_change_zscore_rejects_non_positive_windows() -> None:
    series = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError, match="change_window must be > 0"):
        rolling_change_zscore(series, change_window=0, normalizer_window=3)

    with pytest.raises(ValueError, match="normalizer_window must be > 0"):
        rolling_change_zscore(series, change_window=1, normalizer_window=0)
