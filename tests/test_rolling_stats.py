from __future__ import annotations

import pandas as pd
import pytest

from regime_detection._rolling_stats import (
    period_return,
    rolling_change_zscore,
    simple_moving_average,
)


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


def test_rolling_change_zscore_rejects_non_positive_windows() -> None:
    series = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError, match="change_window must be > 0"):
        rolling_change_zscore(series, change_window=0, normalizer_window=3)

    with pytest.raises(ValueError, match="normalizer_window must be > 0"):
        rolling_change_zscore(series, change_window=1, normalizer_window=0)


def test_simple_moving_average_matches_inline_strict_rolling_mean() -> None:
    close = pd.Series(
        [100.0, 101.5, 102.0, 103.5, 105.0],
        index=pd.bdate_range("2024-01-02", periods=5),
        name="close",
    )

    out = simple_moving_average(close, window=3, output_name="sma_3")
    expected = close.astype(float).rolling(window=3, min_periods=3).mean().rename("sma_3")

    pd.testing.assert_series_equal(out, expected, check_exact=True)


def test_period_return_matches_inline_shift_return() -> None:
    close = pd.Series(
        [100.0, 101.0, 102.0, 104.0, 108.0],
        index=pd.bdate_range("2024-01-02", periods=5),
        name="close",
    )

    out = period_return(close, periods=2, output_name="return_2d")
    expected = (close.astype(float) / close.astype(float).shift(2) - 1.0).rename("return_2d")

    pd.testing.assert_series_equal(out, expected, check_exact=True)


def test_rolling_helpers_reject_non_positive_windows() -> None:
    series = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError, match="window must be > 0"):
        simple_moving_average(series, window=0)

    with pytest.raises(ValueError, match="periods must be > 0"):
        period_return(series, periods=0)
