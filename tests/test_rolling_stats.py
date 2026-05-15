from __future__ import annotations

import pandas as pd
import pytest

from regime_detection._rolling_stats import rolling_change_zscore


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
