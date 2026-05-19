from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from regime_detection.trend_character import (
    TrendCharacterFeatures,
    compute_features,
    raw_label_for_day,
)


def test_trend_character_matches_pinned_fixtures(classified_golden_outputs) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )
    for row in golden["rows"]:
        as_of = date.fromisoformat(row["as_of_date"])
        out = classified_golden_outputs[as_of]
        assert out.trend_character.active_label == row["expected"]["trend_character"], (
            f"{as_of}: expected {row['expected']['trend_character']}, got {out.trend_character.active_label}"
        )


def _trend_character_features(
    *,
    close: float = 100.0,
    sma_50: float = 99.0,
    return_10d: float = 0.0,
    return_21d: float = 0.0,
    prior_63d_drawdown: float = 0.0,
    adx_14: float = 25.0,
) -> TrendCharacterFeatures:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
    return TrendCharacterFeatures(
        close=pd.Series([close], index=idx),
        high=pd.Series([close + 1.0], index=idx),
        low=pd.Series([close - 1.0], index=idx),
        sma_50=pd.Series([sma_50], index=idx),
        return_10d=pd.Series([return_10d], index=idx),
        return_21d=pd.Series([return_21d], index=idx),
        prior_63d_drawdown=pd.Series([prior_63d_drawdown], index=idx),
        adx_14=pd.Series([adx_14], index=idx),
        return_63d=pd.Series([0.0], index=idx),
        midpoint_excursion_20d=pd.Series([0.10], index=idx),
        breakout_20d_or_50d=pd.Series([False], index=idx),
        bb_width_expanding=pd.Series([False], index=idx),
        volume_above_20d_average=pd.Series([False], index=idx),
        followthrough_rate=pd.Series([float("nan")], index=idx),
    )


def test_trend_character_rolling_features_match_legacy_inline_formulas() -> None:
    index = pd.bdate_range("2023-01-02", periods=120)
    close = pd.Series(
        [100.0 + i * 0.5 + (i % 9) * 0.2 for i in range(len(index))],
        index=index,
        name="close",
    )
    high = close + 1.0
    low = close - 1.0

    out = compute_features(close=close, high=high, low=low)

    pd.testing.assert_series_equal(
        out.sma_50,
        close.rolling(50).mean(),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        out.return_63d,
        close / close.shift(63) - 1,
        check_exact=True,
    )


def test_trend_character_raw_label_thresholds_for_v1_labels() -> None:
    dt = pd.Timestamp("2024-01-02")

    assert raw_label_for_day(
        _trend_character_features(
            close=105.0,
            sma_50=100.0,
            return_10d=0.05,
            prior_63d_drawdown=-0.10,
            adx_14=30.0,
        ),
        dt,
        allow_v2_labels=False,
    )[0] == "recovery_attempt"
    assert raw_label_for_day(
        _trend_character_features(return_21d=0.05, adx_14=20.0),
        dt,
        allow_v2_labels=False,
    )[0] == "trending"
    assert raw_label_for_day(
        _trend_character_features(return_10d=0.02, return_21d=0.04, adx_14=19.99),
        dt,
        allow_v2_labels=False,
    )[0] == "chop"
    assert raw_label_for_day(
        _trend_character_features(return_10d=0.04, return_21d=0.04, adx_14=19.99),
        dt,
        allow_v2_labels=False,
    )[0] == "transition"


def test_trend_character_raw_label_unknown_when_required_feature_is_nan() -> None:
    dt = pd.Timestamp("2024-01-02")

    label, evidence = raw_label_for_day(
        _trend_character_features(adx_14=float("nan")),
        dt,
        allow_v2_labels=False,
    )

    assert label == "unknown"
    assert evidence == {"reason": "insufficient_history"}
