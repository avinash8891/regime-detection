from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from regime_detection.trend_direction import TrendDirectionFeatures, raw_label_for_day


def test_trend_direction_matches_pinned_fixtures(classified_golden_outputs) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )

    for row in golden["rows"]:
        as_of = date.fromisoformat(row["as_of_date"])
        out = classified_golden_outputs[as_of]
        assert out.trend_direction.active_label == row["expected"]["trend_direction"]


def _trend_direction_features(
    *,
    close: float,
    sma_50: float,
    sma_200: float,
    return_63d: float,
) -> TrendDirectionFeatures:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
    return TrendDirectionFeatures(
        close=pd.Series([close], index=idx),
        sma_50=pd.Series([sma_50], index=idx),
        sma_200=pd.Series([sma_200], index=idx),
        return_63d=pd.Series([return_63d], index=idx),
    )


def test_trend_direction_raw_label_thresholds_for_v1_labels() -> None:
    dt = pd.Timestamp("2024-01-02")

    assert raw_label_for_day(
        _trend_direction_features(close=105.0, sma_50=102.0, sma_200=100.0, return_63d=0.08),
        dt,
    )[0] == "bull"
    assert raw_label_for_day(
        _trend_direction_features(close=95.0, sma_50=98.0, sma_200=100.0, return_63d=-0.08),
        dt,
    )[0] == "bear"
    assert raw_label_for_day(
        _trend_direction_features(close=101.0, sma_50=99.0, sma_200=100.0, return_63d=0.04),
        dt,
    )[0] == "sideways"
    assert raw_label_for_day(
        _trend_direction_features(close=110.0, sma_50=95.0, sma_200=100.0, return_63d=0.07),
        dt,
    )[0] == "transition"


def test_trend_direction_raw_label_unknown_when_required_feature_is_nan() -> None:
    dt = pd.Timestamp("2024-01-02")

    label, evidence = raw_label_for_day(
        _trend_direction_features(close=105.0, sma_50=float("nan"), sma_200=100.0, return_63d=0.08),
        dt,
    )

    assert label == "unknown"
    assert evidence == {"reason": "insufficient_history"}
