from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from regime_detection.config import load_default_regime_config
from regime_detection.trend_direction import (
    TrendDirectionFeatures,
    build_raw_outputs,
    compute_features,
    raw_label_for_day,
)
from regime_detection.trend_direction_v2 import TrendDirectionV2Features


def test_trend_direction_matches_pinned_fixtures(classified_golden_outputs) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )
    for row in golden["rows"]:
        if "expected" not in row:
            continue  # V2-axis rows run through the V2 harness, not here
        as_of = date.fromisoformat(row["as_of_date"])
        out = classified_golden_outputs[as_of]
        assert (
            out.trend_direction.active_label == row["expected"]["trend_direction"]
        ), f"{as_of}: expected {row['expected']['trend_direction']}, got {out.trend_direction.active_label}"


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


def _trend_direction_v2_features(
    *,
    dt: pd.Timestamp,
    return_63d: float,
    return_126d: float,
    drawdown_252d: float,
    sma_50: float,
    sma_200: float,
    realized_vol_21d: float,
    sentiment_score: float | None = None,
    news_sentiment_score: float | None = None,
    sentiment_concordance: float | None = None,
) -> TrendDirectionV2Features:
    idx = pd.DatetimeIndex([dt])
    nan = pd.Series([float("nan")], index=idx)

    def _optional(value: float | None, name: str) -> pd.Series | None:
        if value is None:
            return None
        return pd.Series([value], index=idx, name=name)

    return TrendDirectionV2Features(
        efficiency_ratio_20d=nan.copy(),
        hurst_250d=nan.copy(),
        slope_sma_50=nan.copy(),
        slope_sma_200=nan.copy(),
        return_63d=pd.Series([return_63d], index=idx),
        return_126d=pd.Series([return_126d], index=idx),
        drawdown_252d=pd.Series([drawdown_252d], index=idx),
        sma_50=pd.Series([sma_50], index=idx),
        sma_200=pd.Series([sma_200], index=idx),
        realized_vol_21d=pd.Series([realized_vol_21d], index=idx),
        sentiment_score=_optional(sentiment_score, "sentiment_score"),
        news_sentiment_score=_optional(news_sentiment_score, "news_sentiment_score"),
        sentiment_concordance=_optional(sentiment_concordance, "sentiment_concordance"),
    )


def test_trend_direction_rolling_features_match_legacy_inline_formulas() -> None:
    close = pd.Series(
        [100.0 + i + (i % 7) * 0.25 for i in range(240)],
        index=pd.bdate_range("2023-01-02", periods=240),
        name="close",
    )

    out = compute_features(close)

    pd.testing.assert_series_equal(
        out.sma_50,
        close.rolling(50).mean(),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        out.sma_200,
        close.rolling(200).mean(),
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        out.return_63d,
        close / close.shift(63) - 1,
        check_exact=True,
    )


def test_raw_label_for_day_is_single_source_of_truth_over_build_raw_outputs() -> None:
    # F-043: the per-day scalar path must be a thin wrapper over the vectorized
    # builder so the §3.5 rule predicates have ONE encoding. This guard fails if
    # either path ever diverges (v1-only mode and the v2 recovery/euphoria mode).
    close = pd.Series(
        [250.0 + i * 0.5 - (i % 11) * 1.5 for i in range(360)],
        index=pd.bdate_range("2022-06-01", periods=360),
        name="close",
    )
    features = compute_features(close)
    cfg = load_default_regime_config()
    assert cfg.trend_direction_v2 is not None

    from regime_detection.trend_direction_v2 import compute_trend_v2_features

    v2_features = compute_trend_v2_features(close, config=cfg.trend_direction_v2)

    def _norm(value: object) -> object:
        # NaN != NaN in Python; normalize so equal-NaN evidence compares equal.
        if isinstance(value, float) and pd.isna(value):
            return "__nan__"
        if isinstance(value, dict):
            return {k: _norm(v) for k, v in value.items()}
        return value

    for v2_kwargs in (
        {},
        {
            "trend_direction_v2_features": v2_features,
            "trend_direction_v2_rules": cfg.trend_direction_v2.rules,
        },
    ):
        labels, evidence = build_raw_outputs(features, **v2_kwargs)
        for idx, dt in enumerate(close.index):
            day_label, day_evidence = raw_label_for_day(features, dt, **v2_kwargs)
            assert day_label == labels[idx], f"{dt}: {day_label} != {labels[idx]}"
            assert _norm(day_evidence) == _norm(
                evidence[idx]
            ), f"{dt}: evidence mismatch"


def test_trend_direction_raw_label_thresholds_for_v1_labels() -> None:
    dt = pd.Timestamp("2024-01-02")

    assert (
        raw_label_for_day(
            _trend_direction_features(
                close=105.0, sma_50=102.0, sma_200=100.0, return_63d=0.08
            ),
            dt,
        )[0]
        == "bull"
    )
    assert (
        raw_label_for_day(
            _trend_direction_features(
                close=95.0, sma_50=98.0, sma_200=100.0, return_63d=-0.08
            ),
            dt,
        )[0]
        == "bear"
    )
    assert (
        raw_label_for_day(
            _trend_direction_features(
                close=101.0, sma_50=99.0, sma_200=100.0, return_63d=0.04
            ),
            dt,
        )[0]
        == "sideways"
    )
    assert (
        raw_label_for_day(
            _trend_direction_features(
                close=110.0, sma_50=95.0, sma_200=100.0, return_63d=0.07
            ),
            dt,
        )[0]
        == "transition"
    )


def test_trend_direction_raw_label_unknown_when_required_feature_is_nan() -> None:
    dt = pd.Timestamp("2024-01-02")

    label, evidence = raw_label_for_day(
        _trend_direction_features(
            close=105.0, sma_50=float("nan"), sma_200=100.0, return_63d=0.08
        ),
        dt,
    )

    assert label == "unknown"
    assert evidence == {"reason": "insufficient_history"}


def test_trend_direction_raw_label_unknown_can_be_overridden_by_v2_recovery() -> None:
    dt = pd.Timestamp("2024-01-02")
    cfg = load_default_regime_config()
    assert cfg.trend_direction_v2 is not None

    label, evidence = raw_label_for_day(
        _trend_direction_features(
            close=290.0,
            sma_50=float("nan"),
            sma_200=250.0,
            return_63d=0.12,
        ),
        dt,
        trend_direction_v2_features=_trend_direction_v2_features(
            dt=dt,
            return_63d=0.20,
            return_126d=0.05,
            drawdown_252d=-0.20,
            sma_50=280.0,
            sma_200=260.0,
            realized_vol_21d=0.10,
        ),
        trend_direction_v2_rules=cfg.trend_direction_v2.rules,
    )

    assert label == "recovery"
    assert evidence["reason"] == "insufficient_history"
    assert evidence["v2_override"] == {
        "from": "unknown",
        "to": "recovery",
        "rule": "recovery",
    }


def test_trend_direction_raw_label_surfaces_optional_v2_evidence_fields() -> None:
    dt = pd.Timestamp("2024-01-02")
    cfg = load_default_regime_config()
    assert cfg.trend_direction_v2 is not None

    label, evidence = raw_label_for_day(
        _trend_direction_features(
            close=105.0,
            sma_50=102.0,
            sma_200=100.0,
            return_63d=0.08,
        ),
        dt,
        trend_direction_v2_features=_trend_direction_v2_features(
            dt=dt,
            return_63d=0.08,
            return_126d=0.05,
            drawdown_252d=-0.02,
            sma_50=102.0,
            sma_200=100.0,
            realized_vol_21d=0.10,
            sentiment_score=12.0,
            news_sentiment_score=0.25,
            sentiment_concordance=1.0,
        ),
        trend_direction_v2_rules=cfg.trend_direction_v2.rules,
    )

    assert label == "bull"
    assert evidence["sentiment_score"] == 12.0
    assert evidence["news_sentiment_score"] == 0.25
    assert evidence["sentiment_concordance"] == 1.0
    assert "v2_override" not in evidence
