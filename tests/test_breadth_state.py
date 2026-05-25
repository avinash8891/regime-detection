from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from regime_detection.breadth_state import (
    BreadthFeatures,
    compute_features,
    raw_label_for_day,
)

_BREADTH_LABELS = {
    "breadth_thrust",
    "divergent_fragile",
    "narrowing_breadth",
    "recovery_breadth",
    "broadening_breadth",
    "weak_breadth",
    "healthy_breadth",
    "neutral_breadth",
    "unknown",
}


def test_breadth_state_matches_pinned_fixtures(classified_golden_outputs) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )
    for row in golden["rows"]:
        as_of = date.fromisoformat(row["as_of_date"])
        out = classified_golden_outputs[as_of]
        assert out.breadth_state.active_label in _BREADTH_LABELS


def test_breadth_state_uses_written_etf_proxy_rules_not_invented_recovery_label(
    market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
) -> None:
    from regime_detection.engine import RegimeEngine

    as_of = date(2023, 12, 14)
    market_data = market_df_for_asof(as_of)
    rsp_recent_idx = (
        market_data[market_data["symbol"] == "RSP"].sort_values("date").tail(20).index
    )
    market_data.loc[rsp_recent_idx, "close"] = (
        market_data.loc[rsp_recent_idx, "close"] * 1.20
    )

    out = RegimeEngine().classify(
        as_of_date=as_of,
        market_data=market_data,
        **synthetic_v2_kwargs_for_market_data(market_data),
    )

    assert out.breadth_state.raw_label == "healthy_breadth"
    assert out.breadth_state.active_label == "healthy_breadth"
    rule_evidence = out.breadth_state.evidence["rule_evidence"]
    assert rule_evidence["healthy_breadth"] is True
    assert "recovery_breadth" not in rule_evidence


def test_index_distance_from_63d_high_requires_full_window() -> None:
    idx = pd.bdate_range("2024-01-02", periods=62)
    spy = pd.Series(range(100, 162), index=idx, dtype="float64")
    rsp = spy.copy()

    features = compute_features(spy_close=spy, rsp_close=rsp)

    assert features.index_distance_from_63d_high.isna().all()


def _breadth_features(
    *,
    ratio: float,
    ratio_sma50: float,
    ratio_ret20: float,
    index_distance_from_63d_high: float,
) -> BreadthFeatures:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
    return BreadthFeatures(
        spy_close=pd.Series([100.0], index=idx),
        rsp_close=pd.Series([ratio * 100.0], index=idx),
        relative_breadth_ratio=pd.Series([ratio], index=idx),
        relative_breadth_sma50=pd.Series([ratio_sma50], index=idx),
        relative_breadth_return_20d=pd.Series([ratio_ret20], index=idx),
        index_distance_from_63d_high=pd.Series(
            [index_distance_from_63d_high], index=idx
        ),
    )


def test_breadth_raw_label_thresholds_for_v1_proxy_labels() -> None:
    dt = pd.Timestamp("2024-01-02")

    assert (
        raw_label_for_day(
            _breadth_features(
                ratio=0.95,
                ratio_sma50=1.0,
                ratio_ret20=-0.03,
                index_distance_from_63d_high=-0.05,
            ),
            dt,
        )[0]
        == "divergent_fragile"
    )
    assert (
        raw_label_for_day(
            _breadth_features(
                ratio=0.99,
                ratio_sma50=1.0,
                ratio_ret20=-0.01,
                index_distance_from_63d_high=-0.10,
            ),
            dt,
        )[0]
        == "weak_breadth"
    )
    assert (
        raw_label_for_day(
            _breadth_features(
                ratio=1.01,
                ratio_sma50=1.0,
                ratio_ret20=0.0,
                index_distance_from_63d_high=-0.10,
            ),
            dt,
        )[0]
        == "healthy_breadth"
    )
    assert (
        raw_label_for_day(
            _breadth_features(
                ratio=1.0,
                ratio_sma50=1.0,
                ratio_ret20=0.01,
                index_distance_from_63d_high=-0.10,
            ),
            dt,
        )[0]
        == "neutral_breadth"
    )


def test_breadth_raw_label_unknown_when_required_feature_is_nan() -> None:
    dt = pd.Timestamp("2024-01-02")

    label, evidence = raw_label_for_day(
        _breadth_features(
            ratio=1.0,
            ratio_sma50=float("nan"),
            ratio_ret20=0.0,
            index_distance_from_63d_high=-0.10,
        ),
        dt,
    )

    assert label == "unknown"
    assert evidence == {"reason": "insufficient_history"}
