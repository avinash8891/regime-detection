from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from regime_detection.breadth_state import (
    BreadthFeatures,
    build_raw_outputs,
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
        if "expected" not in row:
            continue  # V2-axis rows run through the V2 harness, not here
        as_of = date.fromisoformat(row["as_of_date"])
        expected = row["expected"]
        assert expected["breadth_state_raw"] in _BREADTH_LABELS
        assert expected["breadth_state_active"] in _BREADTH_LABELS
        out = classified_golden_outputs[as_of]
        assert (
            out.breadth_state.raw_label == expected["breadth_state_raw"]
        ), f"{as_of}: expected raw {expected['breadth_state_raw']}, got {out.breadth_state.raw_label}"
        assert (
            out.breadth_state.active_label == expected["breadth_state_active"]
        ), f"{as_of}: expected active {expected['breadth_state_active']}, got {out.breadth_state.active_label}"


def test_breadth_state_uses_written_etf_proxy_rules_not_invented_recovery_label(
    v2_market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
) -> None:
    from regime_detection.engine import RegimeEngine

    as_of = date(2023, 12, 14)
    market_data = v2_market_df_for_asof(as_of)
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

    assert out.breadth_state.raw_label == "broadening_breadth"
    assert out.breadth_state.active_label == "broadening_breadth"
    rule_evidence = out.breadth_state.evidence["rule_evidence"]
    assert rule_evidence["healthy_breadth"] is True
    assert rule_evidence["v1_raw_label"] == "healthy_breadth"
    assert rule_evidence["v2_broadening_breadth"] is True
    assert "recovery_breadth" not in rule_evidence


def test_raw_label_for_day_is_single_source_of_truth_over_build_raw_outputs() -> None:
    # F-043: the per-day scalar path must be a thin wrapper over the vectorized
    # builder so the §6.9 ETF-proxy rule predicates have ONE encoding. Guard
    # fails if the two v1 paths ever diverge in label or evidence shape.
    idx = pd.bdate_range("2023-01-02", periods=120)
    spy = pd.Series(
        [400.0 + i * 0.5 - (i % 9) * 1.5 for i in range(120)], index=idx, name="SPY"
    )
    rsp = pd.Series(
        [150.0 + i * 0.15 - (i % 5) * 0.6 for i in range(120)], index=idx, name="RSP"
    )
    features = compute_features(spy_close=spy, rsp_close=rsp)

    labels, evidence = build_raw_outputs(features)
    for i, dt in enumerate(idx):
        day_label, day_evidence = raw_label_for_day(features, dt)
        assert day_label == labels[i], f"{dt}: {day_label} != {labels[i]}"
        assert day_evidence == evidence[i], f"{dt}: evidence mismatch"


def test_index_distance_from_63d_high_warms_at_50th_observation() -> None:
    # F-011: §6.6/§6.8 pin index_distance_from_63d_high to
    # close.rolling(63, min_periods=50) — the 63d high requires 50 observations, NOT a
    # full 63. The prior assertion (62 sessions ⇒ all NaN) encoded a min_periods=63 bug;
    # corrected here to the spec boundary: first 49 NaN, the 50th observation valid.
    idx = pd.bdate_range("2024-01-02", periods=62)
    spy = pd.Series(range(100, 162), index=idx, dtype="float64")
    rsp = spy.copy()

    features = compute_features(spy_close=spy, rsp_close=rsp)
    dist = features.index_distance_from_63d_high

    assert dist.iloc[:49].isna().all()
    assert not pd.isna(dist.iloc[49])


def test_relative_breadth_sma50_requires_full_window() -> None:
    # F-007: §6.8 relative_breadth_sma50 must be NaN-masked until the 50-session
    # window is complete (min_periods=50). 49 sessions → all NaN; the 50th
    # session is the first non-NaN value.
    idx = pd.bdate_range("2024-01-02", periods=50)
    spy = pd.Series(range(100, 150), index=idx, dtype="float64")
    rsp = pd.Series(range(50, 100), index=idx, dtype="float64")

    short = compute_features(spy_close=spy.iloc[:49], rsp_close=rsp.iloc[:49])
    assert short.relative_breadth_sma50.isna().all()

    full = compute_features(spy_close=spy, rsp_close=rsp)
    assert full.relative_breadth_sma50.iloc[:49].isna().all()
    assert not pd.isna(full.relative_breadth_sma50.iloc[49])


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
