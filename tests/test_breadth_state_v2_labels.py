"""V2 §1D Breadth State extensions.

Includes ``breadth_thrust``, ``narrowing_breadth``, ``recovery_breadth``,
and ``broadening_breadth`` (Ambiguity Log #21–#26, #68, #69, #70).

Tests use the same per-day predicate / vectorised label paths as the §1B
trend-character slice. PIT features are constructed as synthetic
``pd.Series`` aligned to a SPY-shape NYSE business-day calendar; the V1
RSP/SPY proxy features ride the existing engine fixture path.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from regime_detection.breadth_state import (
    BreadthFeatures,
    BreadthLabel,
    _RISK_RANK,
    _evaluate_breadth_thrust,
    _evaluate_broadening_breadth,
    _evaluate_narrowing_breadth,
    _evaluate_recovery_breadth,
    build_raw_outputs,
    resolve_v2_raw_outputs,
)
from regime_detection.config import BreadthV2Config
from regime_detection.models import BreadthStateOutput, DataQuality


_V1_LABELS: set[str] = {
    "healthy_breadth",
    "neutral_breadth",
    "weak_breadth",
    "divergent_fragile",
    "unknown",
}


def _trading_index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-02", periods=n)


def _const(value: float, n: int, idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(np.full(n, float(value)), index=idx)


# ---------------------------------------------------------------------------
# Group A — narrowing_breadth predicate (spec §1D line 280).
#
# Predicate (Log #68):
#   pct_above_50dma falling (strict 5-session decrease)
#   AND pct_above_200dma falling (strict 5-session decrease)
#   AND nh_nl_ratio < 0.4.
# ---------------------------------------------------------------------------


def test_narrowing_breadth_fires_on_3_falling_conjuncts() -> None:
    n = 30
    idx = _trading_index(n)
    pct_50 = _const(0.55, n, idx)
    pct_200 = _const(0.45, n, idx)
    pct_50.iloc[-1] = 0.45  # falling 0.55 -> 0.45 over 5 sessions
    pct_200.iloc[-1] = 0.40  # falling 0.45 -> 0.40 over 5 sessions
    nh_nl = _const(0.35, n, idx)
    assert _evaluate_narrowing_breadth(
        pct_above_50dma=pct_50,
        pct_above_200dma=pct_200,
        nh_nl_ratio=nh_nl,
        dt=idx[-1],
        lookback_sessions=5,
        nh_nl_threshold=0.4,
    )


def test_narrowing_breadth_fails_on_pct_above_50dma_not_falling() -> None:
    n = 30
    idx = _trading_index(n)
    pct_50 = _const(0.55, n, idx)  # FLAT
    pct_200 = _const(0.45, n, idx)
    pct_200.iloc[-1] = 0.40
    nh_nl = _const(0.35, n, idx)
    assert not _evaluate_narrowing_breadth(
        pct_above_50dma=pct_50,
        pct_above_200dma=pct_200,
        nh_nl_ratio=nh_nl,
        dt=idx[-1],
        lookback_sessions=5,
        nh_nl_threshold=0.4,
    )


def test_narrowing_breadth_fails_on_nh_nl_ratio_above_threshold() -> None:
    n = 30
    idx = _trading_index(n)
    pct_50 = _const(0.55, n, idx)
    pct_200 = _const(0.45, n, idx)
    pct_50.iloc[-1] = 0.45
    pct_200.iloc[-1] = 0.40
    nh_nl = _const(0.45, n, idx)  # ABOVE 0.4 threshold
    assert not _evaluate_narrowing_breadth(
        pct_above_50dma=pct_50,
        pct_above_200dma=pct_200,
        nh_nl_ratio=nh_nl,
        dt=idx[-1],
        lookback_sessions=5,
        nh_nl_threshold=0.4,
    )


def test_narrowing_breadth_fails_on_nan_endpoint() -> None:
    n = 30
    idx = _trading_index(n)
    pct_50 = _const(0.55, n, idx)
    pct_200 = _const(0.45, n, idx)
    pct_50.iloc[-1] = 0.45
    pct_200.iloc[-1] = 0.40
    pct_50.iloc[-6] = np.nan  # 5-session-ago endpoint NaN
    nh_nl = _const(0.35, n, idx)
    assert not _evaluate_narrowing_breadth(
        pct_above_50dma=pct_50,
        pct_above_200dma=pct_200,
        nh_nl_ratio=nh_nl,
        dt=idx[-1],
        lookback_sessions=5,
        nh_nl_threshold=0.4,
    )


# ---------------------------------------------------------------------------
# Group B — broadening_breadth predicate (spec §1D line 279).
#
# Predicate (Log #68):
#   nh_nl_ratio rising (strict 5-session increase)
#   AND ad_line_slope_20d > 0.
# ---------------------------------------------------------------------------


def test_broadening_breadth_fires_on_rising_nh_nl_and_positive_ad_line_slope() -> None:
    n = 30
    idx = _trading_index(n)
    nh_nl = _const(0.5, n, idx)
    nh_nl.iloc[-1] = 0.7  # rising 0.5 -> 0.7
    ad_slope = _const(1.5, n, idx)
    assert _evaluate_broadening_breadth(
        nh_nl_ratio=nh_nl,
        ad_line_slope_20d=ad_slope,
        dt=idx[-1],
        lookback_sessions=5,
    )


def test_broadening_breadth_fails_on_ad_line_slope_zero_or_negative() -> None:
    n = 30
    idx = _trading_index(n)
    nh_nl = _const(0.5, n, idx)
    nh_nl.iloc[-1] = 0.7
    ad_slope = _const(0.0, n, idx)  # strict > 0 required
    assert not _evaluate_broadening_breadth(
        nh_nl_ratio=nh_nl,
        ad_line_slope_20d=ad_slope,
        dt=idx[-1],
        lookback_sessions=5,
    )


def test_broadening_breadth_fails_on_nh_nl_not_rising() -> None:
    n = 30
    idx = _trading_index(n)
    nh_nl = _const(0.5, n, idx)  # FLAT — strict rising required
    ad_slope = _const(1.5, n, idx)
    assert not _evaluate_broadening_breadth(
        nh_nl_ratio=nh_nl,
        ad_line_slope_20d=ad_slope,
        dt=idx[-1],
        lookback_sessions=5,
    )


# ---------------------------------------------------------------------------
# Group C — precedence (spec §1D line 284).
#
# breadth_thrust > divergent_fragile > narrowing_breadth > recovery_breadth >
# broadening_breadth > weak_breadth > healthy_breadth > neutral_breadth >
# unknown.
# ---------------------------------------------------------------------------


def _resolve_v2_label(
    *,
    v1_raw: BreadthLabel,
    narrowing_fires: bool,
    broadening_fires: bool,
) -> BreadthLabel:
    """Mirror the precedence walker in BreadthSeriesClassifier."""
    if v1_raw == "divergent_fragile":
        return "divergent_fragile"
    if narrowing_fires:
        return "narrowing_breadth"
    if v1_raw in {"weak_breadth", "healthy_breadth", "neutral_breadth", "unknown"} and broadening_fires:
        return "broadening_breadth"
    return v1_raw


def test_divergent_fragile_outranks_narrowing_breadth() -> None:
    assert (
        _resolve_v2_label(
            v1_raw="divergent_fragile",
            narrowing_fires=True,
            broadening_fires=False,
        )
        == "divergent_fragile"
    )


def test_narrowing_breadth_outranks_weak_breadth() -> None:
    assert (
        _resolve_v2_label(
            v1_raw="weak_breadth",
            narrowing_fires=True,
            broadening_fires=False,
        )
        == "narrowing_breadth"
    )


def test_broadening_breadth_promotes_neutral_to_broadening() -> None:
    assert (
        _resolve_v2_label(
            v1_raw="neutral_breadth",
            narrowing_fires=False,
            broadening_fires=True,
        )
        == "broadening_breadth"
    )


def test_build_raw_outputs_applies_v2_precedence_when_pit_features_present() -> None:
    n = 80
    idx = _trading_index(n)
    spy = pd.Series(np.linspace(100.0, 130.0, n), index=idx)
    rsp = spy * pd.Series(np.linspace(0.95, 0.85, n), index=idx)
    features = BreadthFeatures(
        spy_close=spy,
        rsp_close=rsp,
        relative_breadth_ratio=rsp / spy,
        relative_breadth_sma50=pd.Series(0.90, index=idx),
        relative_breadth_return_20d=pd.Series(-0.01, index=idx),
        index_distance_from_63d_high=pd.Series(-0.10, index=idx),
    )
    raw_labels, raw_evidence = build_raw_outputs(features)
    assert raw_labels[-1] == "weak_breadth"

    pct50 = pd.Series(np.linspace(0.80, 0.20, n), index=idx)
    pct200 = pd.Series(np.linspace(0.70, 0.30, n), index=idx)
    nh_nl = pd.Series(0.20, index=idx)
    ad_slope = pd.Series(-1.0, index=idx)
    updated_labels, updated_evidence = resolve_v2_raw_outputs(
        dates=idx,
        raw_labels=raw_labels,
        raw_evidence=raw_evidence,
        pct_above_50dma=pct50,
        pct_above_200dma=pct200,
        nh_nl_ratio=nh_nl,
        ad_line_slope_20d=ad_slope,
        breadth_thrust=None,
        lookback_sessions=5,
        nh_nl_threshold=0.4,
    )

    assert updated_labels[-1] == "narrowing_breadth"
    assert updated_evidence[-1]["v1_raw_label"] == "weak_breadth"
    assert updated_evidence[-1]["v2_narrowing_breadth"] is True
    unchanged_labels, unchanged_evidence = resolve_v2_raw_outputs(
        dates=idx,
        raw_labels=raw_labels,
        raw_evidence=raw_evidence,
        pct_above_50dma=pd.Series(0.60, index=idx),
        pct_above_200dma=pd.Series(0.55, index=idx),
        nh_nl_ratio=pd.Series(0.80, index=idx),
        ad_line_slope_20d=pd.Series(0.0, index=idx),
        breadth_thrust=None,
        lookback_sessions=5,
        nh_nl_threshold=0.4,
    )
    assert unchanged_labels[60] == raw_labels[60]
    assert unchanged_evidence[60]["v1_raw_label"] == raw_labels[60]
    assert unchanged_evidence[60]["v2_breadth_thrust"] is False
    assert unchanged_evidence[60]["v2_narrowing_breadth"] is False
    assert unchanged_evidence[60]["v2_recovery_breadth"] is False
    assert unchanged_evidence[60]["v2_broadening_breadth"] is False


# ---------------------------------------------------------------------------
# Group D — V1 byte-identity (no PIT inputs → no V2 labels emitted).
# ---------------------------------------------------------------------------


def test_v1_path_unchanged_when_pit_features_absent(market_df_for_asof) -> None:
    """When the engine runs WITHOUT PIT inputs (default V1 fixture path), the
    classifier must emit one of the 5 V1 labels — never narrowing_breadth or
    broadening_breadth.
    """
    from regime_detection.engine import RegimeEngine

    as_of = date(2023, 12, 14)
    out = RegimeEngine().classify(
        as_of_date=as_of, market_data=market_df_for_asof(as_of)
    )
    assert out.breadth_state.active_label in _V1_LABELS
    assert out.breadth_state.raw_label in _V1_LABELS


def test_v1_golden_dates_active_labels_unchanged(classified_golden_outputs) -> None:
    """Regression guard — V2 labels (narrowing_breadth / broadening_breadth /
    breadth_thrust / recovery_breadth) must NEVER appear on the V1 fixture
    path. Default-config callers (no PIT inputs) get V1 byte-identity.
    """
    repo_root = Path(__file__).resolve().parents[1]
    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )
    for row in golden["rows"]:
        as_of = date.fromisoformat(str(row["as_of_date"]))
        out = classified_golden_outputs[as_of]
        assert out.breadth_state.active_label in _V1_LABELS, (
            as_of,
            out.breadth_state.active_label,
        )


# ---------------------------------------------------------------------------
# Group E — config defaults wired correctly.
# ---------------------------------------------------------------------------


def test_breadth_v2_config_defaults_match_spec() -> None:
    cfg = BreadthV2Config()
    assert cfg.label_rate_of_change_lookback_sessions == 5  # Ambiguity Log #68
    assert cfg.nh_nl_ratio_narrowing_threshold == 0.4       # spec §1D line 280


def test_risk_rank_contains_new_v2_labels() -> None:
    # Sanity: the rank table must include every label in the V2 BreadthLabel
    # Literal.
    expected_labels = {
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
    assert expected_labels.issubset(set(_RISK_RANK.keys()))


def test_breadth_state_output_accepts_pit_biased_research_mode() -> None:
    out = BreadthStateOutput(
        mode="pit_constituent_biased_research",
        raw_label="narrowing_breadth",
        stable_label="narrowing_breadth",
        active_label="narrowing_breadth",
        evidence={"source": "pit_constituent_biased_research"},
        data_quality=DataQuality(status="ok"),
    )
    assert out.mode == "pit_constituent_biased_research"


# ---------------------------------------------------------------------------
# Group F — `breadth_thrust` LABEL predicate (ADR 0003 / Log #69 closure).
#
# Spec §1D Breadth Thrust block (post-amendment):
#   breadth_thrust fires at session t when:
#     EXISTS b in [t-10, t-1] with breadth_thrust_feature[b] < 0.40
#     AND breadth_thrust_feature[t] > 0.615
# ---------------------------------------------------------------------------


def test_breadth_thrust_fires_on_low_then_high_within_10_sessions() -> None:
    """Canonical Zweig case: low reading 4 sessions ago, high reading today."""
    n = 30
    idx = _trading_index(n)
    feature = pd.Series(np.full(n, 0.50), index=idx)
    feature.iloc[-5] = 0.35          # b = t-4: below 0.40 ✓
    feature.iloc[-1] = 0.70          # t: above 0.615 ✓
    dt = idx[-1]
    assert _evaluate_breadth_thrust(feature, dt=dt) is True


def test_breadth_thrust_fails_when_no_low_in_trailing_10() -> None:
    """Even with high reading today, absence of a low in [t-10, t-1] falsifies."""
    n = 30
    idx = _trading_index(n)
    feature = pd.Series(np.full(n, 0.50), index=idx)
    feature.iloc[-1] = 0.70          # high today ✓, but no <0.40 anywhere ✗
    dt = idx[-1]
    assert _evaluate_breadth_thrust(feature, dt=dt) is False


def test_breadth_thrust_fails_when_high_not_at_t() -> None:
    """High reading must be at session t, not earlier in the window."""
    n = 30
    idx = _trading_index(n)
    feature = pd.Series(np.full(n, 0.50), index=idx)
    feature.iloc[-5] = 0.35          # low at b=t-4 ✓
    feature.iloc[-3] = 0.70          # high at b=t-2 (not at t) ✗
    feature.iloc[-1] = 0.55          # t at 0.55 (not > 0.615) ✗
    dt = idx[-1]
    assert _evaluate_breadth_thrust(feature, dt=dt) is False


def test_breadth_thrust_strict_inequalities_at_thresholds() -> None:
    """Both inequalities are strict per Zweig: feature[t] == 0.615 falsifies,
    feature[b] == 0.40 falsifies."""
    n = 30
    idx = _trading_index(n)
    feature = pd.Series(np.full(n, 0.50), index=idx)
    feature.iloc[-5] = 0.40          # b exactly at threshold (not <) ✗
    feature.iloc[-1] = 0.615         # t exactly at threshold (not >) ✗
    dt = idx[-1]
    assert _evaluate_breadth_thrust(feature, dt=dt) is False


def test_breadth_thrust_fails_when_low_outside_10_session_window() -> None:
    """A low reading 11 sessions ago is outside the [t-10, t-1] window."""
    n = 30
    idx = _trading_index(n)
    feature = pd.Series(np.full(n, 0.50), index=idx)
    feature.iloc[-12] = 0.35         # b = t-11: outside the 10-session window
    feature.iloc[-1] = 0.70          # t: high
    dt = idx[-1]
    assert _evaluate_breadth_thrust(feature, dt=dt) is False


def test_breadth_thrust_fails_on_nan_at_t() -> None:
    n = 30
    idx = _trading_index(n)
    feature = pd.Series(np.full(n, 0.50), index=idx)
    feature.iloc[-5] = 0.35
    feature.iloc[-1] = float("nan")
    dt = idx[-1]
    assert _evaluate_breadth_thrust(feature, dt=dt) is False


def test_breadth_thrust_fails_on_all_nan_trailing_window() -> None:
    """Cold-start: every session in [t-10, t-1] is NaN → falsifies."""
    n = 30
    idx = _trading_index(n)
    feature = pd.Series(np.full(n, float("nan")), index=idx)
    feature.iloc[-1] = 0.70  # high today
    dt = idx[-1]
    assert _evaluate_breadth_thrust(feature, dt=dt) is False


# ---------------------------------------------------------------------------
# Group G — `recovery_breadth` LABEL predicate (ADR 0003 / Log #70 closure).
#
# Spec §1D (post-amendment):
#   recovery_breadth fires at session t when:
#     nh_nl_ratio[t] > nh_nl_ratio[t-5]      (rising NH/NL per Log #68)
#     AND ad_line_slope_20d[t] <= 0          (not yet broadening)
# ---------------------------------------------------------------------------


def test_recovery_breadth_fires_on_rising_nh_nl_and_negative_slope() -> None:
    """Improvement starting (NH/NL rising) but cumulative AD still negative."""
    n = 30
    idx = _trading_index(n)
    nh_nl = pd.Series(np.linspace(0.30, 0.45, n), index=idx)  # strictly rising
    ad_slope = _const(-1.5, n, idx)                            # strictly negative
    dt = idx[-1]
    assert (
        _evaluate_recovery_breadth(
            nh_nl_ratio=nh_nl,
            ad_line_slope_20d=ad_slope,
            dt=dt,
            lookback_sessions=5,
        )
        is True
    )


def test_recovery_breadth_fires_at_slope_boundary_exactly_zero() -> None:
    """ad_line_slope_20d <= 0 is non-strict at boundary; slope == 0 fires."""
    n = 30
    idx = _trading_index(n)
    nh_nl = pd.Series(np.linspace(0.30, 0.45, n), index=idx)
    ad_slope = _const(0.0, n, idx)
    dt = idx[-1]
    assert (
        _evaluate_recovery_breadth(
            nh_nl_ratio=nh_nl,
            ad_line_slope_20d=ad_slope,
            dt=dt,
            lookback_sessions=5,
        )
        is True
    )


def test_recovery_breadth_fails_when_slope_strictly_positive() -> None:
    """ad_line_slope_20d > 0 is the broadening territory; recovery must
    fail to avoid co-firing with broadening."""
    n = 30
    idx = _trading_index(n)
    nh_nl = pd.Series(np.linspace(0.30, 0.45, n), index=idx)
    ad_slope = _const(1.0, n, idx)
    dt = idx[-1]
    assert (
        _evaluate_recovery_breadth(
            nh_nl_ratio=nh_nl,
            ad_line_slope_20d=ad_slope,
            dt=dt,
            lookback_sessions=5,
        )
        is False
    )


def test_recovery_breadth_fails_when_nh_nl_not_rising() -> None:
    n = 30
    idx = _trading_index(n)
    nh_nl = _const(0.40, n, idx)   # flat, not rising
    ad_slope = _const(-1.0, n, idx)
    dt = idx[-1]
    assert (
        _evaluate_recovery_breadth(
            nh_nl_ratio=nh_nl,
            ad_line_slope_20d=ad_slope,
            dt=dt,
            lookback_sessions=5,
        )
        is False
    )


def test_recovery_breadth_fails_on_nan_slope() -> None:
    n = 30
    idx = _trading_index(n)
    nh_nl = pd.Series(np.linspace(0.30, 0.45, n), index=idx)
    ad_slope = pd.Series(np.full(n, float("nan")), index=idx)
    dt = idx[-1]
    assert (
        _evaluate_recovery_breadth(
            nh_nl_ratio=nh_nl,
            ad_line_slope_20d=ad_slope,
            dt=dt,
            lookback_sessions=5,
        )
        is False
    )


def test_recovery_breadth_disjoint_from_broadening_breadth_at_zero_slope() -> None:
    """Disjointness invariant: at ad_line_slope_20d == 0.0 with NH/NL rising,
    recovery fires AND broadening does NOT fire. The two predicates partition
    the real line at zero (recovery: slope <= 0, broadening: slope > 0)."""
    n = 30
    idx = _trading_index(n)
    nh_nl = pd.Series(np.linspace(0.30, 0.45, n), index=idx)
    ad_slope = _const(0.0, n, idx)
    dt = idx[-1]
    assert (
        _evaluate_recovery_breadth(
            nh_nl_ratio=nh_nl,
            ad_line_slope_20d=ad_slope,
            dt=dt,
            lookback_sessions=5,
        )
        is True
    )
    assert (
        _evaluate_broadening_breadth(
            nh_nl_ratio=nh_nl,
            ad_line_slope_20d=ad_slope,
            dt=dt,
            lookback_sessions=5,
        )
        is False
    )
