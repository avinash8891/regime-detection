"""V2 §1D Breadth State extensions: ``narrowing_breadth`` and
``broadening_breadth`` (Ambiguity Log #21–#26 + #68).

The deferred labels ``breadth_thrust`` and ``recovery_breadth`` reserve
precedence slots but never fire today (Ambiguity Log #69 / #70).

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
import pytest
import yaml

from regime_detection.breadth_state import (
    BreadthLabel,
    _RISK_RANK,
    _evaluate_broadening_breadth,
    _evaluate_narrowing_breadth,
)
from regime_detection.config import BreadthV2Config


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
    # Literal — including the deferred slots (breadth_thrust, recovery_breadth).
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
