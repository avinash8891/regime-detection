"""TDD tests for v2 §1A `euphoria` rule + updated precedence (Log #32 closure).

Spec references (docs/regime_engine_v2_spec.md §1A lines 159–172):

    euphoria fires when:
      close > SMA_200
      AND return_126d > 0.20
      AND realized_vol_21d rising         (strict 5-session change per Log #68 analogue)
      AND sentiment_score >= euphoria_sentiment_threshold  (default +20)

    Precedence (§1A line 171, slot already reserved):
      euphoria > bull > recovery > bear > sideways > transition > unknown

Per AGENTS.md rules G/L: realistic SPY-like inputs, no toy names, use
production Pydantic config.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from regime_detection.config import (
    TrendDirectionV2Config,
    TrendDirectionV2RulesConfig,
)
from regime_detection.trend_direction import (
    TrendDirectionFeatures,
    apply_hysteresis,
    build_raw_outputs as build_trend_direction_raw_outputs,
)
from regime_detection.trend_direction_v2 import (
    TrendDirectionV2Features,
    compute_trend_v2_features,
    evaluate_euphoria,
    evaluate_v2_trend_label,
)
from regime_detection.volatility_state import realized_vol


# v2 §1A line 162 — exact spec thresholds.
_SPEC_RETURN_126D_THRESHOLD = 0.20
# ADR 0004 Q3 — default operator threshold (V2 §9.1 calibration placeholder).
_SPEC_EUPHORIA_SENTIMENT_THRESHOLD = 20.0
# ADR 0004 Q2 — 5-session rising-of lookback (Log #68 §1D analogue).
_SPEC_EUPHORIA_VOL_RISING_LOOKBACK = 5


@pytest.fixture
def euphoria_rules() -> TrendDirectionV2RulesConfig:
    return TrendDirectionV2RulesConfig(
        recovery_drawdown_threshold=-0.15,
        recovery_return_threshold=0.10,
        euphoria_sentiment_threshold=_SPEC_EUPHORIA_SENTIMENT_THRESHOLD,
        euphoria_return_126d_threshold=_SPEC_RETURN_126D_THRESHOLD,
        euphoria_vol_rising_lookback_sessions=_SPEC_EUPHORIA_VOL_RISING_LOOKBACK,
    )


def _euphoria_inputs_at(
    *,
    dt: pd.Timestamp,
    close_t: float,
    sma_200: float,
    return_126d: float,
    realized_vol_21d_now: float,
    realized_vol_21d_5d_ago: float,
    sentiment_score: float | None,
) -> tuple[TrendDirectionV2Features, pd.Series]:
    """Build the minimal TrendDirectionV2Features needed to evaluate the
    euphoria predicate at session ``dt``. Fields irrelevant to euphoria are
    NaN-filled. The index is a 6-session business-day range ending at
    ``dt`` so the 5-session-ago vol lookup resolves positionally."""
    # 6 sessions: positions 0..5 with dt at position 5 (the last).
    # Position 0 == dt - 5 BDay → the "5 sessions ago" value for the
    # `vol[t] > vol[t-5]` rising-of conjunct.
    idx = pd.bdate_range(
        end=dt, periods=_SPEC_EUPHORIA_VOL_RISING_LOOKBACK + 1, freq="B"
    )
    nan_series = pd.Series([float("nan")] * len(idx), index=idx)
    five_back = idx[0]

    sma_200_series = nan_series.copy()
    sma_200_series.loc[dt] = sma_200

    return_126d_series = nan_series.copy()
    return_126d_series.loc[dt] = return_126d

    vol_series = nan_series.copy()
    vol_series.loc[dt] = realized_vol_21d_now
    vol_series.loc[five_back] = realized_vol_21d_5d_ago

    sentiment_series: pd.Series | None
    if sentiment_score is None:
        sentiment_series = None
    else:
        sentiment_series = nan_series.copy()
        sentiment_series.loc[dt] = sentiment_score

    features = TrendDirectionV2Features(
        efficiency_ratio_20d=nan_series.copy(),
        hurst_250d=nan_series.copy(),
        slope_sma_50=nan_series.copy(),
        slope_sma_200=nan_series.copy(),
        return_63d=nan_series.copy(),
        return_126d=return_126d_series,
        drawdown_252d=nan_series.copy(),
        sma_50=nan_series.copy(),
        sma_200=sma_200_series,
        realized_vol_21d=vol_series,
        sentiment_score=sentiment_series,
    )
    close = nan_series.copy()
    close.loc[dt] = close_t
    return features, close


def test_evaluate_euphoria_fires_when_all_four_conjuncts_satisfied(
    euphoria_rules: TrendDirectionV2RulesConfig,
) -> None:
    """All four spec conjuncts strictly satisfied → euphoria fires."""
    dt = pd.Timestamp("2024-03-15")
    features, close = _euphoria_inputs_at(
        dt=dt,
        close_t=520.0,            # close > SMA_200 (520 > 450) ✓
        sma_200=450.0,
        return_126d=0.30,         # return_126d > 0.20 ✓
        realized_vol_21d_now=0.18,    # rising: 0.18 > 0.15 ✓
        realized_vol_21d_5d_ago=0.15,
        sentiment_score=25.0,     # sentiment >= +20 ✓
    )
    assert (
        evaluate_euphoria(features, close, dt=dt, rules_config=euphoria_rules) is True
    )


def test_evaluate_euphoria_fails_when_close_at_or_below_sma_200(
    euphoria_rules: TrendDirectionV2RulesConfig,
) -> None:
    """close > SMA_200 is strict per spec line 161 — equality falsifies."""
    dt = pd.Timestamp("2024-03-15")
    features, close = _euphoria_inputs_at(
        dt=dt,
        close_t=450.0,            # close == SMA_200 → strict `>` falsifies
        sma_200=450.0,
        return_126d=0.30,
        realized_vol_21d_now=0.18,
        realized_vol_21d_5d_ago=0.15,
        sentiment_score=25.0,
    )
    assert (
        evaluate_euphoria(features, close, dt=dt, rules_config=euphoria_rules) is False
    )


def test_evaluate_euphoria_fails_when_return_126d_at_or_below_threshold(
    euphoria_rules: TrendDirectionV2RulesConfig,
) -> None:
    """return_126d > 0.20 is strict per spec line 162 — equality falsifies."""
    dt = pd.Timestamp("2024-03-15")
    features, close = _euphoria_inputs_at(
        dt=dt,
        close_t=520.0,
        sma_200=450.0,
        return_126d=0.20,         # exactly at threshold → strict `>` falsifies
        realized_vol_21d_now=0.18,
        realized_vol_21d_5d_ago=0.15,
        sentiment_score=25.0,
    )
    assert (
        evaluate_euphoria(features, close, dt=dt, rules_config=euphoria_rules) is False
    )


def test_evaluate_euphoria_fails_when_vol_not_rising_over_5_sessions(
    euphoria_rules: TrendDirectionV2RulesConfig,
) -> None:
    """realized_vol_21d rising is strict 5-session change per ADR 0004 Q2."""
    dt = pd.Timestamp("2024-03-15")
    features, close = _euphoria_inputs_at(
        dt=dt,
        close_t=520.0,
        sma_200=450.0,
        return_126d=0.30,
        realized_vol_21d_now=0.15,    # not rising: 0.15 == 0.15
        realized_vol_21d_5d_ago=0.15,
        sentiment_score=25.0,
    )
    assert (
        evaluate_euphoria(features, close, dt=dt, rules_config=euphoria_rules) is False
    )


def test_evaluate_euphoria_fails_when_sentiment_below_threshold(
    euphoria_rules: TrendDirectionV2RulesConfig,
) -> None:
    """sentiment_score >= +20 is non-strict at boundary per spec line 164."""
    dt = pd.Timestamp("2024-03-15")
    features, close = _euphoria_inputs_at(
        dt=dt,
        close_t=520.0,
        sma_200=450.0,
        return_126d=0.30,
        realized_vol_21d_now=0.18,
        realized_vol_21d_5d_ago=0.15,
        sentiment_score=19.99,        # below +20 → falsifies
    )
    assert (
        evaluate_euphoria(features, close, dt=dt, rules_config=euphoria_rules) is False
    )


def test_evaluate_euphoria_fires_at_sentiment_boundary_exactly_at_threshold(
    euphoria_rules: TrendDirectionV2RulesConfig,
) -> None:
    """sentiment_score >= +20 — non-strict at boundary per spec line 164.

    sentiment EXACTLY at +20 satisfies the rule (operator is `>=`, not `>`)."""
    dt = pd.Timestamp("2024-03-15")
    features, close = _euphoria_inputs_at(
        dt=dt,
        close_t=520.0,
        sma_200=450.0,
        return_126d=0.30,
        realized_vol_21d_now=0.18,
        realized_vol_21d_5d_ago=0.15,
        sentiment_score=_SPEC_EUPHORIA_SENTIMENT_THRESHOLD,  # exactly +20
    )
    assert (
        evaluate_euphoria(features, close, dt=dt, rules_config=euphoria_rules) is True
    )


def test_evaluate_euphoria_fails_when_sentiment_series_is_none(
    euphoria_rules: TrendDirectionV2RulesConfig,
) -> None:
    """When AAII fetcher not wired (sentiment_score=None on features), the
    rule falsifies — V2 §10 "do not invent a sentiment proxy" inherited
    from Log #32."""
    dt = pd.Timestamp("2024-03-15")
    features, close = _euphoria_inputs_at(
        dt=dt,
        close_t=520.0,
        sma_200=450.0,
        return_126d=0.30,
        realized_vol_21d_now=0.18,
        realized_vol_21d_5d_ago=0.15,
        sentiment_score=None,
    )
    assert (
        evaluate_euphoria(features, close, dt=dt, rules_config=euphoria_rules) is False
    )


def test_evaluate_euphoria_fails_on_nan_sentiment(
    euphoria_rules: TrendDirectionV2RulesConfig,
) -> None:
    """Cold-start NaN on sentiment falsifies per V1 §2.7 inheritance."""
    dt = pd.Timestamp("2024-03-15")
    features, close = _euphoria_inputs_at(
        dt=dt,
        close_t=520.0,
        sma_200=450.0,
        return_126d=0.30,
        realized_vol_21d_now=0.18,
        realized_vol_21d_5d_ago=0.15,
        sentiment_score=float("nan"),
    )
    assert (
        evaluate_euphoria(features, close, dt=dt, rules_config=euphoria_rules) is False
    )


def test_evaluate_euphoria_fails_on_nan_vol_history(
    euphoria_rules: TrendDirectionV2RulesConfig,
) -> None:
    """Vol-rising NaN at either endpoint (t or t-5) falsifies the rule."""
    dt = pd.Timestamp("2024-03-15")
    features, close = _euphoria_inputs_at(
        dt=dt,
        close_t=520.0,
        sma_200=450.0,
        return_126d=0.30,
        realized_vol_21d_now=0.18,
        realized_vol_21d_5d_ago=float("nan"),  # cold-start at t-5
        sentiment_score=25.0,
    )
    assert (
        evaluate_euphoria(features, close, dt=dt, rules_config=euphoria_rules) is False
    )


def test_evaluate_v2_trend_label_returns_euphoria_when_predicate_fires(
    euphoria_rules: TrendDirectionV2RulesConfig,
) -> None:
    """When all four euphoria conjuncts fire, the v2 precedence walker
    returns 'euphoria' regardless of the v1 label (euphoria is top of
    the §1A precedence chain)."""
    dt = pd.Timestamp("2024-03-15")
    features, close = _euphoria_inputs_at(
        dt=dt,
        close_t=520.0,
        sma_200=450.0,
        return_126d=0.30,
        realized_vol_21d_now=0.18,
        realized_vol_21d_5d_ago=0.15,
        sentiment_score=25.0,
    )
    result = evaluate_v2_trend_label(
        v1_label="bull",      # bull would normally win; euphoria outranks
        features=features,
        close=close,
        dt=dt,
        rules_config=euphoria_rules,
    )
    assert result == "euphoria"


def test_evaluate_v2_trend_label_euphoria_outranks_recovery_when_both_fire(
    euphoria_rules: TrendDirectionV2RulesConfig,
) -> None:
    """If recovery and euphoria both fire (edge case), euphoria wins per
    spec line 171: euphoria > bull > recovery > ..."""
    dt = pd.Timestamp("2024-03-15")
    # Build features where BOTH recovery and euphoria predicates can fire.
    # We need drawdown_252d <= -0.15 AND return_63d > 0.10 AND close > sma_50
    # plus the euphoria conjuncts. The recovery predicate needs sma_50 and
    # drawdown_252d which our helper leaves NaN — so we override them here.
    features, close = _euphoria_inputs_at(
        dt=dt,
        close_t=520.0,
        sma_200=450.0,
        return_126d=0.30,
        realized_vol_21d_now=0.18,
        realized_vol_21d_5d_ago=0.15,
        sentiment_score=25.0,
    )
    # Patch features to also satisfy recovery.
    features.sma_50.loc[dt] = 400.0     # close > sma_50 (520 > 400) ✓
    features.return_63d.loc[dt] = 0.15  # return_63d > 0.10 ✓
    features.drawdown_252d.loc[dt] = -0.20  # drawdown <= -0.15 ✓

    result = evaluate_v2_trend_label(
        v1_label="bear",       # bear would normally hold without recovery override
        features=features,
        close=close,
        dt=dt,
        rules_config=euphoria_rules,
    )
    assert result == "euphoria"


def test_realized_vol_21d_and_sma_200_and_sentiment_score_exposed_on_features() -> None:
    """The TrendDirectionV2Features dataclass exposes the three new fields
    so the euphoria predicate can read them without re-computing."""
    fields = set(TrendDirectionV2Features.__dataclass_fields__)
    assert "realized_vol_21d" in fields
    assert "sma_200" in fields
    assert "sentiment_score" in fields


def test_euphoria_realized_vol_21d_reuses_shared_volatility_helper() -> None:
    """§1A euphoria uses the same realized_vol_21d series as the shared
    volatility axis helper, not a private return convention."""
    idx = pd.bdate_range(end="2024-03-15", periods=80, freq="B")
    returns = np.r_[
        np.full(40, 0.002),
        np.linspace(-0.03, 0.035, 40),
    ]
    close = pd.Series(400.0 * (1.0 + returns).cumprod(), index=idx, name="close")
    config = TrendDirectionV2Config(
        efficiency_ratio_lookback_days=20,
        hurst_lookback_days=250,
        slope_lookback_days=20,
        sma_short_period=50,
        sma_long_period=200,
        return_short_period=63,
        return_long_period=126,
        drawdown_lookback_days=252,
    )

    features = compute_trend_v2_features(close, config=config)
    expected = realized_vol(close, window=21).rename("realized_vol_21d")

    pd.testing.assert_series_equal(features.realized_vol_21d, expected)


def test_build_raw_outputs_records_euphoria_override_rule(
    euphoria_rules: TrendDirectionV2RulesConfig,
) -> None:
    """When euphoria wins the V2 overlay, evidence must name the euphoria
    rule rather than the lower-precedence recovery rule."""
    dt = pd.Timestamp("2024-03-15")
    idx = pd.bdate_range(
        end=dt, periods=_SPEC_EUPHORIA_VOL_RISING_LOOKBACK + 1, freq="B"
    )
    close = pd.Series(np.linspace(500.0, 520.0, len(idx)), index=idx, name="close")
    v1_features = TrendDirectionFeatures(
        close=close,
        sma_50=pd.Series(460.0, index=idx),
        sma_200=pd.Series(450.0, index=idx),
        return_63d=pd.Series(0.30, index=idx),
    )
    v2_features, _ = _euphoria_inputs_at(
        dt=dt,
        close_t=520.0,
        sma_200=450.0,
        return_126d=0.30,
        realized_vol_21d_now=0.18,
        realized_vol_21d_5d_ago=0.15,
        sentiment_score=25.0,
    )

    labels, evidence = build_trend_direction_raw_outputs(
        v1_features,
        trend_direction_v2_features=v2_features,
        trend_direction_v2_rules=euphoria_rules,
    )

    assert labels[-1] == "euphoria"
    assert evidence[-1]["v2_override"] == {
        "from": "bull",
        "to": "euphoria",
        "rule": "euphoria",
    }


def test_hysteresis_accepts_euphoria_trend_label() -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2024-03-14"), pd.Timestamp("2024-03-15")])

    stable, active = apply_hysteresis(
        dates=dates,
        raw_labels=["bull", "euphoria"],
        deescalation_days=3,
    )

    assert stable == ["bull", "euphoria"]
    assert active == ["bull", "euphoria"]


def test_build_sentiment_score_series_forward_fills_from_publication_date() -> None:
    """v2 §1A line 164 alignment (ADR 0004 Q4): each NYSE session inherits
    the latest AAII publication-date row on or before it. V1 §2.2
    stateless-replay — never consult a future-dated reading."""
    from regime_detection.feature_store import _build_sentiment_score_series

    aaii = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-03-07"),
                "publication_date": pd.Timestamp("2024-03-07"),
                "bull_bear_spread_8w_ma": 15.0,
            },
            {
                "date": pd.Timestamp("2024-03-14"),
                "publication_date": pd.Timestamp("2024-03-14"),
                "bull_bear_spread_8w_ma": 22.0,
            },
        ]
    )
    sessions = pd.bdate_range(start="2024-03-08", end="2024-03-15", freq="B")

    score = _build_sentiment_score_series(
        aaii_sentiment=aaii, session_index=sessions
    )

    assert score is not None
    # Sessions 03-08, 03-11, 03-12, 03-13: pre-03-14 publication →
    # inherit the 03-07 row's value (15.0).
    assert score.loc[pd.Timestamp("2024-03-08")] == 15.0
    assert score.loc[pd.Timestamp("2024-03-13")] == 15.0
    # Sessions 03-14, 03-15: at or after the 03-14 publication →
    # inherit the new value (22.0).
    assert score.loc[pd.Timestamp("2024-03-14")] == 22.0
    assert score.loc[pd.Timestamp("2024-03-15")] == 22.0


def test_build_sentiment_score_series_cold_start_returns_nan_before_first_row() -> None:
    """Sessions before the first AAII publication date receive NaN; the
    euphoria predicate then falsifies (V1 §2.7 cold-start)."""
    from regime_detection.feature_store import _build_sentiment_score_series

    aaii = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-03-14"),
                "publication_date": pd.Timestamp("2024-03-14"),
                "bull_bear_spread_8w_ma": 22.0,
            },
        ]
    )
    sessions = pd.bdate_range(start="2024-03-01", end="2024-03-15", freq="B")

    score = _build_sentiment_score_series(
        aaii_sentiment=aaii, session_index=sessions
    )

    assert score is not None
    # Sessions before 03-14 have no preceding AAII row → NaN.
    assert pd.isna(score.loc[pd.Timestamp("2024-03-01")])
    assert pd.isna(score.loc[pd.Timestamp("2024-03-13")])
    # 03-14 onward inherits the row.
    assert score.loc[pd.Timestamp("2024-03-14")] == 22.0


def test_build_sentiment_score_series_returns_none_when_no_aaii() -> None:
    """Optional input contract: when AAII is None, helper returns None and
    the euphoria predicate falsifies via the sentiment_score=None branch."""
    from regime_detection.feature_store import _build_sentiment_score_series

    sessions = pd.bdate_range(start="2024-03-01", end="2024-03-15", freq="B")
    assert (
        _build_sentiment_score_series(aaii_sentiment=None, session_index=sessions)
        is None
    )
