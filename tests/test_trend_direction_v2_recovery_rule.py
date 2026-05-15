"""TDD tests for v2 §1A `recovery` rule + updated precedence (Slice 2.5).

Spec references (docs/regime_engine_v2_spec.md):
    §1A line 114-119 — `recovery` rule:
        drawdown_252d <= -0.15
        AND return_63d > 0.10
        AND close > SMA_50
    §1A line 132-134 — precedence:
        euphoria > bull > recovery > bear > sideways > transition > unknown

Per ~/.claude/CLAUDE.md and AGENTS.md G/L: realistic SPY-like price series,
no toy a/b/c names, use the real production Pydantic config.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from regime_detection.calendar import nyse_sessions_between
from regime_detection.config import (
    TrendDirectionV2Config,
    TrendDirectionV2RulesConfig,
    load_default_regime_config,
)
from regime_detection.engine import RegimeEngine
from regime_detection.trend_direction import (
    build_raw_outputs as build_v1_raw_outputs,
    compute_features as compute_v1_features,
    raw_label_for_day,
)
from regime_detection.trend_direction_v2 import (
    TrendDirectionV2Features,
    compute_trend_v2_features,
    evaluate_recovery,
    evaluate_v2_trend_label,
)


# v2 §1A line 116-117 — exact spec thresholds.
_SPEC_DRAWDOWN_THRESHOLD = -0.15
_SPEC_RETURN_THRESHOLD = 0.10


@pytest.fixture
def recovery_rules() -> TrendDirectionV2RulesConfig:
    return TrendDirectionV2RulesConfig(
        recovery_drawdown_threshold=_SPEC_DRAWDOWN_THRESHOLD,
        recovery_return_threshold=_SPEC_RETURN_THRESHOLD,
    )


@pytest.fixture
def v2_trend_config() -> TrendDirectionV2Config:
    """Production defaults from configs/core3-v2.0.0.yaml."""
    return TrendDirectionV2Config(
        efficiency_ratio_lookback_days=20,
        hurst_lookback_days=250,
        slope_lookback_days=20,
        sma_short_period=50,
        sma_long_period=200,
        return_short_period=63,
        return_long_period=126,
        drawdown_lookback_days=252,
    )


def _scalar_features_at(
    *,
    dt: pd.Timestamp,
    close_t: float,
    sma_50: float,
    return_63d: float,
    drawdown_252d: float,
) -> tuple[TrendDirectionV2Features, pd.Series]:
    """Construct a minimal TrendDirectionV2Features with a single date
    populated (other fields NaN — irrelevant for the recovery rule).
    """
    idx = pd.DatetimeIndex([dt])
    nan = pd.Series([float("nan")], index=idx)
    features = TrendDirectionV2Features(
        efficiency_ratio_20d=nan.copy(),
        hurst_250d=nan.copy(),
        slope_sma_50=nan.copy(),
        slope_sma_200=nan.copy(),
        return_63d=pd.Series([return_63d], index=idx),
        return_126d=nan.copy(),
        drawdown_252d=pd.Series([drawdown_252d], index=idx),
        sma_50=pd.Series([sma_50], index=idx),
        sma_200=nan.copy(),
        realized_vol_21d=nan.copy(),
        sentiment_score=None,
    )
    close = pd.Series([close_t], index=idx, name="close")
    return features, close


# ---------- Boundary tests on `evaluate_recovery` ---------------------------


def test_recovery_all_three_conditions_true_returns_true(recovery_rules) -> None:
    dt = pd.Timestamp("2020-04-29")
    features, close = _scalar_features_at(
        dt=dt, close_t=290.0, sma_50=280.0, return_63d=0.20, drawdown_252d=-0.18
    )
    assert evaluate_recovery(features, close, dt=dt, rules_config=recovery_rules) is True


def test_recovery_drawdown_exactly_threshold_is_true(recovery_rules) -> None:
    """v2 §1A line 116 uses `<=` — drawdown == -0.15 satisfies."""
    dt = pd.Timestamp("2020-04-29")
    features, close = _scalar_features_at(
        dt=dt, close_t=290.0, sma_50=280.0, return_63d=0.20, drawdown_252d=-0.15
    )
    assert evaluate_recovery(features, close, dt=dt, rules_config=recovery_rules) is True


def test_recovery_drawdown_just_above_threshold_is_false(recovery_rules) -> None:
    dt = pd.Timestamp("2020-04-29")
    features, close = _scalar_features_at(
        dt=dt, close_t=290.0, sma_50=280.0, return_63d=0.20, drawdown_252d=-0.14
    )
    assert evaluate_recovery(features, close, dt=dt, rules_config=recovery_rules) is False


def test_recovery_return_exactly_threshold_is_false(recovery_rules) -> None:
    """v2 §1A line 117 uses strict `>` — return_63d == 0.10 fails."""
    dt = pd.Timestamp("2020-04-29")
    features, close = _scalar_features_at(
        dt=dt, close_t=290.0, sma_50=280.0, return_63d=0.10, drawdown_252d=-0.20
    )
    assert evaluate_recovery(features, close, dt=dt, rules_config=recovery_rules) is False


def test_recovery_return_just_above_threshold_is_true(recovery_rules) -> None:
    dt = pd.Timestamp("2020-04-29")
    features, close = _scalar_features_at(
        dt=dt, close_t=290.0, sma_50=280.0, return_63d=0.10001, drawdown_252d=-0.20
    )
    assert evaluate_recovery(features, close, dt=dt, rules_config=recovery_rules) is True


def test_recovery_close_equals_sma50_is_false(recovery_rules) -> None:
    """v2 §1A line 118 uses strict `>` — close == SMA_50 fails."""
    dt = pd.Timestamp("2020-04-29")
    features, close = _scalar_features_at(
        dt=dt, close_t=280.0, sma_50=280.0, return_63d=0.20, drawdown_252d=-0.20
    )
    assert evaluate_recovery(features, close, dt=dt, rules_config=recovery_rules) is False


def test_recovery_close_just_above_sma50_is_true(recovery_rules) -> None:
    dt = pd.Timestamp("2020-04-29")
    features, close = _scalar_features_at(
        dt=dt, close_t=280.0 * 1.001, sma_50=280.0, return_63d=0.20, drawdown_252d=-0.20
    )
    assert evaluate_recovery(features, close, dt=dt, rules_config=recovery_rules) is True


@pytest.mark.parametrize(
    "field",
    ["return_63d", "drawdown_252d", "sma_50", "close_t"],
)
def test_recovery_nan_input_returns_false(recovery_rules, field) -> None:
    """NaN cold-start contract — any missing input falsifies the rule."""
    dt = pd.Timestamp("2020-04-29")
    args = {
        "dt": dt,
        "close_t": 290.0,
        "sma_50": 280.0,
        "return_63d": 0.20,
        "drawdown_252d": -0.20,
    }
    args[field] = float("nan")
    features, close = _scalar_features_at(**args)
    assert evaluate_recovery(features, close, dt=dt, rules_config=recovery_rules) is False


# ---------- Precedence tests on `evaluate_v2_trend_label` --------------------


def test_precedence_bull_outranks_recovery_when_both_match(recovery_rules) -> None:
    """v2 §1A line 132-134: bull > recovery. v1 bull + recovery true → keep bull."""
    dt = pd.Timestamp("2020-04-29")
    features, close = _scalar_features_at(
        dt=dt, close_t=290.0, sma_50=280.0, return_63d=0.20, drawdown_252d=-0.20
    )
    assert (
        evaluate_v2_trend_label(
            v1_label="bull",
            features=features,
            close=close,
            dt=dt,
            rules_config=recovery_rules,
        )
        is None
    )


@pytest.mark.parametrize("v1_label", ["bear", "sideways", "transition", "unknown"])
def test_precedence_recovery_overrides_lower_v1_labels(recovery_rules, v1_label) -> None:
    """v2 §1A line 132-134: recovery > bear > sideways > transition > unknown."""
    dt = pd.Timestamp("2020-04-29")
    features, close = _scalar_features_at(
        dt=dt, close_t=290.0, sma_50=280.0, return_63d=0.20, drawdown_252d=-0.20
    )
    assert (
        evaluate_v2_trend_label(
            v1_label=v1_label,
            features=features,
            close=close,
            dt=dt,
            rules_config=recovery_rules,
        )
        == "recovery"
    )


def test_precedence_recovery_predicate_false_returns_none(recovery_rules) -> None:
    """No fire → caller keeps v1 label."""
    dt = pd.Timestamp("2020-04-29")
    features, close = _scalar_features_at(
        dt=dt, close_t=290.0, sma_50=280.0, return_63d=0.05, drawdown_252d=-0.20
    )
    assert (
        evaluate_v2_trend_label(
            v1_label="transition",
            features=features,
            close=close,
            dt=dt,
            rules_config=recovery_rules,
        )
        is None
    )


# ---------- V1 contract preservation ----------------------------------------


def _spy_like_drawdown_recovery_series(*, n_total: int = 432) -> pd.Series:
    """Construct a SPY-like series with a clean drawdown + sharp V-rebound:

    - 252-day warmup near 400,
    - 60-day cliff decline 400 → 240 (~40% drawdown — well past -0.15),
    - 60-day sharp rebound 240 → 340 (~42% return),
    - then a plateau near 340 for remaining sessions.

    Window math for a session 63 sessions into the rebound:
      return_63d ≈ (rebound_high - rebound_low) / rebound_low ≈ 0.42 > 0.10.
      drawdown_252d: trailing peak is the warmup 400, so 340/400 - 1 = -0.15
      — sits right on the threshold; the deeper-in-rebound days are at
      -0.15 to -0.30 because the SMA_50 is still well below close. Sized so
      AT LEAST ONE session in the early rebound satisfies all three.
    """
    rng = np.random.default_rng(seed=20260512)
    warmup = np.full(252, 400.0)
    decline = np.linspace(400.0, 240.0, 60)
    rebound = np.linspace(240.0, 340.0, 60)
    plateau = np.full(n_total - 252 - 60 - 60, 340.0)
    series_values = np.concatenate([warmup, decline, rebound, plateau])
    # Add tiny noise so SMA / return / drawdown are realistic.
    series_values = series_values + rng.normal(0.0, 0.1, size=series_values.size)
    # Use NYSE sessions (not pd.bdate_range — that includes US-holidays
    # like MLK Day, Presidents Day, etc., which fail
    # _require_market_data_contract).
    sessions = nyse_sessions_between(date(2018, 1, 2), date(2020, 12, 31))
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in sessions[:n_total]])
    return pd.Series(series_values, index=index, name="close")


def test_v1_contract_byte_identity_when_v2_features_absent(v2_trend_config) -> None:
    """Calling build_raw_outputs WITHOUT v2 features/rules emits the EXACT
    same labels and evidence as the pre-slice-2.5 v1 path.
    """
    close = _spy_like_drawdown_recovery_series()
    v1_features = compute_v1_features(close)

    labels_no_v2, evidence_no_v2 = build_v1_raw_outputs(v1_features)
    # Sanity — no `recovery` ever appears on the v1 path.
    assert "recovery" not in labels_no_v2
    # And no v2_override key in evidence.
    assert all("v2_override" not in ev for ev in evidence_no_v2)

    # Pass v2_features=None explicitly — same output.
    labels_explicit_none, evidence_explicit_none = build_v1_raw_outputs(
        v1_features,
        trend_direction_v2_features=None,
        trend_direction_v2_rules=None,
    )
    assert labels_explicit_none == labels_no_v2
    assert evidence_explicit_none == evidence_no_v2


def test_v1_raw_label_for_day_byte_identity_when_v2_features_absent() -> None:
    close = _spy_like_drawdown_recovery_series()
    v1_features = compute_v1_features(close)
    dt = close.index[-1]
    label_v1, ev_v1 = raw_label_for_day(v1_features, dt)
    label_explicit, ev_explicit = raw_label_for_day(
        v1_features,
        dt,
        trend_direction_v2_features=None,
        trend_direction_v2_rules=None,
    )
    assert label_v1 == label_explicit
    assert ev_v1 == ev_explicit
    assert "v2_override" not in ev_v1


# ---------- Classifier integration test (build_raw_outputs with v2) ---------


def test_build_raw_outputs_emits_recovery_on_synthetic_rebound(
    v2_trend_config, recovery_rules
) -> None:
    close = _spy_like_drawdown_recovery_series()
    v1_features = compute_v1_features(close)
    v2_features = compute_trend_v2_features(close, config=v2_trend_config)

    labels, evidence = build_v1_raw_outputs(
        v1_features,
        trend_direction_v2_features=v2_features,
        trend_direction_v2_rules=recovery_rules,
    )
    # At least one session emits `recovery`.
    assert "recovery" in labels, (
        f"expected recovery in labels; got distinct: {sorted(set(labels))}"
    )

    # Sanity: every recovery day has the three rule inputs satisfied.
    for idx, label in enumerate(labels):
        if label != "recovery":
            continue
        dt = close.index[idx]
        assert v2_features.drawdown_252d.loc[dt] <= _SPEC_DRAWDOWN_THRESHOLD
        assert v2_features.return_63d.loc[dt] > _SPEC_RETURN_THRESHOLD
        assert close.loc[dt] > v2_features.sma_50.loc[dt]
        assert evidence[idx]["v2_override"] == {
            "from": evidence[idx]["v2_override"]["from"],
            "to": "recovery",
            "rule": "recovery",
        }
        # Override must be from a label lower-ranked than `recovery`
        # (bear / sideways / transition / unknown — NEVER bull).
        assert evidence[idx]["v2_override"]["from"] != "bull"


# ---------- End-to-end engine wire test (AGENTS rule A) ---------------------


def test_end_to_end_engine_emits_recovery_on_synthetic_series() -> None:
    """Wire-first AGENTS rule A: build_regime_timeline with the v2 default
    config and a SPY-like rebound series must emit at least one session
    whose ``trend_direction`` raw_label is `recovery`.
    """
    # n_total=432 places the rebound at sessions ~312-372 — within the
    # last 120 sessions emitted by classify_window(lookback_days=120).
    close_series = _spy_like_drawdown_recovery_series(n_total=432)
    # MarketContext requires open/high/low/close/volume on a single frame;
    # synthesize a minimal frame with open=close (no intraday range needed
    # for the trend_direction axis) and a constant volume.
    market_df = pd.DataFrame(
        {
            "date": [d.date() for d in close_series.index],
            "symbol": "SPY",
            "open": close_series.values,
            "high": close_series.values,
            "low": close_series.values,
            "close": close_series.values,
            "volume": 100_000_000,
        }
    )
    # Add RSP rows (equal-weight proxy) so v1 breadth doesn't fail. Mirror SPY.
    rsp_df = market_df.copy()
    rsp_df["symbol"] = "RSP"
    full_df = pd.concat([market_df, rsp_df], ignore_index=True)

    engine = RegimeEngine()
    end_dt = close_series.index[-1].date()
    timeline = engine.classify_window(
        end_date=end_dt,
        market_data=full_df,
        lookback_days=120,
    )

    raw_labels = [out.trend_direction.raw_label for out in timeline.outputs]
    assert "recovery" in raw_labels, (
        f"expected recovery in end-to-end raw_labels; got distinct: {sorted(set(raw_labels))}"
    )


# ---------- Config tests -----------------------------------------------------


def test_v2_yaml_loads_recovery_rules() -> None:
    cfg = load_default_regime_config()
    assert cfg.trend_direction_v2 is not None
    rules = cfg.trend_direction_v2.rules
    assert rules.recovery_drawdown_threshold == _SPEC_DRAWDOWN_THRESHOLD
    assert rules.recovery_return_threshold == _SPEC_RETURN_THRESHOLD


def test_recovery_drawdown_threshold_must_be_negative() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TrendDirectionV2RulesConfig(
            recovery_drawdown_threshold=0.0,
            recovery_return_threshold=0.10,
        )


def test_recovery_return_threshold_must_be_positive() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TrendDirectionV2RulesConfig(
            recovery_drawdown_threshold=-0.15,
            recovery_return_threshold=0.0,
        )
