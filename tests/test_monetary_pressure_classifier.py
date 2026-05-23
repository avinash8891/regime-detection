"""V2 §2A Monetary Pressure axis classifier tests (Slice 4.x).

Per AGENTS.md G + ~/.claude/CLAUDE.md testing rules:
  - Real FRED series IDs (DGS2, DGS10, broad_usd_index) — no mocks.
  - Production constants imported (MONETARY_PRESSURE_V2_RISK_RANK,
    MonetaryPressureV2Label).
  - End-to-end integration tests for the wire path + cohort routing.

Spec authority: docs/regime_engine_v2_spec.md §2A (lines 1093-1130
Ambiguity Log #46 pins).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from regime_detection.axis_series import build_monetary_pressure_axis_series
from regime_detection.calendar import nyse_sessions_between
from regime_detection.config import (
    MonetaryPressureV2Config,
    MonetaryPressureV2FeaturesConfig,
    MonetaryPressureV2RulesConfig,
    load_default_regime_config,
)
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import build_feature_store
from regime_detection.hysteresis import apply_per_label_asymmetric_hysteresis
from regime_detection.market_context import build_market_context
from regime_detection.models import MonetaryPressureV2Output
from regime_detection.monetary_pressure import (
    MONETARY_PRESSURE_V2_RISK_RANK,
    MonetaryPressureRuleInputs,
    MonetaryPressureV2Features,
    MonetaryPressureV2Label,
    compute_monetary_pressure_features,
    evaluate_rules,
)


_TRAINING_SESSIONS = 1400  # > 1260 normalizer + 63 cold-start
_LAST_SESSION = pd.Timestamp("2025-04-30")
_SEED = 20260514

_LOOKBACK_DAYS = 63
_NORMALIZER_WINDOW = 1260
_RATE_SHOCK_LOOKBACK = 21
_BROAD_USD_LOOKBACK = 63


def _bdate_index(periods: int = _TRAINING_SESSIONS) -> pd.DatetimeIndex:
    sessions = nyse_sessions_between(
        (_LAST_SESSION - pd.Timedelta(days=periods * 2)).date(),
        _LAST_SESSION.date(),
    )
    return pd.DatetimeIndex([pd.Timestamp(d) for d in sessions[-periods:]])


def _features_config() -> MonetaryPressureV2FeaturesConfig:
    return MonetaryPressureV2FeaturesConfig(
        yield_change_lookback_days=_LOOKBACK_DAYS,
        zscore_normalizer_window_days=_NORMALIZER_WINDOW,
        rate_shock_lookback_days=_RATE_SHOCK_LOOKBACK,
        broad_usd_lookback_days=_BROAD_USD_LOOKBACK,
    )


def _default_classifier_config() -> MonetaryPressureV2Config:
    return load_default_regime_config().monetary_pressure_state


def _yield_series(*, index: pd.DatetimeIndex, base: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed=seed)
    innovations = rng.normal(loc=0.0, scale=0.03, size=len(index))
    levels = np.cumsum(innovations) + base
    levels = np.clip(levels, 0.1, None)
    return pd.Series(levels, index=index, name="yield")


def _usd_series(*, index: pd.DatetimeIndex, base: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed=seed)
    innovations = rng.normal(loc=0.0, scale=0.3, size=len(index))
    levels = np.cumsum(innovations) + base
    return pd.Series(levels, index=index, name="broad_usd_index")


# =============================================================================
# Group A — Feature compute: 3 new z-scores
# =============================================================================


def test_compute_monetary_pressure_features_emits_five_series():
    idx = _bdate_index()
    dgs2 = _yield_series(index=idx, base=4.5, seed=_SEED + 1)
    dgs10 = _yield_series(index=idx, base=4.2, seed=_SEED + 2)
    usd = _usd_series(index=idx, base=100.0, seed=_SEED + 3)
    feats = compute_monetary_pressure_features(
        dgs2=dgs2, dgs10=dgs10, broad_usd_index=usd, config=_features_config()
    )
    assert isinstance(feats, MonetaryPressureV2Features)
    names = feats.feature_names
    assert "yield_change_zscore_2y_63d" in names
    assert "yield_change_zscore_10y_63d" in names
    assert "broad_usd_index_zscore_63d" in names
    assert "yield_change_zscore_21d_2y" in names
    assert "yield_change_zscore_21d_10y" in names
    for name in names:
        s = getattr(feats, name)
        assert s.index.equals(idx)


def test_compute_monetary_pressure_features_cold_start_is_nan():
    idx = _bdate_index()
    dgs2 = _yield_series(index=idx, base=4.5, seed=_SEED + 1)
    dgs10 = _yield_series(index=idx, base=4.2, seed=_SEED + 2)
    usd = _usd_series(index=idx, base=100.0, seed=_SEED + 3)
    feats = compute_monetary_pressure_features(
        dgs2=dgs2, dgs10=dgs10, broad_usd_index=usd, config=_features_config()
    )
    # 63d-change z-score with 1260d normalizer → first valid at t=1322
    first_valid_63 = _LOOKBACK_DAYS + _NORMALIZER_WINDOW - 1
    assert pd.isna(feats.yield_change_zscore_2y_63d.iloc[first_valid_63 - 1])
    assert pd.isna(feats.broad_usd_index_zscore_63d.iloc[first_valid_63 - 1])
    # 21d-change z-score with 1260d normalizer → first valid at t=1280
    first_valid_21 = _RATE_SHOCK_LOOKBACK + _NORMALIZER_WINDOW - 1
    assert pd.isna(feats.yield_change_zscore_21d_2y.iloc[first_valid_21 - 1])
    # After warmup non-NaN
    assert not pd.isna(feats.yield_change_zscore_21d_2y.iloc[first_valid_21])


def test_compute_monetary_pressure_features_broad_usd_none_returns_nan_series():
    idx = _bdate_index()
    dgs2 = _yield_series(index=idx, base=4.5, seed=_SEED + 1)
    dgs10 = _yield_series(index=idx, base=4.2, seed=_SEED + 2)
    feats = compute_monetary_pressure_features(
        dgs2=dgs2, dgs10=dgs10, broad_usd_index=None, config=_features_config()
    )
    # broad_usd_index_zscore_63d must be a Series aligned to dgs2 index, all NaN.
    assert feats.broad_usd_index_zscore_63d.index.equals(idx)
    assert feats.broad_usd_index_zscore_63d.isna().all()


# =============================================================================
# Group B — Rule predicates
# =============================================================================


def _rules() -> MonetaryPressureV2RulesConfig:
    return MonetaryPressureV2RulesConfig(
        tightening_pressure_zscore_threshold=1.5,
        easing_pressure_zscore_threshold=-1.5,
        rate_shock_zscore_threshold=2.0,
    )


def _inputs(**kwargs) -> MonetaryPressureRuleInputs:
    base = dict(
        zscore_2y_63d=0.0,
        zscore_10y_63d=0.0,
        broad_usd_zscore_63d=0.0,
        zscore_21d_2y=0.0,
        zscore_21d_10y=0.0,
    )
    base.update(kwargs)
    return MonetaryPressureRuleInputs(**base)


def test_rate_shock_fires_on_2y_21d_abs_threshold():
    label = evaluate_rules(inputs=_inputs(zscore_21d_2y=2.5), config=_rules())
    assert label == "rate_shock"


def test_rate_shock_fires_on_10y_21d_absolute_negative_move():
    label = evaluate_rules(inputs=_inputs(zscore_21d_10y=-2.5), config=_rules())
    assert label == "rate_shock"


def test_tightening_pressure_fires_on_2y_63d_above_threshold():
    label = evaluate_rules(inputs=_inputs(zscore_2y_63d=1.6), config=_rules())
    assert label == "tightening_pressure"


def test_tightening_pressure_fires_on_broad_usd_above_threshold():
    label = evaluate_rules(inputs=_inputs(broad_usd_zscore_63d=1.6), config=_rules())
    assert label == "tightening_pressure"


def test_easing_pressure_fires_when_either_yield_signal_eases():
    label = evaluate_rules(inputs=_inputs(zscore_2y_63d=-1.6), config=_rules())
    assert label == "easing_pressure"

    label = evaluate_rules(inputs=_inputs(zscore_10y_63d=-1.6), config=_rules())
    assert label == "easing_pressure"

    label = evaluate_rules(
        inputs=_inputs(zscore_2y_63d=-1.6, zscore_10y_63d=-1.6), config=_rules()
    )
    assert label == "easing_pressure"


def test_neutral_monetary_when_no_rule_fires():
    label = evaluate_rules(inputs=_inputs(), config=_rules())
    assert label == "neutral_monetary"


def test_rate_shock_outranks_tightening_pressure_when_both_match():
    # Tightening signal AND rate_shock signal both present → rate_shock wins.
    label = evaluate_rules(
        inputs=_inputs(zscore_2y_63d=1.7, zscore_21d_2y=2.5), config=_rules()
    )
    assert label == "rate_shock"


def test_nan_inputs_fall_through_to_neutral():
    label = evaluate_rules(
        inputs=_inputs(
            zscore_2y_63d=float("nan"),
            zscore_10y_63d=float("nan"),
            broad_usd_zscore_63d=float("nan"),
            zscore_21d_2y=float("nan"),
            zscore_21d_10y=float("nan"),
        ),
        config=_rules(),
    )
    # All-NaN doesn't trigger any > / < comparison; falls to neutral.
    assert label == "neutral_monetary"


# =============================================================================
# Group C — Risk rank constant
# =============================================================================


def test_risk_rank_pins_per_log_46_d():
    assert MONETARY_PRESSURE_V2_RISK_RANK == {
        "neutral_monetary": 0,
        "easing_pressure": 1,
        "unknown": 1,
        "tightening_pressure": 2,
        "rate_shock": 3,
    }


# =============================================================================
# Group D — Hysteresis
# =============================================================================


def test_rate_shock_holds_five_sessions_per_log_46_e():
    cfg = _default_classifier_config()
    assert cfg.deescalation_days_by_label["rate_shock"] == 5

    raws: list[MonetaryPressureV2Label] = (
        ["rate_shock"] * 10 + ["neutral_monetary"] * 3 + ["rate_shock"] * 5
    )
    stable, _ = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=MONETARY_PRESSURE_V2_RISK_RANK,
        deescalation_days_by_label=cfg.deescalation_days_by_label,
        default_deescalation_days=cfg.default_deescalation_days,
    )
    # After 3 days of neutral_monetary the stable should still be rate_shock
    # (de-escalation threshold = 5).
    assert stable[10] == "rate_shock"
    assert stable[11] == "rate_shock"
    assert stable[12] == "rate_shock"


def test_neutral_monetary_immediate_de_escalation():
    cfg = _default_classifier_config()
    assert cfg.deescalation_days_by_label["neutral_monetary"] == 0


def test_unknown_quality_gap_clears_immediately_when_monetary_features_recover():
    cfg = _default_classifier_config()
    assert cfg.deescalation_days_by_label["unknown"] == 0
    stable, active = apply_per_label_asymmetric_hysteresis(
        raw_labels=["unknown", "neutral_monetary"],
        risk_rank=MONETARY_PRESSURE_V2_RISK_RANK,
        deescalation_days_by_label=cfg.deescalation_days_by_label,
        default_deescalation_days=cfg.default_deescalation_days,
    )
    assert stable[-1] == "neutral_monetary"
    assert active[-1] == "neutral_monetary"


# =============================================================================
# Group E — Wire integration end-to-end
# =============================================================================


def _synthetic_market_data(index: pd.DatetimeIndex, seed: int = _SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    returns = rng.normal(0.0, 0.01, size=len(index))
    close = (1.0 + returns).cumprod() * 400.0
    rows: list[dict[str, object]] = []
    for i, ts in enumerate(index):
        for symbol in ("SPY", "RSP", "VIXY"):
            mult = {"SPY": 1.0, "RSP": 0.5, "VIXY": 0.05}[symbol]
            rows.append(
                {
                    "date": ts.date(),
                    "symbol": symbol,
                    "open": float(close[i]) * mult,
                    "high": float(close[i]) * mult * 1.005,
                    "low": float(close[i]) * mult * 0.995,
                    "close": float(close[i]) * mult,
                    "volume": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def _build_context_with_macro(
    *,
    dgs2_offset: float = 0.0,
    dgs10_offset: float = 0.0,
    usd_drift: float = 0.0,
):
    index = _bdate_index()
    market_data = _synthetic_market_data(index)
    dgs2 = _yield_series(index=index, base=4.5, seed=_SEED + 1) + dgs2_offset
    dgs10 = _yield_series(index=index, base=4.2, seed=_SEED + 2) + dgs10_offset
    usd = _usd_series(index=index, base=100.0, seed=_SEED + 3)
    if usd_drift != 0.0:
        # Apply a linear drift over the last 100 sessions to push z-score up/down.
        drift = np.linspace(0, usd_drift, len(index))
        usd = usd + drift
    macro = {"2y_yield": dgs2, "10y_yield": dgs10, "broad_usd_index": usd}
    config = RegimeEngine().config
    context = build_market_context(
        end_date=_LAST_SESSION.date(),
        market_data=market_data,
        config=config,
        macro_series=macro,
    )
    return context


def test_classifier_returns_none_when_feature_store_monetary_seam_is_none():
    """No monetary_pressure_v2 config / no macro series → seam None → classifier returns None."""
    index = _bdate_index()
    market_data = _synthetic_market_data(index)
    config = RegimeEngine().config
    context = build_market_context(
        end_date=_LAST_SESSION.date(),
        market_data=market_data,
        config=config,
    )
    bare_store = build_feature_store(context)
    assert bare_store.monetary is None
    out = build_monetary_pressure_axis_series(context, bare_store)
    assert out is None


def test_classifier_emits_outputs_when_seam_lit():
    context = _build_context_with_macro()
    store = build_feature_store(
        context,
        monetary_pressure_v2_config=context.config.monetary_pressure_v2,
    )
    assert store.monetary is not None
    out = build_monetary_pressure_axis_series(context, store)
    assert out is not None
    assert set(out.keys()) == set(context.sessions)
    allowed = set(MONETARY_PRESSURE_V2_RISK_RANK.keys())
    for output in out.values():
        assert output.raw_label in allowed
        assert isinstance(output, MonetaryPressureV2Output)


def test_classifier_emits_central_bank_text_score_as_evidence_only():
    context = _build_context_with_macro()
    store = build_feature_store(
        context,
        monetary_pressure_v2_config=context.config.monetary_pressure_v2,
    )
    assert store.monetary is not None
    score = pd.Series(0.25, index=context.spy_ohlcv.index, name="central_bank_text_score")
    store = store.model_copy(
        update={"monetary": replace(store.monetary, central_bank_text_score=score)}
    )

    out = build_monetary_pressure_axis_series(context, store)

    assert out is not None
    sample = next(
        output for output in out.values() if "rule_evidence" in output.evidence
    )
    assert sample.evidence["rule_evidence"]["central_bank_text_score"] == 0.25


def test_engine_classify_window_populates_monetary_pressure_state(
    synthetic_v2_kwargs_for_market_data,
):
    """Top-level engine + monetary_pressure_state config → axis output populated."""
    index = _bdate_index()
    market_data = _synthetic_market_data(index)
    dgs2 = _yield_series(index=index, base=4.5, seed=_SEED + 1)
    dgs10 = _yield_series(index=index, base=4.2, seed=_SEED + 2)
    usd = _usd_series(index=index, base=100.0, seed=_SEED + 3)
    macro = {"2y_yield": dgs2, "10y_yield": dgs10, "broad_usd_index": usd}
    kwargs = synthetic_v2_kwargs_for_market_data(market_data)
    timeline = RegimeEngine().classify_window(
        end_date=_LAST_SESSION.date(),
        market_data=market_data,
        lookback_days=50,
        event_calendar=kwargs["event_calendar"],
        sector_etf_closes=kwargs["sector_etf_closes"],
        cross_asset_closes=kwargs["cross_asset_closes"],
        pit_constituent_intervals=kwargs["pit_constituent_intervals"],
        constituent_ohlcv=kwargs["constituent_ohlcv"],
        macro_series=macro,
    )
    populated = [
        out for out in timeline.outputs if out.monetary_pressure_state is not None
    ]
    assert populated, "Expected monetary_pressure_state populated when seam lit"
    allowed = set(MONETARY_PRESSURE_V2_RISK_RANK.keys())
    for out in populated:
        assert out.monetary_pressure_state.active_label in allowed
        assert (
            out.structural_causal_state.monetary_pressure.label
            == out.monetary_pressure_state.active_label
        )
        assert (
            out.structural_causal_state.monetary_pressure.evidence
            == out.monetary_pressure_state.evidence
        )
        assert (
            out.structural_causal_state.monetary_pressure.data_quality
            == out.monetary_pressure_state.data_quality
        )


def test_engine_classify_window_monetary_pressure_state_none_in_pure_v1_mode():
    """V1 config lacks transition_score, so the default timeline fails loudly."""
    from pathlib import Path
    from regime_detection.config import load_regime_config

    v1_yaml = (
        Path(__file__).parent.parent
        / "src"
        / "regime_detection"
        / "configs"
        / "core3-v1.0.0.yaml"
    )
    v1_config = load_regime_config(v1_yaml)
    assert v1_config.monetary_pressure_state is None
    index = _bdate_index(periods=400)
    market_data = _synthetic_market_data(index)
    with pytest.raises(RuntimeError, match="context.config.transition_score"):
        RegimeEngine().classify_window(
            end_date=_LAST_SESSION.date(),
            market_data=market_data,
            lookback_days=20,
            event_calendar=pd.DataFrame(columns=["date", "market", "type", "importance"]),
            config=v1_config,
        )


# =============================================================================
# Group F — Cohort routing integration
# =============================================================================


def test_cohort_routing_tightening_specialist_fires_when_label_tightens():
    """When monetary_pressure_state.active_label == tightening_pressure, the
    cohort_routing tightening_specialist rule fires."""
    from regime_detection.cohort_routing import evaluate_cohort_routing

    config = RegimeEngine().config
    routing_config = config.cohort_routing
    assert routing_config is not None

    routing = evaluate_cohort_routing(
        trend_direction_active="bull",
        trend_character_active="trending",
        volatility_state_active="normal_vol",
        breadth_state_active="strong_breadth",
        network_fragility_active="diversified_normal",
        monetary_pressure_active="tightening_pressure",
        config=routing_config,
    )
    assert routing.active_cohort == "tightening_specialist"


def test_cohort_routing_easing_specialist_fires_when_label_eases():
    from regime_detection.cohort_routing import evaluate_cohort_routing

    config = RegimeEngine().config
    routing = evaluate_cohort_routing(
        trend_direction_active="bull",
        trend_character_active="trending",
        volatility_state_active="normal_vol",
        breadth_state_active="strong_breadth",
        network_fragility_active="diversified_normal",
        monetary_pressure_active="easing_pressure",
        config=config.cohort_routing,
    )
    assert routing.active_cohort == "easing_specialist"
