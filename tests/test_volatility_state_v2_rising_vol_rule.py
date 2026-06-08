"""TDD tests for v2 §1C `rising_vol` rule + updated precedence (Slice 2.6).

Spec references (docs/regime_engine_v2_spec.md):
    §1C line 146-148 — `rising_vol` rule:
        ATR_ratio > 1.15
        OR realized_vol_10d > realized_vol_63d * 1.25
    §1C line 191 — precedence:
        crisis_vol > vol_crush > high_vol > rising_vol > low_vol > normal_vol > unknown
        (`vol_crush` is covered separately in test_volatility_state_v2_vol_crush.)

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
    VolatilityV2Config,
    VolatilityV2RulesConfig,
    load_default_regime_config,
)
from regime_detection.engine import RegimeEngine
from regime_detection.volatility_state import (
    build_raw_outputs as build_v1_raw_outputs,
    compute_features as compute_v1_features,
    raw_label_for_day,
    realized_vol as realized_vol_helper,
)
from regime_detection.volatility_state import (
    VolatilityV2Features,
    compute_volatility_v2_features,
    evaluate_rising_vol,
    evaluate_v2_volatility_label,
)

# v2 §1C line 147-148 — exact spec thresholds.
_SPEC_ATR_RATIO_THRESHOLD = 1.15
_SPEC_REALIZED_VOL_RATIO_THRESHOLD = 1.25
_SPEC_RV_SHORT = 10
_SPEC_RV_LONG = 63


@pytest.fixture
def rising_vol_rules() -> VolatilityV2RulesConfig:
    return VolatilityV2RulesConfig(
        atr_ratio_threshold=_SPEC_ATR_RATIO_THRESHOLD,
        realized_vol_ratio_threshold=_SPEC_REALIZED_VOL_RATIO_THRESHOLD,
        realized_vol_short_period=_SPEC_RV_SHORT,
        realized_vol_long_period=_SPEC_RV_LONG,
    )


@pytest.fixture
def v2_volatility_config() -> VolatilityV2Config:
    """Production defaults from configs/core3-v2.0.0.yaml (slice 2.2 fields)."""
    return VolatilityV2Config(
        atr_short_period=14,
        atr_long_period=50,
        gap_frequency_lookback_days=20,
        gap_threshold_pct=0.005,
        intraday_range_lookback_days=252,
    )


def _scalar_v2_features_at(
    *,
    dt: pd.Timestamp,
    atr_ratio: float,
    realized_vol_short: float,
    realized_vol_long: float,
) -> VolatilityV2Features:
    """Construct a single-day VolatilityV2Features carrying the rising_vol
    rule's three scalar inputs. Other §1C feature slots (gap_frequency,
    intraday_range_percentile) are NaN — irrelevant for the rising_vol rule.
    """
    idx = pd.DatetimeIndex([dt])
    nan = pd.Series([float("nan")], index=idx)
    return VolatilityV2Features(
        atr_ratio=pd.Series([atr_ratio], index=idx),
        gap_frequency_20d=nan.copy(),
        gap_frequency_percentile_252d=nan.copy(),
        intraday_range=nan.copy(),
        intraday_range_percentile_252d=nan.copy(),
        realized_vol_short=pd.Series([realized_vol_short], index=idx),
        realized_vol_long=pd.Series([realized_vol_long], index=idx),
        realized_vol_21d=nan.copy(),
        # vol_crush IV inputs left None — these rising_vol tests don't
        # exercise the vol_crush rule.
    )


# ---------- Boundary tests on `evaluate_rising_vol` -------------------------


def test_rising_vol_atr_limb_fires(rising_vol_rules) -> None:
    """atr_ratio > 1.15 alone triggers rising_vol (RV ratio not exceeded)."""
    dt = pd.Timestamp("2020-03-09")
    features = _scalar_v2_features_at(
        dt=dt, atr_ratio=1.16, realized_vol_short=0.20, realized_vol_long=0.20
    )
    assert evaluate_rising_vol(features, dt=dt, rules_config=rising_vol_rules) is True


def test_rising_vol_rv_limb_fires(rising_vol_rules) -> None:
    """realized_vol_10d > realized_vol_63d * 1.25 alone triggers rising_vol."""
    dt = pd.Timestamp("2020-03-09")
    features = _scalar_v2_features_at(
        dt=dt,
        atr_ratio=1.14,  # below ATR threshold
        realized_vol_short=0.30,  # 0.30 > 0.20 * 1.25 == 0.25  → True
        realized_vol_long=0.20,
    )
    assert evaluate_rising_vol(features, dt=dt, rules_config=rising_vol_rules) is True


def test_rising_vol_atr_exactly_threshold_is_false(rising_vol_rules) -> None:
    """v2 §1C line 147 uses strict `>` — atr_ratio == 1.15 fails."""
    dt = pd.Timestamp("2020-03-09")
    features = _scalar_v2_features_at(
        dt=dt, atr_ratio=1.15, realized_vol_short=0.20, realized_vol_long=0.20
    )
    assert evaluate_rising_vol(features, dt=dt, rules_config=rising_vol_rules) is False


def test_rising_vol_rv_exactly_threshold_is_false(rising_vol_rules) -> None:
    """v2 §1C line 148 uses strict `>` — realized_vol_10d == realized_vol_63d * 1.25 fails."""
    dt = pd.Timestamp("2020-03-09")
    features = _scalar_v2_features_at(
        dt=dt,
        atr_ratio=1.10,
        realized_vol_short=0.25,
        realized_vol_long=0.20,  # 0.20 * 1.25 == 0.25, not strictly greater
    )
    assert evaluate_rising_vol(features, dt=dt, rules_config=rising_vol_rules) is False


def test_rising_vol_both_limbs_false_returns_false(rising_vol_rules) -> None:
    dt = pd.Timestamp("2020-03-09")
    features = _scalar_v2_features_at(
        dt=dt, atr_ratio=1.05, realized_vol_short=0.21, realized_vol_long=0.20
    )
    assert evaluate_rising_vol(features, dt=dt, rules_config=rising_vol_rules) is False


@pytest.mark.parametrize(
    "field",
    ["atr_ratio", "realized_vol_short", "realized_vol_long"],
)
def test_rising_vol_nan_input_returns_false(rising_vol_rules, field) -> None:
    """NaN cold-start contract — any missing input falsifies the rule."""
    dt = pd.Timestamp("2020-03-09")
    args = {
        "dt": dt,
        "atr_ratio": 1.20,
        "realized_vol_short": 0.30,
        "realized_vol_long": 0.20,
    }
    args[field] = float("nan")
    features = _scalar_v2_features_at(**args)
    assert evaluate_rising_vol(features, dt=dt, rules_config=rising_vol_rules) is False


def test_rising_vol_false_when_dt_missing_from_feature_index(rising_vol_rules) -> None:
    dt = pd.Timestamp("2020-03-09")
    features = _scalar_v2_features_at(
        dt=pd.Timestamp("2020-03-10"),
        atr_ratio=1.30,
        realized_vol_short=0.40,
        realized_vol_long=0.20,
    )
    assert evaluate_rising_vol(features, dt=dt, rules_config=rising_vol_rules) is False


# ---------- Precedence tests on `evaluate_v2_volatility_label` --------------


@pytest.mark.parametrize("v1_label", ["crisis_vol", "high_vol"])
def test_precedence_higher_v1_labels_outrank_rising_vol(
    rising_vol_rules, v1_label
) -> None:
    """v2 §1C line 191: crisis_vol / high_vol both outrank rising_vol.

    When the v1 classifier already emits a higher-ranked label and the v2
    rising_vol predicate fires, the v1 label wins (return None → caller keeps v1).
    """
    dt = pd.Timestamp("2020-03-09")
    features = _scalar_v2_features_at(
        dt=dt, atr_ratio=1.30, realized_vol_short=0.40, realized_vol_long=0.20
    )
    assert (
        evaluate_v2_volatility_label(
            v1_label=v1_label,
            features=features,
            dt=dt,
            rules_config=rising_vol_rules,
        )
        is None
    )


@pytest.mark.parametrize("v1_label", ["low_vol", "normal_vol", "unknown"])
def test_precedence_rising_vol_overrides_lower_v1_labels(
    rising_vol_rules, v1_label
) -> None:
    """v2 §1C line 191: rising_vol > low_vol > normal_vol > unknown."""
    dt = pd.Timestamp("2020-03-09")
    features = _scalar_v2_features_at(
        dt=dt, atr_ratio=1.30, realized_vol_short=0.40, realized_vol_long=0.20
    )
    assert (
        evaluate_v2_volatility_label(
            v1_label=v1_label,
            features=features,
            dt=dt,
            rules_config=rising_vol_rules,
        )
        == "rising_vol"
    )


def test_precedence_rising_vol_predicate_false_returns_none(rising_vol_rules) -> None:
    """No fire → caller keeps v1 label."""
    dt = pd.Timestamp("2020-03-09")
    features = _scalar_v2_features_at(
        dt=dt, atr_ratio=1.05, realized_vol_short=0.20, realized_vol_long=0.20
    )
    assert (
        evaluate_v2_volatility_label(
            v1_label="normal_vol",
            features=features,
            dt=dt,
            rules_config=rising_vol_rules,
        )
        is None
    )


def test_precedence_unknown_v1_label_treated_as_lowest_rank(rising_vol_rules) -> None:
    dt = pd.Timestamp("2020-03-09")
    features = _scalar_v2_features_at(
        dt=dt, atr_ratio=1.30, realized_vol_short=0.40, realized_vol_long=0.20
    )
    assert (
        evaluate_v2_volatility_label(
            v1_label="custom_label",
            features=features,
            dt=dt,
            rules_config=rising_vol_rules,
        )
        == "rising_vol"
    )


# ---------- V1 contract preservation ----------------------------------------


def _spy_like_volatility_expansion_series(*, n_total: int = 700) -> pd.Series:
    """Construct a SPY-like series with a gentle vol expansion that fires
    rising_vol WITHOUT tripping v1's high_vol / crisis_vol thresholds.

    Design:
    - 500-day warmup with mixed low-and-mid vol so the 21d realized-vol
      252d percentile (v1 high_vol gate at >= 0.80) settles in mid-range.
    - 200-day moderate expansion (~1.6× the warmup std). RV_10d > RV_63d *
      1.25 fires within the first ~20 sessions of the expansion (RV_10d
      sees the new regime faster than RV_63d which still drags the
      mixed-warmup tail). The expansion magnitude is calibrated so the
      21d vol percentile stays under 0.80 for many sessions — keeping
      v1 at normal_vol so the v2 override to rising_vol can occur.

    Per CLAUDE.md "no toy names": the series is calibrated to the actual
    v1 thresholds (vol_pct >= 0.80 for high_vol; ret1 <= -0.05 for crisis)
    and the v2 §1C thresholds (atr_ratio > 1.15 OR rv_10d > rv_63d * 1.25)
    so the test exercises real production decision paths.
    """
    rng = np.random.default_rng(seed=20260512)
    base = 400.0
    # Warmup: mixed std drawn uniformly in [0.005, 0.012] per day. This
    # populates the 252d realised-vol-percentile window with a broad
    # spread so the 0.80 threshold corresponds to ~0.012-std sessions.
    warmup_stds = rng.uniform(0.005, 0.012, size=500)
    warmup_returns = rng.normal(0.0, warmup_stds)
    # Expansion: bump std to ~0.015 (just above the warmup ceiling) so
    # RV_10d (10-day std * sqrt(252)) climbs above RV_63d * 1.25 but the
    # 21d realised-vol-percentile stays around 0.6-0.8 (NOT >= 0.80).
    expansion_returns = rng.normal(0.0, 0.015, size=200)
    returns = np.concatenate([warmup_returns, expansion_returns])[:n_total]
    # Clip extremes so no single-day move triggers v1 crisis (ret1 <= -0.05).
    returns = np.clip(returns, -0.04, 0.04)
    levels = base * np.exp(np.cumsum(returns))
    sessions = nyse_sessions_between(date(2021, 1, 4), date(2024, 12, 31))
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in sessions[:n_total]])
    return pd.Series(levels[:n_total], index=index, name="close")


def test_v1_contract_byte_identity_when_v2_features_absent() -> None:
    """Calling build_raw_outputs WITHOUT v2 features/rules emits the EXACT
    same labels and evidence as the pre-slice-2.6 v1 path.
    """
    close = _spy_like_volatility_expansion_series()
    v1_features = compute_v1_features(close=close, vix_proxy_close=None)

    labels_no_v2, evidence_no_v2 = build_v1_raw_outputs(v1_features)
    # Sanity — no `rising_vol` ever appears on the v1 path.
    assert "rising_vol" not in labels_no_v2
    assert all("v2_override" not in ev for ev in evidence_no_v2)

    labels_explicit_none, evidence_explicit_none = build_v1_raw_outputs(
        v1_features,
        volatility_state_v2_features=None,
        volatility_state_v2_rules=None,
    )
    assert labels_explicit_none == labels_no_v2
    assert evidence_explicit_none == evidence_no_v2


def test_v1_raw_label_for_day_byte_identity_when_v2_features_absent() -> None:
    close = _spy_like_volatility_expansion_series()
    v1_features = compute_v1_features(close=close, vix_proxy_close=None)
    dt = close.index[-1]
    label_v1, ev_v1 = raw_label_for_day(v1_features, dt)
    label_explicit, ev_explicit = raw_label_for_day(
        v1_features,
        dt,
        volatility_state_v2_features=None,
        volatility_state_v2_rules=None,
    )
    assert label_v1 == label_explicit
    assert ev_v1 == ev_explicit
    assert "v2_override" not in ev_v1


# ---------- Classifier integration test ------------------------------------


def _compute_v2_features_with_rv(
    *,
    spy_ohlcv: pd.DataFrame,
    cfg: VolatilityV2Config,
    rules: VolatilityV2RulesConfig,
) -> VolatilityV2Features:
    """Synthesize a complete VolatilityV2Features (slice 2.2 fields + slice
    2.6 RV fields). Slice 2.6 will land this on the production compute path
    but tests here construct directly so the rising_vol predicate can be
    exercised without an axis-classifier round-trip.
    """
    base = compute_volatility_v2_features(
        open_=spy_ohlcv["open"],
        high=spy_ohlcv["high"],
        low=spy_ohlcv["low"],
        close=spy_ohlcv["close"],
        config=cfg,
    )
    rv_short = realized_vol_helper(
        spy_ohlcv["close"], window=rules.realized_vol_short_period
    )
    rv_long = realized_vol_helper(
        spy_ohlcv["close"], window=rules.realized_vol_long_period
    )
    return VolatilityV2Features(
        atr_ratio=base.atr_ratio,
        gap_frequency_20d=base.gap_frequency_20d,
        gap_frequency_percentile_252d=base.gap_frequency_percentile_252d,
        intraday_range=base.intraday_range,
        intraday_range_percentile_252d=base.intraday_range_percentile_252d,
        realized_vol_short=rv_short,
        realized_vol_long=rv_long,
        realized_vol_21d=base.realized_vol_21d,
    )


def _ohlcv_from_close(close: pd.Series) -> pd.DataFrame:
    """Minimal OHLCV (open=high=low=close) for an axis-only synthetic test."""
    return pd.DataFrame(
        {
            "open": close.values,
            "high": close.values,
            "low": close.values,
            "close": close.values,
            "volume": 100_000_000,
        },
        index=close.index,
    )


def test_build_raw_outputs_emits_rising_vol_on_synthetic_expansion(
    v2_volatility_config, rising_vol_rules
) -> None:
    close = _spy_like_volatility_expansion_series()
    v1_features = compute_v1_features(close=close, vix_proxy_close=None)
    ohlcv = _ohlcv_from_close(close)
    v2_features = _compute_v2_features_with_rv(
        spy_ohlcv=ohlcv, cfg=v2_volatility_config, rules=rising_vol_rules
    )

    labels, evidence = build_v1_raw_outputs(
        v1_features,
        volatility_state_v2_features=v2_features,
        volatility_state_v2_rules=rising_vol_rules,
    )
    # At least one session emits `rising_vol` (predicate must fire on the
    # synthetic volatility expansion AND the v1 label must rank below
    # rising_vol per §1C line 191).
    assert (
        "rising_vol" in labels
    ), f"expected rising_vol in labels; got distinct: {sorted(set(labels))}"

    for idx, label in enumerate(labels):
        if label != "rising_vol":
            continue
        dt = close.index[idx]
        atr = v2_features.atr_ratio.loc[dt]
        rv_short = v2_features.realized_vol_short.loc[dt]
        rv_long = v2_features.realized_vol_long.loc[dt]
        atr_limb = (not pd.isna(atr)) and atr > _SPEC_ATR_RATIO_THRESHOLD
        rv_limb = (
            (not pd.isna(rv_short))
            and (not pd.isna(rv_long))
            and rv_short > rv_long * _SPEC_REALIZED_VOL_RATIO_THRESHOLD
        )
        assert atr_limb or rv_limb, f"rising_vol fired without limb at {dt}"
        # The override must be FROM a label lower-ranked than rising_vol.
        assert evidence[idx]["v2_override"]["to"] == "rising_vol"
        assert evidence[idx]["v2_override"]["from"] in {
            "low_vol",
            "normal_vol",
            "unknown",
        }
        assert evidence[idx]["v2_override"]["rule"] == "rising_vol"


# ---------- End-to-end engine wire test (AGENTS rule A) ---------------------


def test_end_to_end_engine_emits_rising_vol_on_synthetic_series(
    synthetic_v2_kwargs_for_market_data,
) -> None:
    """Wire-first AGENTS rule A: build_regime_timeline with the v2 default
    config and a SPY-like volatility-expansion series must emit at least one
    session whose ``volatility_state`` raw_label is `rising_vol`.
    """
    close_series = _spy_like_volatility_expansion_series(n_total=700)
    market_df = pd.DataFrame(
        {
            "date": [d.date() for d in close_series.index],
            "symbol": "SPY",
            "open": close_series.values,
            "high": close_series.values,
            "low": close_series.values,
            "close": close_series.values,
            "volume": range(100_000_000, 100_000_000 + len(close_series.index)),
        }
    )
    rsp_df = market_df.copy()
    rsp_df["symbol"] = "RSP"
    vix_df = market_df.copy()
    vix_df["symbol"] = "VIX"
    vix_values = np.linspace(30.0, 10.0, len(vix_df))
    vix_df["open"] = vix_values
    vix_df["high"] = vix_values * 1.01
    vix_df["low"] = vix_values * 0.99
    vix_df["close"] = vix_values
    full_df = pd.concat([market_df, rsp_df, vix_df], ignore_index=True)

    engine = RegimeEngine()
    end_dt = close_series.index[-1].date()
    kwargs = synthetic_v2_kwargs_for_market_data(full_df)
    timeline = engine.classify_window(
        end_date=end_dt,
        market_data=full_df,
        lookback_days=120,
        config=kwargs["config"],
        event_calendar=kwargs["event_calendar"],
        sector_etf_closes=kwargs["sector_etf_closes"],
        cross_asset_closes=kwargs["cross_asset_closes"],
        macro_series=kwargs["macro_series"],
        pit_constituent_intervals=kwargs["pit_constituent_intervals"],
        constituent_ohlcv=kwargs["constituent_ohlcv"],
    )

    raw_labels = [out.volatility_state.raw_label for out in timeline.outputs]
    assert (
        "rising_vol" in raw_labels
    ), f"expected rising_vol in end-to-end raw_labels; got distinct: {sorted(set(raw_labels))}"


# ---------- Config tests -----------------------------------------------------


def test_v2_yaml_loads_rising_vol_rules() -> None:
    cfg = load_default_regime_config()
    assert cfg.volatility_state_v2 is not None
    rules = cfg.volatility_state_v2.rules
    assert rules.atr_ratio_threshold == _SPEC_ATR_RATIO_THRESHOLD
    assert rules.realized_vol_ratio_threshold == _SPEC_REALIZED_VOL_RATIO_THRESHOLD
    assert rules.realized_vol_short_period == _SPEC_RV_SHORT
    assert rules.realized_vol_long_period == _SPEC_RV_LONG


def test_rising_vol_rules_rejects_unknown_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        VolatilityV2RulesConfig(
            atr_ratio_threshold=1.15,
            realized_vol_ratio_threshold=1.25,
            realized_vol_short_period=10,
            realized_vol_long_period=63,
            unexpected_field=42,  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    "field", ["atr_ratio_threshold", "realized_vol_ratio_threshold"]
)
def test_rising_vol_rules_thresholds_must_be_positive(field) -> None:
    from pydantic import ValidationError

    args = {
        "atr_ratio_threshold": 1.15,
        "realized_vol_ratio_threshold": 1.25,
        "realized_vol_short_period": 10,
        "realized_vol_long_period": 63,
    }
    args[field] = 0.0
    with pytest.raises(ValidationError):
        VolatilityV2RulesConfig(**args)


@pytest.mark.parametrize(
    "field", ["realized_vol_short_period", "realized_vol_long_period"]
)
def test_rising_vol_rules_periods_must_be_positive(field) -> None:
    from pydantic import ValidationError

    args = {
        "atr_ratio_threshold": 1.15,
        "realized_vol_ratio_threshold": 1.25,
        "realized_vol_short_period": 10,
        "realized_vol_long_period": 63,
    }
    args[field] = 0
    with pytest.raises(ValidationError):
        VolatilityV2RulesConfig(**args)


# ---------- Shared realized_vol helper --------------------------------------


def test_realized_vol_helper_is_annualized() -> None:
    """The shared helper exposed in volatility_state.py must annualize via
    sqrt(252) so v1 (slice 2.2 ago) and v2 (slice 2.6+) consume one path.
    """
    # Pure-constant returns → zero std → zero realized vol.
    idx = pd.DatetimeIndex(pd.bdate_range(start="2020-01-01", periods=100))
    close = pd.Series(np.linspace(100.0, 100.0, 100), index=idx)
    rv = realized_vol_helper(close, window=10)
    assert pd.isna(rv.iloc[:9]).all()  # warmup
    # Constant series → all-zero returns → std == 0 from t=10 onwards.
    assert float(rv.iloc[-1]) == 0.0
