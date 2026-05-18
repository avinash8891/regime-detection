"""Slice 2.7 — build_volume_liquidity_axis_series integration tests.

Tests follow the slice-1.4 NetworkFragility pattern:
- Realistic SPY-volume series (~80M shares baseline).
- No toy names; uses production constants (VOLUME_LIQUIDITY_RISK_RANK,
  VolumeLiquidityLabel).
- End-to-end engine test (AGENTS rule A) via build_regime_timeline.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from regime_detection.axis_series import build_volume_liquidity_axis_series
from regime_detection.calendar import nyse_sessions_between
from regime_detection.config import (
    load_default_regime_config,
)
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import build_feature_store
from regime_detection.market_context import build_market_context
from regime_detection.models import VolumeLiquidityStateOutput
from regime_detection.volume_liquidity_rules import (
    VOLUME_LIQUIDITY_RISK_RANK,
    VolumeLiquidityLabel,
)


_TRAINING_SESSIONS = 700
_LAST_SESSION = pd.Timestamp("2025-04-30")
_SEED = 20260514
_SPY_VOLUME_BASELINE = 80_000_000  # realistic SPY daily shares baseline


def _bdate_index(
    periods: int = _TRAINING_SESSIONS, end: pd.Timestamp = _LAST_SESSION
) -> pd.DatetimeIndex:
    sessions = nyse_sessions_between(
        (end - pd.Timedelta(days=periods * 2)).date(),
        end.date(),
    )
    return pd.DatetimeIndex([pd.Timestamp(d) for d in sessions[-periods:]])


def _synthetic_spy_market_data(
    *, index: pd.DatetimeIndex, seed: int = _SEED
) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    returns = rng.normal(0.0, 0.01, size=len(index))
    close = (1.0 + returns).cumprod() * 400.0
    # Volume noise around 80M baseline, with a panic spike on the last 5 sessions
    # so the classifier has a chance to emit panic_volume after warmup.
    volume = rng.normal(
        _SPY_VOLUME_BASELINE, _SPY_VOLUME_BASELINE * 0.10, size=len(index)
    )
    # Inject a panic event near the end: 3x baseline volume AND -3% return.
    panic_idx = len(index) - 3
    volume[panic_idx] = _SPY_VOLUME_BASELINE * 4.0
    close[panic_idx] = close[panic_idx - 1] * (1.0 - 0.03)

    rows: list[dict[str, object]] = []
    for i, ts in enumerate(index):
        rows.append(
            {
                "date": ts.date(),
                "symbol": "SPY",
                "open": float(close[i]),
                "high": float(close[i]) * 1.005,
                "low": float(close[i]) * 0.995,
                "close": float(close[i]),
                "volume": float(volume[i]),
            }
        )
        rows.append(
            {
                "date": ts.date(),
                "symbol": "RSP",
                "open": float(close[i]) * 0.5,
                "high": float(close[i]) * 0.5 * 1.005,
                "low": float(close[i]) * 0.5 * 0.995,
                "close": float(close[i]) * 0.5,
                "volume": float(volume[i]) * 0.5,
            }
        )
        rows.append(
            {
                "date": ts.date(),
                "symbol": "VIXY",
                "open": 20.0,
                "high": 20.5,
                "low": 19.5,
                "close": 20.0,
                "volume": 100_000.0,
            }
        )
    return pd.DataFrame(rows)


def _build_context_with_volume():
    index = _bdate_index()
    market_data = _synthetic_spy_market_data(index=index)
    config = RegimeEngine().config
    context = build_market_context(
        end_date=_LAST_SESSION.date(),
        market_data=market_data,
        config=config,
    )
    return context, market_data


# ---------- Classifier wiring -----------------------------------------------


def test_classifier_returns_none_when_feature_store_volume_seam_is_none():
    """If volume_liquidity_v2 config is absent on context, the feature store
    seam is None — classifier propagates None (no v2 axis output)."""
    index = _bdate_index()
    market_data = _synthetic_spy_market_data(index=index)
    config = RegimeEngine().config
    context = build_market_context(
        end_date=_LAST_SESSION.date(),
        market_data=market_data,
        config=config,
    )
    # Build a store WITHOUT passing the v2 config — seam stays None.
    bare_store = build_feature_store(context)
    assert bare_store.volume_liquidity_v2 is None

    out = build_volume_liquidity_axis_series(context, bare_store)
    assert out is None


def test_classifier_emits_one_output_per_session_in_context():
    context, _ = _build_context_with_volume()
    store = build_feature_store(
        context, volume_liquidity_v2_config=context.config.volume_liquidity_v2
    )
    out = build_volume_liquidity_axis_series(context, store)

    assert out is not None
    assert set(out.keys()) == set(context.sessions)


def test_classifier_emits_labels_from_v2_label_set():
    context, _ = _build_context_with_volume()
    store = build_feature_store(
        context, volume_liquidity_v2_config=context.config.volume_liquidity_v2
    )
    out = build_volume_liquidity_axis_series(context, store)

    allowed = set(VOLUME_LIQUIDITY_RISK_RANK.keys())
    for day, output in out.items():
        assert output.raw_label in allowed, f"{day}: {output.raw_label!r}"
        assert output.stable_label in allowed
        assert output.active_label in allowed


def test_classifier_evidence_reports_live_liquidity_gap_inputs():
    """Log #40 closure: classifier evidence must expose the percentile inputs
    actually used by the live liquidity_gap_behavior predicate."""
    context, _ = _build_context_with_volume()
    store = build_feature_store(
        context,
        volume_liquidity_v2_config=context.config.volume_liquidity_v2,
        volatility_state_v2_config=context.config.volatility_state_v2,
    )
    out = build_volume_liquidity_axis_series(context, store)
    assert out is not None

    last_day = context.sessions[-1]
    evidence = out[last_day].evidence
    assert set(evidence) == {"rule_evidence"}
    rule_evidence = evidence["rule_evidence"]
    assert set(rule_evidence) == {
        "volume_zscore_20d",
        "return_1d",
        "gap_frequency_percentile_252d",
        "intraday_range_percentile_252d",
    }
    assert isinstance(rule_evidence["gap_frequency_percentile_252d"], float)
    assert isinstance(rule_evidence["intraday_range_percentile_252d"], float)


def test_classifier_emits_normal_volume_after_warmup():
    """After ≥20 sessions, the rules must produce at least one normal_volume
    label across the post-warmup window (most days are normal)."""
    context, _ = _build_context_with_volume()
    store = build_feature_store(
        context, volume_liquidity_v2_config=context.config.volume_liquidity_v2
    )
    out = build_volume_liquidity_axis_series(context, store)

    last_100 = list(context.sessions)[-100:]
    raw_labels = [out[day].raw_label for day in last_100]
    assert "normal_volume" in raw_labels, raw_labels


def test_classifier_emits_panic_volume_when_injected():
    """The injected panic event (3x volume, -3% return) must yield at least
    one panic_volume raw label."""
    context, _ = _build_context_with_volume()
    store = build_feature_store(
        context, volume_liquidity_v2_config=context.config.volume_liquidity_v2
    )
    out = build_volume_liquidity_axis_series(context, store)

    seen = {out[day].raw_label for day in context.sessions}
    assert "panic_volume" in seen, seen


def test_classifier_forces_unknown_when_volume_zscore_is_all_nan():
    """Quality gating: if the volume_zscore_20d series is all-NaN at the as-of
    date, assess_series_input_quality marks the day insufficient and the
    classifier emits unknown."""
    context, _ = _build_context_with_volume()
    store = build_feature_store(
        context, volume_liquidity_v2_config=context.config.volume_liquidity_v2
    )
    vl = store.volume_liquidity_v2
    assert vl is not None
    nan_series = pd.Series(np.nan, index=vl.volume_zscore_20d.index)
    broken = vl.__class__(volume_zscore_20d=nan_series)
    broken_store = store.model_copy(update={"volume_liquidity_v2": broken})

    out = build_volume_liquidity_axis_series(context, broken_store)
    last_100 = list(context.sessions)[-100:]
    for day in last_100:
        assert out[day].raw_label == "unknown"
        assert out[day].stable_label == "unknown"
        assert out[day].active_label == "unknown"


def test_classifier_applies_per_label_hysteresis_so_single_day_panic_flip_does_not_propagate():
    """Per Ambiguity Log #41: panic_volume de-escalation requires 3 days.
    A single one-off raw=normal_volume in a run of panic_volume must NOT
    flip the stable label."""
    from regime_detection.hysteresis import apply_per_label_asymmetric_hysteresis

    cfg = load_default_regime_config().volume_liquidity_state
    raws: list[VolumeLiquidityLabel] = (
        ["panic_volume"] * 10 + ["normal_volume"] + ["panic_volume"] * 10
    )
    stable, _active = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=VOLUME_LIQUIDITY_RISK_RANK,
        deescalation_days_by_label=cfg.deescalation_days_by_label,
        default_deescalation_days=cfg.default_deescalation_days,
    )
    assert stable[10] == "panic_volume"
    assert stable[11] == "panic_volume"


# ---------- Engine end-to-end (AGENTS rule A) -------------------------------


def test_engine_classify_window_emits_real_volume_liquidity_labels():
    """Top-level engine entrypoint: with volume data + v2 config, the
    volume_liquidity_state axis must emit non-None outputs on every session."""
    index = _bdate_index()
    market_data = _synthetic_spy_market_data(index=index)
    timeline = RegimeEngine().classify_window(
        end_date=_LAST_SESSION.date(),
        market_data=market_data,
        lookback_days=200,
    )
    seen = {
        out.volume_liquidity_state.active_label
        for out in timeline.outputs
        if out.volume_liquidity_state is not None
    }
    assert seen, "engine emitted no volume_liquidity_state outputs"
    assert seen <= set(VOLUME_LIQUIDITY_RISK_RANK.keys())
    for out in timeline.outputs:
        assert isinstance(out.volume_liquidity_state, VolumeLiquidityStateOutput)


def test_engine_classify_window_volume_liquidity_state_none_in_pure_v1_mode():
    """V1 byte-identity: when running with the V1 config (no
    volume_liquidity_v2 sub-config), volume_liquidity_state stays None
    on every RegimeOutput."""
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
    assert v1_config.volume_liquidity_v2 is None

    index = _bdate_index()
    market_data = _synthetic_spy_market_data(index=index)
    timeline = RegimeEngine().classify_window(
        end_date=_LAST_SESSION.date(),
        market_data=market_data,
        lookback_days=20,
        config=v1_config,
    )
    for out in timeline.outputs:
        assert out.volume_liquidity_state is None
