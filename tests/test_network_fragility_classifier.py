"""Slice 1.4 — NetworkFragilitySeriesClassifier integration tests.

TDD per AGENTS.md rules A/G + ~/.claude/CLAUDE.md testing rules:

- No toy names. All symbols come from project enums/constants
  (NETWORK_FRAGILITY_UNIVERSE, BreadthLabel, VolatilityLabel,
  NetworkFragilityLabel).
- Real spec values (v2 §3.5/§3.6/§3.7 thresholds, percentiles, days).
- Integration path: feature_store.network_fragility → rule inputs → rule
  engine → per-label asymmetric hysteresis → NetworkFragilityOutput.
- One end-to-end engine test (AGENTS rule A) via RegimeEngine.classify_window.

The classifier signatures referenced here:
    NetworkFragilitySeriesClassifier.build(context, feature_store)
        -> dict[date, NetworkFragilityOutput] | None
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from regime_detection.axis_series import NetworkFragilitySeriesClassifier
from regime_detection.calendar import nyse_sessions_between
from regime_detection.config import (
    NetworkFragilityConfig,
    NetworkFragilityRulesConfig,
    load_default_regime_config,
)
from regime_detection.engine import RegimeEngine
from regime_detection.feature_store import build_feature_store
from regime_detection.fragility_universe import (
    CROSS_ASSET_SYMBOLS,
    INDEX_SYMBOL,
    NETWORK_FRAGILITY_UNIVERSE,
    SECTOR_ETFS,
)
from regime_detection.market_context import build_market_context
from regime_detection.models import NetworkFragilityOutput
from regime_detection.network_fragility_rules import (
    NETWORK_FRAGILITY_RISK_RANK,
    NetworkFragilityLabel,
)


# ---------- Synthetic full-universe fixtures ---------------------------------

# v2 §3.2 requires 504d percentile history; +21d for trailing windows.
# The engine slices context to ENGINE_MINIMUM_HISTORY (320) + lookback − 1
# sessions, and our integration test runs lookback_days=200. We need the
# kept-window to contain enough post-warmup sessions for the rule engine
# to fire (504d cold start + 200d lookback >> 519). Use 1100 sessions.
_TRAINING_SESSIONS = 1100
_LAST_SESSION = pd.Timestamp("2025-04-30")
_SEED = 20260513


def _bdate_index(periods: int = _TRAINING_SESSIONS, end: pd.Timestamp = _LAST_SESSION) -> pd.DatetimeIndex:
    # NYSE sessions (not business days) — required by V1's market_data contract.
    sessions = nyse_sessions_between(
        (end - pd.Timedelta(days=periods * 2)).date(),
        end.date(),
    )
    return pd.DatetimeIndex([pd.Timestamp(d) for d in sessions[-periods:]])


def _synthetic_universe_prices(*, index: pd.DatetimeIndex, seed: int = _SEED) -> pd.DataFrame:
    """Random-walk closes for every symbol in NETWORK_FRAGILITY_UNIVERSE."""
    rng = np.random.default_rng(seed=seed)
    returns = rng.normal(0.0, 0.01, size=(len(index), len(NETWORK_FRAGILITY_UNIVERSE)))
    prices = (1.0 + returns).cumprod(axis=0) * 100.0
    return pd.DataFrame(
        prices,
        index=index,
        columns=list(NETWORK_FRAGILITY_UNIVERSE),
    )


def _market_data_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Long-form market_data for V1 ingestion. SPY + RSP + VIXY rows synthesized
    from SPY (RSP/VIXY shape only matters for the V1 ingestion contract — they
    do not drive the network_fragility test). RSP rides SPY (the breadth axis
    uses RSP/SPY ratio; for these tests we just need a valid context)."""
    rows: list[dict[str, object]] = []
    spy = prices[INDEX_SYMBOL]
    for ts, close in spy.items():
        rows.append({
            "date": ts.date(),
            "symbol": "SPY",
            "open": float(close),
            "high": float(close) * 1.005,
            "low": float(close) * 0.995,
            "close": float(close),
            "volume": 1_000_000,
        })
        rows.append({
            "date": ts.date(),
            "symbol": "RSP",
            "open": float(close) * 0.5,
            "high": float(close) * 0.5 * 1.005,
            "low": float(close) * 0.5 * 0.995,
            "close": float(close) * 0.5,
            "volume": 500_000,
        })
        rows.append({
            "date": ts.date(),
            "symbol": "VIXY",
            "open": 20.0,
            "high": 20.5,
            "low": 19.5,
            "close": 20.0,
            "volume": 100_000,
        })
    return pd.DataFrame(rows)


def _build_context_with_full_universe(*, end_session: pd.Timestamp = _LAST_SESSION):
    """Build a MarketContext with SPY+RSP+VIXY in market_data AND sector/cross-asset
    closes for every symbol in NETWORK_FRAGILITY_UNIVERSE (so feature_store
    materializes a non-NaN network_fragility seam)."""
    index = _bdate_index(end=end_session)
    prices = _synthetic_universe_prices(index=index)
    market_data = _market_data_from_prices(prices)

    sector_etf_closes: dict[str, pd.Series] = {s: prices[s] for s in SECTOR_ETFS}
    cross_asset_closes: dict[str, pd.Series] = {s: prices[s] for s in CROSS_ASSET_SYMBOLS}

    config = RegimeEngine().config
    context = build_market_context(
        end_date=end_session.date(),
        market_data=market_data,
        config=config,
        sector_etf_closes=sector_etf_closes,
        cross_asset_closes=cross_asset_closes,
    )
    return context, prices


# ---------- Unit tests on NETWORK_FRAGILITY_RISK_RANK (v2 §3.6) --------------


def test_network_fragility_risk_rank_matches_v2_spec_3_6():
    # v2 spec §3.6 lines 661–668 verbatim.
    assert NETWORK_FRAGILITY_RISK_RANK == {
        "diversified_normal": 0,
        "stock_picker_dispersion": 1,
        "rising_fragility": 2,
        "correlation_concentration": 2,
        "correlation_to_one": 3,
        "systemic_stress": 3,
        "unknown": 2,
    }


# ---------- Config: deescalation_days_by_label is wired ----------------------


def test_default_yaml_loads_deescalation_days_by_label_per_v2_spec_3_7():
    cfg = load_default_regime_config()
    assert cfg.network_fragility is not None
    deesc = cfg.network_fragility.deescalation_days_by_label
    # v2 spec §3.7 lines 675–679 verbatim PLUS Implementation Ambiguity Log
    # entry #8: `unknown` is treated as a high-risk hold (5d) so a single-day
    # quality flicker does NOT fast-track de-escalation through unknown.
    assert deesc == {
        "rising_fragility": 3,
        "correlation_concentration": 3,
        "correlation_to_one": 5,
        "systemic_stress": 5,
        "unknown": 5,
    }
    # Default for labels NOT listed (diversified_normal, stock_picker_dispersion)
    # is exposed explicitly per Implementation Ambiguity Log entry #6.
    assert cfg.network_fragility.default_deescalation_days == 0


def test_network_fragility_config_rejects_unknown_keys():
    from pydantic import ValidationError

    cfg = load_default_regime_config()
    base = cfg.network_fragility.model_dump()
    base["unexpected_key"] = "x"
    with pytest.raises(ValidationError):
        NetworkFragilityConfig.model_validate(base)


def test_rules_config_bounds_are_enforced():
    from pydantic import ValidationError

    cfg = load_default_regime_config()
    base = cfg.network_fragility.rules.model_dump()
    base["diversified_normal_percentile_lo"] = 1.5  # out of [0, 1]
    with pytest.raises(ValidationError):
        NetworkFragilityRulesConfig.model_validate(base)


# ---------- Integration: classifier.build() end-to-end -----------------------


def test_classifier_returns_none_when_feature_store_network_fragility_is_none():
    """Without sector ETF data the feature store seam is None — classifier
    must propagate None (no v2 axis output)."""
    context, _ = _build_context_with_full_universe()
    # Rebuild a feature store WITHOUT the network_fragility seam by stripping
    # sector data from the context.
    bare_context = build_market_context(
        end_date=context.end_date,
        market_data=_market_data_from_prices(
            _synthetic_universe_prices(index=_bdate_index())
        ),
        config=context.config,
    )
    store = build_feature_store(bare_context)
    assert store.network_fragility is None
    out = NetworkFragilitySeriesClassifier().build(bare_context, store)
    assert out is None


def test_classifier_emits_one_output_per_session_in_context():
    context, _ = _build_context_with_full_universe()
    store = build_feature_store(
        context, network_fragility_config=context.config.network_fragility
    )
    out = NetworkFragilitySeriesClassifier().build(context, store)

    assert out is not None
    assert set(out.keys()) == set(context.sessions)


def test_classifier_emits_labels_from_v2_label_set():
    context, _ = _build_context_with_full_universe()
    store = build_feature_store(
        context, network_fragility_config=context.config.network_fragility
    )
    out = NetworkFragilitySeriesClassifier().build(context, store)

    allowed: set[str] = set(NETWORK_FRAGILITY_RISK_RANK.keys())
    for day, output in out.items():
        assert output.raw_label in allowed, f"{day}: {output.raw_label!r}"
        assert output.stable_label in allowed
        assert output.active_label in allowed


def test_classifier_emits_non_unknown_labels_after_warmup():
    """After 504+21 sessions the rules must produce at least one
    non-unknown label across the post-warmup window."""
    context, _ = _build_context_with_full_universe()
    store = build_feature_store(
        context, network_fragility_config=context.config.network_fragility
    )
    out = NetworkFragilitySeriesClassifier().build(context, store)

    # Look only at the last 100 sessions (well past warmup).
    last_100 = list(context.sessions)[-100:]
    seen_non_unknown = {
        out[day].active_label for day in last_100 if out[day].active_label != "unknown"
    }
    assert seen_non_unknown, "classifier produced only unknowns on a 100-session window"


def test_classifier_forces_unknown_when_feature_column_is_all_nan():
    """Quality gating: if a required feature series is all NaN at the as-of
    date, assess_series_input_quality must mark the day insufficient and the
    classifier emits unknown."""
    context, _ = _build_context_with_full_universe()
    store = build_feature_store(
        context, network_fragility_config=context.config.network_fragility
    )
    # Mutate the feature store: blow away avg_pairwise_corr_63d entirely.
    nf = store.network_fragility
    assert nf is not None
    nan_series = pd.Series(np.nan, index=nf.avg_pairwise_corr_63d.index)
    broken = nf.__class__(
        avg_pairwise_corr_63d=nan_series,
        avg_pairwise_corr_percentile_504d=nan_series,
        largest_eigenvalue_share=nf.largest_eigenvalue_share,
        largest_eigenvalue_share_percentile_504d=nf.largest_eigenvalue_share_percentile_504d,
        effective_rank=nf.effective_rank,
        effective_rank_percentile_504d=nf.effective_rank_percentile_504d,
        absorption_ratio_top3=nf.absorption_ratio_top3,
        dispersion_ratio=nf.dispersion_ratio,
        dispersion_ratio_percentile_252d=nf.dispersion_ratio_percentile_252d,
    )
    broken_store = store.model_copy(update={"network_fragility": broken})
    out = NetworkFragilitySeriesClassifier().build(context, broken_store)

    # Every session must be forced to unknown by the data-quality gate.
    last_100 = list(context.sessions)[-100:]
    for day in last_100:
        assert out[day].raw_label == "unknown"
        assert out[day].stable_label == "unknown"
        assert out[day].active_label == "unknown"


def test_classifier_applies_per_label_hysteresis_so_single_day_flip_does_not_propagate():
    """Per v2 §3.7: rising_fragility de-escalation requires 3 consecutive
    days. A single one-off raw=diversified_normal in a run of
    rising_fragility must NOT flip the stable label."""
    from regime_detection.hysteresis import apply_per_label_asymmetric_hysteresis

    # Direct hysteresis check at the same call site the classifier uses.
    deesc = load_default_regime_config().network_fragility.deescalation_days_by_label
    raws: list[NetworkFragilityLabel] = (
        ["rising_fragility"] * 10
        + ["diversified_normal"]
        + ["rising_fragility"] * 10
    )
    stable, _active = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=NETWORK_FRAGILITY_RISK_RANK,
        deescalation_days_by_label=deesc,
    )
    # The one-day diversified_normal flip must NOT propagate (threshold = 3).
    assert stable[10] == "rising_fragility"
    assert stable[11] == "rising_fragility"


# ---------- Engine end-to-end (AGENTS rule A) --------------------------------


def test_engine_classify_window_emits_real_network_fragility_labels():
    """Top-level engine entrypoint: with sector + cross-asset data, the
    network_fragility axis MUST emit non-unknown labels for at least some of
    the days in the window (proves end-to-end wiring is live, not stubbed)."""
    context, _prices = _build_context_with_full_universe()
    market_data = _market_data_from_prices(
        _synthetic_universe_prices(index=_bdate_index())
    )
    sector_closes = {s: context.sector_etf_closes[s] for s in SECTOR_ETFS}
    cross_asset_closes = {
        s: context.cross_asset_closes[s] for s in CROSS_ASSET_SYMBOLS
    }

    # ENGINE_MINIMUM_HISTORY (320) + lookback_days − 1 = working sessions kept
    # by slice_context_to_recent_sessions. We need that working window to
    # exceed the 504d v2 percentile cold-start with margin so the rules fire
    # on multiple post-warmup days.
    timeline = RegimeEngine().classify_window(
        end_date=_LAST_SESSION.date(),
        market_data=market_data,
        lookback_days=600,
        sector_etf_closes=sector_closes,
        cross_asset_closes=cross_asset_closes,
    )

    labels = {out.network_fragility.active_label for out in timeline.outputs}
    assert labels - {"unknown"}, (
        f"engine emitted only unknown network_fragility labels: {labels}"
    )
    for out in timeline.outputs:
        assert isinstance(out.network_fragility, NetworkFragilityOutput)
        # mode must remain pinned to the v2 §3.1 closed-universe identifier.
        assert out.network_fragility.mode == "sector_cross_asset_22"


def test_engine_classify_window_forces_unknown_when_universe_data_missing():
    """Quality gating end-to-end: when sector ETF data is absent, the v2
    fallback in timeline._resolve_network_fragility_by_date emits 'unknown'
    placeholders (preserves pure-v1 mode)."""
    market_data = _market_data_from_prices(
        _synthetic_universe_prices(index=_bdate_index())
    )
    timeline = RegimeEngine().classify_window(
        end_date=_LAST_SESSION.date(),
        market_data=market_data,
        lookback_days=20,
    )
    for out in timeline.outputs:
        assert out.network_fragility.active_label == "unknown"
        assert out.network_fragility.raw_label == "unknown"


# ---------- Slice-1 cleanup: I1 + I2 regression tests ------------------------


def test_classifier_raises_on_v1_axis_calendar_drift_breadth():
    """I1: when caller supplies a v1 breadth_active_labels_by_date dict that
    is missing a session in ``context.sessions``, the classifier MUST raise
    rather than silently substituting 'unknown' (which would defang
    systemic_stress on any drifted session)."""
    context, _ = _build_context_with_full_universe()
    store = build_feature_store(
        context, network_fragility_config=context.config.network_fragility
    )
    # Build a complete breadth dict, then drop one session.
    breadth = {day: "healthy_breadth" for day in context.sessions}
    volatility = {day: "normal_vol" for day in context.sessions}
    dropped = list(context.sessions)[-50]
    del breadth[dropped]

    with pytest.raises(KeyError, match="breadth_active_labels_by_date missing session"):
        NetworkFragilitySeriesClassifier().build(
            context,
            store,
            breadth_active_labels_by_date=breadth,
            volatility_active_labels_by_date=volatility,
        )


def test_classifier_raises_on_v1_axis_calendar_drift_volatility():
    """I1: same contract for volatility."""
    context, _ = _build_context_with_full_universe()
    store = build_feature_store(
        context, network_fragility_config=context.config.network_fragility
    )
    breadth = {day: "healthy_breadth" for day in context.sessions}
    volatility = {day: "normal_vol" for day in context.sessions}
    dropped = list(context.sessions)[-50]
    del volatility[dropped]

    with pytest.raises(KeyError, match="volatility_active_labels_by_date missing session"):
        NetworkFragilitySeriesClassifier().build(
            context,
            store,
            breadth_active_labels_by_date=breadth,
            volatility_active_labels_by_date=volatility,
        )


def test_unknown_flicker_does_not_fast_track_deescalation_through_correlation_to_one():
    """I2(b): a single-day `unknown` flicker in the middle of a stable
    `correlation_to_one` run must NOT cause fast de-escalation. With
    `unknown: 5` in deescalation_days_by_label, the stable label holds at
    `correlation_to_one` on the day after the flicker."""
    from regime_detection.hysteresis import apply_per_label_asymmetric_hysteresis

    cfg = load_default_regime_config().network_fragility
    raws: list[NetworkFragilityLabel] = (
        ["correlation_to_one"] * 10
        + ["unknown"]  # one-day quality flicker
        + ["correlation_to_one"] * 5
    )
    stable, _active = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=NETWORK_FRAGILITY_RISK_RANK,
        deescalation_days_by_label=cfg.deescalation_days_by_label,
        default_deescalation_days=cfg.default_deescalation_days,
    )
    # The single-day unknown must not flip stable away from correlation_to_one.
    assert stable[10] == "correlation_to_one"
    assert stable[11] == "correlation_to_one"
