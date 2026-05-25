"""Slice 1.4 — build_network_fragility_axis_series integration tests.

TDD per AGENTS.md rules A/G + ~/.claude/CLAUDE.md testing rules:

- No toy names. All symbols come from project enums/constants
  (NETWORK_FRAGILITY_UNIVERSE, BreadthLabel, VolatilityLabel,
  NetworkFragilityLabel).
- Real spec values (v2 §3.5/§3.6/§3.7 thresholds, percentiles, days).
- Integration path: feature_store.network_fragility → rule inputs → rule
  engine → per-label asymmetric hysteresis → NetworkFragilityOutput.
- One end-to-end engine test (AGENTS rule A) via RegimeEngine.classify_window.

The builder signature referenced here:
    build_network_fragility_axis_series(context, feature_store)
        -> dict[date, NetworkFragilityOutput] | None
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from regime_detection.axis_series import (
    build_network_fragility_axis_series,
    build_axis_series_bundle,
)
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
from regime_detection.network_fragility import NetworkFragilityFeatures
from regime_detection.volatility_state import VolatilityFeatures
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


def _bdate_index(
    periods: int = _TRAINING_SESSIONS, end: pd.Timestamp = _LAST_SESSION
) -> pd.DatetimeIndex:
    # NYSE sessions (not business days) — required by V1's market_data contract.
    sessions = nyse_sessions_between(
        (end - pd.Timedelta(days=periods * 2)).date(),
        end.date(),
    )
    return pd.DatetimeIndex([pd.Timestamp(d) for d in sessions[-periods:]])


def _synthetic_universe_prices(
    *, index: pd.DatetimeIndex, seed: int = _SEED
) -> pd.DataFrame:
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
    """Long-form market_data for V1 ingestion. SPY + RSP + VIX rows synthesized
    from SPY (RSP/VIX shape only matters for the V1 ingestion contract — they
    do not drive the network_fragility test). RSP rides SPY (the breadth axis
    uses RSP/SPY ratio; for these tests we just need a valid context)."""
    rows: list[dict[str, object]] = []
    spy = prices[INDEX_SYMBOL]
    for ts, close in spy.items():
        rows.append(
            {
                "date": ts.date(),
                "symbol": "SPY",
                "open": float(close),
                "high": float(close) * 1.005,
                "low": float(close) * 0.995,
                "close": float(close),
                "volume": 1_000_000,
            }
        )
        rows.append(
            {
                "date": ts.date(),
                "symbol": "RSP",
                "open": float(close) * 0.5,
                "high": float(close) * 0.5 * 1.005,
                "low": float(close) * 0.5 * 0.995,
                "close": float(close) * 0.5,
                "volume": 500_000,
            }
        )
        rows.append(
            {
                "date": ts.date(),
                "symbol": "VIX",
                "open": 20.0,
                "high": 20.5,
                "low": 19.5,
                "close": 20.0,
                "volume": 100_000,
            }
        )
    market_data = pd.DataFrame(rows)
    spy_mask = market_data["symbol"] == "SPY"
    market_data.loc[spy_mask, "volume"] = range(
        1_000_000, 1_000_000 + int(spy_mask.sum())
    )
    return market_data


def _network_fixture_config():
    cfg = RegimeEngine().config
    assert cfg.hmm is not None
    assert cfg.clustering is not None
    assert cfg.change_point is not None
    assert cfg.network_fragility is not None
    return cfg.model_copy(
        update={
            "network_fragility": cfg.network_fragility.model_copy(
                update={
                    "percentile_lookback_days": 100,
                    "dispersion_percentile_lookback_days": 100,
                }
            ),
            "hmm": cfg.hmm.model_copy(
                update={
                    "n_states": 2,
                    "training_window_days": 100,
                    "random_seeds": (42, 7, 13),
                }
            ),
            "clustering": cfg.clustering.model_copy(
                update={"training_window_days": 100}
            ),
            "change_point": cfg.change_point.model_copy(
                update={"training_window_days": 100}
            ),
        }
    )


def _macro_series_for_index(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    trend = pd.Series(range(len(index)), index=index, dtype="float64")
    return {
        "2y_yield": (4.00 + trend * 0.0002).rename("2y_yield"),
        "10y_yield": (4.25 + trend * 0.0001).rename("10y_yield"),
        "broad_usd_index": (100.0 + trend * 0.001).rename("broad_usd_index"),
    }


def _build_context_with_full_universe(*, end_session: pd.Timestamp = _LAST_SESSION):
    """Build a MarketContext with SPY+RSP+VIX in market_data AND sector/cross-asset
    closes for every symbol in NETWORK_FRAGILITY_UNIVERSE (so feature_store
    materializes a non-NaN network_fragility seam)."""
    index = _bdate_index(end=end_session)
    prices = _synthetic_universe_prices(index=index)
    market_data = _market_data_from_prices(prices)

    sector_etf_closes: dict[str, pd.Series] = {s: prices[s] for s in SECTOR_ETFS}
    cross_asset_closes: dict[str, pd.Series] = {
        s: prices[s] for s in CROSS_ASSET_SYMBOLS
    }

    config = _network_fixture_config()
    context = build_market_context(
        end_date=end_session.date(),
        market_data=market_data,
        config=config,
        macro_series=_macro_series_for_index(index),
        sector_etf_closes=sector_etf_closes,
        cross_asset_closes=cross_asset_closes,
    )
    return context, prices


def _build_real_v2_network_context(
    *,
    as_of: date,
    market_df_for_asof,
    v2_close_series_by_symbol: dict[str, pd.Series],
):
    missing = sorted(
        set(NETWORK_FRAGILITY_UNIVERSE).difference(v2_close_series_by_symbol)
    )
    if missing:
        raise AssertionError(f"real V2 OHLCV fixture missing symbols: {missing}")

    config = RegimeEngine().config
    return build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=config,
        sector_etf_closes={s: v2_close_series_by_symbol[s] for s in SECTOR_ETFS},
        cross_asset_closes={
            s: v2_close_series_by_symbol[s] for s in CROSS_ASSET_SYMBOLS
        },
    )


# ---------- Unit tests on NETWORK_FRAGILITY_RISK_RANK (v2 §3.6) --------------


def test_network_fragility_risk_rank_matches_v2_spec_3_6():
    # v2 spec §3.6 lines 661–668 verbatim.
    assert NETWORK_FRAGILITY_RISK_RANK == {
        "diversified_normal": 0,
        "stock_picker_dispersion": 1,
        "rising_fragility": 2,
        "correlation_concentration": 2,
        "correlation_to_one": 3,
        "systemic_stress_unconfirmed": 3,
        "systemic_stress": 3,
        "unknown": 2,
    }


# ---------- Config: deescalation_days_by_label is wired ----------------------


def test_default_yaml_loads_deescalation_days_by_label_per_v2_spec_3_7():
    cfg = load_default_regime_config()
    assert cfg.network_fragility is not None
    deesc = cfg.network_fragility.deescalation_days_by_label
    # `unknown` is absence of signal, not a sticky regime. It must not delay
    # recovery into a valid classified label; high-risk labels still use their
    # own hold periods when they are the stable label being left.
    assert deesc == {
        "rising_fragility": 3,
        "correlation_concentration": 3,
        "correlation_to_one": 5,
        "systemic_stress_unconfirmed": 5,
        "systemic_stress": 5,
        "unknown": 0,
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


# ---------- Integration: axis builder end-to-end -----------------------------


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
    out = build_network_fragility_axis_series(bare_context, store)
    assert out is None


def test_classifier_emits_one_output_per_session_in_context():
    context, _ = _build_context_with_full_universe()
    store = build_feature_store(
        context, network_fragility_config=context.config.network_fragility
    )
    out = build_network_fragility_axis_series(context, store)

    assert out is not None
    assert set(out.keys()) == set(context.sessions)


def test_classifier_emits_labels_from_v2_label_set():
    context, _ = _build_context_with_full_universe()
    store = build_feature_store(
        context, network_fragility_config=context.config.network_fragility
    )
    out = build_network_fragility_axis_series(context, store)

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
    out = build_network_fragility_axis_series(context, store)

    # Look only at the last 100 sessions (well past warmup).
    last_100 = list(context.sessions)[-100:]
    seen_non_unknown = {
        out[day].active_label for day in last_100 if out[day].active_label != "unknown"
    }
    assert seen_non_unknown, "classifier produced only unknowns on a 100-session window"


def test_real_v2_ohlcv_fixture_network_fragility_golden_labels(
    v2_market_df_for_asof,
    v2_close_series_by_symbol,
):
    """Real-market fixture acceptance: prove the V2 network-fragility
    classifier emits deterministic golden labels from tracked OHLCV, not only
    from synthetic random walks."""
    context = _build_real_v2_network_context(
        as_of=date(2026, 5, 13),
        market_df_for_asof=v2_market_df_for_asof,
        v2_close_series_by_symbol=v2_close_series_by_symbol,
    )
    store = build_feature_store(
        context, network_fragility_config=context.config.network_fragility
    )
    out = build_network_fragility_axis_series(context, store)

    assert out is not None
    golden_rows = [
        (
            date(2022, 6, 16),
            "correlation_to_one",
            "avg_pairwise_corr_percentile_504d",
            0.998015873015873,
        ),
        (
            date(2024, 1, 3),
            "diversified_normal",
            "dispersion_ratio_percentile_252d",
            0.9246031746031746,
        ),
        (
            date(2026, 5, 13),
            "correlation_concentration",
            "largest_eigenvalue_share_percentile_504d",
            0.8630952380952381,
        ),
    ]
    for day, expected_label, evidence_key, expected_value in golden_rows:
        output = out[day]
        assert output.raw_label == expected_label
        assert output.stable_label == expected_label
        assert output.active_label == expected_label
        assert output.data_quality.status == "ok"
        assert output.evidence["rule_evidence"][evidence_key] == pytest.approx(
            expected_value
        )


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
    out = build_network_fragility_axis_series(context, broken_store)

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
        ["rising_fragility"] * 10 + ["diversified_normal"] + ["rising_fragility"] * 10
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


def test_engine_classify_window_emits_network_fragility_labels_on_full_universe():
    """Top-level engine entrypoint: with sector + cross-asset data, the
    network_fragility axis MUST emit non-unknown labels for at least some of
    the days in the window (proves end-to-end wiring is live, not stubbed)."""
    context, _prices = _build_context_with_full_universe()
    market_data = _market_data_from_prices(
        _synthetic_universe_prices(index=_bdate_index())
    )
    sector_closes = {s: context.sector_etf_closes[s] for s in SECTOR_ETFS}
    cross_asset_closes = {s: context.cross_asset_closes[s] for s in CROSS_ASSET_SYMBOLS}
    pit_intervals = pd.DataFrame(
        {
            "ticker": list(SECTOR_ETFS),
            "start_date": [_bdate_index()[0].date()] * len(SECTOR_ETFS),
            "end_date": [None] * len(SECTOR_ETFS),
        }
    )
    constituent_ohlcv = {
        symbol: pd.DataFrame(
            {
                "open": series,
                "high": series,
                "low": series,
                "close": series,
                "volume": pd.Series(1_000_000, index=series.index),
                "adjusted_close": series,
            }
        )
        for symbol, series in sector_closes.items()
    }

    # ENGINE_MINIMUM_HISTORY (320) + lookback_days − 1 = working sessions kept
    # by slice_context_to_recent_sessions. We need that working window to
    # exceed the 504d v2 percentile cold-start with margin so the rules fire
    # on multiple post-warmup days.
    timeline = RegimeEngine().classify_window(
        end_date=_LAST_SESSION.date(),
        market_data=market_data,
        lookback_days=600,
        config=context.config,
        event_calendar=pd.DataFrame(columns=["date", "market", "type", "importance"]),
        sector_etf_closes=sector_closes,
        cross_asset_closes=cross_asset_closes,
        macro_series=context.macro_series,
        pit_constituent_intervals=pit_intervals,
        constituent_ohlcv=constituent_ohlcv,
    )

    labels = {out.network_fragility.active_label for out in timeline.outputs}
    assert labels - {
        "unknown"
    }, f"engine emitted only unknown network_fragility labels: {labels}"
    for out in timeline.outputs:
        assert isinstance(out.network_fragility, NetworkFragilityOutput)
        # mode must remain pinned to the v2 §3.1 closed-universe identifier.
        assert out.network_fragility.mode == "sector_cross_asset_24"


def test_engine_classify_window_emits_real_fixture_network_fragility_label(
    real_v2_classify_window_2026_05_13,
):
    """Top-level engine entrypoint over tracked real OHLCV: protects the
    RegimeEngine → MarketContext → FeatureStore → AxisSeriesBundle seam.

    Uses the cross-worker cached classify_window (see conftest).
    """
    as_of = date(2026, 5, 13)
    timeline = real_v2_classify_window_2026_05_13

    by_date = {out.as_of_date: out for out in timeline.outputs}
    network_fragility = by_date[as_of].network_fragility
    assert network_fragility.raw_label == "correlation_concentration"
    assert network_fragility.stable_label == "correlation_concentration"
    assert network_fragility.active_label == "correlation_concentration"
    assert network_fragility.evidence["rule_evidence"][
        "largest_eigenvalue_share_percentile_504d"
    ] == pytest.approx(0.8630952380952381)


def test_engine_classify_window_forces_unknown_when_universe_data_missing():
    """Default V2 timeline fails loudly when required universe data is absent."""
    market_data = _market_data_from_prices(
        _synthetic_universe_prices(index=_bdate_index())
    )
    with pytest.raises(RuntimeError, match="transition_risk requires score inputs"):
        RegimeEngine().classify_window(
            end_date=_LAST_SESSION.date(),
            market_data=market_data,
            lookback_days=20,
            config=_network_fixture_config(),
            macro_series=_macro_series_for_index(_bdate_index()),
            event_calendar=pd.DataFrame(
                columns=["date", "market", "type", "importance"]
            ),
        )


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
        build_network_fragility_axis_series(
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

    with pytest.raises(
        KeyError, match="volatility_active_labels_by_date missing session"
    ):
        build_network_fragility_axis_series(
            context,
            store,
            breadth_active_labels_by_date=breadth,
            volatility_active_labels_by_date=volatility,
        )


def test_classifier_emits_systemic_stress_when_credit_funding_confirms_it():
    """`systemic_stress` is a cross-axis label: the fragility classifier must
    consume credit_funding.active_label when the §2C axis is already built.
    """
    context, _ = _build_context_with_full_universe()
    store = build_feature_store(
        context, network_fragility_config=context.config.network_fragility
    )
    nf = store.network_fragility
    assert nf is not None

    idx = nf.avg_pairwise_corr_63d.index
    corr_pct = pd.Series(0.95, index=idx)
    eig_pct = pd.Series(0.50, index=idx)
    eff_rank_pct = pd.Series(0.50, index=idx)
    disp_pct = pd.Series(0.50, index=idx)
    corr_level = pd.Series(0.60, index=idx)
    eig_level = pd.Series(0.60, index=idx)
    eff_rank = pd.Series(0.40, index=idx)
    absorb = pd.Series(0.70, index=idx)
    dispersion = pd.Series(0.30, index=idx)
    stressed_nf = NetworkFragilityFeatures(
        avg_pairwise_corr_63d=corr_level,
        avg_pairwise_corr_percentile_504d=corr_pct,
        largest_eigenvalue_share=eig_level,
        largest_eigenvalue_share_percentile_504d=eig_pct,
        effective_rank=eff_rank,
        effective_rank_percentile_504d=eff_rank_pct,
        absorption_ratio_top3=absorb,
        dispersion_ratio=dispersion,
        dispersion_ratio_percentile_252d=disp_pct,
    )

    vol = store.volatility
    spy_ohlcv = context.spy_ohlcv.copy()
    spy_ohlcv["close"] = np.linspace(100.0, 90.0, len(spy_ohlcv))
    context = context.model_copy(update={"spy_ohlcv": spy_ohlcv})
    stressed_volatility = VolatilityFeatures(
        close=vol.close,
        return_1d=vol.return_1d,
        return_5d=vol.return_5d,
        return_21d=pd.Series(-0.05, index=idx),
        realized_vol_21d=pd.Series(0.30, index=idx),
        realized_vol_percentile_252d=pd.Series(0.85, index=idx),
        vix_percentile_252d=pd.Series(0.85, index=idx),
    )
    stressed_store = store.model_copy(
        update={
            "network_fragility": stressed_nf,
            "volatility": stressed_volatility,
        }
    )

    breadth = {day: "weak_breadth" for day in context.sessions}
    volatility = {day: "normal_vol" for day in context.sessions}
    credit_funding = {day: "credit_stress" for day in context.sessions}

    out = build_network_fragility_axis_series(
        context,
        stressed_store,
        breadth_active_labels_by_date=breadth,
        volatility_active_labels_by_date=volatility,
        credit_funding_active_labels_by_date=credit_funding,
    )

    assert out is not None
    assert out[context.sessions[-1]].raw_label == "systemic_stress"
    assert out[context.sessions[-1]].active_label == "systemic_stress"


def test_classifier_emits_unconfirmed_systemic_stress_when_credit_funding_unavailable():
    """Missing §2C credit/funding must not silently downgrade otherwise
    systemic market stress to correlation_to_one."""
    context, _ = _build_context_with_full_universe()
    store = build_feature_store(
        context, network_fragility_config=context.config.network_fragility
    )
    nf = store.network_fragility
    assert nf is not None

    idx = nf.avg_pairwise_corr_63d.index
    stressed_nf = NetworkFragilityFeatures(
        avg_pairwise_corr_63d=pd.Series(0.60, index=idx),
        avg_pairwise_corr_percentile_504d=pd.Series(0.95, index=idx),
        largest_eigenvalue_share=pd.Series(0.60, index=idx),
        largest_eigenvalue_share_percentile_504d=pd.Series(0.50, index=idx),
        effective_rank=pd.Series(0.40, index=idx),
        effective_rank_percentile_504d=pd.Series(0.50, index=idx),
        absorption_ratio_top3=pd.Series(0.70, index=idx),
        dispersion_ratio=pd.Series(0.30, index=idx),
        dispersion_ratio_percentile_252d=pd.Series(0.50, index=idx),
    )

    vol = store.volatility
    spy_ohlcv = context.spy_ohlcv.copy()
    spy_ohlcv["close"] = np.linspace(100.0, 90.0, len(spy_ohlcv))
    context = context.model_copy(update={"spy_ohlcv": spy_ohlcv})
    stressed_volatility = VolatilityFeatures(
        close=vol.close,
        return_1d=vol.return_1d,
        return_5d=vol.return_5d,
        return_21d=pd.Series(-0.05, index=idx),
        realized_vol_21d=pd.Series(0.30, index=idx),
        realized_vol_percentile_252d=pd.Series(0.85, index=idx),
        vix_percentile_252d=pd.Series(0.85, index=idx),
    )
    stressed_store = store.model_copy(
        update={
            "network_fragility": stressed_nf,
            "volatility": stressed_volatility,
        }
    )

    breadth = {day: "weak_breadth" for day in context.sessions}
    volatility = {day: "normal_vol" for day in context.sessions}

    out = build_network_fragility_axis_series(
        context,
        stressed_store,
        breadth_active_labels_by_date=breadth,
        volatility_active_labels_by_date=volatility,
        credit_funding_active_labels_by_date=None,
    )

    assert out is not None
    final = out[context.sessions[-1]]
    assert final.raw_label == "systemic_stress"
    assert final.active_label == "systemic_stress"
    assert (
        final.evidence["rule_evidence"]["rule_reason"] == "credit_funding_unavailable"
    )


def test_axis_bundle_threads_credit_funding_into_network_fragility_systemic_stress():
    """The live bundle path must pass the authoritative §2C active label into
    network fragility so `systemic_stress` can outrank `correlation_to_one`.
    """
    from test_credit_funding_axis_engine import _build_full_synthetic_context

    context = _build_full_synthetic_context()
    idx = pd.DatetimeIndex(context.spy_ohlcv.index)
    spy_factor = pd.Series(
        np.r_[np.ones(len(idx) - 21), np.linspace(1.0, 0.90, 21)],
        index=idx,
    )
    rsp_factor = pd.Series(
        np.r_[np.ones(len(idx) - 21), np.linspace(1.0, 0.50, 21)],
        index=idx,
    )
    stressed_spy_ohlcv = context.spy_ohlcv.copy()
    for col in ["open", "high", "low", "close"]:
        stressed_spy_ohlcv[col] = stressed_spy_ohlcv[col] * spy_factor
    stressed_context = context.model_copy(
        update={
            "spy_ohlcv": stressed_spy_ohlcv,
            "rsp_close": context.spy_ohlcv["close"] * rsp_factor,
        }
    )
    store = build_feature_store(
        stressed_context,
        network_fragility_config=stressed_context.config.network_fragility,
        credit_funding_config=stressed_context.config.credit_funding,
    )
    assert store.network_fragility is not None
    assert store.credit_funding is not None

    idx = store.network_fragility.avg_pairwise_corr_63d.index
    stressed_nf = replace(
        store.network_fragility,
        avg_pairwise_corr_63d=pd.Series(0.60, index=idx),
        avg_pairwise_corr_percentile_504d=pd.Series(0.95, index=idx),
        largest_eigenvalue_share=pd.Series(0.60, index=idx),
        largest_eigenvalue_share_percentile_504d=pd.Series(0.50, index=idx),
        effective_rank=pd.Series(0.40, index=idx),
        effective_rank_percentile_504d=pd.Series(0.50, index=idx),
        absorption_ratio_top3=pd.Series(0.70, index=idx),
        dispersion_ratio=pd.Series(0.30, index=idx),
        dispersion_ratio_percentile_252d=pd.Series(0.50, index=idx),
    )
    stressed_cf = replace(
        store.credit_funding,
        hy_oas_percentile_504d=pd.Series(0.95, index=idx),
        spy_21d_return=pd.Series(-0.06, index=idx),
    )
    stressed_volatility = VolatilityFeatures(
        close=store.volatility.close,
        return_1d=store.volatility.return_1d,
        return_5d=store.volatility.return_5d,
        return_21d=pd.Series(-0.05, index=idx),
        realized_vol_21d=pd.Series(0.30, index=idx),
        realized_vol_percentile_252d=pd.Series(0.85, index=idx),
        vix_percentile_252d=pd.Series(0.85, index=idx),
    )
    stressed_store = store.model_copy(
        update={
            "network_fragility": stressed_nf,
            "credit_funding": stressed_cf,
            "volatility": stressed_volatility,
        }
    )

    bundle = build_axis_series_bundle(
        context=stressed_context, feature_store=stressed_store
    )

    assert bundle.credit_funding is not None
    assert bundle.network_fragility is not None
    assert (
        bundle.breadth_state.active_labels_by_date[context.sessions[-1]]
        == "weak_breadth"
    )
    assert bundle.credit_funding[context.sessions[-1]].active_label == "credit_stress"
    assert (
        bundle.network_fragility[context.sessions[-1]].active_label == "systemic_stress"
    )


def test_unknown_flicker_does_not_fast_track_deescalation_through_correlation_to_one():
    """I2(b): a single-day `unknown` flicker in the middle of a stable
    `correlation_to_one` run must NOT cause fast de-escalation. The hold comes
    from the stable label being left (`correlation_to_one: 5`), not from
    `unknown` itself."""
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


def test_unknown_does_not_delay_recovery_into_classified_network_label():
    from regime_detection.hysteresis import apply_per_label_asymmetric_hysteresis

    cfg = load_default_regime_config().network_fragility
    raws: list[NetworkFragilityLabel] = ["unknown", "diversified_normal"]

    stable, active = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=NETWORK_FRAGILITY_RISK_RANK,
        deescalation_days_by_label=cfg.deescalation_days_by_label,
        default_deescalation_days=cfg.default_deescalation_days,
    )

    assert stable == ["unknown", "diversified_normal"]
    assert active == ["unknown", "diversified_normal"]
