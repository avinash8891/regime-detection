"""Slice 5 — v2 §2B Inflation/Growth axis end-to-end tests.

TDD per AGENTS.md / ~/.claude/CLAUDE.md testing rules:
  - Real ticker symbols (DBC, TLT, XLY, XLI, XLP, XLU, SPY) + real macro
    series keys (cpi_all_items, pmi_manufacturing, dgs10).
  - Real config (load_default_regime_config). No mocks.
  - Hand-computed expected values for numeric assertions.

Spec authority: docs/regime_engine_v2_spec.md §2B lines 2174-2326.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_detection.axis_series import build_inflation_growth_axis_series
from regime_detection.calendar import nyse_sessions_between
from regime_detection.config import (
    InflationGrowthRulesConfig,
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
from regime_detection.hysteresis import apply_per_label_asymmetric_hysteresis
from regime_detection.inflation_growth import (
    INFLATION_GROWTH_RISK_RANK,
    INFLATION_SURPRISE_NOWCAST_BIAS_WARNING_CODE,
    InflationGrowthFeatures,
    InflationGrowthRuleInputs,
    compute_inflation_growth_features,
    compute_inflation_surprise_zscore,
)
from regime_detection.market_context import build_market_context


# --- Synthetic fixtures ------------------------------------------------------

_TRAINING_SESSIONS = 650
_LAST_SESSION = pd.Timestamp("2025-04-30")
_SEED = 20260513


def _bdate_index(periods: int = _TRAINING_SESSIONS) -> pd.DatetimeIndex:
    sessions = nyse_sessions_between(
        (_LAST_SESSION - pd.Timedelta(days=periods * 2)).date(),
        _LAST_SESSION.date(),
    )
    return pd.DatetimeIndex([pd.Timestamp(d) for d in sessions[-periods:]])


def _default_rules() -> InflationGrowthRulesConfig:
    return load_default_regime_config().inflation_growth.rules


def _rule_inputs(**overrides) -> InflationGrowthRuleInputs:
    defaults: dict[str, object] = dict(
        cpi_6m_change_pct=0.02,
        cpi_6m_change_pct_lag_21=0.02,
        cpi_6m_change_pct_slope_21d=0.0,
        # ADR 0006 — NaN by default so the inflation_shock single-signal
        # limb is silent unless a test explicitly supplies a z-score.
        inflation_surprise_zscore=float("nan"),
        # Log #48 closure — NaN by default so the earnings_expansion /
        # earnings_contraction labels are silent unless a test supplies
        # a revision value (mirrors the accumulator cold-start state).
        aggregate_forward_eps_revision_direction_4w=float("nan"),
        pmi_manufacturing=52.0,
        pmi_manufacturing_slope_21d=0.0,
        commodity_return_63d=0.0,
        treasury_10y_yield_slope_21d=0.0,
        cyclical_defensive_slope_21d=0.0,
        spy_21d_return=0.01,
        tlt_21d_return=0.0,
        credit_funding_active_label="credit_calm",
    )
    defaults.update(overrides)
    return InflationGrowthRuleInputs(**defaults)


# --- Group A — Feature compute (4 tests) ------------------------------------


def _build_synthetic_context(
    *,
    cpi_truncate_calendar_days: int | None = None,
    pmi_truncate_calendar_days: int | None = None,
    dgs10_truncate_sessions: int | None = None,
    include_nowcast_and_eps_revision: bool = False,
):
    """Build a full MarketContext with §2B inputs."""
    idx = _bdate_index(periods=_TRAINING_SESSIONS)
    n = len(idx)
    rng = np.random.default_rng(_SEED)

    universe_prices = pd.DataFrame(
        (
            1.0 + rng.normal(0.0, 0.01, size=(n, len(NETWORK_FRAGILITY_UNIVERSE)))
        ).cumprod(axis=0)
        * 100.0,
        index=idx,
        columns=list(NETWORK_FRAGILITY_UNIVERSE),
    )
    spy_close = universe_prices[INDEX_SYMBOL]
    market_rows: list[dict[str, object]] = []
    for ts in idx:
        close = float(spy_close.loc[ts])
        market_rows.append(
            {
                "date": ts.date(),
                "symbol": "SPY",
                "open": close,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": 1_000_000,
            }
        )
        market_rows.append(
            {
                "date": ts.date(),
                "symbol": "RSP",
                "open": close * 0.5,
                "high": close * 0.5 * 1.005,
                "low": close * 0.5 * 0.995,
                "close": close * 0.5,
                "volume": 500_000,
            }
        )
        market_rows.append(
            {
                "date": ts.date(),
                "symbol": "VIXY",
                "open": 20.0,
                "high": 20.5,
                "low": 19.5,
                "close": 20.0,
                "volume": 100_000,
            }
        )
    market_data = pd.DataFrame(market_rows)

    sector_etf_closes = {s: universe_prices[s] for s in SECTOR_ETFS}
    cross_asset_closes: dict[str, pd.Series] = {
        s: universe_prices[s] for s in CROSS_ASSET_SYMBOLS
    }
    # Add KRE for credit_funding seam.
    cross_asset_closes["KRE"] = pd.Series(
        np.linspace(50.0, 55.0, n), index=idx, dtype=float, name="KRE"
    )
    # Add DBC for §2B inflation_growth (cross_asset_closes uses upper).
    cross_asset_closes["DBC"] = pd.Series(
        np.linspace(20.0, 22.0, n), index=idx, dtype=float, name="DBC"
    )
    # XLY/XLI/XLP/XLU are sector ETFs; the §2B classifier reads them from
    # cross_asset_closes (per slice 5 plan). Add them.
    for sym in ("XLY", "XLI", "XLP", "XLU"):
        cross_asset_closes[sym] = sector_etf_closes[sym]

    # Macro series — CPI/PMI monthly (every ~21 sessions); DGS10 daily; plus
    # credit_funding macro inputs for the §2C axis dependency.
    cpi = pd.Series(np.nan, index=idx, dtype=float)
    cpi_release_positions = list(range(0, n, 21))
    cpi.iloc[cpi_release_positions] = np.linspace(
        300.0, 320.0, len(cpi_release_positions)
    )
    if cpi_truncate_calendar_days is not None:
        cutoff = idx[-1] - pd.Timedelta(days=cpi_truncate_calendar_days)
        cpi.loc[cpi.index > cutoff] = np.nan

    pmi = pd.Series(np.nan, index=idx, dtype=float)
    pmi_release_positions = list(range(0, n, 21))
    pmi.iloc[pmi_release_positions] = 51.0
    if pmi_truncate_calendar_days is not None:
        cutoff = idx[-1] - pd.Timedelta(days=pmi_truncate_calendar_days)
        pmi.loc[pmi.index > cutoff] = np.nan

    dgs10 = pd.Series(4.0, index=idx, dtype=float, name="dgs10")
    if dgs10_truncate_sessions is not None:
        dgs10 = dgs10.copy()
        dgs10.iloc[-dgs10_truncate_sessions:] = np.nan

    # Macro for §2C (credit_funding) so cross-axis label populates.
    sofr = pd.Series(5.0, index=idx, dtype=float, name="SOFR")
    iorb = pd.Series(4.9, index=idx, dtype=float, name="IORB")
    nfci_w = pd.Series(np.nan, index=idx, dtype=float, name="NFCI")
    for pos in range(0, n, 5):
        nfci_w.iloc[pos] = -0.5
    usd = pd.Series(
        np.linspace(100.0, 102.0, n), index=idx, dtype=float, name="broad_usd_index"
    )

    macro_series = {
        "cpi_all_items": cpi,
        "pmi_manufacturing": pmi,
        "dgs10": dgs10,
        "SOFR": sofr,
        "IORB": iorb,
        "NFCI": nfci_w,
        "broad_usd_index": usd,
        "DGS2": pd.Series(4.5, index=idx, dtype=float),
        "DGS10": pd.Series(4.0, index=idx, dtype=float),
    }
    if include_nowcast_and_eps_revision:
        macro_series["cpi_nowcast"] = pd.Series(0.01, index=idx, dtype=float)
        macro_series["aggregate_forward_eps_revision"] = pd.Series(
            0.03, index=idx, dtype=float
        )

    config = RegimeEngine().config
    context = build_market_context(
        end_date=idx[-1].date(),
        market_data=market_data,
        config=config,
        sector_etf_closes=sector_etf_closes,
        cross_asset_closes=cross_asset_closes,
        macro_series=macro_series,
    )
    return context


def _build_store_and_outputs(context, *, credit_funding_active_labels_by_date=None):
    cfg = context.config
    store = build_feature_store(
        context,
        network_fragility_config=cfg.network_fragility,
        credit_funding_config=cfg.credit_funding,
        inflation_growth_config=cfg.inflation_growth,
    )
    outputs = build_inflation_growth_axis_series(
        context,
        store,
        credit_funding_active_labels_by_date=credit_funding_active_labels_by_date,
    )
    return store, outputs


def test_unknown_when_cpi_stale_more_than_60_days() -> None:
    """§2B line 2309: CPI stale > 60 days → unknown."""
    context = _build_synthetic_context(cpi_truncate_calendar_days=90)
    _, outputs = _build_store_and_outputs(context)
    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label == "unknown"
    assert "cpi_stale" in (out.data_quality.reason or "")


def test_unknown_when_pmi_stale_more_than_45_days() -> None:
    """§2B line 2310: PMI stale > 45 days → unknown."""
    context = _build_synthetic_context(pmi_truncate_calendar_days=60)
    _, outputs = _build_store_and_outputs(context)
    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label == "unknown"
    assert "pmi_stale" in (out.data_quality.reason or "")


def test_unknown_when_dgs10_stale_more_than_5_sessions() -> None:
    """§2B line 2311: DGS10 stale > 5 sessions → unknown."""
    context = _build_synthetic_context(dgs10_truncate_sessions=10)
    _, outputs = _build_store_and_outputs(context)
    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label == "unknown"
    assert "dgs10_stale" in (out.data_quality.reason or "")


def test_unknown_when_assess_series_input_quality_fails() -> None:
    """§2B line 2312: assess_series_input_quality fails → unknown.

    Force this by mutating the features so the spy_21d_return series is all NaN.
    """
    context = _build_synthetic_context()
    store = build_feature_store(
        context,
        network_fragility_config=context.config.network_fragility,
        credit_funding_config=context.config.credit_funding,
        inflation_growth_config=context.config.inflation_growth,
    )
    ig = store.inflation_growth
    assert ig is not None
    nan_series = pd.Series(np.nan, index=ig.cpi_6m_change_pct.index)
    broken = InflationGrowthFeatures(
        cpi_3m_change_pct=nan_series,
        cpi_6m_change_pct=nan_series,
        cpi_6m_change_pct_slope_21d=nan_series,
        inflation_surprise_zscore=nan_series,
        pmi_manufacturing=nan_series,
        pmi_manufacturing_slope_21d=nan_series,
        aggregate_forward_eps_revision_direction_4w=nan_series,
        commodity_return_63d=nan_series,
        treasury_10y_yield_slope_21d=nan_series,
        cyclical_defensive_ratio=nan_series,
        cyclical_defensive_slope_21d=nan_series,
        spy_21d_return=nan_series,
        tlt_21d_return=nan_series,
        bias_warnings=ig.bias_warnings,
    )
    broken_store = store.model_copy(update={"inflation_growth": broken})
    outputs = build_inflation_growth_axis_series(context, broken_store)
    assert outputs is not None
    last_day = context.sessions[-1]
    assert outputs[last_day].raw_label == "unknown"


def test_feature_store_seam_lit_with_all_inputs() -> None:
    """All 9 §2B inputs present → feature_store.inflation_growth populated."""
    context = _build_synthetic_context()
    store = build_feature_store(
        context,
        inflation_growth_config=context.config.inflation_growth,
    )
    assert store.inflation_growth is not None
    assert isinstance(store.inflation_growth, InflationGrowthFeatures)


def test_classifier_rule_evidence_includes_nowcast_and_eps_revision_inputs() -> None:
    """2B evidence must surface the optional nowcast/EPS scalars once wired."""
    context = _build_synthetic_context(include_nowcast_and_eps_revision=True)
    _, outputs = _build_store_and_outputs(context)
    assert outputs is not None

    last_day = context.sessions[-1]
    evidence = outputs[last_day].evidence["rule_evidence"]
    assert "inflation_surprise_zscore" in evidence
    assert "aggregate_forward_eps_revision_direction_4w" in evidence


def test_feature_store_seam_none_when_dbc_missing() -> None:
    """Missing DBC → feature_store.inflation_growth is None."""
    context = _build_synthetic_context()
    # Strip DBC.
    stripped = {
        k: v for k, v in (context.cross_asset_closes or {}).items() if k != "DBC"
    }
    new_context = build_market_context(
        end_date=context.end_date,
        market_data=pd.DataFrame(
            [
                {
                    "date": ts.date(),
                    "symbol": "SPY",
                    "open": float(context.spy_ohlcv["open"].loc[ts]),
                    "high": float(context.spy_ohlcv["high"].loc[ts]),
                    "low": float(context.spy_ohlcv["low"].loc[ts]),
                    "close": float(context.spy_ohlcv["close"].loc[ts]),
                    "volume": float(context.spy_ohlcv["volume"].loc[ts]),
                }
                for ts in context.spy_ohlcv.index
            ]
            + [
                {
                    "date": ts.date(),
                    "symbol": "RSP",
                    "open": float(context.rsp_close.loc[ts]),
                    "high": float(context.rsp_close.loc[ts]),
                    "low": float(context.rsp_close.loc[ts]),
                    "close": float(context.rsp_close.loc[ts]),
                    "volume": 500_000,
                }
                for ts in context.spy_ohlcv.index
            ]
        ),
        config=context.config,
        sector_etf_closes=context.sector_etf_closes,
        cross_asset_closes=stripped,
        macro_series=context.macro_series,
    )
    store = build_feature_store(
        new_context, inflation_growth_config=new_context.config.inflation_growth
    )
    assert store.inflation_growth is None


# --- Group D — Hysteresis (2 tests) ------------------------------------------


def test_inflation_shock_holds_for_5_deescalation_days() -> None:
    """§2B line 2295: inflation_shock → ... holds 5 days."""
    deesc = load_default_regime_config().inflation_growth.deescalation_days_by_label
    raws = ["inflation_shock"] * 10 + ["goldilocks"] * 10
    stable, _ = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=INFLATION_GROWTH_RISK_RANK,
        deescalation_days_by_label=deesc,
        default_deescalation_days=0,
    )
    for i in range(10, 14):
        assert stable[i] == "inflation_shock", f"position {i}: {stable[i]}"
    assert stable[14] == "goldilocks"


def test_goldilocks_deescalates_immediately() -> None:
    """§2B line 2299: goldilocks deescalates in 0 days (immediate)."""
    deesc = load_default_regime_config().inflation_growth.deescalation_days_by_label
    # goldilocks (rank 0) → inflation_shock (rank 3) is immediate escalation.
    raws = ["goldilocks"] * 5 + ["inflation_shock"] * 5
    stable, _ = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=INFLATION_GROWTH_RISK_RANK,
        deescalation_days_by_label=deesc,
        default_deescalation_days=0,
    )
    assert stable[5] == "inflation_shock"


# --- Group E — End-to-end wire integration (1 test) --------------------------


def test_regime_output_carries_inflation_growth_state_when_configured() -> None:
    """End-to-end: classify_window populates RegimeOutput.inflation_growth_state."""
    context = _build_synthetic_context()
    engine = RegimeEngine()
    timeline = engine.classify_window(
        end_date=context.end_date,
        market_data=pd.DataFrame(
            [
                {
                    "date": ts.date(),
                    "symbol": "SPY",
                    "open": float(context.spy_ohlcv["open"].loc[ts]),
                    "high": float(context.spy_ohlcv["high"].loc[ts]),
                    "low": float(context.spy_ohlcv["low"].loc[ts]),
                    "close": float(context.spy_ohlcv["close"].loc[ts]),
                    "volume": float(context.spy_ohlcv["volume"].loc[ts]),
                }
                for ts in context.spy_ohlcv.index
            ]
            + [
                {
                    "date": ts.date(),
                    "symbol": "RSP",
                    "open": float(context.rsp_close.loc[ts]),
                    "high": float(context.rsp_close.loc[ts]),
                    "low": float(context.rsp_close.loc[ts]),
                    "close": float(context.rsp_close.loc[ts]),
                    "volume": 500_000,
                }
                for ts in context.spy_ohlcv.index
            ]
        ),
        lookback_days=1,
        sector_etf_closes=context.sector_etf_closes,
        cross_asset_closes=context.cross_asset_closes,
        macro_series=context.macro_series,
    )
    out = timeline.outputs[-1]
    assert out.inflation_growth_state is not None
    allowed = set(INFLATION_GROWTH_RISK_RANK.keys())
    assert out.inflation_growth_state.active_label in allowed


# ---------------------------------------------------------------------------
# ADR 0006 — inflation_surprise_zscore via the Cleveland Fed nowcast.
# ---------------------------------------------------------------------------


def test_compute_inflation_surprise_zscore_hand_computed() -> None:
    """ADR 0006: surprise = realized_cpi_rate - cpi_nowcast, z-scored over
    a rolling std window. With a short normalizer window and a hand-built
    CPI/nowcast pair the z-score at the trailing session is exact."""
    idx = _bdate_index(periods=40)
    # CPIAUCSL index level rising ~0.5%/month over the window.
    cpi = pd.Series(np.linspace(300.0, 306.0, len(idx)), index=idx, dtype=float)
    # Nowcast: a constant inflation-rate estimate of 1.0% (0.01).
    nowcast = pd.Series(0.01, index=idx, dtype=float)
    zscore = compute_inflation_surprise_zscore(
        cpi_all_items=cpi,
        cpi_nowcast=nowcast,
        session_index=idx,
        realized_rate_lookback=5,
        normalizer_window=10,
    )
    assert isinstance(zscore, pd.Series)
    assert zscore.name == "inflation_surprise_zscore"
    # First (realized_rate_lookback + normalizer_window - 1) rows are NaN
    # (cold-start — the 5y/normalizer std needs a full window).
    assert zscore.iloc[: 5 + 10 - 2].isna().all()
    # Past the cold-start window the z-score is finite.
    assert zscore.dropna().shape[0] > 0


def test_compute_inflation_surprise_zscore_cold_start_all_nan_below_window() -> None:
    """Below `normalizer_window` of surprise history, the z-score is
    entirely NaN — the single-signal limb stays silent (V1 §2.7)."""
    idx = _bdate_index(periods=15)
    cpi = pd.Series(np.linspace(300.0, 302.0, len(idx)), index=idx, dtype=float)
    nowcast = pd.Series(0.01, index=idx, dtype=float)
    zscore = compute_inflation_surprise_zscore(
        cpi_all_items=cpi,
        cpi_nowcast=nowcast,
        session_index=idx,
        realized_rate_lookback=5,
        normalizer_window=1260,  # far longer than the 15-session input
    )
    assert zscore.isna().all()


def test_compute_inflation_growth_features_emits_real_zscore_with_nowcast() -> None:
    """When cpi_nowcast is supplied, compute_inflation_growth_features
    computes a real (non-all-NaN) inflation_surprise_zscore and emits the
    Cleveland-Fed-nowcast bias-warning provenance row."""
    idx = _bdate_index(periods=400)
    n = len(idx)
    cpi = pd.Series(np.linspace(300.0, 312.0, n), index=idx, dtype=float)
    nowcast = pd.Series(0.01, index=idx, dtype=float)
    pmi = pd.Series(51.0, index=idx, dtype=float)
    dgs10 = pd.Series(4.0, index=idx, dtype=float)
    dbc = pd.Series(np.linspace(20.0, 25.0, n), index=idx, dtype=float)
    spy = pd.Series(np.linspace(400.0, 420.0, n), index=idx, dtype=float)
    tlt = pd.Series(100.0, index=idx, dtype=float)
    xly = xli = xlp = xlu = pd.Series(100.0, index=idx, dtype=float)

    # Short normalizer window so a 400-session input produces non-NaN values.
    rules = _default_rules().model_copy(
        update={"inflation_surprise_normalizer_window_sessions": 60}
    )
    feats = compute_inflation_growth_features(
        cpi_all_items=cpi,
        pmi_manufacturing=pmi,
        dgs10=dgs10,
        dbc_close=dbc,
        spy_close=spy,
        tlt_close=tlt,
        xly_close=xly,
        xli_close=xli,
        xlp_close=xlp,
        xlu_close=xlu,
        config=rules,
        cpi_nowcast=nowcast,
    )
    # The z-score is no longer the all-NaN placeholder.
    assert feats.inflation_surprise_zscore.notna().any()
    # The bias-warning frame carries the Cleveland Fed nowcast provenance row.
    bw = feats.bias_warnings
    assert (bw["warning_code"] == INFLATION_SURPRISE_NOWCAST_BIAS_WARNING_CODE).any()
    nowcast_row = bw[bw["warning_code"] == INFLATION_SURPRISE_NOWCAST_BIAS_WARNING_CODE]
    assert list(nowcast_row["feature_name"]) == ["inflation_surprise_zscore"]


def test_compute_inflation_growth_features_all_nan_zscore_without_nowcast() -> None:
    """When cpi_nowcast is NOT supplied, inflation_surprise_zscore stays the
    all-NaN placeholder and NO Cleveland-Fed bias-warning row is emitted —
    V1 byte-identity preserved (the pre-ADR-0006 behaviour)."""
    idx = _bdate_index(periods=120)
    n = len(idx)
    cpi = pd.Series(np.linspace(300.0, 304.0, n), index=idx, dtype=float)
    pmi = pd.Series(51.0, index=idx, dtype=float)
    dgs10 = pd.Series(4.0, index=idx, dtype=float)
    dbc = pd.Series(20.0, index=idx, dtype=float)
    spy = pd.Series(400.0, index=idx, dtype=float)
    tlt = pd.Series(100.0, index=idx, dtype=float)
    xly = xli = xlp = xlu = pd.Series(100.0, index=idx, dtype=float)

    feats = compute_inflation_growth_features(
        cpi_all_items=cpi,
        pmi_manufacturing=pmi,
        dgs10=dgs10,
        dbc_close=dbc,
        spy_close=spy,
        tlt_close=tlt,
        xly_close=xly,
        xli_close=xli,
        xlp_close=xlp,
        xlu_close=xlu,
        config=_default_rules(),
        # cpi_nowcast omitted
    )
    assert feats.inflation_surprise_zscore.isna().all()
    bw = feats.bias_warnings
    assert not (
        bw["warning_code"] == INFLATION_SURPRISE_NOWCAST_BIAS_WARNING_CODE
    ).any()
