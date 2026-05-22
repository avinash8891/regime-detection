"""Slice 4 — v2 §2C Credit/Funding axis end-to-end tests.

TDD per AGENTS.md / ~/.claude/CLAUDE.md testing rules:
  - Real ticker symbols (HYG, LQD, TLT, KRE, SOFR, IORB, NFCI, broad_usd_index).
  - Real config (load_default_regime_config). No mocks of pandas/fetchers.
  - Hand-computed expected values for numeric assertions.
  - One end-to-end engine test via RegimeEngine.classify.

Spec authority: docs/regime_engine_v2_spec.md §2C lines 2005-2130.
"""

from __future__ import annotations


from datetime import date

import numpy as np
import pandas as pd
import pytest

from regime_detection.axis_series import (
    build_credit_funding_axis_series,
    build_credit_funding_proxy_axis_series,
    resolve_credit_funding_effective_output,
)
from regime_detection.models import CreditFundingOutput, DataQuality
from regime_detection.calendar import nyse_sessions_between
from regime_detection.config import (
    CreditFundingRulesConfig,
    load_default_regime_config,
)
from regime_detection.credit_funding import (
    CREDIT_FUNDING_RISK_RANK,
    CreditFundingFeatures,
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
from regime_detection.market_context import build_market_context


# --- Synthetic fixtures ------------------------------------------------------

_TRAINING_SESSIONS = 650  # > 504 + 63 cold-start
_LAST_SESSION = pd.Timestamp("2025-04-30")
_SEED = 20260513
_REAL_FIXTURE_CREDIT_AS_OF = date(2026, 5, 12)


def _bdate_index(periods: int = _TRAINING_SESSIONS) -> pd.DatetimeIndex:
    sessions = nyse_sessions_between(
        (_LAST_SESSION - pd.Timedelta(days=periods * 2)).date(),
        _LAST_SESSION.date(),
    )
    return pd.DatetimeIndex([pd.Timestamp(d) for d in sessions[-periods:]])


def _make_constant_series(
    index: pd.DatetimeIndex, value: float, name: str
) -> pd.Series:
    return pd.Series(value, index=index, name=name)


def _make_random_walk(
    index: pd.DatetimeIndex, *, seed: int, start: float, sigma: float
) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, sigma, size=len(index))
    closes = start * (1.0 + rets).cumprod()
    return pd.Series(closes, index=index, dtype=float)


def _default_rules() -> CreditFundingRulesConfig:
    return load_default_regime_config().credit_funding.rules


# --- Group A — Feature compute (5 tests) -------------------------------------


def _build_full_synthetic_context(
    *,
    hyg_truncate_sessions: int | None = None,
    nfci_truncate_calendar_days: int | None = None,
    sofr_drop_last: bool = False,
    iorb_truncate_calendar_days: int | None = None,
    legacy_funding_splice: bool = False,
    ioer_legacy_truncate_calendar_days: int | None = None,
    hy_oas_truncate_calendar_days: int | None = None,
    omit_oas_series: bool = False,
):
    """Build a MarketContext with full cross_asset_closes and macro_series.

    Optional knobs simulate spec unknown-gate failure modes.
    """
    idx = _bdate_index(periods=_TRAINING_SESSIONS)
    n = len(idx)
    rng = np.random.default_rng(_SEED)

    # Build full NETWORK_FRAGILITY_UNIVERSE prices (so feature_store.network_fragility lights up).
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
    # Add KRE on cross_asset_closes alongside the §3.1 cross-asset symbols.
    kre_series = _make_random_walk(idx, seed=_SEED + 99, start=50.0, sigma=0.012)
    cross_asset_closes = {s: universe_prices[s] for s in CROSS_ASSET_SYMBOLS}
    cross_asset_closes["KRE"] = kre_series

    # HYG truncation: zero out the last N sessions of HYG to simulate staleness.
    if hyg_truncate_sessions is not None:
        hyg_copy = cross_asset_closes["HYG"].copy()
        hyg_copy.iloc[-hyg_truncate_sessions:] = np.nan
        cross_asset_closes["HYG"] = hyg_copy

    # Macro series — daily SOFR/IORB, weekly NFCI, daily broad_usd_index.
    sofr = _make_constant_series(idx, 5.0, "sofr")
    iorb = _make_constant_series(idx, 4.9, "iorb")
    if sofr_drop_last:
        sofr = sofr.copy()
        sofr.iloc[-1] = np.nan
    if iorb_truncate_calendar_days is not None:
        iorb = iorb.copy()
        cutoff = idx[-1] - pd.Timedelta(days=iorb_truncate_calendar_days)
        iorb.loc[iorb.index > cutoff] = np.nan
    fedfunds = None
    ioer_legacy = None
    if legacy_funding_splice:
        sofr = pd.Series(np.nan, index=idx, dtype=float, name="sofr")
        iorb = pd.Series(np.nan, index=idx, dtype=float, name="iorb")
        fedfunds = _make_constant_series(idx, 0.41, "fedfunds")
        ioer_legacy = _make_constant_series(idx, 0.40, "ioer_legacy")
        if ioer_legacy_truncate_calendar_days is not None:
            cutoff = idx[-1] - pd.Timedelta(days=ioer_legacy_truncate_calendar_days)
            ioer_legacy.loc[ioer_legacy.index > cutoff] = np.nan
    nfci_w = pd.Series(np.nan, index=idx, dtype=float, name="nfci")
    weekly_positions = list(range(0, n, 5))
    nfci_values = rng.normal(-0.5, 0.2, size=len(weekly_positions))
    for pos, val in zip(weekly_positions, nfci_values):
        nfci_w.iloc[pos] = val
    if nfci_truncate_calendar_days is not None:
        # Wipe NFCI for the last `nfci_truncate_calendar_days` calendar days.
        cutoff = idx[-1] - pd.Timedelta(days=nfci_truncate_calendar_days)
        nfci_w.loc[nfci_w.index > cutoff] = np.nan
    usd = _make_random_walk(idx, seed=_SEED + 100, start=100.0, sigma=0.003)

    hy_oas = _make_random_walk(idx, seed=_SEED + 101, start=400.0, sigma=0.01)
    ig_oas = _make_random_walk(idx, seed=_SEED + 102, start=150.0, sigma=0.01)
    if hy_oas_truncate_calendar_days is not None:
        cutoff = idx[-1] - pd.Timedelta(days=hy_oas_truncate_calendar_days)
        hy_oas.loc[hy_oas.index > cutoff] = np.nan

    macro_series = {
        "sofr": sofr,
        "iorb": iorb,
        "nfci": nfci_w,
        "broad_usd_index": usd,
        # ICE BofA OAS series — single source for the §2C credit-spread
        # metric. Required by `_CF_MACRO_KEYS`, so the §2C seam does not
        # build without them.
        "hy_oas": hy_oas,
        "ig_bbb_oas": ig_oas,
        # Add yield series for monetary slice compatibility.
        "2y_yield": _make_constant_series(idx, 4.5, "2y_yield"),
        "10y_yield": _make_constant_series(idx, 4.0, "10y_yield"),
    }
    if omit_oas_series:
        macro_series.pop("hy_oas")
        macro_series.pop("ig_bbb_oas")
    if fedfunds is not None and ioer_legacy is not None:
        macro_series["fedfunds"] = fedfunds
        macro_series["ioer_legacy"] = ioer_legacy

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


def _build_store_and_outputs(context):
    cfg = context.config
    store = build_feature_store(
        context,
        network_fragility_config=cfg.network_fragility,
        monetary_pressure_v2_config=cfg.monetary_pressure_v2,
        credit_funding_config=cfg.credit_funding,
    )
    return store, build_credit_funding_axis_series(context, store)


def _build_real_v2_credit_context(
    as_of: date,
    v2_market_df_for_asof,
    v2_close_series_by_symbol: dict[str, pd.Series],
    v2_macro_series_by_key: dict[str, pd.Series],
):
    required_symbols = set(SECTOR_ETFS) | set(CROSS_ASSET_SYMBOLS) | {"KRE"}
    missing = required_symbols - set(v2_close_series_by_symbol)
    assert not missing, f"V2 OHLCV fixture missing symbols: {sorted(missing)}"
    sector_etf_closes = {
        symbol: v2_close_series_by_symbol[symbol] for symbol in SECTOR_ETFS
    }
    cross_asset_closes = {
        symbol: v2_close_series_by_symbol[symbol]
        for symbol in set(CROSS_ASSET_SYMBOLS) | {"KRE"}
    }
    return build_market_context(
        end_date=as_of,
        market_data=v2_market_df_for_asof(as_of),
        config=RegimeEngine().config,
        sector_etf_closes=sector_etf_closes,
        cross_asset_closes=cross_asset_closes,
        macro_series=v2_macro_series_by_key,
    )


def test_build_proxy_runs_parallel_to_build_with_proxy_bias_code() -> None:
    """build_proxy() runs the identical §2C rule schema on the TLT-proxy
    series, producing a parallel output keyed exactly like build() — but
    tagged with the proxy bias-warning code, never blended (Log #71)."""
    context = _build_full_synthetic_context()
    cfg = context.config
    store = build_feature_store(
        context,
        network_fragility_config=cfg.network_fragility,
        monetary_pressure_v2_config=cfg.monetary_pressure_v2,
        credit_funding_config=cfg.credit_funding,
    )
    real = build_credit_funding_axis_series(context, store)
    proxy = build_credit_funding_proxy_axis_series(context, store)

    assert real is not None and proxy is not None
    # One output per session from both runs.
    assert set(real.keys()) == set(proxy.keys())

    # A session where the rule engine fired (past every cold-start / gate) in
    # both runs — its evidence must carry the source-specific bias code.
    rule_day = next(
        d
        for d in real
        if "rule_evidence" in real[d].evidence and "rule_evidence" in proxy[d].evidence
    )
    assert (
        real[rule_day].evidence["bias_warning_code"]
        == "credit_spread_ice_bofa_oas_fred"
    )
    assert (
        proxy[rule_day].evidence["bias_warning_code"]
        == "credit_spread_proxy_total_return_differential"
    )
    assert real[rule_day].evidence["spread_source"] == "ice_bofa_oas"
    assert proxy[rule_day].evidence["spread_source"] == "tlt_total_return_differential"


def _credit_output(
    *,
    label: str,
    source: str,
    status: str = "ok",
) -> CreditFundingOutput:
    return CreditFundingOutput(
        raw_label=label,
        stable_label=label,
        active_label=label,
        evidence={"spread_source": source},
        data_quality=DataQuality(status=status),
    )


def test_effective_credit_funding_uses_higher_risk_when_oas_and_proxy_diverge() -> None:
    oas = _credit_output(label="credit_calm", source="ice_bofa_oas")
    proxy = _credit_output(
        label="spread_widening",
        source="tlt_total_return_differential",
    )

    effective = resolve_credit_funding_effective_output(oas=oas, proxy=proxy)

    assert effective is not None
    assert effective.active_label == "spread_widening"
    assert effective.evidence["source_used"] == "proxy_higher_risk"
    assert effective.evidence["agreement_status"] == "divergent"
    assert effective.evidence["oas_label"] == "credit_calm"
    assert effective.evidence["proxy_label"] == "spread_widening"


def test_effective_credit_funding_falls_back_to_proxy_when_oas_unavailable() -> None:
    oas = _credit_output(
        label="unknown",
        source="ice_bofa_oas",
        status="insufficient_data",
    )
    proxy = _credit_output(
        label="credit_calm",
        source="tlt_total_return_differential",
    )

    effective = resolve_credit_funding_effective_output(oas=oas, proxy=proxy)

    assert effective is not None
    assert effective.active_label == "credit_calm"
    assert effective.evidence["source_used"] == "proxy_fallback"
    assert effective.evidence["agreement_status"] == "proxy_only"


def test_credit_funding_proxy_builds_when_oas_series_are_absent() -> None:
    context = _build_full_synthetic_context(omit_oas_series=True)
    cfg = context.config
    store = build_feature_store(
        context,
        network_fragility_config=cfg.network_fragility,
        monetary_pressure_v2_config=cfg.monetary_pressure_v2,
        credit_funding_config=cfg.credit_funding,
    )
    real = build_credit_funding_axis_series(context, store)
    proxy = build_credit_funding_proxy_axis_series(context, store)

    assert real is not None
    assert proxy is not None
    rule_day = next(
        d
        for d in proxy
        if "rule_evidence" in proxy[d].evidence and proxy[d].active_label != "unknown"
    )
    assert real[rule_day].active_label == "unknown"
    assert proxy[rule_day].active_label in CREDIT_FUNDING_RISK_RANK
    effective = resolve_credit_funding_effective_output(
        oas=real[rule_day],
        proxy=proxy[rule_day],
    )
    assert effective is not None
    assert effective.evidence["source_used"] == "proxy_fallback"


def test_real_oas_percentile_warmup_is_insufficient_history_not_missing_feature() -> None:
    context = _build_full_synthetic_context()
    store = build_feature_store(
        context,
        network_fragility_config=context.config.network_fragility,
        monetary_pressure_v2_config=context.config.monetary_pressure_v2,
        credit_funding_config=context.config.credit_funding,
    )
    cf = store.credit_funding
    assert cf is not None

    # Simulate the real FRED OAS truncation class: raw OAS has enough recent
    # observations to pass the generic completeness floor, but the derived
    # 504d percentile is still NaN on the current session.
    hy_oas = cf.hy_oas_63d.copy()
    ig_oas = cf.ig_oas_63d.copy()
    hy_oas.iloc[:-400] = np.nan
    ig_oas.iloc[:-400] = np.nan
    hy_percentile = pd.Series(np.nan, index=hy_oas.index, dtype=float)
    warmed_store = store.model_copy(
        update={
            "credit_funding": CreditFundingFeatures(
                hy_oas_63d=hy_oas,
                ig_oas_63d=ig_oas,
                hy_oas_percentile_504d=hy_percentile,
                hy_oas_slope_21d=cf.hy_oas_slope_21d,
                ig_oas_slope_21d=cf.ig_oas_slope_21d,
                hy_tr_differential_63d=cf.hy_tr_differential_63d,
                ig_tr_differential_63d=cf.ig_tr_differential_63d,
                hy_tr_differential_percentile_504d=cf.hy_tr_differential_percentile_504d,
                hy_tr_differential_slope_21d=cf.hy_tr_differential_slope_21d,
                ig_tr_differential_slope_21d=cf.ig_tr_differential_slope_21d,
                kre_spy_ratio=cf.kre_spy_ratio,
                kre_spy_slope_63d=cf.kre_spy_slope_63d,
                nfci_daily_carried=cf.nfci_daily_carried,
                sofr_iorb_spread=cf.sofr_iorb_spread,
                sofr_iorb_slope_21d=cf.sofr_iorb_slope_21d,
                broad_usd_index_zscore_21d=cf.broad_usd_index_zscore_21d,
                spy_21d_return=cf.spy_21d_return,
                tlt_21d_return=cf.tlt_21d_return,
                bias_warnings=cf.bias_warnings,
            )
        }
    )

    real = build_credit_funding_axis_series(context, warmed_store)
    proxy = build_credit_funding_proxy_axis_series(context, warmed_store)
    assert real is not None
    assert proxy is not None
    day = context.sessions[-1]
    assert real[day].classification_status == "insufficient_history"
    assert real[day].classification_reason == "hy_spread_percentile_504d_warmup"
    assert real[day].reporting_label == "insufficient_history"

    effective = resolve_credit_funding_effective_output(oas=real[day], proxy=proxy[day])
    assert effective is not None
    assert effective.evidence["source_used"] == "proxy_fallback"


def test_unknown_when_hyg_stale_more_than_5_sessions() -> None:
    """§2C line 2123: HYG stale > 5 sessions → unknown gate trip."""
    context = _build_full_synthetic_context(hyg_truncate_sessions=10)
    _, outputs = _build_store_and_outputs(context)
    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label == "unknown"
    assert "etf_stale:HYG" in (out.data_quality.reason or "")


def test_unknown_when_nfci_stale_more_than_14_days() -> None:
    """§2C line 2124: NFCI stale > 14 calendar days → unknown gate trip."""
    context = _build_full_synthetic_context(nfci_truncate_calendar_days=20)
    _, outputs = _build_store_and_outputs(context)
    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label == "unknown"
    assert "nfci_stale" in (out.data_quality.reason or "")


def test_credit_funding_carries_one_session_sofr_publication_lag() -> None:
    """SOFR can be absent on the latest NYSE session until publication catches up."""
    context = _build_full_synthetic_context(sofr_drop_last=True)
    _, outputs = _build_store_and_outputs(context)
    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label != "unknown"
    assert out.data_quality.status != "insufficient_data"


def test_unknown_when_oas_spread_source_is_stale() -> None:
    context = _build_full_synthetic_context(hy_oas_truncate_calendar_days=70)
    _, outputs = _build_store_and_outputs(context)

    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label == "unknown"
    assert "hy_oas_stale" in (out.data_quality.reason or "")


def test_unknown_when_iorb_component_is_stale() -> None:
    context = _build_full_synthetic_context(iorb_truncate_calendar_days=70)
    _, outputs = _build_store_and_outputs(context)

    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label == "unknown"
    assert "funding_spread_stale" in (out.data_quality.reason or "")


def test_unknown_when_legacy_ioer_component_is_stale() -> None:
    context = _build_full_synthetic_context(
        legacy_funding_splice=True,
        ioer_legacy_truncate_calendar_days=70,
    )
    _, outputs = _build_store_and_outputs(context)

    assert outputs is not None
    last_day = context.sessions[-1]
    out = outputs[last_day]
    assert out.raw_label == "unknown"
    assert "funding_spread_stale" in (out.data_quality.reason or "")


def test_unknown_when_assess_series_input_quality_fails() -> None:
    """§2C line 2126: assess_series_input_quality fails → unknown.

    Forced by mutating the feature store so the spread-proxy series is all NaN
    (insufficient history) — staleness gate passes because the underlying ETF
    closes are intact, so the secondary quality gate must be what catches us.
    """
    context = _build_full_synthetic_context()
    store = build_feature_store(
        context,
        network_fragility_config=context.config.network_fragility,
        credit_funding_config=context.config.credit_funding,
    )
    cf = store.credit_funding
    assert cf is not None
    nan_series = pd.Series(np.nan, index=cf.hy_oas_63d.index)
    broken = CreditFundingFeatures(
        hy_oas_63d=nan_series,
        ig_oas_63d=nan_series,
        hy_oas_percentile_504d=nan_series,
        hy_oas_slope_21d=nan_series,
        ig_oas_slope_21d=nan_series,
        hy_tr_differential_63d=nan_series,
        ig_tr_differential_63d=nan_series,
        hy_tr_differential_percentile_504d=nan_series,
        hy_tr_differential_slope_21d=nan_series,
        ig_tr_differential_slope_21d=nan_series,
        kre_spy_ratio=nan_series,
        kre_spy_slope_63d=nan_series,
        nfci_daily_carried=nan_series,
        sofr_iorb_spread=nan_series,
        sofr_iorb_slope_21d=nan_series,
        broad_usd_index_zscore_21d=nan_series,
        spy_21d_return=nan_series,
        tlt_21d_return=nan_series,
        bias_warnings=cf.bias_warnings,
    )
    broken_store = store.model_copy(update={"credit_funding": broken})
    outputs = build_credit_funding_axis_series(context, broken_store)
    assert outputs is not None
    last_day = context.sessions[-1]
    assert outputs[last_day].raw_label == "unknown"


# --- Group D — Hysteresis (2 tests) ------------------------------------------


def test_deleveraging_holds_for_5_deescalation_days() -> None:
    """§2C lines 2111-2117: deleveraging→credit_calm transitions held 5d."""
    deesc = load_default_regime_config().credit_funding.deescalation_days_by_label
    # 10 days deleveraging, then switch to credit_calm. Hold period = 5.
    raws = ["deleveraging"] * 10 + ["credit_calm"] * 10
    stable, active = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=CREDIT_FUNDING_RISK_RANK,
        deescalation_days_by_label=deesc,
        default_deescalation_days=0,
    )
    # Stable still deleveraging on the first 4 post-flip days (positions 10..13).
    for i in range(10, 14):
        assert stable[i] == "deleveraging", f"position {i}: {stable[i]}"
    # On position 14 (the 5th post-flip day) the hold expires; per the
    # hysteresis implementation, pending_count >= threshold triggers the flip.
    assert stable[14] == "credit_calm"


def test_credit_calm_deescalates_immediately() -> None:
    """§2C line 2115: credit_calm holds 0 days (immediate de-escalation)."""
    deesc = load_default_regime_config().credit_funding.deescalation_days_by_label
    # Start in credit_calm (rank 0), flip to spread_widening (rank 1).
    # spread_widening has HIGHER risk_rank — escalation must be immediate.
    raws = ["credit_calm"] * 5 + ["spread_widening"] * 5
    stable, _active = apply_per_label_asymmetric_hysteresis(
        raw_labels=raws,
        risk_rank=CREDIT_FUNDING_RISK_RANK,
        deescalation_days_by_label=deesc,
        default_deescalation_days=0,
    )
    # Immediate escalation: position 5 must already be spread_widening.
    assert stable[5] == "spread_widening"


# --- Group E — Wire integration (3 tests) ------------------------------------


def test_feature_store_credit_funding_seam_none_without_kre_in_cross_asset_closes() -> (
    None
):
    """Missing KRE on cross_asset_closes → feature_store.credit_funding is None."""
    context = _build_full_synthetic_context()
    # Strip KRE from the cross_asset_closes dict.
    stripped = {
        k: v for k, v in (context.cross_asset_closes or {}).items() if k != "KRE"
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
        new_context, credit_funding_config=new_context.config.credit_funding
    )
    assert store.credit_funding is None


def test_feature_store_credit_funding_seam_lit_with_all_inputs() -> None:
    """All 8 §2C inputs present → feature_store.credit_funding is populated."""
    context = _build_full_synthetic_context()
    store = build_feature_store(
        context,
        network_fragility_config=context.config.network_fragility,
        credit_funding_config=context.config.credit_funding,
    )
    assert store.credit_funding is not None
    assert isinstance(store.credit_funding, CreditFundingFeatures)


def test_real_v2_fixture_credit_funding_golden_label(
    v2_market_df_for_asof,
    v2_close_series_by_symbol: dict[str, pd.Series],
    v2_macro_series_by_key: dict[str, pd.Series],
) -> None:
    """Real V2 OHLCV + FRED fixture lights §2C and pins current labels."""
    as_of = _REAL_FIXTURE_CREDIT_AS_OF
    context = _build_real_v2_credit_context(
        as_of,
        v2_market_df_for_asof,
        v2_close_series_by_symbol,
        v2_macro_series_by_key,
    )
    store = build_feature_store(
        context,
        network_fragility_config=context.config.network_fragility,
        monetary_pressure_v2_config=context.config.monetary_pressure_v2,
        credit_funding_config=context.config.credit_funding,
    )
    assert store.credit_funding is not None

    real_outputs = build_credit_funding_axis_series(context, store)
    proxy_outputs = build_credit_funding_proxy_axis_series(context, store)
    assert real_outputs is not None
    assert proxy_outputs is not None

    real = real_outputs[as_of]
    proxy = proxy_outputs[as_of]
    assert real.raw_label == "credit_calm"
    assert real.stable_label == "credit_calm"
    assert real.active_label == "credit_calm"
    assert real.data_quality.status == "ok"
    assert real.data_quality.reason is None
    assert real.evidence["spread_source"] == "ice_bofa_oas"
    assert real.evidence["bias_warning_code"] == "credit_spread_ice_bofa_oas_fred"
    assert real.evidence["nfci_daily_carried"] == pytest.approx(-0.524)
    assert real.evidence["kre_spy_slope_63d"] == pytest.approx(-7.786519147306989e-05)
    real_rule = real.evidence["rule_evidence"]
    assert real_rule["hy_spread_percentile_504d"] == pytest.approx(0.24305555555555555)
    assert real_rule["hy_spread_slope_21d"] == pytest.approx(-0.004064935064935065)
    assert real_rule["spy_21d_return"] == pytest.approx(0.07590730214254471)
    assert real_rule["avg_pairwise_corr_percentile_504d"] == pytest.approx(
        0.3055555555555556
    )

    assert proxy.raw_label == "credit_calm"
    assert proxy.stable_label == "credit_calm"
    assert proxy.active_label == "credit_calm"
    assert proxy.data_quality.status == "ok"
    assert proxy.evidence["spread_source"] == "tlt_total_return_differential"
    assert (
        proxy.evidence["bias_warning_code"]
        == "credit_spread_proxy_total_return_differential"
    )
    proxy_rule = proxy.evidence["rule_evidence"]
    assert proxy_rule["hy_spread_percentile_504d"] == pytest.approx(0.31746031746031744)
    assert proxy_rule["hy_spread_slope_21d"] == pytest.approx(-0.0005359273106014766)


@pytest.mark.slow
def test_regime_output_carries_real_fixture_credit_funding_state_when_configured(
    v2_market_df_for_asof,
    v2_close_series_by_symbol: dict[str, pd.Series],
    v2_macro_series_by_key: dict[str, pd.Series],
) -> None:
    """End-to-end: real fixture reaches both §2C wire fields.

    Marked ``slow`` because this single test consumes 105-145s under the
    default ``-n auto`` configuration (real V2 fixture + macro + full
    classify_window pipeline). The default suite retains equivalent
    coverage via:
      - ``test_regime_output_carries_credit_funding_state_when_configured``
        (synthetic context, same engine.classify_window path,
        asserts credit_funding_state / _proxy / _effective_state are
        populated with allowed labels).
      - ``test_real_v2_fixture_credit_funding_golden_label`` (real V2
        fixture at axis-series-builder level, asserts exact
        spread_source / rule_evidence numerical values).
    Run this test with ``python3.14 -m pytest -m slow`` (see pytest.ini).
    """
    as_of = _REAL_FIXTURE_CREDIT_AS_OF
    context = _build_real_v2_credit_context(
        as_of,
        v2_market_df_for_asof,
        v2_close_series_by_symbol,
        v2_macro_series_by_key,
    )
    engine = RegimeEngine()
    timeline = engine.classify_window(
        end_date=as_of,
        market_data=v2_market_df_for_asof(as_of),
        lookback_days=1,
        sector_etf_closes=context.sector_etf_closes,
        cross_asset_closes=context.cross_asset_closes,
        macro_series=v2_macro_series_by_key,
    )
    out = timeline.outputs[-1]
    assert out.as_of_date == as_of
    assert out.credit_funding_state is not None
    assert out.credit_funding_state.raw_label == "credit_calm"
    assert out.credit_funding_state.active_label == "credit_calm"
    assert out.credit_funding_state.evidence["spread_source"] == "ice_bofa_oas"
    assert out.credit_funding_state_proxy is not None
    assert out.credit_funding_state_proxy.raw_label == "credit_calm"
    assert out.credit_funding_state_proxy.active_label == "credit_calm"
    assert (
        out.credit_funding_state_proxy.evidence["spread_source"]
        == "tlt_total_return_differential"
    )
    assert out.credit_funding_effective_state is not None
    assert out.credit_funding_effective_state.active_label == "credit_calm"
    assert out.credit_funding_effective_state.evidence["agreement_status"] in {
        "confirmed",
        "divergent",
    }


def test_regime_output_carries_credit_funding_state_when_configured() -> None:
    """End-to-end: classify_window populates RegimeOutput.credit_funding_state."""
    context = _build_full_synthetic_context()
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
    assert out.credit_funding_state is not None
    allowed = set(CREDIT_FUNDING_RISK_RANK.keys())
    assert out.credit_funding_state.active_label in allowed
    # §2C parallel proxy label (Ambiguity Log #71) — emitted alongside the
    # real-OAS label, a distinct CreditFundingOutput, never blended.
    assert out.credit_funding_state_proxy is not None
    assert out.credit_funding_state_proxy is not out.credit_funding_state
    assert out.credit_funding_state_proxy.active_label in allowed
    assert out.credit_funding_effective_state is not None
    assert out.credit_funding_effective_state.active_label in allowed


# --- Group F — Pre-SOFR/IORB splice regression (ADR 0009) --------------------
#
# Regression guard: feature_store.py must pass fedfunds/ioer_legacy through to
# compute_credit_funding_features (feature_store.py:616-617). If those two lines
# are removed, sofr_iorb_spread is all-NaN for pre-SOFR/IORB eras and the axis
# builder emits stale_data for 67% of full history. This has regressed multiple
# times because there was no test guarding the routing.


def test_feature_store_routes_fedfunds_ioer_legacy_to_splice() -> None:
    """feature_store.py must pass fedfunds/ioer_legacy to compute_credit_funding_features.

    Simulates a pre-SOFR/IORB window by zeroing out sofr and iorb in
    macro_series while keeping fedfunds and ioer_legacy. The resulting
    sofr_iorb_spread must be fully non-NaN — the FEDFUNDS-IOER splice filled it.

    Regression: if feature_store.py:616-617 are removed, the splice
    receives fedfunds=None, ioer_legacy=None and sofr_iorb_spread is all-NaN.
    """
    base_context = _build_full_synthetic_context()
    idx = base_context.spy_ohlcv.index

    # Simulate pre-SOFR/IORB: zero out both series so the splice must carry the load.
    nan_series = pd.Series(float("nan"), index=idx, dtype=float)
    fedfunds = pd.Series(0.41, index=idx, dtype=float, name="fedfunds")
    ioer_legacy = pd.Series(0.40, index=idx, dtype=float, name="ioer_legacy")

    patched_macro = dict(base_context.macro_series or {})
    patched_macro["sofr"] = nan_series
    patched_macro["iorb"] = nan_series
    patched_macro["fedfunds"] = fedfunds
    patched_macro["ioer_legacy"] = ioer_legacy

    patched_context = build_market_context(
        end_date=base_context.end_date,
        market_data=pd.DataFrame(
            [
                {
                    "date": ts.date(),
                    "symbol": "SPY",
                    "open": float(base_context.spy_ohlcv["open"].loc[ts]),
                    "high": float(base_context.spy_ohlcv["high"].loc[ts]),
                    "low": float(base_context.spy_ohlcv["low"].loc[ts]),
                    "close": float(base_context.spy_ohlcv["close"].loc[ts]),
                    "volume": float(base_context.spy_ohlcv["volume"].loc[ts]),
                }
                for ts in idx
            ]
            + [
                {
                    "date": ts.date(),
                    "symbol": "RSP",
                    "open": float(base_context.rsp_close.loc[ts]),
                    "high": float(base_context.rsp_close.loc[ts]),
                    "low": float(base_context.rsp_close.loc[ts]),
                    "close": float(base_context.rsp_close.loc[ts]),
                    "volume": 500_000,
                }
                for ts in idx
            ]
        ),
        config=base_context.config,
        sector_etf_closes=base_context.sector_etf_closes,
        cross_asset_closes=base_context.cross_asset_closes,
        macro_series=patched_macro,
    )

    cfg = patched_context.config
    store = build_feature_store(
        patched_context,
        network_fragility_config=cfg.network_fragility,
        credit_funding_config=cfg.credit_funding,
    )
    assert store.credit_funding is not None, "credit_funding seam should be lit"

    spread = store.credit_funding.sofr_iorb_spread
    null_count = spread.isna().sum()
    assert null_count == 0, (
        f"sofr_iorb_spread has {null_count} NaN values — "
        "feature_store.py is not routing fedfunds/ioer_legacy to the splice "
        "(check feature_store.py:616-617 and credit_funding.py:409-416)."
    )
