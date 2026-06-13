"""Macro-axis feature spec resolvers for the feature store."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from regime_detection.central_bank_text import to_daily_score_series
from regime_detection.config import (
    CentralBankTextConfig,
    CreditFundingConfig,
    InflationGrowthConfig,
    MonetaryPressureV2FeaturesConfig,
)
from regime_detection.credit_funding_rules import (
    BROAD_USD_INDEX_KEY as CF_BROAD_USD_KEY,
    FEDFUNDS_KEY as CF_FEDFUNDS_KEY,
    HYG_KEY as CF_HYG_KEY,
    HY_OAS_KEY as CF_HY_OAS_KEY,
    IG_OAS_KEY as CF_IG_OAS_KEY,
    IOER_LEGACY_KEY as CF_IOER_LEGACY_KEY,
    IORB_KEY as CF_IORB_KEY,
    KRE_KEY as CF_KRE_KEY,
    LQD_KEY as CF_LQD_KEY,
    NFCI_KEY as CF_NFCI_KEY,
    REQUIRED_CROSS_ASSET_KEYS as CF_CROSS_ASSET_KEYS,
    REQUIRED_MACRO_KEYS as CF_MACRO_KEYS,
    SOFR_KEY as CF_SOFR_KEY,
    TLT_KEY as CF_TLT_KEY,
)
from regime_detection.feature_store_runtime import (
    _Unavailable,
    _require_build_input,
)
from regime_detection.inflation_growth_rules import (
    AGG_FORWARD_EPS_REVISION_KEY as IG_AGG_FORWARD_EPS_REVISION_KEY,
    CPI_KEY as IG_CPI_KEY,
    CPI_NOWCAST_KEY as IG_CPI_NOWCAST_KEY,
    DBC_KEY as IG_DBC_KEY,
    DGS10_KEY as IG_DGS10_KEY,
    PMI_KEY as IG_PMI_KEY,
    REQUIRED_CROSS_ASSET_KEYS as IG_CROSS_ASSET_KEYS,
    REQUIRED_MACRO_KEYS as IG_MACRO_KEYS,
    TLT_KEY as IG_TLT_KEY,
    XLI_KEY as IG_XLI_KEY,
    XLP_KEY as IG_XLP_KEY,
    XLU_KEY as IG_XLU_KEY,
    XLY_KEY as IG_XLY_KEY,
)
from regime_detection.market_context import MarketContext

FRED_DGS2_KEY = "2y_yield"


class MacroFeatureState(Protocol):
    context: MarketContext
    spy_close: pd.Series
    monetary_pressure_v2_config: MonetaryPressureV2FeaturesConfig | None
    credit_funding_config: CreditFundingConfig | None
    inflation_growth_config: InflationGrowthConfig | None
    central_bank_text_config: CentralBankTextConfig | None


def missing_macro_keys(
    macro_series: dict[str, pd.Series] | None, required_keys: tuple[str, ...]
) -> tuple[str, ...]:
    if macro_series is None:
        return ("macro_series",) + required_keys
    return tuple(key for key in required_keys if key not in macro_series)


def missing_cross_asset_keys(
    cross_asset_closes: dict[str, pd.Series] | None, required_keys: tuple[str, ...]
) -> tuple[str, ...]:
    if cross_asset_closes is None:
        return ("cross_asset_closes",) + required_keys
    return tuple(key for key in required_keys if key not in cross_asset_closes)


def resolve_credit_funding(
    state: MacroFeatureState,
) -> dict[str, object] | _Unavailable:
    missing: list[str] = []
    if state.credit_funding_config is None:
        missing.append("credit_funding_config")
        return _Unavailable(missing_inputs=tuple(missing))
    cross_missing = missing_cross_asset_keys(
        state.context.cross_asset_closes, tuple(CF_CROSS_ASSET_KEYS)
    )
    macro_missing = missing_macro_keys(
        state.context.macro_series, tuple(CF_MACRO_KEYS)
    )
    missing.extend(cross_missing)
    missing.extend(macro_missing)
    if missing:
        return _Unavailable(missing_inputs=tuple(missing))
    cross_asset_closes = _require_build_input(
        state.context.cross_asset_closes, "cross_asset_closes"
    )
    macro_series = _require_build_input(state.context.macro_series, "macro_series")
    nan_oas = pd.Series(float("nan"), index=state.spy_close.index)
    return {
        "hyg_close": cross_asset_closes[CF_HYG_KEY],
        "lqd_close": cross_asset_closes[CF_LQD_KEY],
        "tlt_close": cross_asset_closes[CF_TLT_KEY],
        "kre_close": cross_asset_closes[CF_KRE_KEY],
        "spy_close": state.spy_close,
        "sofr": macro_series[CF_SOFR_KEY],
        "iorb": macro_series[CF_IORB_KEY],
        "nfci_weekly": macro_series[CF_NFCI_KEY],
        "broad_usd_index": macro_series[CF_BROAD_USD_KEY],
        "hy_oas": macro_series.get(CF_HY_OAS_KEY, nan_oas),
        "ig_oas": macro_series.get(CF_IG_OAS_KEY, nan_oas),
        "config": state.credit_funding_config.rules,
        "fedfunds": macro_series.get(CF_FEDFUNDS_KEY),
        "ioer_legacy": macro_series.get(CF_IOER_LEGACY_KEY),
    }


def resolve_inflation_growth(
    state: MacroFeatureState,
) -> dict[str, object] | _Unavailable:
    missing: list[str] = []
    if state.inflation_growth_config is None:
        missing.append("inflation_growth_config")
        return _Unavailable(missing_inputs=tuple(missing))
    cross_missing = missing_cross_asset_keys(
        state.context.cross_asset_closes, tuple(IG_CROSS_ASSET_KEYS)
    )
    macro_missing = missing_macro_keys(
        state.context.macro_series, tuple(IG_MACRO_KEYS)
    )
    missing.extend(cross_missing)
    missing.extend(macro_missing)
    if missing:
        return _Unavailable(missing_inputs=tuple(missing))
    cross_asset_closes = _require_build_input(
        state.context.cross_asset_closes, "cross_asset_closes"
    )
    macro_series = _require_build_input(state.context.macro_series, "macro_series")
    return {
        "cpi_all_items": macro_series[IG_CPI_KEY],
        "pmi_manufacturing": macro_series[IG_PMI_KEY],
        "dgs10": macro_series[IG_DGS10_KEY],
        "dbc_close": cross_asset_closes[IG_DBC_KEY],
        "spy_close": state.spy_close,
        "tlt_close": cross_asset_closes[IG_TLT_KEY],
        "xly_close": cross_asset_closes[IG_XLY_KEY],
        "xli_close": cross_asset_closes[IG_XLI_KEY],
        "xlp_close": cross_asset_closes[IG_XLP_KEY],
        "xlu_close": cross_asset_closes[IG_XLU_KEY],
        "config": state.inflation_growth_config.rules,
        "cpi_nowcast": macro_series.get(IG_CPI_NOWCAST_KEY),
        "aggregate_forward_eps_revision": macro_series.get(
            IG_AGG_FORWARD_EPS_REVISION_KEY
        ),
        "cpi_first_release": state.context.cpi_first_release,
        "use_first_release_cpi_when_available": (
            state.inflation_growth_config.rules.use_first_release_cpi_when_available
        ),
    }


def resolve_monetary(
    state: MacroFeatureState,
) -> dict[str, object] | _Unavailable:
    if state.monetary_pressure_v2_config is None:
        return _Unavailable(missing_inputs=())
    macro_missing = missing_macro_keys(
        state.context.macro_series,
        (FRED_DGS2_KEY, IG_DGS10_KEY, "broad_usd_index"),
    )
    if macro_missing:
        return _Unavailable(
            missing_inputs=tuple(macro_missing),
            policy_override="raise",
        )
    macro_series = _require_build_input(state.context.macro_series, "macro_series")
    cb_text_score_series: pd.Series | None = None
    if state.central_bank_text_config is not None:
        if (
            state.context.central_bank_text_releases is None
            or state.context.central_bank_text_releases.empty
        ):
            return _Unavailable(
                missing_inputs=("central_bank_text_releases",),
                policy_override="raise",
            )
        cb_text_score_series = to_daily_score_series(
            state.context.central_bank_text_releases,
            session_index=_as_datetime_index(state.spy_close.index),
            smoothing_window_sessions=state.central_bank_text_config.smoothing_window_sessions,
            same_date_aggregation=state.central_bank_text_config.same_date_aggregation,
            max_release_age_days=state.central_bank_text_config.max_release_age_days,
        )
    return {
        "dgs2": macro_series[FRED_DGS2_KEY],
        "dgs10": macro_series[IG_DGS10_KEY],
        "broad_usd_index": macro_series["broad_usd_index"],
        "central_bank_text_score": cb_text_score_series,
        "config": state.monetary_pressure_v2_config,
    }


def _as_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise RuntimeError("feature store requires a DatetimeIndex-backed SPY frame")
    return index
