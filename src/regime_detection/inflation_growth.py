"""v2 §2B Inflation / Growth axis — feature compute + rule materialisation (Slice 5).

Implements the 8-label axis classifier from spec lines 2174-2326. Per
Implementation Ambiguity Log #48 two labels short-circuit to False until
their paid-data dependency lands:

  - ``earnings_expansion`` / ``earnings_contraction`` need the weekly
    ``aggregate_forward_eps_revision_direction_4w`` time series (S&P Global
    weekly EPS revision feed — paid).
  - ``inflation_shock``'s single-signal limb
    (``inflation_surprise_zscore > +1.5``) needs the BLS consensus-vs-
    actual feed; the composite-shock limb remains active.

Labels (§2B lines 2177-2186):
    goldilocks, inflation_shock, disinflation, recession_scare,
    recovery_growth, earnings_expansion, earnings_contraction, unknown

Precedence (§2B line 2190):
    inflation_shock > recession_scare > disinflation > goldilocks >
    recovery_growth > earnings_contraction > earnings_expansion > unknown

Per Ambiguity Log #48: DBC ETF substitutes for the Bloomberg Commodity
Index (paid feed unavailable). The classifier emits a bias-warning row
with code ``commodity_proxy_dbc_substitute``.

Inputs:
  - ``cpi_all_items`` via ``MarketContext.macro_series["cpi_all_items"]``
    (FRED CPIAUCSL monthly → forward-fill to daily).
  - ``pmi_manufacturing`` via ``MarketContext.macro_series["pmi_manufacturing"]``
    (Investing release-history; monthly → forward-fill to daily).
  - ``dgs10`` via ``MarketContext.macro_series["dgs10"]`` (slice 4.1 loader).
  - ``dbc_close`` via ``MarketContext.cross_asset_closes["DBC"]``.
  - ``spy_close`` via ``MarketContext.spy_ohlcv["close"]``.
  - ``tlt_close`` via ``MarketContext.cross_asset_closes["TLT"]``.
  - Sector ETF closes (XLY/XLI/XLP/XLU) via ``MarketContext.cross_asset_closes``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from regime_detection.breadth_state_v2 import make_bias_warnings_frame
from regime_detection.config import InflationGrowthRulesConfig
from regime_detection.credit_funding import _rolling_ols_slope


# ---------------------------------------------------------------------------
# Spec labels (§2B lines 2177-2186) + risk rank (§2B lines 2274-2284).
# ---------------------------------------------------------------------------

InflationGrowthLabel = Literal[
    "goldilocks",
    "inflation_shock",
    "disinflation",
    "recession_scare",
    "recovery_growth",
    "earnings_expansion",
    "earnings_contraction",
    "unknown",
]


# v2 §2B lines 2274-2284 verbatim.
INFLATION_GROWTH_RISK_RANK: dict[InflationGrowthLabel, int] = {
    "goldilocks": 0,
    "recovery_growth": 0,
    "earnings_expansion": 0,
    "unknown": 1,
    "disinflation": 1,
    "earnings_contraction": 2,
    "recession_scare": 3,
    "inflation_shock": 3,
}


# v2 §2B line 2190 precedence (highest-severity-first walk).
RULE_PRECEDENCE: tuple[InflationGrowthLabel, ...] = (
    "inflation_shock",
    "recession_scare",
    "disinflation",
    "goldilocks",
    "recovery_growth",
    "earnings_contraction",
    "earnings_expansion",
)


# ---------------------------------------------------------------------------
# Required input keys. Pinned here as single source of truth.
# ---------------------------------------------------------------------------

CPI_KEY = "cpi_all_items"
PMI_KEY = "pmi_manufacturing"
DGS10_KEY = "dgs10"
DBC_KEY = "DBC"
TLT_KEY = "TLT"
XLY_KEY = "XLY"
XLI_KEY = "XLI"
XLP_KEY = "XLP"
XLU_KEY = "XLU"

REQUIRED_CROSS_ASSET_KEYS: tuple[str, ...] = (
    DBC_KEY,
    TLT_KEY,
    XLY_KEY,
    XLI_KEY,
    XLP_KEY,
    XLU_KEY,
)
REQUIRED_MACRO_KEYS: tuple[str, ...] = (CPI_KEY, PMI_KEY, DGS10_KEY)


# ---------------------------------------------------------------------------
# Bias-warning constants (§2B Ambiguity Log #48 — DBC substitute for the
# paid Bloomberg Commodity Index).
# ---------------------------------------------------------------------------

COMMODITY_PROXY_BIAS_WARNING_CODE = "commodity_proxy_dbc_substitute"
COMMODITY_PROXY_BIAS_SOURCE = "dbc_etf_close_total_return_substitute"
COMMODITY_PROXY_BIAS_SOURCE_URL = "internal:dbc_etf_close_total_return_substitute"

_BIAS_FEATURE_NAMES: tuple[str, ...] = ("commodity_return_63d",)


# ---------------------------------------------------------------------------
# Feature dataclass — per-session §2B feature seam.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InflationGrowthFeatures:
    """v2 §2B per-session inflation/growth feature series (Slice 5).

    All series are aligned to the SPY DatetimeIndex. NaN cold-start at the
    head of each series until the corresponding lookback fills.

    The two paid-data series (``inflation_surprise_zscore`` and
    ``aggregate_forward_eps_revision_direction_4w``) are exposed as all-NaN
    series per Ambiguity Log #48 so consumers can wire them in once the
    feeds land without a feature-store schema change.
    """

    cpi_3m_change_pct: pd.Series
    cpi_6m_change_pct: pd.Series
    cpi_6m_change_pct_slope_21d: pd.Series
    inflation_surprise_zscore: pd.Series  # all-NaN; deferred (Log #48)
    pmi_manufacturing: pd.Series
    pmi_manufacturing_slope_21d: pd.Series
    aggregate_forward_eps_revision_direction_4w: pd.Series  # all-NaN; deferred
    commodity_return_63d: pd.Series
    treasury_10y_yield_slope_21d: pd.Series
    cyclical_defensive_ratio: pd.Series
    cyclical_defensive_slope_21d: pd.Series
    spy_21d_return: pd.Series
    tlt_21d_return: pd.Series
    bias_warnings: pd.DataFrame

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "cpi_3m_change_pct",
            "cpi_6m_change_pct",
            "cpi_6m_change_pct_slope_21d",
            "inflation_surprise_zscore",
            "pmi_manufacturing",
            "pmi_manufacturing_slope_21d",
            "aggregate_forward_eps_revision_direction_4w",
            "commodity_return_63d",
            "treasury_10y_yield_slope_21d",
            "cyclical_defensive_ratio",
            "cyclical_defensive_slope_21d",
            "spy_21d_return",
            "tlt_21d_return",
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: getattr(self, name) for name in self.feature_names}
        )


# ---------------------------------------------------------------------------
# Feature compute (§2B lines 2193-2228).
# ---------------------------------------------------------------------------


def _pct_change_lookback(series: pd.Series, lookback: int) -> pd.Series:
    base = series.shift(lookback)
    return (series - base) / base.where(base != 0)


def compute_inflation_growth_features(
    *,
    cpi_all_items: pd.Series,
    pmi_manufacturing: pd.Series,
    dgs10: pd.Series,
    dbc_close: pd.Series,
    spy_close: pd.Series,
    tlt_close: pd.Series,
    xly_close: pd.Series,
    xli_close: pd.Series,
    xlp_close: pd.Series,
    xlu_close: pd.Series,
    config: InflationGrowthRulesConfig,
) -> InflationGrowthFeatures:
    """Compute the v2 §2B inflation/growth feature seam from raw inputs.

    All inputs are aligned to ``spy_close.index``. CPI and PMI are monthly
    series and are forward-filled to daily (§2B line 2208 PMI; CPI follows
    the same NFCI-style pattern slice 4 uses).
    """
    spy_index = spy_close.index

    # Reindex every input to the SPY calendar so all returned series share
    # the same DatetimeIndex (single source of truth for the rule engine).
    cpi = cpi_all_items.reindex(spy_index).astype(float).ffill()
    pmi = pmi_manufacturing.reindex(spy_index).astype(float).ffill()
    dgs10_s = dgs10.reindex(spy_index).astype(float)
    dbc = dbc_close.reindex(spy_index).astype(float)
    spy = spy_close.reindex(spy_index).astype(float)
    tlt = tlt_close.reindex(spy_index).astype(float)
    xly = xly_close.reindex(spy_index).astype(float)
    xli = xli_close.reindex(spy_index).astype(float)
    xlp = xlp_close.reindex(spy_index).astype(float)
    xlu = xlu_close.reindex(spy_index).astype(float)

    # CPI trend (§2B lines 2197-2198).
    cpi_3m_change_pct = _pct_change_lookback(
        cpi, config.cpi_lookback_3m_sessions
    ).rename("cpi_3m_change_pct")
    cpi_6m_change_pct = _pct_change_lookback(
        cpi, config.cpi_lookback_6m_sessions
    ).rename("cpi_6m_change_pct")
    # §2B line 2235 — 21d OLS slope of cpi_6m_change_pct.
    cpi_6m_change_pct_slope_21d = _rolling_ols_slope(
        cpi_6m_change_pct, window=config.cpi_slope_lookback_sessions
    ).rename("cpi_6m_change_pct_slope_21d")

    # Inflation surprise — deferred per Log #48; emit as all-NaN series so
    # the rule engine naturally falsifies the single-signal limb.
    inflation_surprise_zscore = pd.Series(
        np.nan, index=spy_index, name="inflation_surprise_zscore", dtype=float
    )

    # PMI (§2B lines 2208-2209). 21d slope on the forward-filled daily series.
    pmi_manufacturing_series = pmi.rename("pmi_manufacturing")
    pmi_manufacturing_slope_21d = _rolling_ols_slope(
        pmi_manufacturing_series, window=config.pmi_slope_lookback_sessions
    ).rename("pmi_manufacturing_slope_21d")

    # Aggregate forward EPS revision — deferred per Log #48; all-NaN.
    aggregate_forward_eps_revision_direction_4w = pd.Series(
        np.nan,
        index=spy_index,
        name="aggregate_forward_eps_revision_direction_4w",
        dtype=float,
    )

    # Commodity return (§2B line 2220) — DBC 63d total return.
    commodity_return_63d = (
        (dbc / dbc.shift(config.commodity_return_lookback_sessions)) - 1.0
    ).rename("commodity_return_63d")

    # Treasury yield slope (§2B line 2223).
    treasury_10y_yield_slope_21d = _rolling_ols_slope(
        dgs10_s, window=config.treasury_slope_lookback_sessions
    ).rename("treasury_10y_yield_slope_21d")

    # Cyclical vs defensive (§2B lines 2225-2227).
    cyclical_sum = xly + xli
    defensive_sum = xlp + xlu
    cyclical_defensive_ratio = (
        cyclical_sum / defensive_sum.where(defensive_sum != 0)
    ).rename("cyclical_defensive_ratio")
    cyclical_defensive_slope_21d = _rolling_ols_slope(
        cyclical_defensive_ratio,
        window=config.cyclical_defensive_slope_lookback_sessions,
    ).rename("cyclical_defensive_slope_21d")

    # SPY / TLT 21d returns (§2B lines 2237 / 2245).
    spy_21d_return = (
        (spy / spy.shift(config.spy_return_lookback_sessions)) - 1.0
    ).rename("spy_21d_return")
    tlt_21d_return = (
        (tlt / tlt.shift(config.tlt_return_lookback_sessions)) - 1.0
    ).rename("tlt_21d_return")

    bias_warnings = make_bias_warnings_frame(
        [
            {
                "warning_code": COMMODITY_PROXY_BIAS_WARNING_CODE,
                "feature_name": feat,
                "source": COMMODITY_PROXY_BIAS_SOURCE,
                "source_url": COMMODITY_PROXY_BIAS_SOURCE_URL,
            }
            for feat in _BIAS_FEATURE_NAMES
        ]
    )

    return InflationGrowthFeatures(
        cpi_3m_change_pct=cpi_3m_change_pct,
        cpi_6m_change_pct=cpi_6m_change_pct,
        cpi_6m_change_pct_slope_21d=cpi_6m_change_pct_slope_21d,
        inflation_surprise_zscore=inflation_surprise_zscore,
        pmi_manufacturing=pmi_manufacturing_series,
        pmi_manufacturing_slope_21d=pmi_manufacturing_slope_21d,
        aggregate_forward_eps_revision_direction_4w=aggregate_forward_eps_revision_direction_4w,
        commodity_return_63d=commodity_return_63d,
        treasury_10y_yield_slope_21d=treasury_10y_yield_slope_21d,
        cyclical_defensive_ratio=cyclical_defensive_ratio,
        cyclical_defensive_slope_21d=cyclical_defensive_slope_21d,
        spy_21d_return=spy_21d_return,
        tlt_21d_return=tlt_21d_return,
        bias_warnings=bias_warnings,
    )


# ---------------------------------------------------------------------------
# Per-day scalar rule inputs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InflationGrowthRuleInputs:
    """Per-day scalars consumed by the §2B rule predicates.

    ``credit_funding_active_label`` carries the cross-axis dependency from
    §2C; ``None`` signals the §2C axis is unbuilt (cross-axis short-circuit
    per spec lines 2314-2316).
    """

    cpi_6m_change_pct: float
    cpi_6m_change_pct_lag_21: float
    cpi_6m_change_pct_slope_21d: float
    pmi_manufacturing: float
    pmi_manufacturing_slope_21d: float
    commodity_return_63d: float
    treasury_10y_yield_slope_21d: float
    cyclical_defensive_slope_21d: float
    spy_21d_return: float
    tlt_21d_return: float
    credit_funding_active_label: str | None


def _scalar_at(series: pd.Series, dt: pd.Timestamp) -> float:
    if dt not in series.index:
        return float("nan")
    val = series.loc[dt]
    if pd.isna(val):
        return float("nan")
    return float(val)


def _scalar_at_lag(series: pd.Series, dt: pd.Timestamp, lag: int) -> float:
    if dt not in series.index:
        return float("nan")
    pos = series.index.get_loc(dt)
    if pos - lag < 0:
        return float("nan")
    val = series.iloc[pos - lag]
    if pd.isna(val):
        return float("nan")
    return float(val)


def build_rule_inputs_for_date(
    *,
    features: InflationGrowthFeatures,
    dt: pd.Timestamp,
    config: InflationGrowthRulesConfig,
    credit_funding_active_label: str | None,
) -> InflationGrowthRuleInputs:
    """Materialize the per-day scalar rule inputs at session ``dt``."""
    return InflationGrowthRuleInputs(
        cpi_6m_change_pct=_scalar_at(features.cpi_6m_change_pct, dt),
        cpi_6m_change_pct_lag_21=_scalar_at_lag(
            features.cpi_6m_change_pct, dt, config.cpi_slope_lookback_sessions
        ),
        cpi_6m_change_pct_slope_21d=_scalar_at(
            features.cpi_6m_change_pct_slope_21d, dt
        ),
        pmi_manufacturing=_scalar_at(features.pmi_manufacturing, dt),
        pmi_manufacturing_slope_21d=_scalar_at(
            features.pmi_manufacturing_slope_21d, dt
        ),
        commodity_return_63d=_scalar_at(features.commodity_return_63d, dt),
        treasury_10y_yield_slope_21d=_scalar_at(
            features.treasury_10y_yield_slope_21d, dt
        ),
        cyclical_defensive_slope_21d=_scalar_at(
            features.cyclical_defensive_slope_21d, dt
        ),
        spy_21d_return=_scalar_at(features.spy_21d_return, dt),
        tlt_21d_return=_scalar_at(features.tlt_21d_return, dt),
        credit_funding_active_label=credit_funding_active_label,
    )


# ---------------------------------------------------------------------------
# Rule predicates (§2B lines 2232-2270).
# ---------------------------------------------------------------------------


def _any_nan(*values: float) -> bool:
    return any(np.isnan(v) for v in values)


def evaluate_goldilocks(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 2233-2238.

    ``(abs(cpi_6m_change_pct[t] - cpi_6m_change_pct[t-21]) <= 0.005
       OR cpi_6m_change_pct_slope_21d <= 0)
       AND pmi_manufacturing > 50
       AND spy_21d_return > 0
       AND credit_funding.active_label == "credit_calm"``
    """
    # Cross-axis short-circuit (§2B line 2316).
    if inputs.credit_funding_active_label is None:
        return False
    if inputs.credit_funding_active_label != "credit_calm":
        return False
    if _any_nan(
        inputs.pmi_manufacturing,
        inputs.spy_21d_return,
    ):
        return False
    # CPI drift OR slope leg. NaN in BOTH cpi components → no signal.
    drift_ok = False
    if not _any_nan(inputs.cpi_6m_change_pct, inputs.cpi_6m_change_pct_lag_21):
        drift_ok = (
            abs(inputs.cpi_6m_change_pct - inputs.cpi_6m_change_pct_lag_21)
            <= config.cpi_drift_threshold
        )
    slope_ok = False
    if not np.isnan(inputs.cpi_6m_change_pct_slope_21d):
        slope_ok = inputs.cpi_6m_change_pct_slope_21d <= 0.0
    if not (drift_ok or slope_ok):
        return False
    return bool(
        inputs.pmi_manufacturing > config.pmi_goldilocks_threshold
        and inputs.spy_21d_return > 0.0
    )


def evaluate_inflation_shock(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 2240-2245 (composite limb only).

    The single-signal limb (``inflation_surprise_zscore > +1.5``) short-
    circuits to False per spec line 2320 + Ambiguity Log #48 (BLS consensus
    feed not ingested).
    """
    if _any_nan(
        inputs.commodity_return_63d,
        inputs.treasury_10y_yield_slope_21d,
        inputs.spy_21d_return,
        inputs.tlt_21d_return,
    ):
        return False
    return bool(
        inputs.commodity_return_63d > config.commodity_return_threshold
        and inputs.treasury_10y_yield_slope_21d > 0.0
        and inputs.spy_21d_return < 0.0
        and inputs.tlt_21d_return < 0.0
    )


def evaluate_disinflation(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 2247-2250."""
    if _any_nan(
        inputs.cpi_6m_change_pct_slope_21d,
        inputs.treasury_10y_yield_slope_21d,
        inputs.pmi_manufacturing,
    ):
        return False
    return bool(
        inputs.cpi_6m_change_pct_slope_21d < 0.0
        and inputs.treasury_10y_yield_slope_21d < 0.0
        and inputs.pmi_manufacturing > config.pmi_disinflation_threshold
    )


def evaluate_recession_scare(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 2252-2256."""
    if inputs.credit_funding_active_label is None:
        return False
    if inputs.credit_funding_active_label not in {"spread_widening", "credit_stress"}:
        return False
    if _any_nan(
        inputs.treasury_10y_yield_slope_21d,
        inputs.cyclical_defensive_slope_21d,
        inputs.spy_21d_return,
    ):
        return False
    return bool(
        inputs.treasury_10y_yield_slope_21d < 0.0
        and inputs.cyclical_defensive_slope_21d < 0.0
        and inputs.spy_21d_return < config.spy_recession_threshold
    )


def evaluate_recovery_growth(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B lines 2258-2261."""
    if inputs.credit_funding_active_label is None:
        return False
    if inputs.credit_funding_active_label != "credit_calm":
        return False
    if _any_nan(
        inputs.pmi_manufacturing_slope_21d,
        inputs.pmi_manufacturing,
        inputs.cyclical_defensive_slope_21d,
    ):
        return False
    return bool(
        inputs.pmi_manufacturing_slope_21d > 0.0
        and inputs.pmi_manufacturing > config.pmi_recovery_threshold
        and inputs.cyclical_defensive_slope_21d > 0.0
    )


def evaluate_earnings_expansion(
    inputs: InflationGrowthRuleInputs,  # noqa: ARG001
    config: InflationGrowthRulesConfig,  # noqa: ARG001
) -> bool:
    """v2 §2B lines 2263-2265. Short-circuits to False until weekly EPS
    revision time series ships (Ambiguity Log #48)."""
    return False


def evaluate_earnings_contraction(
    inputs: InflationGrowthRuleInputs,  # noqa: ARG001
    config: InflationGrowthRulesConfig,  # noqa: ARG001
) -> bool:
    """v2 §2B lines 2267-2269. Short-circuits to False (Ambiguity Log #48)."""
    return False


def evaluate_rules(
    *,
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> InflationGrowthLabel:
    """Walk v2 §2B precedence and return the first matching label.

    Falls through to ``unknown`` when no rule fires.
    """
    if evaluate_inflation_shock(inputs, config):
        return "inflation_shock"
    if evaluate_recession_scare(inputs, config):
        return "recession_scare"
    if evaluate_disinflation(inputs, config):
        return "disinflation"
    if evaluate_goldilocks(inputs, config):
        return "goldilocks"
    if evaluate_recovery_growth(inputs, config):
        return "recovery_growth"
    if evaluate_earnings_contraction(inputs, config):
        return "earnings_contraction"
    if evaluate_earnings_expansion(inputs, config):
        return "earnings_expansion"
    return "unknown"
