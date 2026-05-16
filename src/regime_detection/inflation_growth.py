"""v2 §2B Inflation / Growth axis — feature compute + rule materialisation (Slice 5).

Implements the 8-label axis classifier from spec lines 2174-2326. Optional
nowcast and EPS-revision source series are consumed when present and
otherwise falsify via NaN:

  - ``earnings_expansion`` / ``earnings_contraction`` need the weekly
    ``aggregate_forward_eps_revision_direction_4w`` time series from the
    S&P Global aggregate forward-EPS weekly snapshot accumulator.
  - ``inflation_shock``'s single-signal limb
    (``inflation_surprise_zscore > +1.5``) uses the Cleveland Fed nowcast
    substitution when ``cpi_nowcast`` is wired; the composite-shock limb
    remains active.

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
from datetime import date
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from regime_detection._staleness_utils import (
    calendar_staleness_days_series as _calendar_staleness_days_series,
    trading_staleness_series as _trading_staleness_series,
)
from regime_detection.breadth_state_v2 import make_bias_warnings_frame
from collections.abc import Sequence

from regime_detection.config import InflationGrowthConfig, InflationGrowthRulesConfig
from regime_detection.credit_funding import _rolling_ols_slope
from regime_detection.data_quality import assess_series_input_quality, quality_forces_unknown
from regime_detection.hysteresis import apply_per_label_asymmetric_hysteresis
from regime_detection.models import DataQuality, InflationGrowthOutput

if TYPE_CHECKING:
    from regime_detection.feature_store import FeatureStore
    from regime_detection.market_context import MarketContext


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
INFLATION_GROWTH_RISK_RANK: dict[str, int] = {
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

# ADR 0006 — `inflation_surprise_zscore` provenance. The `consensus_estimate`
# term of the §2B surprise formula is substituted with the Cleveland Fed
# inflation nowcast (a free, model-derived current-period CPI rate estimate).
# The bias-warning row flags the surprise as MODEL-relative, not
# survey-relative — emitted only when `cpi_nowcast` is actually wired.
INFLATION_SURPRISE_NOWCAST_BIAS_WARNING_CODE = "inflation_surprise_cleveland_fed_nowcast"
INFLATION_SURPRISE_NOWCAST_BIAS_SOURCE = "cleveland_fed_inflation_nowcast"
INFLATION_SURPRISE_NOWCAST_BIAS_SOURCE_URL = (
    "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
)

# v2 §2A lines 2587-2593 — `value_first_release` provenance row. Emitted
# only when the historical-replay first-release CPI substitution is in
# effect (audit M2 / docs/spec_code_data_audit_2026_05_15.md §3.2).
FIRST_RELEASE_CPI_PROVENANCE_CODE = "cpi_first_release_vintage_replay"
FIRST_RELEASE_CPI_PROVENANCE_SOURCE = "fred_cpiaucsl_realtime_vintages"
FIRST_RELEASE_CPI_PROVENANCE_SOURCE_URL = (
    "https://fred.stlouisfed.org/series/CPIAUCSL"
)


# ---------------------------------------------------------------------------
# Feature dataclass — per-session §2B feature seam.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InflationGrowthFeatures:
    """v2 §2B per-session inflation/growth feature series (Slice 5).

    All series are aligned to the SPY DatetimeIndex. NaN cold-start at the
    head of each series until the corresponding lookback fills.

    The optional series (``inflation_surprise_zscore`` and
    ``aggregate_forward_eps_revision_direction_4w``) are real when their
    source inputs are wired and all-NaN otherwise, preserving the rule
    engine's NaN-falsifies behavior without a feature-store schema change.
    """

    cpi_3m_change_pct: pd.Series
    cpi_6m_change_pct: pd.Series
    cpi_6m_change_pct_slope_21d: pd.Series
    inflation_surprise_zscore: pd.Series
    pmi_manufacturing: pd.Series
    pmi_manufacturing_slope_21d: pd.Series
    aggregate_forward_eps_revision_direction_4w: pd.Series
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


def compute_inflation_surprise_zscore(
    *,
    cpi_all_items: pd.Series,
    cpi_nowcast: pd.Series,
    session_index: pd.DatetimeIndex,
    realized_rate_lookback: int,
    normalizer_window: int,
) -> pd.Series:
    """v2 §2B `inflation_surprise_zscore` from CPI vs the Cleveland Fed
    inflation nowcast (ADR 0006).

        realized_cpi_rate   = `realized_rate_lookback`-session % change of CPIAUCSL
        inflation_surprise  = realized_cpi_rate - cpi_nowcast
        zscore              = inflation_surprise / rolling_std(inflation_surprise,
                                                               normalizer_window)

    Both CPI and the nowcast are monthly series forward-filled onto the
    SPY session index (the daily classifier reads the most-recent-release
    value carried forward — same pattern as `cpi_all_items` already uses).
    The z-score is NaN until a full `normalizer_window` of surprise
    history exists (V1 §2.7 cold-start) and wherever either operand is
    NaN. The 5y normalizer window matches the §2A yield z-score
    convention.
    """
    cpi = cpi_all_items.reindex(session_index).astype(float).ffill()
    nowcast = cpi_nowcast.reindex(session_index).astype(float).ffill()
    realized_cpi_rate = _pct_change_lookback(cpi, realized_rate_lookback)
    inflation_surprise = realized_cpi_rate - nowcast
    rolling_std = inflation_surprise.rolling(
        normalizer_window, min_periods=normalizer_window
    ).std()
    zscore = inflation_surprise / rolling_std.where(rolling_std > 0)
    zscore.name = "inflation_surprise_zscore"
    return zscore


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
    cpi_nowcast: pd.Series | None = None,
    aggregate_forward_eps_revision: pd.Series | None = None,
    cpi_first_release: pd.Series | None = None,
    use_first_release_cpi_when_available: bool = True,
) -> InflationGrowthFeatures:
    """Compute the v2 §2B inflation/growth feature seam from raw inputs.

    All inputs are aligned to ``spy_close.index``. CPI and PMI are monthly
    series and are forward-filled to daily (§2B line 2208 PMI; CPI follows
    the same NFCI-style pattern slice 4 uses).

    ``cpi_nowcast`` (the Cleveland Fed inflation nowcast — ADR 0006) is
    optional. When supplied, ``inflation_surprise_zscore`` is computed
    from the realized CPI rate vs the nowcast and a model-relative
    bias-warning row is emitted; when None, the all-NaN placeholder
    stays and the `inflation_shock` single-signal limb falsifies (V1
    byte-identity preserved).

    ``aggregate_forward_eps_revision`` (the weekly 4-week forward-EPS
    revision-direction series from
    ``regime_data_fetch.aggregate_eps.compute_eps_revision_direction_4w``)
    is optional. When supplied, it is forward-filled onto the SPY session
    index and the `earnings_expansion` / `earnings_contraction` labels
    can fire; when None, the all-NaN placeholder stays and both labels
    falsify. No bias warning — this is S&P's own forward-EPS data, not a
    proxy.
    """
    spy_index = spy_close.index

    # v2 §2A lines 2587-2593 — first-release vs latest-revision CPI for
    # historical replay (audit M2). When the vintage seam is supplied
    # AND the config flag enables the substitution, replace the
    # latest-revision `cpi_all_items` with the release-date-keyed
    # first-release Series. Both series live on the same SPY calendar
    # after the standard reindex/ffill.
    if cpi_first_release is not None and use_first_release_cpi_when_available:
        cpi_source = cpi_first_release
    else:
        cpi_source = cpi_all_items

    # Reindex every input to the SPY calendar so all returned series share
    # the same DatetimeIndex (single source of truth for the rule engine).
    cpi = cpi_source.reindex(spy_index).astype(float).ffill()
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

    # Inflation surprise (ADR 0006 / Log #48 closure). When `cpi_nowcast`
    # (the Cleveland Fed inflation nowcast) is supplied, compute the real
    # z-score; otherwise emit an all-NaN series so the rule engine
    # naturally falsifies the `inflation_shock` single-signal limb.
    if cpi_nowcast is not None:
        inflation_surprise_zscore = compute_inflation_surprise_zscore(
            cpi_all_items=cpi_source,
            cpi_nowcast=cpi_nowcast,
            session_index=spy_index,
            realized_rate_lookback=config.inflation_surprise_realized_rate_lookback_sessions,
            normalizer_window=config.inflation_surprise_normalizer_window_sessions,
        )
    else:
        inflation_surprise_zscore = pd.Series(
            np.nan, index=spy_index, name="inflation_surprise_zscore", dtype=float
        )

    # PMI (§2B lines 2208-2209). 21d slope on the forward-filled daily series.
    pmi_manufacturing_series = pmi.rename("pmi_manufacturing")
    pmi_manufacturing_slope_21d = _rolling_ols_slope(
        pmi_manufacturing_series, window=config.pmi_slope_lookback_sessions
    ).rename("pmi_manufacturing_slope_21d")

    # Aggregate forward EPS revision (Log #48 closure). When the weekly
    # revision series from the EPS accumulator is supplied, forward-fill
    # it onto the SPY session index. `reindex(method="ffill")` carries
    # each weekly value forward even when its observation_date is not
    # itself an NYSE session (the accumulator is keyed by workbook
    # observation_date, not the trading calendar). When None, emit an
    # all-NaN series so `earnings_expansion` / `earnings_contraction`
    # falsify.
    if aggregate_forward_eps_revision is not None:
        eps_revision_sorted = aggregate_forward_eps_revision.astype(float).copy()
        eps_revision_sorted.index = pd.DatetimeIndex(
            pd.to_datetime(eps_revision_sorted.index)
        )
        eps_revision_sorted = eps_revision_sorted.sort_index()
        aggregate_forward_eps_revision_direction_4w = (
            eps_revision_sorted.reindex(spy_index, method="ffill").rename(
                "aggregate_forward_eps_revision_direction_4w"
            )
        )
    else:
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

    bias_rows = [
        {
            "warning_code": COMMODITY_PROXY_BIAS_WARNING_CODE,
            "feature_name": feat,
            "source": COMMODITY_PROXY_BIAS_SOURCE,
            "source_url": COMMODITY_PROXY_BIAS_SOURCE_URL,
        }
        for feat in _BIAS_FEATURE_NAMES
    ]
    # ADR 0006 — emit the model-relative provenance row only when the
    # Cleveland Fed nowcast substitution is actually in effect.
    if cpi_nowcast is not None:
        bias_rows.append(
            {
                "warning_code": INFLATION_SURPRISE_NOWCAST_BIAS_WARNING_CODE,
                "feature_name": "inflation_surprise_zscore",
                "source": INFLATION_SURPRISE_NOWCAST_BIAS_SOURCE,
                "source_url": INFLATION_SURPRISE_NOWCAST_BIAS_SOURCE_URL,
            }
        )
    # v2 §2A lines 2587-2593 — first-release CPI provenance row (audit M2).
    # Surfaces in the feature store output so replay consumers can audit
    # which CPI vintage powered each `as_of_date`.
    if cpi_first_release is not None and use_first_release_cpi_when_available:
        for feat in ("cpi_3m_change_pct", "cpi_6m_change_pct", "inflation_surprise_zscore"):
            bias_rows.append(
                {
                    "warning_code": FIRST_RELEASE_CPI_PROVENANCE_CODE,
                    "feature_name": feat,
                    "source": FIRST_RELEASE_CPI_PROVENANCE_SOURCE,
                    "source_url": FIRST_RELEASE_CPI_PROVENANCE_SOURCE_URL,
                }
            )
    bias_warnings = make_bias_warnings_frame(bias_rows)

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
    inflation_surprise_zscore: float
    aggregate_forward_eps_revision_direction_4w: float
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
        inflation_surprise_zscore=_scalar_at(features.inflation_surprise_zscore, dt),
        aggregate_forward_eps_revision_direction_4w=_scalar_at(
            features.aggregate_forward_eps_revision_direction_4w, dt
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


def build_rule_inputs_by_date(
    *,
    features: InflationGrowthFeatures,
    config: InflationGrowthRulesConfig,
    credit_funding_active_labels_by_date: dict[pd.Timestamp, str | None] | None,
) -> dict[pd.Timestamp, InflationGrowthRuleInputs]:
    index = features.cpi_6m_change_pct.index
    cpi_lag_21 = features.cpi_6m_change_pct.shift(config.cpi_slope_lookback_sessions)
    outputs: dict[pd.Timestamp, InflationGrowthRuleInputs] = {}
    for dt in index:
        credit_funding_active_label = None
        if credit_funding_active_labels_by_date is not None:
            credit_funding_active_label = credit_funding_active_labels_by_date.get(dt)
        outputs[dt] = InflationGrowthRuleInputs(
            cpi_6m_change_pct=_scalar_at(features.cpi_6m_change_pct, dt),
            cpi_6m_change_pct_lag_21=_scalar_at(cpi_lag_21, dt),
            cpi_6m_change_pct_slope_21d=_scalar_at(
                features.cpi_6m_change_pct_slope_21d, dt
            ),
            inflation_surprise_zscore=_scalar_at(
                features.inflation_surprise_zscore, dt
            ),
            aggregate_forward_eps_revision_direction_4w=_scalar_at(
                features.aggregate_forward_eps_revision_direction_4w, dt
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
    return outputs


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
    """v2 §2B lines 2550-2555 — `inflation_shock` two-limb OR rule.

    Fires when EITHER limb is satisfied:

    * Single-signal limb (ADR 0006 / Log #48 closure):
      ``inflation_surprise_zscore > inflation_surprise_zscore_threshold``
      (default +1.5). NaN falsifies this limb — the z-score is NaN when
      `cpi_nowcast` is unwired or during the 5y cold-start, so the limb
      is simply silent then (it does not block the composite limb).
    * Composite limb: commodity return high AND 10y yield rising AND
      equities falling AND bonds falling.
    """
    # Single-signal limb — fires on a large positive (hotter-than-nowcast)
    # inflation surprise. NaN falsifies (does not block the OR).
    if not _any_nan(inputs.inflation_surprise_zscore) and (
        inputs.inflation_surprise_zscore
        > config.inflation_surprise_zscore_threshold
    ):
        return True

    # Composite limb — equities AND bonds both weak under a commodity surge.
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
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B line 2605 — `earnings_expansion`:
    ``aggregate_forward_eps_revision_direction_4w > +0.02`` (strict).

    Log #48 closure: the revision series is built by the EPS weekly-
    snapshot accumulator. NaN falsifies — the label is silent during
    the accumulator cold-start (< 5 weekly fetches) or when the
    revision series is not wired into ``macro_series``.
    """
    if _any_nan(inputs.aggregate_forward_eps_revision_direction_4w):
        return False
    return bool(
        inputs.aggregate_forward_eps_revision_direction_4w
        > config.eps_revision_expansion_threshold
    )


def evaluate_earnings_contraction(
    inputs: InflationGrowthRuleInputs,
    config: InflationGrowthRulesConfig,
) -> bool:
    """v2 §2B line 2609 — `earnings_contraction`:
    ``aggregate_forward_eps_revision_direction_4w < -0.02`` (strict).

    Log #48 closure: NaN falsifies (accumulator cold-start / unwired).
    """
    if _any_nan(inputs.aggregate_forward_eps_revision_direction_4w):
        return False
    return bool(
        inputs.aggregate_forward_eps_revision_direction_4w
        < config.eps_revision_contraction_threshold
    )


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


# ---------------------------------------------------------------------------
# Per-session classification helpers — extracted from build_axis_series.
# ---------------------------------------------------------------------------


def _check_staleness_gate(
    *,
    dt: pd.Timestamp,
    day: date,
    cpi_staleness_by_date: pd.Series,
    pmi_staleness_by_date: pd.Series,
    dgs10_staleness_by_date: pd.Series,
    ig_config: InflationGrowthConfig,
) -> tuple[bool, str]:
    """Return (is_stale, gate_reason). gate_reason is empty when not stale."""
    cpi_staleness_days = int(cpi_staleness_by_date.loc[dt])
    pmi_staleness_days = int(pmi_staleness_by_date.loc[dt])
    dgs10_staleness_sessions = int(dgs10_staleness_by_date.loc[dt])
    cpi_stale = cpi_staleness_days > ig_config.cpi_stale_calendar_days
    pmi_stale = pmi_staleness_days > ig_config.pmi_stale_calendar_days
    dgs10_stale = dgs10_staleness_sessions > ig_config.dgs10_stale_sessions
    if not (cpi_stale or pmi_stale or dgs10_stale):
        return False, ""
    reason_parts: list[str] = []
    if cpi_stale:
        reason_parts.append(f"cpi_stale_{cpi_staleness_days}d")
    if pmi_stale:
        reason_parts.append(f"pmi_stale_{pmi_staleness_days}d")
    if dgs10_stale:
        reason_parts.append(f"dgs10_stale_{dgs10_staleness_sessions}s")
    return True, ",".join(reason_parts)


def _build_inflation_growth_evidence(
    rule_inputs: InflationGrowthRuleInputs,
    credit_funding_active_label: str | None,
) -> dict[str, object]:
    """Build the evidence dict for a single classified session."""
    return {
        "rule_evidence": {
            "cpi_6m_change_pct": rule_inputs.cpi_6m_change_pct,
            "cpi_6m_change_pct_lag_21": rule_inputs.cpi_6m_change_pct_lag_21,
            "cpi_6m_change_pct_slope_21d": rule_inputs.cpi_6m_change_pct_slope_21d,
            "inflation_surprise_zscore": rule_inputs.inflation_surprise_zscore,
            "pmi_manufacturing": rule_inputs.pmi_manufacturing,
            "pmi_manufacturing_slope_21d": rule_inputs.pmi_manufacturing_slope_21d,
            "aggregate_forward_eps_revision_direction_4w": rule_inputs.aggregate_forward_eps_revision_direction_4w,
            "commodity_return_63d": rule_inputs.commodity_return_63d,
            "treasury_10y_yield_slope_21d": rule_inputs.treasury_10y_yield_slope_21d,
            "cyclical_defensive_slope_21d": rule_inputs.cyclical_defensive_slope_21d,
            "spy_21d_return": rule_inputs.spy_21d_return,
            "tlt_21d_return": rule_inputs.tlt_21d_return,
        },
        "credit_funding_active_label": credit_funding_active_label,
        "bias_warning_code": "commodity_proxy_dbc_substitute",
    }


def _classify_inflation_growth_session(
    *,
    day: date,
    dt: pd.Timestamp,
    required_inputs: list[pd.Series],
    required_trading_days: int,
    max_freshness_days: int,
    min_completeness: float,
    credit_funding_active_labels_by_date: dict[date, str] | None,
    rule_inputs_by_date: dict[pd.Timestamp, InflationGrowthRuleInputs],
    ig_config: InflationGrowthConfig,
) -> tuple[InflationGrowthLabel, DataQuality, dict[str, object]]:
    """Classify a single session after the staleness gate has passed.

    Returns (raw_label, data_quality, evidence).
    """
    day_quality = assess_series_input_quality(
        as_of_date=day,
        required_inputs=required_inputs,
        required_trading_days=required_trading_days,
        raw_label="",
        max_freshness_days=max_freshness_days,
        min_completeness=min_completeness,
        skip_raw_label_short_circuit=True,
    )
    if quality_forces_unknown(day_quality):
        return "unknown", day_quality, {"reason": day_quality.reason or "insufficient_data"}

    credit_funding_active_label: str | None = None
    if credit_funding_active_labels_by_date is not None:
        if day not in credit_funding_active_labels_by_date:
            raise KeyError(
                f"credit_funding_active_labels_by_date missing session {day!r} "
                "(v1/v2 calendar drift would silently downgrade §2B cross-axis rules)"
            )
        credit_funding_active_label = credit_funding_active_labels_by_date[day]

    rule_inputs = rule_inputs_by_date[dt]
    label = evaluate_rules(inputs=rule_inputs, config=ig_config.rules)
    return label, day_quality, _build_inflation_growth_evidence(rule_inputs, credit_funding_active_label)


def _accumulate_ig_session_lists(
    *,
    sessions: Sequence[date],
    cpi_staleness_by_date: pd.Series,
    pmi_staleness_by_date: pd.Series,
    dgs10_staleness_by_date: pd.Series,
    ig_config: InflationGrowthConfig,
    required_inputs: list[pd.Series],
    required_trading_days: int,
    max_freshness_days: int,
    min_completeness: float,
    credit_funding_active_labels_by_date: dict[date, str] | None,
    rule_inputs_by_date: dict[pd.Timestamp, InflationGrowthRuleInputs],
) -> tuple[list[InflationGrowthLabel], list[DataQuality], list[dict[str, object]]]:
    """Iterate sessions and return (raw_labels, per_day_dq, per_day_evidence)."""
    raw_labels: list[InflationGrowthLabel] = []
    per_day_data_quality: list[DataQuality] = []
    per_day_evidence: list[dict[str, object]] = []
    for day in sessions:
        dt = pd.Timestamp(day)
        is_stale, gate_reason = _check_staleness_gate(
            dt=dt, day=day,
            cpi_staleness_by_date=cpi_staleness_by_date,
            pmi_staleness_by_date=pmi_staleness_by_date,
            dgs10_staleness_by_date=dgs10_staleness_by_date,
            ig_config=ig_config,
        )
        if is_stale:
            raw_labels.append("unknown")
            per_day_data_quality.append(
                DataQuality(status="stale_data", freshness_days=None, completeness=None, reason=gate_reason)
            )
            per_day_evidence.append({"reason": gate_reason})
            continue
        label, dq, evidence = _classify_inflation_growth_session(
            day=day, dt=dt,
            required_inputs=required_inputs,
            required_trading_days=required_trading_days,
            max_freshness_days=max_freshness_days,
            min_completeness=min_completeness,
            credit_funding_active_labels_by_date=credit_funding_active_labels_by_date,
            rule_inputs_by_date=rule_inputs_by_date,
            ig_config=ig_config,
        )
        raw_labels.append(label)
        per_day_data_quality.append(dq)
        per_day_evidence.append(evidence)
    return raw_labels, per_day_data_quality, per_day_evidence


def _prepare_ig_rule_inputs(
    *,
    features: InflationGrowthFeatures,
    ig_config: InflationGrowthConfig,
    spy_close: pd.Series,
    macro_series: dict,
    credit_funding_active_labels_by_date: dict[date, str] | None,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[pd.Timestamp, InflationGrowthRuleInputs]]:
    """Build staleness series and rule_inputs_by_date for the axis loop."""
    session_index = spy_close.index
    cpi_staleness = _calendar_staleness_days_series(macro_series.get("cpi_all_items"), session_index)
    pmi_staleness = _calendar_staleness_days_series(macro_series.get("pmi_manufacturing"), session_index)
    dgs10_staleness = _trading_staleness_series(macro_series.get("dgs10"), session_index)
    credit_funding_labels_by_ts: dict[pd.Timestamp, str | None] | None = None
    if credit_funding_active_labels_by_date is not None:
        credit_funding_labels_by_ts = {
            pd.Timestamp(d): lbl for d, lbl in credit_funding_active_labels_by_date.items()
        }
    rule_inputs_by_date = build_rule_inputs_by_date(
        features=features, config=ig_config.rules,
        credit_funding_active_labels_by_date=credit_funding_labels_by_ts,
    )
    return cpi_staleness, pmi_staleness, dgs10_staleness, rule_inputs_by_date


def _build_ig_outputs(
    sessions: Sequence[date],
    ig_config: InflationGrowthConfig,
    raw_labels: list[InflationGrowthLabel],
    per_day_dq: list[DataQuality],
    per_day_evidence: list[dict[str, object]],
) -> dict[date, InflationGrowthOutput]:
    """Apply hysteresis and zip per-session lists into the final output dict."""
    stable_labels, active_labels = apply_per_label_asymmetric_hysteresis(
        raw_labels=raw_labels, risk_rank=INFLATION_GROWTH_RISK_RANK,
        deescalation_days_by_label=ig_config.deescalation_days_by_label,
        default_deescalation_days=ig_config.default_deescalation_days,
    )
    return {
        day: InflationGrowthOutput(
            raw_label=raw, stable_label=stable, active_label=active,
            evidence=evidence, data_quality=dq,
        )
        for day, raw, stable, active, dq, evidence in zip(
            sessions, raw_labels, stable_labels, active_labels,
            per_day_dq, per_day_evidence, strict=True,
        )
    }


def build_axis_series(
    context: MarketContext,
    feature_store: FeatureStore,
    credit_funding_active_labels_by_date: dict[date, str] | None = None,
) -> dict[date, InflationGrowthOutput] | None:
    """Free-function replacement for InflationGrowthSeriesClassifier.build()."""
    features = feature_store.inflation_growth
    if features is None:
        return None
    ig_config = context.config.inflation_growth
    if ig_config is None:
        return None
    spy_close = context.spy_ohlcv["close"]
    cpi_staleness, pmi_staleness, dgs10_staleness, rule_inputs_by_date = _prepare_ig_rule_inputs(
        features=features, ig_config=ig_config, spy_close=spy_close,
        macro_series=context.macro_series or {},
        credit_funding_active_labels_by_date=credit_funding_active_labels_by_date,
    )
    raw_labels, per_day_dq, per_day_evidence = _accumulate_ig_session_lists(
        sessions=context.sessions,
        cpi_staleness_by_date=cpi_staleness,
        pmi_staleness_by_date=pmi_staleness,
        dgs10_staleness_by_date=dgs10_staleness,
        ig_config=ig_config,
        required_inputs=[
            features.cpi_6m_change_pct, features.pmi_manufacturing,
            features.treasury_10y_yield_slope_21d, features.commodity_return_63d, spy_close,
        ],
        required_trading_days=ig_config.rules.cpi_lookback_6m_sessions,
        max_freshness_days=context.config.data_quality.max_freshness_days,
        min_completeness=context.config.data_quality.min_completeness,
        credit_funding_active_labels_by_date=credit_funding_active_labels_by_date,
        rule_inputs_by_date=rule_inputs_by_date,
    )
    return _build_ig_outputs(context.sessions, ig_config, raw_labels, per_day_dq, per_day_evidence)
