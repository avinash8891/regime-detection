"""v2 §2B Inflation / Growth axis — feature compute + rule materialisation.

Implements the 11-label axis classifier from V2 §2B (spec lines 2961-3170).
Optional nowcast and EPS-revision source series are consumed when present and
otherwise falsify via NaN:

  - ``earnings_expansion`` / ``earnings_contraction`` need the weekly
    ``aggregate_forward_eps_revision_direction_4w`` time series from the
    S&P Global aggregate forward-EPS weekly snapshot accumulator.
  - ``inflation_shock``'s single-signal limb
    (``inflation_surprise_zscore > +1.5``) uses the Cleveland Fed nowcast
    substitution when ``cpi_nowcast`` is wired; the composite-shock limb
    remains active.

Labels (V2 §2B spec lines 2965-2975):
    goldilocks, inflation_shock, disinflation, recession_scare,
    risk_off_mild, recovery_growth, reflation, stagflation_lite,
    earnings_expansion, earnings_contraction, unknown

Precedence (V2 §2B spec line 2980):
    inflation_shock > recession_scare > risk_off_mild > disinflation >
    goldilocks > recovery_growth > reflation > stagflation_lite >
    earnings_contraction > earnings_expansion > unknown

Per implementation decision: DBC ETF substitutes for the Bloomberg Commodity
Index (paid feed unavailable). The classifier emits a bias-warning row
with code ``commodity_proxy_dbc_substitute``.

Inputs:
  - ``cpi_all_items`` via ``MarketContext.macro_series["cpi_all_items"]``
    (FRED CPIAUCSL monthly → forward-fill to daily).
  - ``pmi_manufacturing`` via ``MarketContext.macro_series["pmi_manufacturing"]``
    (Investing release-history; monthly → forward-fill to daily).
  - ``dgs10`` via ``MarketContext.macro_series["10y_yield"]`` (FRED DGS10).
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

from regime_detection._series_alignment import aligned_float_values
from regime_shared.pit_provenance import make_bias_warnings_frame
from regime_detection.config import InflationGrowthRulesConfig
from regime_detection._rolling_stats import rolling_ols_slope
from regime_detection._rule_helpers import (
    scalar_at as _scalar_at,
    scalar_at_lag as _scalar_at_lag,
)

# ---------------------------------------------------------------------------
# Spec labels (V2 §2B spec lines 2965-2975) + risk rank (V2 §2B spec lines 3109-3124).
# ---------------------------------------------------------------------------

InflationGrowthLabel = Literal[
    "goldilocks",
    "inflation_shock",
    "disinflation",
    "recession_scare",
    "risk_off_mild",
    "recovery_growth",
    "recovery_growth_unconfirmed",
    "reflation",
    "late_cycle_inflation_stress",
    "stagflation_lite",
    "contractionary_disinflation",
    "macro_neutral",
    "earnings_expansion",
    "earnings_contraction",
    "unknown",
]


INFLATION_GROWTH_RISK_RANK: dict[InflationGrowthLabel, int] = {
    "goldilocks": 0,
    "recovery_growth": 0,
    "earnings_expansion": 0,
    "recovery_growth_unconfirmed": 1,
    "reflation": 1,
    "macro_neutral": 1,
    "unknown": 1,
    "disinflation": 1,
    "contractionary_disinflation": 2,
    "late_cycle_inflation_stress": 2,
    "stagflation_lite": 2,
    "risk_off_mild": 2,
    "earnings_contraction": 2,
    "recession_scare": 3,
    "inflation_shock": 3,
}


# ---------------------------------------------------------------------------
# Required input keys. Pinned here as single source of truth.
# ---------------------------------------------------------------------------

CPI_KEY = "cpi_all_items"
PMI_KEY = "pmi_manufacturing"
DGS10_KEY = "10y_yield"
DBC_KEY = "DBC"
TLT_KEY = "TLT"
XLY_KEY = "XLY"
XLI_KEY = "XLI"
XLP_KEY = "XLP"
XLU_KEY = "XLU"
CPI_NOWCAST_KEY = "cpi_nowcast"
AGG_FORWARD_EPS_REVISION_KEY = "aggregate_forward_eps_revision"

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
# Bias-warning constants (implementation decision — DBC substitute for the
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
INFLATION_SURPRISE_NOWCAST_BIAS_WARNING_CODE = (
    "inflation_surprise_cleveland_fed_nowcast"
)
INFLATION_SURPRISE_NOWCAST_BIAS_SOURCE = "cleveland_fed_inflation_nowcast"
INFLATION_SURPRISE_NOWCAST_BIAS_SOURCE_URL = (
    "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
)

# implementation decision — `value_first_release` provenance row. Emitted
# only when the historical-replay first-release CPI substitution is in
# effect.
FIRST_RELEASE_CPI_PROVENANCE_CODE = "cpi_first_release_vintage_replay"
FIRST_RELEASE_CPI_PROVENANCE_SOURCE = "fred_cpiaucsl_realtime_vintages"
FIRST_RELEASE_CPI_PROVENANCE_SOURCE_URL = "https://fred.stlouisfed.org/series/CPIAUCSL"


# ---------------------------------------------------------------------------
# Feature dataclass — per-session §2B feature seam.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InflationGrowthFeatures:
    """v2 §2B per-session inflation/growth feature series.

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
            "cyclical_defensive_slope_21d",
            "spy_21d_return",
            "tlt_21d_return",
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({name: getattr(self, name) for name in self.feature_names})


# ---------------------------------------------------------------------------
# Feature compute (V2 §2B spec lines 2983-3042).
# ---------------------------------------------------------------------------


def _pct_change_lookback(series: pd.Series, lookback: int) -> pd.Series:
    base = series.shift(lookback)
    return (series - base) / base.where(base != 0)


def _cpi_with_first_release_fallback(
    *,
    latest_cpi: pd.Series,
    first_release_cpi: pd.Series,
    session_index: pd.DatetimeIndex,
) -> pd.Series:
    """Produce the daily CPI **value** series used by feature math.

    Both vintages are reindexed onto ``session_index`` and forward-filled,
    then first-release values take precedence wherever they exist. Earlier
    history (before first-release coverage begins) falls back to latest-
    revision values so cold-start lookbacks don't NaN out.

    Note: this is the value-series path. The matching staleness/release-
    timestamp logic lives in ``axis_builders.inflation_growth.
    _cpi_staleness_source`` — that one preserves the union of observation
    *timestamps* without reindexing, because what it feeds only cares about
    the index, not the values.
    """
    # F-010: forward-fill from the release/observation date (latest reading with date
    # <= each session), matching the AAII/EPS leak-safe pattern. A bare
    # reindex(session_index) lands values only on exact-match dates, so a CPI release on
    # a NYSE-closed day would be dropped and the prior month carried. method="ffill"
    # honors the most-recent on-or-before reading; sort_index guards the requirement.
    latest = (
        latest_cpi.sort_index().reindex(session_index, method="ffill").astype(float)
    )
    first_release = (
        first_release_cpi.sort_index()
        .reindex(session_index, method="ffill")
        .astype(float)
    )
    return first_release.combine_first(latest).rename(latest_cpi.name)


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
    the same NFCI-style forward-fill pattern).

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

    # implementation decision — first-release vs latest-revision CPI for
    # historical replay. When the vintage seam is supplied and enabled, use
    # first-release CPI where it exists, but preserve latest-revision history
    # before vintage coverage begins.
    if cpi_first_release is not None and use_first_release_cpi_when_available:
        cpi_source = _cpi_with_first_release_fallback(
            latest_cpi=cpi_all_items,
            first_release_cpi=cpi_first_release,
            session_index=spy_index,
        )
    else:
        cpi_source = cpi_all_items

    # Reindex every input to the SPY calendar so all returned series share
    # the same DatetimeIndex (single source of truth for the rule engine).
    cpi = cpi_source.reindex(spy_index).astype(float).ffill()
    pmi = pmi_manufacturing.reindex(spy_index).astype(float).ffill()
    dgs10_s = dgs10.reindex(spy_index).astype(float).ffill()
    dbc = dbc_close.reindex(spy_index).astype(float)
    spy = spy_close.reindex(spy_index).astype(float)
    tlt = tlt_close.reindex(spy_index).astype(float)
    xly = xly_close.reindex(spy_index).astype(float)
    xli = xli_close.reindex(spy_index).astype(float)
    xlp = xlp_close.reindex(spy_index).astype(float)
    xlu = xlu_close.reindex(spy_index).astype(float)

    # CPI trend (V2 §2B spec lines 2987-2988).
    # Ambiguity #3 (DECISION): the 3m/6m CPI change is computed as a fixed SESSION
    # offset (cpi_lookback_3m_sessions=63, 6m=126) on the daily forward-filled CPI
    # series — an APPROXIMATION of the exact calendar-month offset, NOT a lookup of the
    # CPI observation exactly 3/6 calendar months prior. 63/126 NYSE sessions ≈ 3/6
    # months (~21 sessions/month). The approximation is intentional: CPI is monthly and
    # forward-filled, so a session-count offset lands on the same monthly vintage as the
    # calendar offset for all but boundary days, while keeping the reducer purely
    # positional (consistent with every other §-window in this engine). Pinned by
    # test_cpi_3m_6m_change_uses_session_offset_approximation.
    cpi_3m_change_pct = _pct_change_lookback(
        cpi, config.cpi_lookback_3m_sessions
    ).rename("cpi_3m_change_pct")
    cpi_6m_change_pct = _pct_change_lookback(
        cpi, config.cpi_lookback_6m_sessions
    ).rename("cpi_6m_change_pct")
    # V2 §2B spec lines 3049 / 3065 — 21d OLS slope of cpi_6m_change_pct (rule operand).
    cpi_6m_change_pct_slope_21d = rolling_ols_slope(
        cpi_6m_change_pct, window=config.cpi_slope_lookback_sessions
    ).rename("cpi_6m_change_pct_slope_21d")

    # Inflation surprise (ADR 0006). When `cpi_nowcast`
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

    # PMI (V2 §2B spec lines 3013-3017). 21d slope on the forward-filled daily series.
    pmi_manufacturing_series = pmi.rename("pmi_manufacturing")
    pmi_manufacturing_slope_21d = rolling_ols_slope(
        pmi_manufacturing_series, window=config.pmi_slope_lookback_sessions
    ).rename("pmi_manufacturing_slope_21d")

    # Aggregate forward EPS revision (implementation decision). When the weekly
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
        eps_revision_sorted = eps_revision_sorted.dropna().sort_index()
        aggregate_forward_eps_revision_direction_4w = eps_revision_sorted.reindex(
            spy_index, method="ffill"
        ).rename("aggregate_forward_eps_revision_direction_4w")
    else:
        aggregate_forward_eps_revision_direction_4w = pd.Series(
            np.nan,
            index=spy_index,
            name="aggregate_forward_eps_revision_direction_4w",
            dtype=float,
        )

    # Commodity return (V2 §2B spec line 3034) — DBC 63d total return.
    commodity_return_63d = (
        (dbc / dbc.shift(config.commodity_return_lookback_sessions)) - 1.0
    ).rename("commodity_return_63d")

    # Treasury yield slope (V2 §2B spec line 3037).
    treasury_10y_yield_slope_21d = rolling_ols_slope(
        dgs10_s, window=config.treasury_slope_lookback_sessions
    ).rename("treasury_10y_yield_slope_21d")

    # Cyclical vs defensive (V2 §2B spec lines 3039-3041).
    cyclical_sum = xly + xli
    defensive_sum = xlp + xlu
    cyclical_defensive_ratio = (
        cyclical_sum / defensive_sum.where(defensive_sum != 0)
    ).rename("cyclical_defensive_ratio")
    cyclical_defensive_slope_21d = rolling_ols_slope(
        cyclical_defensive_ratio,
        window=config.cyclical_defensive_slope_lookback_sessions,
    ).rename("cyclical_defensive_slope_21d")

    # SPY / TLT 21d returns (V2 §2B rule operands — spec lines 3052 / 3060).
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
    # implementation decision — first-release CPI provenance row.
    # Surfaces in the feature store output so replay consumers can audit
    # which CPI vintage powered each `as_of_date`.
    if cpi_first_release is not None and use_first_release_cpi_when_available:
        for feat in (
            "cpi_3m_change_pct",
            "cpi_6m_change_pct",
            "inflation_surprise_zscore",
        ):
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
    per V2 §2B "Cross-Axis Short-Circuit" subsection ~spec line 3159).
    """

    cpi_3m_change_pct: float
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


def build_rule_inputs_for_date(
    *,
    features: InflationGrowthFeatures,
    dt: pd.Timestamp,
    config: InflationGrowthRulesConfig,
    credit_funding_active_label: str | None,
) -> InflationGrowthRuleInputs:
    """Materialize the per-day scalar rule inputs at session ``dt``."""
    return InflationGrowthRuleInputs(
        cpi_3m_change_pct=_scalar_at(features.cpi_3m_change_pct, dt),
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
    cpi_3m_values = aligned_float_values(features.cpi_3m_change_pct, index)
    cpi_6m_values = aligned_float_values(features.cpi_6m_change_pct, index)
    cpi_lag_values = aligned_float_values(cpi_lag_21, index)
    cpi_slope_values = aligned_float_values(features.cpi_6m_change_pct_slope_21d, index)
    inflation_surprise_values = aligned_float_values(
        features.inflation_surprise_zscore, index
    )
    eps_revision_values = aligned_float_values(
        features.aggregate_forward_eps_revision_direction_4w, index
    )
    pmi_values = aligned_float_values(features.pmi_manufacturing, index)
    pmi_slope_values = aligned_float_values(features.pmi_manufacturing_slope_21d, index)
    commodity_return_values = aligned_float_values(features.commodity_return_63d, index)
    treasury_slope_values = aligned_float_values(
        features.treasury_10y_yield_slope_21d, index
    )
    cyclical_defensive_slope_values = aligned_float_values(
        features.cyclical_defensive_slope_21d, index
    )
    spy_return_values = aligned_float_values(features.spy_21d_return, index)
    tlt_return_values = aligned_float_values(features.tlt_21d_return, index)

    outputs: dict[pd.Timestamp, InflationGrowthRuleInputs] = {}
    for pos, dt in enumerate(index):
        credit_funding_active_label = None
        if credit_funding_active_labels_by_date is not None:
            credit_funding_active_label = credit_funding_active_labels_by_date.get(dt)
        outputs[dt] = InflationGrowthRuleInputs(
            cpi_3m_change_pct=float(cpi_3m_values[pos]),
            cpi_6m_change_pct=float(cpi_6m_values[pos]),
            cpi_6m_change_pct_lag_21=float(cpi_lag_values[pos]),
            cpi_6m_change_pct_slope_21d=float(cpi_slope_values[pos]),
            inflation_surprise_zscore=float(inflation_surprise_values[pos]),
            aggregate_forward_eps_revision_direction_4w=float(eps_revision_values[pos]),
            pmi_manufacturing=float(pmi_values[pos]),
            pmi_manufacturing_slope_21d=float(pmi_slope_values[pos]),
            commodity_return_63d=float(commodity_return_values[pos]),
            treasury_10y_yield_slope_21d=float(treasury_slope_values[pos]),
            cyclical_defensive_slope_21d=float(cyclical_defensive_slope_values[pos]),
            spy_21d_return=float(spy_return_values[pos]),
            tlt_21d_return=float(tlt_return_values[pos]),
            credit_funding_active_label=credit_funding_active_label,
        )
    return outputs


from regime_detection.inflation_growth_rules import (  # noqa: E402
    evaluate_disinflation as evaluate_disinflation,
    evaluate_earnings_contraction as evaluate_earnings_contraction,
    evaluate_earnings_expansion as evaluate_earnings_expansion,
    evaluate_goldilocks as evaluate_goldilocks,
    evaluate_inflation_shock as evaluate_inflation_shock,
    evaluate_recession_scare as evaluate_recession_scare,
    evaluate_reflation as evaluate_reflation,
    evaluate_recovery_growth as evaluate_recovery_growth,
    evaluate_risk_off_mild as evaluate_risk_off_mild,
    evaluate_stagflation_lite as evaluate_stagflation_lite,
    evaluate_rules as evaluate_rules,
    goldilocks_limb_evidence as goldilocks_limb_evidence,
)
