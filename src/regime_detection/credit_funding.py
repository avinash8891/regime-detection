"""v2 §2C Credit/Funding axis — feature compute + per-day rule materialisation.

Implements the 5-label axis classifier from spec §2C (lines 3169-3358):

  Labels (§2C lines 3173-3178):
    credit_calm, credit_recovery, credit_mixed, spread_widening, credit_stress,
    funding_squeeze, deleveraging, unknown

  Precedence (§2C line 3183):
    deleveraging > funding_squeeze > credit_stress > spread_widening >
    credit_recovery > credit_calm > credit_mixed > unknown

Credit-spread metrics — two parallel sources (ADR 0007; implementation decision + #71):

  §2C carries two distinct credit-spread metrics:

  1. Real ICE BofA OAS — ``hy_oas_*`` / ``ig_oas_*``, from the
     FRED-redistributed ICE BofA Option-Adjusted Spread series
     (``BAMLH0A0HYM2`` HY Master II OAS, ``BAMLC0A4CBBB`` BBB Corporate
     OAS). The authoritative metric. ``MarketContext.macro_series`` keys
     ``hy_oas`` / ``ig_bbb_oas`` (set by ``V2_FRED_SERIES`` in
     ``regime_data_fetch.fetch_workflow``) feed the ``hy_oas`` / ``ig_oas``
     params of ``compute_credit_funding_features``. FRED exposes only a
     trailing ~3-year window of these series (start 2023-05-15 — implementation decision),
     so the real-OAS label (``credit_funding_state``) is ``unknown``
     before ~2023.

  2. TLT-vs-HYG/LQD total-return-differential proxy — ``hy_tr_differential_*``
     / ``ig_tr_differential_*``, computed from the HYG/LQD/TLT closes. A
     SEPARATE parallel metric covering the full history; it produces its
     own label (``credit_funding_state_proxy``) via the same
     scale-invariant rule schema, and carries a
     ``credit_spread_proxy_total_return_differential`` bias-warning row.
     The proxy exists because FRED's OAS series lack pre-2023 history — it
     is a *similar* measure (credit-spread direction), kept strictly
     parallel at the raw-series level.

  When the OAS series are absent from ``macro_series``, real-OAS features are
  all-NaN and ``credit_funding_state`` emits unknown/data-unavailable. The
  proxy still builds from ETF closes, so ``credit_funding_effective_state`` can
  use proxy fallback with source evidence.

Inputs:
  - HYG, LQD, TLT, KRE close series via ``MarketContext.cross_asset_closes``
  - SOFR, IORB daily FRED series via ``MarketContext.macro_series``
  - NFCI weekly FRED series via ``MarketContext.macro_series`` (forward-filled to daily)
  - ``broad_usd_index`` series via ``MarketContext.macro_series``
  - ``avg_pairwise_corr_percentile_504d`` from FeatureStore.network_fragility
  - ``realized_vol_21d_percentile_252d`` from FeatureStore.volatility (V1 path)

The module also defines:
  - ``CreditFundingFeatures`` dataclass (the per-session feature seam)
  - ``CreditFundingRuleInputs`` dataclass (per-day scalars consumed by predicates)
  - rule predicates ``evaluate_*`` + ``evaluate_rules`` walker (§2C lines 3249-3271)
  - ``CREDIT_FUNDING_RISK_RANK`` + ``CreditFundingLabel`` enum (§2C lines 3277-3283)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from regime_detection._rolling_stats import rolling_change_zscore as _change_zscore
from regime_detection.breadth_state_v2 import make_bias_warnings_frame
from regime_detection.config import (
    CreditFundingRulesConfig,
)

# ---------------------------------------------------------------------------
# Spec labels (§2C lines 3173-3178) + risk rank (§2C lines 3277-3283).
# ---------------------------------------------------------------------------

CreditFundingLabel = Literal[
    "credit_calm",
    "credit_recovery",
    "credit_mixed",
    "spread_widening",
    "credit_stress",
    "funding_squeeze",
    "deleveraging",
    "unknown",
]


# v2 §2C lines 3277-3283 verbatim. ``deleveraging: 4`` is the ONLY V2 axis label
# with risk_rank>3 — reflects that the rule fires only when five distinct stress
# signals coincide (spec line 3286).
CREDIT_FUNDING_RISK_RANK: dict[CreditFundingLabel, int] = {
    "credit_calm": 0,
    "credit_recovery": 0,
    "credit_mixed": 0,
    "unknown": 1,
    "spread_widening": 1,
    "credit_stress": 2,
    "funding_squeeze": 3,
    "deleveraging": 4,
}


# ---------------------------------------------------------------------------
# Required FRED / cross-asset symbol keys. Pinned here as single source of
# truth so feature_store + classifier read one constant.
# ---------------------------------------------------------------------------

HYG_KEY = "HYG"
LQD_KEY = "LQD"
TLT_KEY = "TLT"
KRE_KEY = "KRE"

SOFR_KEY = "sofr"
IORB_KEY = "iorb"
FEDFUNDS_KEY = "fedfunds"
IOER_LEGACY_KEY = "ioer_legacy"
NFCI_KEY = "nfci"
BROAD_USD_INDEX_KEY = "broad_usd_index"
# ICE BofA Option-Adjusted Spread series — FRED-redistributed under ICE
# license, free at the FRED endpoint. macro_series keys set by
# `V2_FRED_SERIES` in `regime_data_fetch.fetch_workflow`.
HY_OAS_KEY = "hy_oas"  # FRED BAMLH0A0HYM2 — ICE BofA US High Yield OAS
IG_OAS_KEY = "ig_bbb_oas"  # FRED BAMLC0A4CBBB — ICE BofA BBB Corporate OAS


REQUIRED_CROSS_ASSET_KEYS: tuple[str, ...] = (HYG_KEY, LQD_KEY, TLT_KEY, KRE_KEY)
REQUIRED_MACRO_KEYS: tuple[str, ...] = (
    SOFR_KEY,
    IORB_KEY,
    NFCI_KEY,
    BROAD_USD_INDEX_KEY,
)


# ---------------------------------------------------------------------------
# Credit-spread provenance (§2C lines 2128-2130).
# ---------------------------------------------------------------------------

# §2C authoritative credit-spread source: ICE BofA Option-Adjusted Spread
# series, FRED-redistributed under ICE license. The parallel TLT-vs-HYG/LQD
# metric below is a separate proxy output, not a fallback or splice into OAS.
CREDIT_SPREAD_SOURCE_CODE = "credit_spread_ice_bofa_oas_fred"
CREDIT_SPREAD_SOURCE = "fred:BAMLH0A0HYM2+BAMLC0A4CBBB"
CREDIT_SPREAD_SOURCE_URL = "https://fred.stlouisfed.org/series/BAMLH0A0HYM2"

# Feature names carrying the OAS provenance row (one row per emitted §2C feature
# derived from the ICE BofA OAS series).
_BIAS_FEATURE_NAMES: tuple[str, ...] = (
    "hy_oas_63d",
    "ig_oas_63d",
    "hy_oas_percentile_504d",
    "hy_oas_slope_21d",
    "ig_oas_slope_21d",
)

# Proxy provenance — the TLT-vs-HYG/LQD total-return-differential metric
# (ADR 0007; implementation decision + #71). Distinct from the real-OAS source code above; the
# proxy is a similar measure that exists because FRED's ICE BofA OAS
# series lack pre-2023 history.
CREDIT_SPREAD_PROXY_BIAS_WARNING_CODE = "credit_spread_proxy_total_return_differential"
CREDIT_SPREAD_PROXY_BIAS_SOURCE = "tlt_minus_hyg_lqd_total_return_differential"
CREDIT_SPREAD_PROXY_BIAS_SOURCE_URL = (
    "internal:tlt_minus_hyg_lqd_total_return_differential"
)

# Pre-SOFR/IORB funding stress proxy — FEDFUNDS minus IOER, spliced into
# sofr_iorb_spread for sessions before SOFR (Apr 2018) and IORB (Jul 2021)
# availability. Emitted only when the splice is actually active.
FUNDING_SPREAD_PROXY_BIAS_WARNING_CODE = "funding_spread_fedfunds_ioer_proxy"
FUNDING_SPREAD_PROXY_BIAS_SOURCE = "fred:DFF-IOER"
FUNDING_SPREAD_PROXY_BIAS_SOURCE_URL = "https://fred.stlouisfed.org/series/DFF"

_PROXY_BIAS_FEATURE_NAMES: tuple[str, ...] = (
    "hy_tr_differential_63d",
    "ig_tr_differential_63d",
    "hy_tr_differential_percentile_504d",
    "hy_tr_differential_slope_21d",
    "ig_tr_differential_slope_21d",
)


# ---------------------------------------------------------------------------
# Feature dataclass — single source of truth for §2C feature seam.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CreditFundingFeatures:
    """v2 §2C per-session credit/funding feature series.

    All series are aligned to the SPY DatetimeIndex. NaN cold-start at the
    head of each series until the corresponding lookback fills.
    """

    hy_oas_63d: pd.Series
    ig_oas_63d: pd.Series
    hy_oas_percentile_504d: pd.Series
    hy_oas_slope_21d: pd.Series
    ig_oas_slope_21d: pd.Series
    hy_tr_differential_63d: pd.Series
    ig_tr_differential_63d: pd.Series
    hy_tr_differential_percentile_504d: pd.Series
    hy_tr_differential_slope_21d: pd.Series
    ig_tr_differential_slope_21d: pd.Series
    kre_spy_ratio: pd.Series
    kre_spy_slope_63d: pd.Series
    nfci_daily_carried: pd.Series
    sofr_iorb_spread: pd.Series
    sofr_iorb_slope_21d: pd.Series
    broad_usd_index_zscore_21d: pd.Series
    spy_21d_return: pd.Series
    tlt_21d_return: pd.Series
    bias_warnings: pd.DataFrame

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "hy_oas_63d",
            "ig_oas_63d",
            "hy_oas_percentile_504d",
            "hy_oas_slope_21d",
            "ig_oas_slope_21d",
            "hy_tr_differential_63d",
            "ig_tr_differential_63d",
            "hy_tr_differential_percentile_504d",
            "hy_tr_differential_slope_21d",
            "ig_tr_differential_slope_21d",
            "kre_spy_ratio",
            "kre_spy_slope_63d",
            "nfci_daily_carried",
            "sofr_iorb_spread",
            "sofr_iorb_slope_21d",
            "broad_usd_index_zscore_21d",
            "spy_21d_return",
            "tlt_21d_return",
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({name: getattr(self, name) for name in self.feature_names})


# ---------------------------------------------------------------------------
# Rolling helpers — `_rolling_ols_slope` lives here because the per-§2C
# slope helper differs from `network_fragility_rules._trailing_slope`
# (vectorised rolling form vs per-day scalar). The z-score helper is
# shared with §2A — imported from `_rolling_stats` (single home for
# shared helpers).
# ---------------------------------------------------------------------------


def _rolling_ols_slope(series: pd.Series, *, window: int) -> pd.Series:
    """Rolling OLS slope of ``series`` vs a unit trading-day index.

    Closed-form OLS slope: ``cov(x, y) / var(x)`` where x = [0, 1, ..., window-1].
    Returns NaN until ``window`` non-NaN observations are available (any NaN in
    the window propagates to NaN).

    Mirrors the per-day ``_trailing_slope`` helper in network_fragility_rules
    (which uses ``np.polyfit``); both produce numerically identical slopes on
    finite input. The rolling vectorised form is preferred here because §2C
    computes slopes on the full series rather than a single per-day scalar.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")
    series = series.astype(float)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_centered = x - x_mean
    x_var = float((x_centered**2).sum())  # constant

    def _slope(window_arr: np.ndarray) -> float:
        if np.isnan(window_arr).any():
            return float("nan")
        y_mean = window_arr.mean()
        return float(((x_centered) * (window_arr - y_mean)).sum() / x_var)

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


# ---------------------------------------------------------------------------
# Feature compute (§2C lines 3208-3243).
# ---------------------------------------------------------------------------


def compute_credit_funding_features(
    *,
    hyg_close: pd.Series,
    lqd_close: pd.Series,
    tlt_close: pd.Series,
    kre_close: pd.Series,
    spy_close: pd.Series,
    sofr: pd.Series,
    iorb: pd.Series,
    nfci_weekly: pd.Series,
    broad_usd_index: pd.Series,
    hy_oas: pd.Series,
    ig_oas: pd.Series,
    config: CreditFundingRulesConfig,
    fedfunds: pd.Series | None = None,
    ioer_legacy: pd.Series | None = None,
) -> CreditFundingFeatures:
    """Compute the v2 §2C credit/funding feature seam from raw inputs.

    All inputs are aligned to ``spy_close.index``; missing dates within the
    NYSE calendar produce NaN at those rows.

    Credit-spread metrics are parallel. ``hy_oas`` / ``ig_oas`` are the
    authoritative FRED-redistributed ICE BofA Option-Adjusted Spread series
    (BAMLH0A0HYM2 for HY, BAMLC0A4CBBB for BBB IG). The TLT-vs-HYG/LQD
    total-return differential is computed separately below and produces
    ``credit_funding_state_proxy`` through the axis-series classifier. The
    raw spread series are never spliced. When the OAS series are absent from
    ``macro_series``, real-OAS features are all-NaN and the real-OAS label
    emits unknown/data-unavailable; proxy features still build from ETF closes.
    """
    spy_index = spy_close.index

    # Reindex every input to the SPY calendar so all returned series share
    # the same DatetimeIndex (single-source-of-truth for the rule engine).
    hyg = hyg_close.reindex(spy_index).astype(float)
    lqd = lqd_close.reindex(spy_index).astype(float)
    tlt = tlt_close.reindex(spy_index).astype(float)
    kre = kre_close.reindex(spy_index).astype(float)
    spy = spy_close.reindex(spy_index).astype(float)
    sofr_s = sofr.reindex(spy_index).astype(float).ffill()
    iorb_s = iorb.reindex(spy_index).astype(float).ffill()
    nfci_w = nfci_weekly.reindex(spy_index).astype(float)
    usd = broad_usd_index.reindex(spy_index).astype(float).ffill()

    pct_window = config.hy_percentile_504d_lookback
    slope_21d = config.slope_21d_lookback
    slope_63d = config.slope_63d_lookback
    spy_window = config.spy_return_lookback_days
    tlt_window = config.tlt_return_lookback_days
    usd_change_window = config.broad_usd_change_window_days
    usd_norm_window = config.broad_usd_normalizer_window_days

    # §2C lines 3211-3212 — authoritative ICE BofA OAS. Rising OAS =
    # wider spread (matches the §2C line 3210 sign convention by construction).
    hy_oas_63d = hy_oas.reindex(spy_index).astype(float).ffill().rename("hy_oas_63d")
    ig_oas_63d = ig_oas.reindex(spy_index).astype(float).ffill().rename("ig_oas_63d")

    # §2C line 3221: 504d percentile (pct=True).
    hy_oas_percentile_504d = (
        hy_oas_63d.rolling(pct_window).rank(pct=True).rename("hy_oas_percentile_504d")
    )

    # §2C lines 3222-3223: 21d OLS slope.
    hy_oas_slope_21d = _rolling_ols_slope(hy_oas_63d, window=slope_21d).rename(
        "hy_oas_slope_21d"
    )
    ig_oas_slope_21d = _rolling_ols_slope(ig_oas_63d, window=slope_21d).rename(
        "ig_oas_slope_21d"
    )

    # §2C proxy metric (ADR 0007; implementation decision + #71) — TLT-vs-HYG/LQD total-return
    # differential. Rising = Treasury outperforming credit = widening
    # spreads (matches the §2C line 3215 sign convention). A SEPARATE
    # parallel metric — never blended with the real-OAS series above.
    total_return_window = config.total_return_lookback_days
    hyg_tr = (hyg / hyg.shift(total_return_window)) - 1.0
    lqd_tr = (lqd / lqd.shift(total_return_window)) - 1.0
    tlt_tr = (tlt / tlt.shift(total_return_window)) - 1.0
    hy_tr_differential_63d = (tlt_tr - hyg_tr).rename("hy_tr_differential_63d")
    ig_tr_differential_63d = (tlt_tr - lqd_tr).rename("ig_tr_differential_63d")
    hy_tr_differential_percentile_504d = (
        hy_tr_differential_63d.rolling(pct_window)
        .rank(pct=True)
        .rename("hy_tr_differential_percentile_504d")
    )
    hy_tr_differential_slope_21d = _rolling_ols_slope(
        hy_tr_differential_63d, window=slope_21d
    ).rename("hy_tr_differential_slope_21d")
    ig_tr_differential_slope_21d = _rolling_ols_slope(
        ig_tr_differential_63d, window=slope_21d
    ).rename("ig_tr_differential_slope_21d")

    # §2C lines 3229-3230: bank-index relative strength.
    kre_spy_ratio = (kre / spy.where(spy > 0)).rename("kre_spy_ratio")
    kre_spy_slope_63d = _rolling_ols_slope(kre_spy_ratio, window=slope_63d).rename(
        "kre_spy_slope_63d"
    )

    # §2C lines 3232-3233: NFCI weekly → daily via forward-fill (last-known-value).
    nfci_daily_carried = nfci_w.ffill().rename("nfci_daily_carried")

    # §2C lines 3235-3238: broad-USD-index 21d-change z-score.
    broad_usd_index_zscore_21d = _change_zscore(
        usd,
        change_window=usd_change_window,
        normalizer_window=usd_norm_window,
    ).rename("broad_usd_index_zscore_21d")

    # §2C lines 3242-3243: SOFR-IORB spread + 21d slope.
    # Splice priority for pre-SOFR/IORB eras (ADR 0009):
    #   SOFR-IORB (Jul 2021+) > SOFR-IOER (Apr 2018 - Jul 2021) > FEDFUNDS-IOER (Oct 2008+)
    # When fedfunds/ioer_legacy are supplied, NaN gaps in SOFR-IORB are filled,
    # eliminating spurious credit_funding "unknown" for 2016-2021 sessions.
    sofr_iorb_raw = sofr_s - iorb_s
    funding_spread_proxy_active = False
    if fedfunds is not None and ioer_legacy is not None:
        fedfunds_s = fedfunds.reindex(spy_index).astype(float).ffill()
        ioer_s = ioer_legacy.reindex(spy_index).astype(float).ffill()
        sofr_ioer = sofr_s - ioer_s
        fedfunds_ioer = fedfunds_s - ioer_s
        spliced = sofr_iorb_raw.fillna(sofr_ioer).fillna(fedfunds_ioer)
        funding_spread_proxy_active = (
            spliced.notna().sum() > sofr_iorb_raw.notna().sum()
        )
        sofr_iorb_spread = spliced.rename("sofr_iorb_spread")
    else:
        sofr_iorb_spread = sofr_iorb_raw.rename("sofr_iorb_spread")
    sofr_iorb_slope_21d = _rolling_ols_slope(sofr_iorb_spread, window=slope_21d).rename(
        "sofr_iorb_slope_21d"
    )

    # SPY / TLT 21d returns (consumed by §2C credit_stress / funding_squeeze /
    # deleveraging rules — spec lines 3259/3264/3267-3268).
    spy_21d_return = ((spy / spy.shift(spy_window)) - 1.0).rename("spy_21d_return")
    tlt_21d_return = ((tlt / tlt.shift(tlt_window)) - 1.0).rename("tlt_21d_return")

    # Single-source provenance row per spread feature — ICE BofA OAS via
    # FRED. Retained on the `bias_warnings` frame (rather than dropped)
    # so downstream consumers have an explicit, machine-readable record
    # of the credit-spread metric's origin; it is provenance, not a bias.
    proxy_funding_rows = (
        [
            {
                "warning_code": FUNDING_SPREAD_PROXY_BIAS_WARNING_CODE,
                "feature_name": "sofr_iorb_spread",
                "source": FUNDING_SPREAD_PROXY_BIAS_SOURCE,
                "source_url": FUNDING_SPREAD_PROXY_BIAS_SOURCE_URL,
            }
        ]
        if funding_spread_proxy_active
        else []
    )
    bias_warnings = make_bias_warnings_frame(
        [
            {
                "warning_code": CREDIT_SPREAD_SOURCE_CODE,
                "feature_name": feat,
                "source": CREDIT_SPREAD_SOURCE,
                "source_url": CREDIT_SPREAD_SOURCE_URL,
            }
            for feat in _BIAS_FEATURE_NAMES
        ]
        + [
            {
                "warning_code": CREDIT_SPREAD_PROXY_BIAS_WARNING_CODE,
                "feature_name": feat,
                "source": CREDIT_SPREAD_PROXY_BIAS_SOURCE,
                "source_url": CREDIT_SPREAD_PROXY_BIAS_SOURCE_URL,
            }
            for feat in _PROXY_BIAS_FEATURE_NAMES
        ]
        + proxy_funding_rows
    )

    return CreditFundingFeatures(
        hy_oas_63d=hy_oas_63d,
        ig_oas_63d=ig_oas_63d,
        hy_oas_percentile_504d=hy_oas_percentile_504d,
        hy_oas_slope_21d=hy_oas_slope_21d,
        ig_oas_slope_21d=ig_oas_slope_21d,
        hy_tr_differential_63d=hy_tr_differential_63d,
        ig_tr_differential_63d=ig_tr_differential_63d,
        hy_tr_differential_percentile_504d=hy_tr_differential_percentile_504d,
        hy_tr_differential_slope_21d=hy_tr_differential_slope_21d,
        ig_tr_differential_slope_21d=ig_tr_differential_slope_21d,
        kre_spy_ratio=kre_spy_ratio,
        kre_spy_slope_63d=kre_spy_slope_63d,
        nfci_daily_carried=nfci_daily_carried,
        sofr_iorb_spread=sofr_iorb_spread,
        sofr_iorb_slope_21d=sofr_iorb_slope_21d,
        broad_usd_index_zscore_21d=broad_usd_index_zscore_21d,
        spy_21d_return=spy_21d_return,
        tlt_21d_return=tlt_21d_return,
        bias_warnings=bias_warnings,
    )


# ---------------------------------------------------------------------------
# Per-day scalar rule inputs (mirrors network_fragility_rules pattern).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CreditFundingRuleInputs:
    """Per-day scalars consumed by the §2C rule predicates."""

    hy_spread_percentile_504d: float
    hy_spread_slope_21d: float
    ig_spread_slope_21d: float
    broad_usd_index_zscore_21d: float
    sofr_iorb_slope_21d: float
    spy_21d_return: float
    tlt_21d_return: float
    realized_vol_21d_percentile_252d: float
    realized_vol_21d: float
    avg_pairwise_corr_percentile_504d: float
    avg_pairwise_corr_63d: float


@dataclass(frozen=True)
class CreditFundingRuleEvaluation:
    label: CreditFundingLabel
    rule_path: str
    reason: str | None = None


def _scalar_at(series: pd.Series, dt: pd.Timestamp) -> float:
    if dt not in series.index:
        return float("nan")
    val = series.loc[dt]
    if pd.isna(val):
        return float("nan")
    return float(val)


def build_rule_inputs_for_date(
    *,
    features: CreditFundingFeatures,
    dt: pd.Timestamp,
    hy_spread_percentile_504d: pd.Series,
    hy_spread_slope_21d: pd.Series,
    ig_spread_slope_21d: pd.Series,
    realized_vol_21d_percentile_252d: pd.Series,
    avg_pairwise_corr_percentile_504d: pd.Series,
    realized_vol_21d: pd.Series | None = None,
    avg_pairwise_corr_63d: pd.Series | None = None,
) -> CreditFundingRuleInputs:
    """Materialize the per-day scalar rule inputs at session ``dt``.

    The spread triple is passed explicitly (source-neutral) so the same
    builder serves both the real-OAS run (pass ``features.hy_oas_*``) and
    the proxy run (pass ``features.hy_tr_differential_*``) — ADR 0007; implementation decision + #71.
    """
    return CreditFundingRuleInputs(
        hy_spread_percentile_504d=_scalar_at(hy_spread_percentile_504d, dt),
        hy_spread_slope_21d=_scalar_at(hy_spread_slope_21d, dt),
        ig_spread_slope_21d=_scalar_at(ig_spread_slope_21d, dt),
        broad_usd_index_zscore_21d=_scalar_at(features.broad_usd_index_zscore_21d, dt),
        sofr_iorb_slope_21d=_scalar_at(features.sofr_iorb_slope_21d, dt),
        spy_21d_return=_scalar_at(features.spy_21d_return, dt),
        tlt_21d_return=_scalar_at(features.tlt_21d_return, dt),
        realized_vol_21d_percentile_252d=_scalar_at(
            realized_vol_21d_percentile_252d, dt
        ),
        realized_vol_21d=(
            _scalar_at(realized_vol_21d, dt)
            if realized_vol_21d is not None
            else float("nan")
        ),
        avg_pairwise_corr_percentile_504d=_scalar_at(
            avg_pairwise_corr_percentile_504d, dt
        ),
        avg_pairwise_corr_63d=(
            _scalar_at(avg_pairwise_corr_63d, dt)
            if avg_pairwise_corr_63d is not None
            else float("nan")
        ),
    )


def build_rule_inputs_by_date(
    *,
    features: CreditFundingFeatures,
    hy_spread_percentile_504d: pd.Series,
    hy_spread_slope_21d: pd.Series,
    ig_spread_slope_21d: pd.Series,
    realized_vol_21d_percentile_252d: pd.Series,
    avg_pairwise_corr_percentile_504d: pd.Series,
    realized_vol_21d: pd.Series | None = None,
    avg_pairwise_corr_63d: pd.Series | None = None,
) -> dict[pd.Timestamp, CreditFundingRuleInputs]:
    """Per-date rule inputs. The spread triple is source-neutral — pass
    ``features.hy_oas_*`` for the real-OAS run or ``features.hy_tr_differential_*``
    for the proxy run (ADR 0007; implementation decision + #71)."""
    index = hy_spread_percentile_504d.index
    outputs: dict[pd.Timestamp, CreditFundingRuleInputs] = {}
    for dt in index:
        outputs[dt] = CreditFundingRuleInputs(
            hy_spread_percentile_504d=_scalar_at(hy_spread_percentile_504d, dt),
            hy_spread_slope_21d=_scalar_at(hy_spread_slope_21d, dt),
            ig_spread_slope_21d=_scalar_at(ig_spread_slope_21d, dt),
            broad_usd_index_zscore_21d=_scalar_at(
                features.broad_usd_index_zscore_21d, dt
            ),
            sofr_iorb_slope_21d=_scalar_at(features.sofr_iorb_slope_21d, dt),
            spy_21d_return=_scalar_at(features.spy_21d_return, dt),
            tlt_21d_return=_scalar_at(features.tlt_21d_return, dt),
            realized_vol_21d_percentile_252d=_scalar_at(
                realized_vol_21d_percentile_252d, dt
            ),
            realized_vol_21d=(
                _scalar_at(realized_vol_21d, dt)
                if realized_vol_21d is not None
                else float("nan")
            ),
            avg_pairwise_corr_percentile_504d=_scalar_at(
                avg_pairwise_corr_percentile_504d, dt
            ),
            avg_pairwise_corr_63d=(
                _scalar_at(avg_pairwise_corr_63d, dt)
                if avg_pairwise_corr_63d is not None
                else float("nan")
            ),
        )
    return outputs


# ---------------------------------------------------------------------------
# Rule predicates (§2C lines 3249-3271).
# ---------------------------------------------------------------------------


def _any_nan(*values: float) -> bool:
    return any(np.isnan(v) for v in values)


def evaluate_credit_calm(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """v2 §2C lines 3249-3251.

    ``hy_spread_percentile_504d < 0.50
       AND hy_spread_slope_21d <= 0`` (non-rising slope).
    """
    if _any_nan(
        inputs.hy_spread_percentile_504d,
        inputs.hy_spread_slope_21d,
    ):
        return False
    return bool(
        inputs.hy_spread_percentile_504d < config.hy_percentile_calm_max
        and inputs.hy_spread_slope_21d <= 0.0
    )


def evaluate_spread_widening(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """v2 §2C lines 3253-3255.

    ``hy_spread_slope_21d > 0 AND ig_spread_slope_21d > 0``
    (strict positive slope on BOTH HY and IG legs).

    Extended path: HY slope > 0 alone (IG not required). Captures
    early-stage widening and divergent credit moves where HY leads.
    """
    if _any_nan(inputs.hy_spread_slope_21d):
        return False
    if not np.isnan(inputs.ig_spread_slope_21d) and (
        inputs.hy_spread_slope_21d > 0.0 and inputs.ig_spread_slope_21d > 0.0
    ):
        return True
    if getattr(config, "spread_widening_hy_only", False) and (
        inputs.hy_spread_slope_21d > 0.0
    ):
        return True
    return False


def evaluate_credit_recovery(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """Elevated spreads (percentile 0.50-0.80) that are narrowing (slope < 0).

    Economically: credit conditions are improving from stressed levels.
    Distinct from credit_calm (percentile < 0.50) and spread_widening (slope > 0).
    """
    if _any_nan(
        inputs.hy_spread_percentile_504d,
        inputs.hy_spread_slope_21d,
    ):
        return False
    calm_max = getattr(config, "hy_percentile_calm_max", 0.50)
    stress_min = getattr(config, "hy_percentile_stress_min", 0.80)
    return bool(
        inputs.hy_spread_percentile_504d >= calm_max
        and inputs.hy_spread_percentile_504d < stress_min
        and inputs.hy_spread_slope_21d < 0.0
    )


def evaluate_credit_stress(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """v2 §2C lines 3257-3259.

    ``hy_spread_percentile_504d > 0.80 AND spy_21d_return < -0.05``.
    """
    if _any_nan(
        inputs.hy_spread_percentile_504d,
        inputs.spy_21d_return,
    ):
        return False
    return bool(
        inputs.hy_spread_percentile_504d > config.hy_percentile_stress_min
        and inputs.spy_21d_return < config.spy_drop_threshold
    )


def evaluate_funding_squeeze(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """v2 §2C lines 3261-3264.

    ``broad_usd_index_zscore_21d > +1.5 AND sofr_iorb_slope_21d > 0
       AND spy_21d_return < 0``.
    """
    if _any_nan(
        inputs.broad_usd_index_zscore_21d,
        inputs.sofr_iorb_slope_21d,
        inputs.spy_21d_return,
    ):
        return False
    return bool(
        inputs.broad_usd_index_zscore_21d > config.broad_usd_zscore_funding_threshold
        and inputs.sofr_iorb_slope_21d > 0.0
        and inputs.spy_21d_return < 0.0
    )


def _deleveraging_percentile_path(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    if _any_nan(
        inputs.spy_21d_return,
        inputs.tlt_21d_return,
        inputs.broad_usd_index_zscore_21d,
        inputs.realized_vol_21d_percentile_252d,
        inputs.avg_pairwise_corr_percentile_504d,
    ):
        return False
    return bool(
        inputs.spy_21d_return < config.spy_drop_threshold
        and inputs.tlt_21d_return < 0.0
        and inputs.broad_usd_index_zscore_21d
        > config.broad_usd_zscore_deleveraging_threshold
        and inputs.realized_vol_21d_percentile_252d
        > config.realized_vol_percentile_threshold
        and inputs.avg_pairwise_corr_percentile_504d
        > config.correlation_percentile_threshold
    )


def _deleveraging_cold_start_path(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    if not config.cold_start_deleveraging_enabled:
        return False
    if not (
        np.isnan(inputs.realized_vol_21d_percentile_252d)
        or np.isnan(inputs.avg_pairwise_corr_percentile_504d)
    ):
        return False
    if _any_nan(
        inputs.spy_21d_return,
        inputs.tlt_21d_return,
        inputs.broad_usd_index_zscore_21d,
        inputs.realized_vol_21d,
        inputs.avg_pairwise_corr_63d,
    ):
        return False
    return bool(
        inputs.spy_21d_return < config.spy_drop_threshold
        and inputs.tlt_21d_return < 0.0
        and inputs.broad_usd_index_zscore_21d
        > config.broad_usd_zscore_deleveraging_threshold
        and inputs.realized_vol_21d
        >= config.cold_start_deleveraging_realized_vol_21d_min
        and inputs.avg_pairwise_corr_63d
        >= config.cold_start_deleveraging_avg_corr_63d_min
    )


def deleveraging_rule_path(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> str | None:
    if _deleveraging_percentile_path(inputs, config):
        return "percentile"
    if _deleveraging_cold_start_path(inputs, config):
        return "cold_start_fallback"
    return None


def evaluate_deleveraging(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """v2 §2C lines 3266-3271 — 5-condition composite.

    ``spy_21d_return < -0.05 AND tlt_21d_return < 0
       AND broad_usd_index_zscore_21d > 0
       AND realized_vol_21d_percentile_252d > 0.75
       AND avg_pairwise_corr_percentile_504d > 0.75``.
    """
    return deleveraging_rule_path(inputs, config) is not None


def evaluate_rules(
    *,
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> CreditFundingLabel:
    """Walk v2 §2C precedence and return the first matching label.

    Falls through to ``credit_mixed`` when valid data has no dominant
    credit/funding signal. Data-quality failures still emit ``unknown``
    upstream.
    """
    return evaluate_rules_with_evidence(inputs=inputs, config=config).label


def evaluate_rules_with_evidence(
    *,
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> CreditFundingRuleEvaluation:
    """Walk v2 §2C precedence and return the label plus matched rule path."""
    deleveraging_path = deleveraging_rule_path(inputs, config)
    if deleveraging_path is not None:
        return CreditFundingRuleEvaluation(
            label="deleveraging",
            rule_path=deleveraging_path,
        )
    if evaluate_funding_squeeze(inputs, config):
        return CreditFundingRuleEvaluation(
            label="funding_squeeze", rule_path="standard"
        )
    if evaluate_credit_stress(inputs, config):
        return CreditFundingRuleEvaluation(label="credit_stress", rule_path="standard")
    if evaluate_spread_widening(inputs, config):
        return CreditFundingRuleEvaluation(
            label="spread_widening", rule_path="standard"
        )
    if evaluate_credit_recovery(inputs, config):
        return CreditFundingRuleEvaluation(
            label="credit_recovery", rule_path="standard"
        )
    if evaluate_credit_calm(inputs, config):
        return CreditFundingRuleEvaluation(label="credit_calm", rule_path="standard")
    return CreditFundingRuleEvaluation(
        label="credit_mixed",
        rule_path="valid_data_fallback",
        reason="no_dominant_credit_funding_signal",
    )
