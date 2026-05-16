"""v2 §2C Credit/Funding axis — feature compute + per-day rule materialisation (Slice 4).

Implements the 5-label axis classifier from spec lines 2005-2130:

  Labels (§2C lines 2007-2015):
    credit_calm, spread_widening, credit_stress, funding_squeeze, deleveraging, unknown

  Precedence (§2C line 2019):
    deleveraging > funding_squeeze > credit_stress > spread_widening > credit_calm > unknown

Credit-spread metrics — two parallel sources (Ambiguity Log #49 + #71):

  §2C carries two distinct, never-blended credit-spread metrics:

  1. Real ICE BofA OAS — ``hy_oas_*`` / ``ig_oas_*``, from the
     FRED-redistributed ICE BofA Option-Adjusted Spread series
     (``BAMLH0A0HYM2`` HY Master II OAS, ``BAMLC0A4CBBB`` BBB Corporate
     OAS). The authoritative metric. ``MarketContext.macro_series`` keys
     ``hy_oas`` / ``ig_bbb_oas`` (set by ``V2_FRED_SERIES`` in
     ``regime_data_fetch.fetch_workflow``) feed the ``hy_oas`` / ``ig_oas``
     params of ``compute_credit_funding_features``. FRED exposes only a
     trailing ~3-year window of these series (start 2023-05-15 — Log #71),
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
     parallel, never spliced with the real-OAS series.

  When the OAS series are absent from ``macro_series``, the §2C seam is
  simply not built (``REQUIRED_MACRO_KEYS`` gate in ``feature_store``) and
  both ``credit_funding_state`` / ``credit_funding_state_proxy`` stay
  ``None`` — V1 byte-identity preserved, same as every other unbuilt V2
  seam.

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
  - rule predicates ``evaluate_*`` + ``evaluate_rules`` walker (§2C lines 2064-2088)
  - ``CREDIT_FUNDING_RISK_RANK`` + ``CreditFundingLabel`` enum (§2C lines 2092-2099)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from regime_detection._rolling_stats import rolling_change_zscore as _change_zscore
from regime_detection._staleness_utils import (
    calendar_staleness_days_series as _calendar_staleness_days_series,
    safe_float as _safe_float,
    trading_staleness_series as _trading_staleness_series,
)
from regime_detection.breadth_state_v2 import make_bias_warnings_frame
from regime_detection.config import (
    CreditFundingRulesConfig,
)
from regime_detection.data_quality import assess_series_input_quality, quality_forces_unknown
from regime_detection.hysteresis import apply_per_label_asymmetric_hysteresis
from regime_detection.models import CreditFundingOutput, DataQuality

if TYPE_CHECKING:
    from regime_detection.feature_store import FeatureStore
    from regime_detection.market_context import MarketContext


# ---------------------------------------------------------------------------
# Spec labels (§2C lines 2007-2015) + risk rank (§2C lines 2092-2099).
# ---------------------------------------------------------------------------

CreditFundingLabel = Literal[
    "credit_calm",
    "spread_widening",
    "credit_stress",
    "funding_squeeze",
    "deleveraging",
    "unknown",
]


# v2 §2C lines 2092-2099 verbatim. ``deleveraging: 4`` is the ONLY V2 axis label
# with risk_rank>3 — reflects that the rule fires only when five distinct stress
# signals coincide (spec line 2102).
CREDIT_FUNDING_RISK_RANK: dict[str, int] = {
    "credit_calm": 0,
    "unknown": 1,
    "spread_widening": 1,
    "credit_stress": 2,
    "funding_squeeze": 3,
    "deleveraging": 4,
}


# v2 §2C line 2019 precedence (highest-severity-first walk).
RULE_PRECEDENCE: tuple[CreditFundingLabel, ...] = (
    "deleveraging",
    "funding_squeeze",
    "credit_stress",
    "spread_widening",
    "credit_calm",
)


# ---------------------------------------------------------------------------
# Required FRED / cross-asset symbol keys. Pinned here as single source of
# truth so feature_store + classifier read one constant.
# ---------------------------------------------------------------------------

HYG_KEY = "HYG"
LQD_KEY = "LQD"
TLT_KEY = "TLT"
KRE_KEY = "KRE"

SOFR_KEY = "SOFR"
IORB_KEY = "IORB"
NFCI_KEY = "NFCI"
BROAD_USD_INDEX_KEY = "broad_usd_index"
# ICE BofA Option-Adjusted Spread series — FRED-redistributed under ICE
# license, free at the FRED endpoint. macro_series keys set by
# `V2_FRED_SERIES` in `regime_data_fetch.fetch_workflow`.
HY_OAS_KEY = "hy_oas"          # FRED BAMLH0A0HYM2 — ICE BofA US High Yield OAS
IG_OAS_KEY = "ig_bbb_oas"      # FRED BAMLC0A4CBBB — ICE BofA BBB Corporate OAS


REQUIRED_CROSS_ASSET_KEYS: tuple[str, ...] = (HYG_KEY, LQD_KEY, TLT_KEY, KRE_KEY)
REQUIRED_MACRO_KEYS: tuple[str, ...] = (
    SOFR_KEY,
    IORB_KEY,
    NFCI_KEY,
    BROAD_USD_INDEX_KEY,
    HY_OAS_KEY,
    IG_OAS_KEY,
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
# (Ambiguity Log #71). Distinct from the real-OAS source code above; the
# proxy is a similar measure that exists because FRED's ICE BofA OAS
# series lack pre-2023 history.
CREDIT_SPREAD_PROXY_BIAS_WARNING_CODE = "credit_spread_proxy_total_return_differential"
CREDIT_SPREAD_PROXY_BIAS_SOURCE = "tlt_minus_hyg_lqd_total_return_differential"
CREDIT_SPREAD_PROXY_BIAS_SOURCE_URL = "internal:tlt_minus_hyg_lqd_total_return_differential"

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
    """v2 §2C per-session credit/funding feature series (Slice 4).

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
        return pd.DataFrame(
            {name: getattr(self, name) for name in self.feature_names}
        )


# ---------------------------------------------------------------------------
# Rolling helpers — `_rolling_ols_slope` lives here because the per-§2C
# slope helper differs from `network_fragility_rules._trailing_slope`
# (vectorised rolling form vs per-day scalar). The z-score helper is
# shared with §2A — imported from `_rolling_stats` (one home, AGENTS
# rule B).
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
    x_var = float((x_centered ** 2).sum())  # constant

    def _slope(window_arr: np.ndarray) -> float:
        if np.isnan(window_arr).any():
            return float("nan")
        y_mean = window_arr.mean()
        return float(((x_centered) * (window_arr - y_mean)).sum() / x_var)

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


# ---------------------------------------------------------------------------
# Feature compute (§2C lines 2031-2060).
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
) -> CreditFundingFeatures:
    """Compute the v2 §2C credit/funding feature seam from raw inputs.

    All inputs are aligned to ``spy_close.index``; missing dates within the
    NYSE calendar produce NaN at those rows.

    Credit-spread metrics are parallel. ``hy_oas`` / ``ig_oas`` are the
    authoritative FRED-redistributed ICE BofA Option-Adjusted Spread series
    (BAMLH0A0HYM2 for HY, BAMLC0A4CBBB for BBB IG). The TLT-vs-HYG/LQD
    total-return differential is computed separately below and produces
    ``credit_funding_state_proxy`` through the axis-series classifier. The
    two metrics are never blended. When the OAS series are absent from
    ``macro_series``, the §2C seam simply is not built (handled by the
    ``REQUIRED_MACRO_KEYS`` gate in ``feature_store``) and
    ``credit_funding_state`` / ``credit_funding_state_proxy`` stay ``None`` —
    V1 byte-identity preserved, same as every other unbuilt V2 seam.
    """
    spy_index = spy_close.index

    # Reindex every input to the SPY calendar so all returned series share
    # the same DatetimeIndex (single-source-of-truth for the rule engine).
    hyg = hyg_close.reindex(spy_index).astype(float)
    lqd = lqd_close.reindex(spy_index).astype(float)
    tlt = tlt_close.reindex(spy_index).astype(float)
    kre = kre_close.reindex(spy_index).astype(float)
    spy = spy_close.reindex(spy_index).astype(float)
    sofr_s = sofr.reindex(spy_index).astype(float)
    iorb_s = iorb.reindex(spy_index).astype(float)
    nfci_w = nfci_weekly.reindex(spy_index).astype(float)
    usd = broad_usd_index.reindex(spy_index).astype(float)

    pct_window = config.hy_percentile_504d_lookback
    slope_21d = config.slope_21d_lookback
    slope_63d = config.slope_63d_lookback
    spy_window = config.spy_return_lookback_days
    tlt_window = config.tlt_return_lookback_days
    usd_change_window = config.broad_usd_change_window_days
    usd_norm_window = config.broad_usd_normalizer_window_days

    # §2C lines 2032-2035 — authoritative ICE BofA OAS. Rising OAS =
    # wider spread (matches the §2C line 2033 sign convention by construction).
    hy_oas_63d = (
        hy_oas.reindex(spy_index).astype(float).rename("hy_oas_63d")
    )
    ig_oas_63d = (
        ig_oas.reindex(spy_index).astype(float).rename("ig_oas_63d")
    )

    # §2C line 2038: 504d percentile (pct=True).
    hy_oas_percentile_504d = (
        hy_oas_63d.rolling(pct_window).rank(pct=True)
        .rename("hy_oas_percentile_504d")
    )

    # §2C lines 2041-2042: 21d OLS slope.
    hy_oas_slope_21d = _rolling_ols_slope(
        hy_oas_63d, window=slope_21d
    ).rename("hy_oas_slope_21d")
    ig_oas_slope_21d = _rolling_ols_slope(
        ig_oas_63d, window=slope_21d
    ).rename("ig_oas_slope_21d")

    # §2C proxy metric (Ambiguity Log #71) — TLT-vs-HYG/LQD total-return
    # differential. Rising = Treasury outperforming credit = widening
    # spreads (matches the §2C line 2033 sign convention). A SEPARATE
    # parallel metric — never blended with the real-OAS series above.
    total_return_window = config.total_return_lookback_days
    hyg_tr = (hyg / hyg.shift(total_return_window)) - 1.0
    lqd_tr = (lqd / lqd.shift(total_return_window)) - 1.0
    tlt_tr = (tlt / tlt.shift(total_return_window)) - 1.0
    hy_tr_differential_63d = (tlt_tr - hyg_tr).rename("hy_tr_differential_63d")
    ig_tr_differential_63d = (tlt_tr - lqd_tr).rename("ig_tr_differential_63d")
    hy_tr_differential_percentile_504d = (
        hy_tr_differential_63d.rolling(pct_window).rank(pct=True)
        .rename("hy_tr_differential_percentile_504d")
    )
    hy_tr_differential_slope_21d = _rolling_ols_slope(
        hy_tr_differential_63d, window=slope_21d
    ).rename("hy_tr_differential_slope_21d")
    ig_tr_differential_slope_21d = _rolling_ols_slope(
        ig_tr_differential_63d, window=slope_21d
    ).rename("ig_tr_differential_slope_21d")

    # §2C lines 2045-2046: bank-index relative strength.
    kre_spy_ratio = (kre / spy.where(spy > 0)).rename("kre_spy_ratio")
    kre_spy_slope_63d = _rolling_ols_slope(
        kre_spy_ratio, window=slope_63d
    ).rename("kre_spy_slope_63d")

    # §2C line 2049: NFCI weekly → daily via forward-fill (last-known-value).
    nfci_daily_carried = nfci_w.ffill().rename("nfci_daily_carried")

    # §2C lines 2052-2055: broad-USD-index 21d-change z-score.
    broad_usd_index_zscore_21d = _change_zscore(
        usd,
        change_window=usd_change_window,
        normalizer_window=usd_norm_window,
    ).rename("broad_usd_index_zscore_21d")

    # §2C lines 2058-2059: SOFR-IORB spread + 21d slope.
    sofr_iorb_spread = (sofr_s - iorb_s).rename("sofr_iorb_spread")
    sofr_iorb_slope_21d = _rolling_ols_slope(
        sofr_iorb_spread, window=slope_21d
    ).rename("sofr_iorb_slope_21d")

    # SPY / TLT 21d returns (consumed by §2C credit_stress / funding_squeeze /
    # deleveraging rules — spec lines 2075/2080/2083/2084).
    spy_21d_return = ((spy / spy.shift(spy_window)) - 1.0).rename("spy_21d_return")
    tlt_21d_return = ((tlt / tlt.shift(tlt_window)) - 1.0).rename("tlt_21d_return")

    # Single-source provenance row per spread feature — ICE BofA OAS via
    # FRED. Retained on the `bias_warnings` frame (rather than dropped)
    # so downstream consumers have an explicit, machine-readable record
    # of the credit-spread metric's origin; it is provenance, not a bias.
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
    avg_pairwise_corr_percentile_504d: float


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
) -> CreditFundingRuleInputs:
    """Materialize the per-day scalar rule inputs at session ``dt``.

    The spread triple is passed explicitly (source-neutral) so the same
    builder serves both the real-OAS run (pass ``features.hy_oas_*``) and
    the proxy run (pass ``features.hy_tr_differential_*``) — Ambiguity Log #71.
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
        avg_pairwise_corr_percentile_504d=_scalar_at(
            avg_pairwise_corr_percentile_504d, dt
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
) -> dict[pd.Timestamp, CreditFundingRuleInputs]:
    """Per-date rule inputs. The spread triple is source-neutral — pass
    ``features.hy_oas_*`` for the real-OAS run or ``features.hy_tr_differential_*``
    for the proxy run (Ambiguity Log #71)."""
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
            avg_pairwise_corr_percentile_504d=_scalar_at(
                avg_pairwise_corr_percentile_504d, dt
            ),
        )
    return outputs


# ---------------------------------------------------------------------------
# Rule predicates (§2C lines 2064-2088).
# ---------------------------------------------------------------------------


def _any_nan(*values: float) -> bool:
    return any(np.isnan(v) for v in values)


def evaluate_credit_calm(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """v2 §2C lines 2065-2067.

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
    config: CreditFundingRulesConfig,  # noqa: ARG001 (uniform signature; no thresholds)
) -> bool:
    """v2 §2C lines 2069-2071.

    ``hy_spread_slope_21d > 0 AND ig_spread_slope_21d > 0``
    (strict positive slope on BOTH HY and IG legs).
    """
    if _any_nan(
        inputs.hy_spread_slope_21d,
        inputs.ig_spread_slope_21d,
    ):
        return False
    return bool(
        inputs.hy_spread_slope_21d > 0.0
        and inputs.ig_spread_slope_21d > 0.0
    )


def evaluate_credit_stress(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """v2 §2C lines 2073-2075.

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
    """v2 §2C lines 2077-2080.

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


def evaluate_deleveraging(
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> bool:
    """v2 §2C lines 2082-2087 — 5-condition composite.

    ``spy_21d_return < -0.05 AND tlt_21d_return < 0
       AND broad_usd_index_zscore_21d > 0
       AND realized_vol_21d_percentile_252d > 0.75
       AND avg_pairwise_corr_percentile_504d > 0.75``.
    """
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


def evaluate_rules(
    *,
    inputs: CreditFundingRuleInputs,
    config: CreditFundingRulesConfig,
) -> CreditFundingLabel:
    """Walk v2 §2C precedence and return the first matching label.

    Falls through to ``unknown`` when no rule fires (§2C line 2019 tail).
    """
    if evaluate_deleveraging(inputs, config):
        return "deleveraging"
    if evaluate_funding_squeeze(inputs, config):
        return "funding_squeeze"
    if evaluate_credit_stress(inputs, config):
        return "credit_stress"
    if evaluate_spread_widening(inputs, config):
        return "spread_widening"
    if evaluate_credit_calm(inputs, config):
        return "credit_calm"
    return "unknown"


def _build_for_spread_source(
    context: MarketContext,
    feature_store: FeatureStore,
    *,
    spread_source: Literal["oas", "proxy"],
) -> dict[date, CreditFundingOutput] | None:
    """Shared pipeline for build_axis_series() and build_axis_series_proxy()."""
    features = feature_store.credit_funding
    if features is None:
        return None
    cf_config = context.config.credit_funding
    if cf_config is None:
        return None

    if spread_source == "oas":
        hy_spread_63d = features.hy_oas_63d
        ig_spread_63d = features.ig_oas_63d
        hy_spread_percentile_504d = features.hy_oas_percentile_504d
        hy_spread_slope_21d = features.hy_oas_slope_21d
        ig_spread_slope_21d = features.ig_oas_slope_21d
        bias_warning_code = CREDIT_SPREAD_SOURCE_CODE
        evidence_spread_source = "ice_bofa_oas"
    else:  # "proxy"
        hy_spread_63d = features.hy_tr_differential_63d
        ig_spread_63d = features.ig_tr_differential_63d
        hy_spread_percentile_504d = features.hy_tr_differential_percentile_504d
        hy_spread_slope_21d = features.hy_tr_differential_slope_21d
        ig_spread_slope_21d = features.ig_tr_differential_slope_21d
        bias_warning_code = CREDIT_SPREAD_PROXY_BIAS_WARNING_CODE
        evidence_spread_source = "tlt_total_return_differential"

    spy_close = context.spy_ohlcv["close"]
    volatility_features = feature_store.volatility
    realized_vol_pct = volatility_features.realized_vol_percentile_252d
    nf_features = feature_store.network_fragility
    if nf_features is None:
        avg_corr_pct_series = pd.Series(float("nan"), index=spy_close.index)
    else:
        avg_corr_pct_series = nf_features.avg_pairwise_corr_percentile_504d

    cross_asset_closes = context.cross_asset_closes or {}
    macro_series = context.macro_series or {}
    hyg_close = cross_asset_closes.get("HYG")
    lqd_close = cross_asset_closes.get("LQD")
    tlt_close = cross_asset_closes.get("TLT")
    sofr_series = macro_series.get("SOFR")
    iorb_series = macro_series.get("IORB")
    nfci_series = macro_series.get("NFCI")

    required_inputs: list[pd.Series] = [
        hy_spread_63d,
        ig_spread_63d,
        features.kre_spy_ratio,
        features.sofr_iorb_spread,
        spy_close,
    ]
    required_trading_days = cf_config.rules.hy_percentile_504d_lookback
    max_freshness_days = context.config.data_quality.max_freshness_days
    min_completeness = context.config.data_quality.min_completeness

    raw_labels: list[CreditFundingLabel] = []
    per_day_data_quality: list[DataQuality] = []
    per_day_evidence: list[dict[str, object]] = []

    nfci_carried = features.nfci_daily_carried
    session_index = spy_close.index
    hyg_staleness_by_date = _trading_staleness_series(hyg_close, session_index)
    lqd_staleness_by_date = _trading_staleness_series(lqd_close, session_index)
    tlt_staleness_by_date = _trading_staleness_series(tlt_close, session_index)
    nfci_staleness_by_date = _calendar_staleness_days_series(nfci_series, session_index)
    sofr_missing_by_date = (
        pd.Series(True, index=session_index)
        if sofr_series is None
        else sofr_series.reindex(session_index).isna()
    )
    iorb_missing_by_date = (
        pd.Series(True, index=session_index)
        if iorb_series is None
        else iorb_series.reindex(session_index).isna()
    )
    rule_inputs_by_date = build_rule_inputs_by_date(
        features=features,
        hy_spread_percentile_504d=hy_spread_percentile_504d,
        hy_spread_slope_21d=hy_spread_slope_21d,
        ig_spread_slope_21d=ig_spread_slope_21d,
        realized_vol_21d_percentile_252d=realized_vol_pct,
        avg_pairwise_corr_percentile_504d=avg_corr_pct_series,
    )

    for day in context.sessions:
        dt = pd.Timestamp(day)

        etf_staleness_breach = False
        etf_stale_label: str | None = None
        hyg_staleness = int(hyg_staleness_by_date.loc[dt])
        lqd_staleness = int(lqd_staleness_by_date.loc[dt])
        tlt_staleness = int(tlt_staleness_by_date.loc[dt])
        if hyg_staleness > cf_config.etf_stale_sessions:
            etf_staleness_breach = True
            etf_stale_label = "HYG"
        elif lqd_staleness > cf_config.etf_stale_sessions:
            etf_staleness_breach = True
            etf_stale_label = "LQD"
        elif tlt_staleness > cf_config.etf_stale_sessions:
            etf_staleness_breach = True
            etf_stale_label = "TLT"

        sofr_missing = bool(sofr_missing_by_date.loc[dt])
        iorb_missing = bool(iorb_missing_by_date.loc[dt])
        nfci_staleness_days = int(nfci_staleness_by_date.loc[dt])
        nfci_stale = nfci_staleness_days > cf_config.nfci_stale_days

        if etf_staleness_breach or sofr_missing or iorb_missing or nfci_stale:
            reason_parts: list[str] = []
            if etf_staleness_breach:
                reason_parts.append(f"etf_stale:{etf_stale_label}")
            if sofr_missing:
                reason_parts.append("sofr_missing")
            if iorb_missing:
                reason_parts.append("iorb_missing")
            if nfci_stale:
                reason_parts.append(f"nfci_stale_{nfci_staleness_days}d")
            gate_reason = ",".join(reason_parts)
            raw_labels.append("unknown")
            per_day_data_quality.append(
                DataQuality(
                    status="stale_data" if (etf_staleness_breach or nfci_stale) else "insufficient_data",
                    freshness_days=None,
                    completeness=None,
                    reason=gate_reason,
                )
            )
            per_day_evidence.append({"reason": gate_reason})
            continue

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
            raw_labels.append("unknown")
            per_day_data_quality.append(day_quality)
            per_day_evidence.append({"reason": day_quality.reason or "insufficient_data"})
            continue

        rule_inputs = rule_inputs_by_date[dt]
        label = evaluate_rules(
            inputs=rule_inputs,
            config=cf_config.rules,
        )
        raw_labels.append(label)
        per_day_data_quality.append(day_quality)
        per_day_evidence.append(
            {
                "rule_evidence": {
                    "hy_spread_percentile_504d": rule_inputs.hy_spread_percentile_504d,
                    "hy_spread_slope_21d": rule_inputs.hy_spread_slope_21d,
                    "ig_spread_slope_21d": rule_inputs.ig_spread_slope_21d,
                    "broad_usd_index_zscore_21d": rule_inputs.broad_usd_index_zscore_21d,
                    "sofr_iorb_slope_21d": rule_inputs.sofr_iorb_slope_21d,
                    "spy_21d_return": rule_inputs.spy_21d_return,
                    "tlt_21d_return": rule_inputs.tlt_21d_return,
                    "realized_vol_21d_percentile_252d": rule_inputs.realized_vol_21d_percentile_252d,
                    "avg_pairwise_corr_percentile_504d": rule_inputs.avg_pairwise_corr_percentile_504d,
                },
                "spread_source": evidence_spread_source,
                "nfci_daily_carried": _safe_float(nfci_carried, dt),
                "kre_spy_slope_63d": _safe_float(features.kre_spy_slope_63d, dt),
                "bias_warning_code": bias_warning_code,
            }
        )

    stable_labels, active_labels = apply_per_label_asymmetric_hysteresis(
        raw_labels=raw_labels,
        risk_rank=CREDIT_FUNDING_RISK_RANK,
        deescalation_days_by_label=cf_config.deescalation_days_by_label,
        default_deescalation_days=cf_config.default_deescalation_days,
    )

    outputs: dict[date, CreditFundingOutput] = {}
    for day, raw, stable, active, dq, evidence in zip(
        context.sessions,
        raw_labels,
        stable_labels,
        active_labels,
        per_day_data_quality,
        per_day_evidence,
        strict=True,
    ):
        outputs[day] = CreditFundingOutput(
            raw_label=raw,
            stable_label=stable,
            active_label=active,
            evidence=evidence,
            data_quality=dq,
        )
    return outputs


def build_axis_series(
    context: MarketContext,
    feature_store: FeatureStore,
) -> dict[date, CreditFundingOutput] | None:
    """Real-OAS §2C credit/funding labels (free-function replacement for CreditFundingSeriesClassifier.build())."""
    return _build_for_spread_source(context, feature_store, spread_source="oas")


def build_axis_series_proxy(
    context: MarketContext,
    feature_store: FeatureStore,
) -> dict[date, CreditFundingOutput] | None:
    """TLT-proxy §2C labels (free-function replacement for CreditFundingSeriesClassifier.build_proxy())."""
    return _build_for_spread_source(context, feature_store, spread_source="proxy")
