"""v2 §2C Credit/Funding axis — feature compute.

Computes the §2C feature seam: the ``CreditFundingFeatures`` dataclass plus
``compute_credit_funding_features`` and provenance constants. The classify
layer (labels, rule inputs, predicates, precedence walker) lives in
``credit_funding_rules.py``.

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
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from regime_detection._rolling_stats import (
    rolling_change_zscore as _change_zscore,
    rolling_ols_slope,
)
from regime_detection.config import CreditFundingRulesConfig
from regime_shared.pit_provenance import make_bias_warnings_frame

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
    # F-010: forward-fill from the publication stamp (latest reading with date <= each
    # session), matching the AAII/EPS leak-safe pattern. A bare reindex(spy_index) lands
    # values only on exact-match dates, so a week-ending NFCI stamp on a NYSE-closed day
    # (e.g. Good Friday) would be dropped and the prior week carried. method="ffill"
    # honors the most-recent on-or-before reading; sort_index guards the requirement.
    nfci_w = nfci_weekly.sort_index().reindex(spy_index, method="ffill").astype(float)
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
    hy_oas_slope_21d = rolling_ols_slope(hy_oas_63d, window=slope_21d).rename(
        "hy_oas_slope_21d"
    )
    ig_oas_slope_21d = rolling_ols_slope(ig_oas_63d, window=slope_21d).rename(
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
    hy_tr_differential_slope_21d = rolling_ols_slope(
        hy_tr_differential_63d, window=slope_21d
    ).rename("hy_tr_differential_slope_21d")
    ig_tr_differential_slope_21d = rolling_ols_slope(
        ig_tr_differential_63d, window=slope_21d
    ).rename("ig_tr_differential_slope_21d")

    # §2C lines 3229-3230: bank-index relative strength.
    kre_spy_ratio = (kre / spy.where(spy > 0)).rename("kre_spy_ratio")
    kre_spy_slope_63d = rolling_ols_slope(kre_spy_ratio, window=slope_63d).rename(
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
    sofr_iorb_slope_21d = rolling_ols_slope(sofr_iorb_spread, window=slope_21d).rename(
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
