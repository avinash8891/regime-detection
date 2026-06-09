"""v2 §1C Volatility axis — feature compute.

Both V1 and V2 features:
  - VolatilityFeatures: V1 (close, returns, realized_vol_21d, percentile,
    VIX percentile)
  - VolatilityV2Features: V2 (ATR ratio, gap_frequency, intraday_range,
    IV/RV spread, realized_vol_short/long, event_window_just_passed, etc.)

The classify layer (labels, risk rank, V1 walker, V2 rule predicates +
precedence) lives in ``volatility_state_rules.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_detection.config import VolatilityV2Config, VolatilityV2RulesConfig

# v2 §1C — annualization constant. Pinned here as the single source of truth
# for the shared ``realized_vol`` helper so v1 and v2
# (``rising_vol`` rule) consume one annualization convention.
_TRADING_DAYS_PER_YEAR = 252


def realized_vol(close: pd.Series, window: int, *, ddof: int = 1) -> pd.Series:
    """Rolling annualised realised volatility of ``close`` log/pct returns.

    Shared helper for v1 volatility classifiers and the v2 §1C
    ``rising_vol`` rule.

    Algorithm:
        daily_returns = close.pct_change(fill_method=None)
        realized_vol[t] = rolling(window).std(ddof=ddof) * sqrt(252)

    Args:
        close: per-session close prices (date-indexed).
        window: rolling-window length in trading days.
        ddof: delta degrees-of-freedom for ``Series.std``. Pinned default
            ``1`` (sample std) matches pandas/numpy convention for
            financial time series; v2 §1C is silent so the convention is
            recorded here as the engine pin.

    Returns a date-indexed ``pd.Series`` aligned to ``close.index``; NaN
    until ``t >= window`` (cold-start: pandas ``rolling.std`` requires
    ``window`` observations before emitting).
    """
    if window <= 0:
        raise ValueError(f"window must be > 0: got {window}")
    daily_returns = close.pct_change(fill_method=None)
    return daily_returns.rolling(window).std(ddof=ddof) * np.sqrt(
        _TRADING_DAYS_PER_YEAR
    )


def wilders_atr(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Wilder's Average True Range over `period` sessions.

    Shared helper for v1 volatility classifiers and v2 §1C features
    (atr_ratio = ATR_14 / ATR_50). Wilder's smoothing is the standard
    estimator referenced by v2 §1C lines 247-249 (implementation decision #15
    pins classical Wilder recursive smoothing).

    NOTE: a separate ``_wilder_ewm`` helper lives in
    ``regime_detection.trend_character`` for the v1 ADX cold-start
    path. ``_wilder_ewm`` uses pandas-EWM seeding (first value of the
    TR series), whereas ``wilders_atr`` here uses the textbook
    Wilder-1978 mean-seeded form (seed = simple-mean(TR[0..period-1])).
    Both converge for large ``t`` but differ at cold-start. The two
    implementations intentionally coexist (v1 ADX cold-start values are
    frozen; V2 §1C ATR ratio uses the more faithful textbook form). A
    future cleanup may unify them after V2 walk-forward validation per
    v2 §9.1.

    Algorithm:
        TR[t] = max(
            high[t] - low[t],
            abs(high[t] - close[t-1]),
            abs(low[t]  - close[t-1]),
        )
        ATR[0..period-2] = NaN
        ATR[period-1]    = mean(TR[0..period-1])              # seed
        ATR[t]           = (ATR[t-1] * (period-1) + TR[t]) / period   # t >= period

    Returns a date-indexed pd.Series aligned to ``close.index``.
    """
    if period <= 0:
        raise ValueError(f"period must be > 0: got {period}")
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, close must have identical length")

    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    n = len(tr)
    out = np.full(n, np.nan, dtype=float)
    tr_arr = tr.to_numpy(dtype=float)
    if n < period:
        return pd.Series(out, index=close.index, name=f"atr_{period}")

    # Seed: simple mean of the first `period` true-range values. If any of
    # those is NaN the seed is NaN and Wilder's recursion stays NaN forever.
    seed_window = tr_arr[:period]
    if np.isnan(seed_window).any():
        seed = float("nan")
    else:
        seed = float(seed_window.mean())
    out[period - 1] = seed

    for t in range(period, n):
        prev = out[t - 1]
        cur = tr_arr[t]
        if np.isnan(prev) or np.isnan(cur):
            out[t] = float("nan")
            continue
        out[t] = (prev * (period - 1) + cur) / period

    return pd.Series(out, index=close.index, name=f"atr_{period}")


def _pct_rank_last(arr: np.ndarray) -> float:
    x = arr[-1]
    if np.isnan(x):
        return float("nan")
    arr2 = arr[~np.isnan(arr)]
    if arr2.size == 0:
        return float("nan")
    return float(np.mean(arr2 <= x))


@dataclass(frozen=True)
class VolatilityFeatures:
    close: pd.Series
    return_1d: pd.Series
    return_5d: pd.Series
    return_21d: pd.Series
    realized_vol_21d: pd.Series
    realized_vol_percentile_252d: pd.Series
    vix_percentile_252d: pd.Series | None


def compute_features(
    *, close: pd.Series, vix_proxy_close: pd.Series | None
) -> VolatilityFeatures:
    close = close.astype(float)
    vix_pct: pd.Series | None = None
    if vix_proxy_close is not None:
        vix_proxy_close = vix_proxy_close.astype(float)
        # Align VIX proxy series to the SPY trading-day index; missing dates become NaN.
        vix_proxy_close = vix_proxy_close.reindex(close.index)
        vix_pct = vix_proxy_close.rolling(252, min_periods=252).apply(
            _pct_rank_last, raw=True
        )

    return_1d = close / close.shift(1) - 1
    return_5d = close / close.shift(5) - 1
    return_21d = close / close.shift(21) - 1

    # v1 RV percentile feeds the v1 high_vol/low_vol thresholds. Uses the
    # shared ``realized_vol`` helper — preserves the v1 byte-
    # identical output (window=21, ddof default — pandas .std() is ddof=1).
    realized_vol_21d = realized_vol(close, window=21)
    realized_vol_percentile_252d = realized_vol_21d.rolling(252, min_periods=252).apply(
        _pct_rank_last, raw=True
    )

    return VolatilityFeatures(
        close=close,
        return_1d=return_1d,
        return_5d=return_5d,
        return_21d=return_21d,
        realized_vol_21d=realized_vol_21d,
        realized_vol_percentile_252d=realized_vol_percentile_252d,
        vix_percentile_252d=vix_pct,
    )


@dataclass(frozen=True)
class VolatilityV2Features:
    """v2 §1C — per-session continuous volatility features.

    Fields: atr_ratio, gap_frequency_20d, intraday_range,
    intraday_range_percentile_252d,
    plus the two realized-vol windows used by the `rising_vol` rule
    (v2 §1C lines 251-252): a short-window realised vol (default 10d) and a
    long-window realised vol (default 63d), both annualised via the shared
    ``regime_detection.volatility_state.realized_vol`` helper.
    """

    atr_ratio: pd.Series
    gap_frequency_20d: pd.Series
    # v2 §1E line 417 / engine-pinned implementation decision — 252d percentile rank of
    # `gap_frequency_20d`. Consumed by the §1E `liquidity_gap_behavior`
    # rule. Computed here (rather than at the rule layer) so the percentile
    # shares the volatility seam's session index and the rule layer reads
    # only scalars.
    gap_frequency_percentile_252d: pd.Series
    intraday_range: pd.Series
    intraday_range_percentile_252d: pd.Series
    # v2 §1C lines 251-252 — `rising_vol` rule inputs.
    realized_vol_short: pd.Series
    realized_vol_long: pd.Series
    # v2 §1C `vol_crush` rule input (engine-pinned implementation decision). 21-session
    # realized vol — the mid window for `realized_vol_10d < realized_vol_21d
    # * 0.75`. Always computable from close; never None.
    realized_vol_21d: pd.Series
    # v2 §1C IV features (engine-pinned implementation decision). Optional — populated
    # only when `implied_vol_30d` (FRED VIXCLS / 100) is supplied to
    # `compute_volatility_v2_features`. When None, `vol_crush` falsifies
    # (V1 byte-identity preserved).
    implied_vol_30d: pd.Series | None = None
    implied_vol_5d_change: pd.Series | None = None  # relative 5-session change
    iv_rv_spread: pd.Series | None = None  # implied_vol_30d - realized_vol_21d
    # v2 §1C `vol_crush` rule input (ADR 0005 Q3). Optional per-session
    # boolean — populated only when an event calendar is supplied. When
    # None, `vol_crush` falsifies.
    event_window_just_passed: pd.Series | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "atr_ratio",
            "gap_frequency_20d",
            "gap_frequency_percentile_252d",
            "intraday_range_percentile_252d",
            "realized_vol_short",
            "realized_vol_long",
            "realized_vol_21d",
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({name: getattr(self, name) for name in self.feature_names})


def _atr_ratio(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    short_period: int,
    long_period: int,
) -> pd.Series:
    """v2 §1C lines 247-249: ATR_short / ATR_long (Wilder).

    NaN until ``t >= long_period - 1`` (long ATR cold-start) and when the
    long ATR is zero (constant-OHLC series → 0/0 = NaN by definition).
    """
    atr_short = wilders_atr(high=high, low=low, close=close, period=short_period)
    atr_long = wilders_atr(high=high, low=low, close=close, period=long_period)
    return (atr_short / atr_long.where(atr_long > 0)).rename("atr_ratio")


def _gap_frequency(
    *,
    open_: pd.Series,
    close: pd.Series,
    lookback: int,
    threshold_pct: float,
) -> pd.Series:
    """v2 §1C lines 297-301 (implementation decision #16).

        gap[t] = abs(open[t] - close[t-1]) / close[t-1]
        gap_frequency_20d[t] = count(gap[i] > threshold for i in [t-N+1..t]) / N

    Strictly ``> threshold`` (an exact-threshold gap does NOT count, per
    spec text "gap > 0.005").
    """
    open_ = open_.astype(float)
    close = close.astype(float)
    prev_close = close.shift(1)
    gap = (open_ - prev_close).abs() / prev_close.where(prev_close > 0)
    is_large = (gap > threshold_pct).astype(float)
    # NaN propagation: keep NaN where the input gap is NaN (first session).
    is_large = is_large.where(gap.notna())
    return (
        is_large.rolling(window=lookback, min_periods=lookback).sum() / lookback
    ).rename("gap_frequency_20d")


def _intraday_range_series(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    return ((high - low) / close.where(close > 0)).rename("intraday_range")


def _intraday_range_percentile(
    *,
    intraday_range: pd.Series,
    lookback: int,
) -> pd.Series:
    """v2 §1C lines 304-306 (implementation decision #17).

        intraday_range[t]                  = (high[t] - low[t]) / close[t]
        intraday_range_percentile_252d[t]  = rolling(252).rank(pct=True) on the series

    The rolling rank is computed with the default ``ascending=True`` so a
    rising intraday range maps to a rising percentile (1.0 == current
    value is the maximum within the window). Mirrors the pattern
    in ``network_fragility.py``.
    """
    return (
        intraday_range.rolling(window=lookback, min_periods=lookback).rank(pct=True)
    ).rename("intraday_range_percentile_252d")


def compute_volatility_v2_features(
    *,
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    config: VolatilityV2Config,
    rules_config: VolatilityV2RulesConfig | None = None,
    implied_vol_30d: pd.Series | None = None,
    event_window_just_passed: pd.Series | None = None,
) -> VolatilityV2Features:
    """Compute the v2 §1C volatility features from a SPY-like OHLC.

    All parameters are sourced from ``VolatilityV2Config``; no magic
    numbers in the function body. Returns a frozen dataclass with each
    feature as a date-indexed ``pd.Series`` aligned to ``close.index``.

    ``implied_vol_30d`` (FRED VIXCLS / 100, decimal-annualized) is
    optional — when supplied, the `vol_crush` IV features
    (`implied_vol_5d_change`, `iv_rv_spread`) are computed; when None,
    those features stay None and the `vol_crush` rule falsifies (V1
    byte-identity preserved). ``event_window_just_passed`` is the
    optional per-session boolean from
    ``regime_detection.event_calendar.compute_event_window_just_passed``
    — same Optional contract.
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        open_ = open_.copy()
        high = high.copy()
        low = low.copy()
        close = close.copy()
        new_index = pd.to_datetime(close.index)
        open_.index = new_index
        high.index = new_index
        low.index = new_index
        close.index = new_index

    atr_ratio = _atr_ratio(
        high=high,
        low=low,
        close=close,
        short_period=config.atr_short_period,
        long_period=config.atr_long_period,
    )
    gap_freq = _gap_frequency(
        open_=open_,
        close=close,
        lookback=config.gap_frequency_lookback_days,
        threshold_pct=config.gap_threshold_pct,
    )
    # v2 §1E line 417 — 252d percentile rank of `gap_frequency_20d`. Same
    # rolling-rank shape as `intraday_range_percentile_252d` below and the
    # §1D `nh_nl_ratio` percentile pattern. Implements the documented input contract by computing
    # the previously-missing percentile input for `liquidity_gap_behavior`.
    gap_freq_pct = (
        gap_freq.rolling(
            config.intraday_range_lookback_days,
            min_periods=config.intraday_range_lookback_days,
        )
        .rank(pct=True)
        .rename("gap_frequency_percentile_252d")
    )
    intraday_range = _intraday_range_series(
        high=high,
        low=low,
        close=close,
    )
    intraday_pct = _intraday_range_percentile(
        intraday_range=intraday_range,
        lookback=config.intraday_range_lookback_days,
    )

    # v2 §1C lines 251-252 — `rising_vol` rule inputs. Computed via
    # the shared ``regime_detection.volatility_state.realized_vol`` helper
    # so v1 (realized_vol_21d) and v2 (rv_10d/rv_63d) consume one
    # annualisation path. When no rules_config is supplied,
    # default to spec windows so callers that read the feature seam without
    # explicit rule configuration still get a complete struct.
    if rules_config is not None:
        rv_short_window = rules_config.realized_vol_short_period
        rv_long_window = rules_config.realized_vol_long_period
    else:
        # Spec defaults — v2 §1C lines 251-252 (realized_vol_10d / realized_vol_63d).
        # Hardcoded fallback values intentionally match VolatilityV2RulesConfig
        # defaults; both citations point at v2 §1C lines 251-252.
        rv_short_window = 10
        rv_long_window = 63
    rv_short = realized_vol(close, window=rv_short_window).rename("realized_vol_short")
    rv_long = realized_vol(close, window=rv_long_window).rename("realized_vol_long")

    # v2 §1C `vol_crush` rule input (ADR 0005). The 21-session mid window
    # for `realized_vol_10d < realized_vol_21d * 0.75`.
    rv_mid_window = (
        rules_config.vol_crush_realized_vol_mid_period
        if rules_config is not None
        else 21  # spec default — "realized_vol_21d"
    )
    rv_mid = realized_vol(close, window=rv_mid_window).rename("realized_vol_21d")

    # v2 §1C IV features (ADR 0005). Computed only when implied_vol_30d
    # is supplied; otherwise None — the vol_crush rule then falsifies.
    iv_change_lookback = (
        rules_config.vol_crush_implied_vol_change_lookback_sessions
        if rules_config is not None
        else 5  # ADR 0005 Q1 default
    )
    implied_vol_aligned: pd.Series | None = None
    implied_vol_5d_change: pd.Series | None = None
    iv_rv_spread: pd.Series | None = None
    if implied_vol_30d is not None:
        implied_vol_aligned = (
            implied_vol_30d.reindex(close.index).astype(float).rename("implied_vol_30d")
        )
        # Relative N-session change (ADR 0005 Q1 — unit-agnostic).
        prior_iv = implied_vol_aligned.shift(iv_change_lookback)
        implied_vol_5d_change = (
            (implied_vol_aligned - prior_iv) / prior_iv.where(prior_iv != 0)
        ).rename("implied_vol_5d_change")
        # iv_rv_spread (§1C) — both operands decimal-annualized.
        iv_rv_spread = (implied_vol_aligned - rv_mid).rename("iv_rv_spread")

    event_window_aligned: pd.Series | None = None
    if event_window_just_passed is not None:
        event_window_aligned = (
            event_window_just_passed.astype("boolean")
            .reindex(close.index, fill_value=False)
            .fillna(False)
            .astype(bool)
            .rename("event_window_just_passed")
        )

    return VolatilityV2Features(
        atr_ratio=atr_ratio,
        gap_frequency_20d=gap_freq,
        gap_frequency_percentile_252d=gap_freq_pct,
        intraday_range=intraday_range,
        intraday_range_percentile_252d=intraday_pct,
        realized_vol_short=rv_short,
        realized_vol_long=rv_long,
        realized_vol_21d=rv_mid,
        implied_vol_30d=implied_vol_aligned,
        implied_vol_5d_change=implied_vol_5d_change,
        iv_rv_spread=iv_rv_spread,
        event_window_just_passed=event_window_aligned,
    )
