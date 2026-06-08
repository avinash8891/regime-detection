from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from regime_detection._rule_helpers import ev_float as _ev_float
from regime_detection.config import VolatilityV2Config, VolatilityV2RulesConfig
from regime_shared.pandas_compat import require_single_session

# v2 §1C line 311 precedence:
#   crisis_vol > vol_crush > high_vol > rising_vol > low_vol > normal_vol > unknown
# `rising_vol` added per v2 §1C lines 250-254;
# `vol_crush` added per ADR 0005 using FRED VIXCLS as
# implied_vol_30d plus the event-window seam.
VolatilityLabel = Literal[
    "low_vol",
    "normal_vol",
    "high_vol",
    "crisis_vol",
    "unknown",
    "rising_vol",
    "vol_crush",
]


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


# v2 §1C line 311 precedence:
#   crisis_vol > vol_crush > high_vol > rising_vol > low_vol >
#   normal_vol > unknown.
# V1 risk-rank contract is frozen in replay fixtures; crisis_vol remains 3.
# V2 crisis-vs-vol_crush precedence is resolved before hysteresis in
# volatility_state_v2, not by changing the V1 evidence rank.
_RISK_RANK: dict[VolatilityLabel, int] = {
    "low_vol": 0,
    "normal_vol": 1,
    "high_vol": 2,
    "crisis_vol": 3,
    "unknown": 2,
    "rising_vol": 2,
    "vol_crush": 3,
}


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


def raw_label_for_day(
    f: VolatilityFeatures,
    dt: pd.Timestamp,
    *,
    volatility_state_v2_features: "VolatilityV2Features | None" = None,
    volatility_state_v2_rules: "VolatilityV2RulesConfig | None" = None,
) -> tuple[VolatilityLabel, dict[str, Any]]:
    """Per-day raw volatility_state label.

    When ``volatility_state_v2_features`` AND ``volatility_state_v2_rules``
    are both supplied, the v2 §1C precedence (line 191:
    ``crisis_vol > vol_crush > high_vol > rising_vol > low_vol >
    normal_vol > unknown``) is layered ON TOP of the v1 label. When either
    is ``None`` the function returns the v1 label and evidence unchanged.

    F-043: this is a thin wrapper over :func:`build_raw_outputs` so the §5.5
    rule predicates, evidence shape, and v2 override have a single encoding.
    Slicing each feature to ``[dt]`` is safe because the vectorized builder and
    ``evaluate_v2_volatility_label`` only read values at the target session.
    """
    # Guard: dt must resolve to exactly one session — a duplicate-date index would make
    # .loc[[dt]] return multiple rows and labels[0] silently mask the data issue.
    require_single_session(f.close.index, dt)
    day_features = VolatilityFeatures(
        close=f.close.loc[[dt]],
        return_1d=f.return_1d.loc[[dt]],
        return_5d=f.return_5d.loc[[dt]],
        return_21d=f.return_21d.loc[[dt]],
        realized_vol_21d=f.realized_vol_21d.loc[[dt]],
        realized_vol_percentile_252d=f.realized_vol_percentile_252d.loc[[dt]],
        vix_percentile_252d=(
            None if f.vix_percentile_252d is None else f.vix_percentile_252d.loc[[dt]]
        ),
    )
    labels, evidence = build_raw_outputs(
        day_features,
        volatility_state_v2_features=volatility_state_v2_features,
        volatility_state_v2_rules=volatility_state_v2_rules,
    )
    return labels[0], evidence[0]


def build_raw_outputs(
    f: VolatilityFeatures,
    *,
    volatility_state_v2_features: "VolatilityV2Features | None" = None,
    volatility_state_v2_rules: "VolatilityV2RulesConfig | None" = None,
) -> tuple[list[VolatilityLabel], list[dict[str, Any]]]:
    """Vectorized v1 raw labels + optional v2 §1C `rising_vol` override.

    When both v2 args are supplied, the v2 §1C precedence at line 191 is applied per-day AFTER
    the v1 pass — `rising_vol` overrides v1 `low_vol` / `normal_vol` /
    `unknown` (NOT `crisis_vol` / `high_vol`, which outrank `rising_vol`).
    When either argument is None, output is byte-identical to v1.
    """
    ret1 = f.return_1d
    ret5 = f.return_5d
    ret21 = f.return_21d
    vol_pct = f.realized_vol_percentile_252d
    valid = ~(ret1.isna() | ret5.isna() | ret21.isna() | vol_pct.isna())

    rv21 = f.realized_vol_21d
    vix_pct_series = pd.Series(float("nan"), index=ret1.index, dtype=float)
    vix_present = pd.Series(False, index=ret1.index, dtype="bool")
    vix_crisis = pd.Series(False, index=ret1.index, dtype="bool")
    vix_high = pd.Series(False, index=ret1.index, dtype="bool")
    if f.vix_percentile_252d is not None:
        vix_pct_series = f.vix_percentile_252d.reindex(ret1.index)
        vix_present = vix_pct_series.notna()
        vix_crisis = vix_present & vix_pct_series.ge(0.95)
        vix_high = vix_present & vix_pct_series.ge(0.80)

    crisis = valid & (
        ret1.le(-0.05)
        | ret5.le(-0.08)
        | (vol_pct.ge(0.90) & ret21.le(-0.05))
        | vix_crisis
    )
    high_vol = valid & (vol_pct.ge(0.80) | vix_high)
    low_vol = valid & vol_pct.le(0.30)
    normal_vol = valid & ~(crisis | high_vol | low_vol)

    labels = np.full(len(ret1), "unknown", dtype=object)
    labels[normal_vol.to_numpy()] = "normal_vol"
    labels[low_vol.to_numpy()] = "low_vol"
    labels[high_vol.to_numpy()] = "high_vol"
    labels[crisis.to_numpy()] = "crisis_vol"

    evidence: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        if label == "unknown":
            evidence.append({"reason": "insufficient_history"})
            continue
        evidence.append(
            {
                "realized_vol_21d": _ev_float(rv21.iat[idx]),
                "realized_vol_percentile_252d": _ev_float(vol_pct.iat[idx]),
                "vix_percentile_252d": (
                    _ev_float(vix_pct_series.iat[idx])
                    if bool(vix_present.iat[idx])
                    else None
                ),
                "crisis_vol": bool(crisis.iat[idx]),
                "high_vol": bool(high_vol.iat[idx]),
                "low_vol": bool(low_vol.iat[idx]),
            }
        )

    # Both v2 evidence enrichment (iv_rv_spread) and the v2 §1C label override apply
    # ONLY when BOTH v2 args are present — otherwise the output is byte-identical to v1
    # (docstring contract). Gating the iv_rv_spread block on features alone would change
    # the evidence shape on a partial v2 arg.
    if (
        volatility_state_v2_features is not None
        and volatility_state_v2_rules is not None
    ):
        iv_rv = volatility_state_v2_features.iv_rv_spread
        for idx, dt in enumerate(ret1.index):
            if labels[idx] != "unknown" and iv_rv is not None and dt in iv_rv.index:
                val = iv_rv.loc[dt]
                if not pd.isna(val):
                    evidence[idx]["iv_rv_spread"] = _ev_float(val)

        # v2 §1C line 311 precedence — applied per-day on top of v1.
        for idx, dt in enumerate(ret1.index):
            v1_label = str(labels[idx])
            v2_label = evaluate_v2_volatility_label(
                v1_label=v1_label,
                features=volatility_state_v2_features,
                dt=dt,
                rules_config=volatility_state_v2_rules,
            )
            if v2_label is None:
                continue
            evidence[idx]["v2_override"] = {
                "from": v1_label,
                "to": v2_label,
                "rule": v2_label,  # the winning v2 §1C rule (rising_vol / vol_crush)
            }
            labels[idx] = v2_label

    return list(labels), evidence


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


# ---------------------------------------------------------------------------
# v2 §1C `rising_vol` rule + precedence wrapper.
#
# Rule (v2 §1C lines 250-254, verbatim):
#     ATR_ratio > 1.15
#     OR realized_vol_10d > realized_vol_63d * 1.25
#
# Precedence (v2 §1C line 311):
#     crisis_vol > vol_crush > high_vol > rising_vol > low_vol > normal_vol > unknown
#
# `vol_crush` is wired via engine-pinned implementation decision using FRED VIXCLS-derived
# implied_vol_30d plus event_window_just_passed.
# ---------------------------------------------------------------------------


def evaluate_rising_vol(
    features: VolatilityV2Features,
    *,
    dt: pd.Timestamp,
    rules_config: VolatilityV2RulesConfig,
) -> bool:
    """v2 §1C lines 250-254 `rising_vol` predicate at a single session.

    Returns False when ANY of the three inputs is NaN — strict cold-start
    contract (no silent "partial-input → True" substitution). Both limbs
    use strict ``>`` per spec text:

    * line 251 — ``atr_ratio > atr_ratio_threshold`` (1.15)
    * line 252 — ``realized_vol_short > realized_vol_long * realized_vol_ratio_threshold`` (1.25)
    * Combined: ATR limb OR realised-vol limb.

    The all-inputs-must-be-present contract is recorded in the
    engine-pinned implementation decision — spec §1C is silent on
    partial-NaN behavior so the conservative choice is "any NaN
    falsifies the rule" (matches recovery cold-start).
    """
    if dt not in features.atr_ratio.index:
        return False
    atr = features.atr_ratio.loc[dt]
    rv_short = features.realized_vol_short.loc[dt]
    rv_long = features.realized_vol_long.loc[dt]

    # Strict cold-start: any missing input falsifies the rule
    # (engine-pinned implementation decision — partial-NaN handling).
    if any(pd.isna(x) for x in (atr, rv_short, rv_long)):
        return False

    atr_limb = bool(atr > rules_config.atr_ratio_threshold)  # line 251
    rv_limb = bool(
        rv_short > rv_long * rules_config.realized_vol_ratio_threshold
    )  # line 252
    return atr_limb or rv_limb


def evaluate_vol_crush(
    features: VolatilityV2Features,
    *,
    dt: pd.Timestamp,
    rules_config: VolatilityV2RulesConfig,
) -> bool:
    """v2 §1C `vol_crush` predicate at a single session (engine-pinned implementation decision).

    Rule (spec §1C):
      realized_vol_short < realized_vol_21d * vol_crush_realized_vol_ratio_threshold
      AND implied_vol_5d_change <= vol_crush_implied_vol_change_threshold
      AND event_window_just_passed

    Returns False when:
      - the Optional IV features are absent (no `implied_vol_30d` was
        supplied — `implied_vol_5d_change` is None),
      - the Optional `event_window_just_passed` series is absent (no
        event calendar was supplied),
      - any required input is NaN at ``dt`` (V1 §2.7 cold-start), or
      - ``dt`` is outside any of the input series' indices.

    All three guards collapse to the same outcome: when `vol_crush`'s
    extra data inputs are not wired, the rule simply does not fire and
    the precedence walker keeps the v1/`rising_vol` label.
    """
    iv_change = features.implied_vol_5d_change
    event_window = features.event_window_just_passed
    if iv_change is None or event_window is None:
        return False
    if (
        dt not in features.realized_vol_short.index
        or dt not in features.realized_vol_21d.index
        or dt not in iv_change.index
        or dt not in event_window.index
    ):
        return False

    rv_short = features.realized_vol_short.loc[dt]
    rv_mid = features.realized_vol_21d.loc[dt]
    iv_change_t = iv_change.loc[dt]
    if any(pd.isna(x) for x in (rv_short, rv_mid, iv_change_t)):
        return False

    rv_collapsed = bool(
        rv_short < rv_mid * rules_config.vol_crush_realized_vol_ratio_threshold
    )
    iv_falling_sharply = bool(
        iv_change_t <= rules_config.vol_crush_implied_vol_change_threshold
    )
    event_just_passed = bool(event_window.loc[dt])
    return rv_collapsed and iv_falling_sharply and event_just_passed


# v2 §1C line 311 ranking (lower index = higher precedence).
# `vol_crush` (index 1) was reserved-but-inert before engine-pinned implementation decision
# closure; it is now wired to a real predicate.
_V2_VOLATILITY_PRECEDENCE: tuple[str, ...] = (
    "crisis_vol",
    "vol_crush",
    "high_vol",
    "rising_vol",
    "low_vol",
    "normal_vol",
    "unknown",
)


def evaluate_v2_volatility_label(
    *,
    v1_label: str,
    features: VolatilityV2Features,
    dt: pd.Timestamp,
    rules_config: VolatilityV2RulesConfig,
) -> str | None:
    """Apply v2 §1C volatility precedence on top of a v1 raw label.

    Returns the winning v2 label per the §1C line 311 ordering, or
    ``None`` when no v2 rule fires and the caller should keep ``v1_label``.

    Precedence (line 191): ``crisis_vol > vol_crush > high_vol >
    rising_vol > low_vol > normal_vol > unknown``.

    Dispatch order: `vol_crush` first (rank 1 — outranks high_vol /
    rising_vol; only `crisis_vol` outranks it). When `vol_crush` fires
    AND the v1 label is not `crisis_vol`, returns ``"vol_crush"``. Then
    `rising_vol` (rank 3): fires only when the v1 label is ranked
    strictly LOWER — i.e. v1 emitted ``low_vol`` / ``normal_vol`` /
    ``unknown``.
    """
    try:
        v1_rank = _V2_VOLATILITY_PRECEDENCE.index(v1_label)
    except ValueError:
        # Unknown v1 label — treat as lowest precedence.
        v1_rank = len(_V2_VOLATILITY_PRECEDENCE)

    # vol_crush (rank 1) — only crisis_vol outranks it. Fires when the
    # predicate is true AND v1 did not emit crisis_vol.
    vol_crush_rank = _V2_VOLATILITY_PRECEDENCE.index("vol_crush")
    if v1_rank >= vol_crush_rank and evaluate_vol_crush(
        features, dt=dt, rules_config=rules_config
    ):
        return "vol_crush"

    # rising_vol (rank 3) — fires only when v1 is ranked strictly lower.
    rising_vol_rank = _V2_VOLATILITY_PRECEDENCE.index("rising_vol")
    if v1_rank < rising_vol_rank:
        # v1 label outranks rising_vol (crisis_vol / vol_crush / high_vol).
        return None
    if evaluate_rising_vol(features, dt=dt, rules_config=rules_config):
        return "rising_vol"
    return None
