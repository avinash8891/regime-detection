"""v2 §1C Layer 1 V2 Volatility features — evidence-only compute (Slice 2.2).

Pure pandas/numpy implementation of the §1C continuous features that DO
NOT require options data. The new ``rising_vol`` / ``vol_crush`` labels
and the updated precedence at v2 §1C line 191 are deferred to a later
slice (per v2 §8: "Adds to existing classifiers without changing V1
contracts").

Features (all per-session series aligned to the input close index):

- ``atr_ratio``                       v2 §1C lines 140–143 (ATR_14 / ATR_50, Wilder)
- ``gap_frequency_20d``               v2 §1C lines 176–181
- ``intraday_range_percentile_252d``  v2 §1C lines 183–187

Deferred features (require external data not yet ingested — per v2 §10
absolute rule, do NOT invent missing inputs):

- ``iv_rv_spread`` (§1C lines 151–155) — needs options/implied-vol feed.
- ``vol_crush`` rule inputs (§1C lines 157–174) — needs implied_vol_5d_change
  AND the §2D event-window calendar.

Both deferrals are recorded in the Implementation Ambiguity Log in
``docs/regime_engine_v2_spec.md``.

Implementation choices that resolve ambiguities:

- **ATR estimator**: Wilder's recursive smoothing (the textbook standard).
  Pinned in the shared ``regime_detection.volatility_state.wilders_atr``
  helper so the future labels slice reuses one implementation.
- **gap_frequency_20d window inclusion**: 20 gap observations ending at
  ``t`` inclusive (consistent with slice 2.1 ``efficiency_ratio_20d``'s
  rolling-N convention).
- **intraday_range_percentile_252d**: ``Series.rolling(252).rank(pct=True)``
  with default ``ascending=True`` so a rising intraday range maps to a
  rising percentile (1.0 = current value is the highest in the window).
- **gap_threshold = 0.005**: pinned single US default; the spec note
  "configurable per market" (§1C line 181) is honored by the config knob
  ``VolatilityV2Config.gap_threshold_pct`` rather than per-market
  branching (V2 universe is US-only).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from regime_detection.config import VolatilityV2Config
from regime_detection.volatility_state import wilders_atr


@dataclass(frozen=True)
class VolatilityV2Features:
    """v2 §1C — per-session continuous volatility features (slice 2.2)."""

    atr_ratio: pd.Series
    gap_frequency_20d: pd.Series
    intraday_range_percentile_252d: pd.Series

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "atr_ratio",
            "gap_frequency_20d",
            "intraday_range_percentile_252d",
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: getattr(self, name) for name in self.feature_names}
        )


def _atr_ratio(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    short_period: int,
    long_period: int,
) -> pd.Series:
    """v2 §1C lines 140–143: ATR_short / ATR_long (Wilder).

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
    """v2 §1C lines 176–181.

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


def _intraday_range_percentile(
    *,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int,
) -> pd.Series:
    """v2 §1C lines 183–187.

        intraday_range[t]                  = (high[t] - low[t]) / close[t]
        intraday_range_percentile_252d[t]  = rolling(252).rank(pct=True) on the series

    The rolling rank is computed with the default ``ascending=True`` so a
    rising intraday range maps to a rising percentile (1.0 == current
    value is the maximum within the window). Mirrors slice 1.2's pattern
    in ``network_fragility.py``.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    intraday = (high - low) / close.where(close > 0)
    return (
        intraday.rolling(window=lookback, min_periods=lookback).rank(pct=True)
    ).rename("intraday_range_percentile_252d")


def compute_volatility_v2_features(
    *,
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    config: VolatilityV2Config,
) -> VolatilityV2Features:
    """Compute the three v2 §1C volatility features from a SPY-like OHLC.

    All parameters are sourced from ``VolatilityV2Config``; no magic
    numbers in the function body. Returns a frozen dataclass with each
    feature as a date-indexed ``pd.Series`` aligned to ``close.index``.
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
    intraday_pct = _intraday_range_percentile(
        high=high,
        low=low,
        close=close,
        lookback=config.intraday_range_lookback_days,
    )

    return VolatilityV2Features(
        atr_ratio=atr_ratio,
        gap_frequency_20d=gap_freq,
        intraday_range_percentile_252d=intraday_pct,
    )
