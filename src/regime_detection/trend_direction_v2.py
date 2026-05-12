"""v2 §1A Layer 1 V2 Trend Direction features — evidence-only compute.

Pure pandas/numpy implementation of the §1A continuous features that feed
the future V2 trend_direction classifier (slice 2.1 ships features only,
no classifier — see ``docs/regime_engine_v2_spec.md`` §8 line 1181).

Features (all per-session series aligned to the input close index):

- ``efficiency_ratio_20d``   v2 §1A lines 63–69
- ``hurst_250d``             v2 §1A lines 73–79
- ``slope_sma_50``           v2 §1A lines 105–108
- ``slope_sma_200``          v2 §1A lines 105–108
- ``return_63d``             v2 §1A line 117 (recovery evidence)
- ``return_126d``            v2 §1A line 124 (euphoria evidence)
- ``drawdown_252d``          v2 §1A line 116 (recovery evidence)

The downstream classifier wiring (new trend labels, updated precedence
at v2 §1A line 133) is deferred to a later slice until ``sentiment_score``
(§1A line 126) and ``followthrough_rate`` (§1A line 90) are resolved.

Implementation choices that resolve ambiguities are documented in
``docs/regime_engine_v2_spec.md`` Implementation Ambiguity Log:

- Hurst estimator: classical Rescaled-Range (R/S) analysis applied to
  log-returns (Mandelbrot–Wallis). See Ambiguity Log entry #11.
- ``drawdown_252d`` peak window: trailing 252 sessions INCLUDING ``t``
  (so drawdown == 0 at a fresh 252d high), matching the
  ``_trailing_drawdown`` convention in ``network_fragility_rules.py``.
  See Ambiguity Log entry #13.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_detection.config import TrendDirectionV2Config


# Spec-fixed constants (not configurable — v2 §1A line 67 defines the
# directional-move and path-length sums over the same N=lookback window).
_PATH_DIFF_SHIFT = 1


@dataclass(frozen=True)
class TrendDirectionV2Features:
    """v2 §1A — per-session continuous features for the future V2 trend
    classifier. Names match the spec line citations in the module docstring."""

    efficiency_ratio_20d: pd.Series
    hurst_250d: pd.Series
    slope_sma_50: pd.Series
    slope_sma_200: pd.Series
    return_63d: pd.Series
    return_126d: pd.Series
    drawdown_252d: pd.Series

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "efficiency_ratio_20d",
            "hurst_250d",
            "slope_sma_50",
            "slope_sma_200",
            "return_63d",
            "return_126d",
            "drawdown_252d",
        )

    def to_frame(self) -> pd.DataFrame:
        """All features as a single date-indexed DataFrame (one column per name)."""
        return pd.DataFrame(
            {name: getattr(self, name) for name in self.feature_names}
        )


def _efficiency_ratio(close: pd.Series, lookback: int) -> pd.Series:
    """v2 §1A lines 63–69.

        directional_move = abs(close[t] - close[t - N])
        path_length      = sum(abs(close[i] - close[i-1]) for i in [t-N+1..t])
        efficiency_ratio = directional_move / path_length

    Returns NaN when fewer than `lookback` prior sessions exist, when the
    window contains any NaN, or when path_length == 0 (constant series).
    """
    directional = (close - close.shift(lookback)).abs()
    abs_step = close.diff(_PATH_DIFF_SHIFT).abs()
    path_length = abs_step.rolling(window=lookback, min_periods=lookback).sum()
    ratio = directional / path_length.where(path_length > 0)
    return ratio.rename("efficiency_ratio_20d")


def _rs_hurst_window(values: np.ndarray) -> float:
    """Classical Mandelbrot–Wallis R/S Hurst estimator on a single window.

    Applied to log-returns of the window's price levels (literature
    standard — see Ambiguity Log entry #12). Returns NaN when the window
    is too short, has any NaN, or has zero variance.

    Algorithm (single chunk; no chunk averaging because the spec pins
    a single 250d window):
        r = log(P[1:]) - log(P[:-1])
        mean = r.mean()
        Z = cumsum(r - mean)
        R = max(Z) - min(Z)
        S = std(r, ddof=1)
        H = log(R/S) / log(N), where N = len(r)
    """
    if values.size < 4:
        return float("nan")
    if not np.all(np.isfinite(values)):
        return float("nan")
    if (values <= 0).any():
        return float("nan")
    log_returns = np.diff(np.log(values))
    if log_returns.size < 3:
        return float("nan")
    mean = log_returns.mean()
    deviations = log_returns - mean
    z = np.cumsum(deviations)
    r = z.max() - z.min()
    s = log_returns.std(ddof=1)
    if s <= 0 or r <= 0:
        return float("nan")
    n = float(log_returns.size)
    return float(np.log(r / s) / np.log(n))


def _hurst_series(close: pd.Series, lookback: int) -> pd.Series:
    """Rolling R/S Hurst exponent over `lookback` sessions. NaN until t>=lookback-1."""
    arr = close.to_numpy(dtype=float)
    n = arr.size
    out = np.full(n, np.nan, dtype=float)
    for t in range(lookback - 1, n):
        window = arr[t - lookback + 1 : t + 1]
        out[t] = _rs_hurst_window(window)
    return pd.Series(out, index=close.index, name="hurst_250d")


def _slope_of_sma(close: pd.Series, sma_period: int, slope_lookback: int) -> pd.Series:
    """v2 §1A line 106: (sma[t] - sma[t-N]) / sma[t-N].

    NaN propagates from the SMA (until t >= sma_period-1) and from the
    `slope_lookback` shift (until t >= sma_period-1+slope_lookback).
    """
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()
    prior = sma.shift(slope_lookback)
    return (sma - prior) / prior.where(prior != 0)


def _trailing_drawdown(close: pd.Series, lookback: int) -> pd.Series:
    """v2 §1A line 116: (close[t] / max(close[t-N+1..t])) - 1.

    Peak window is inclusive of t (matching
    ``network_fragility_rules._trailing_drawdown`` — Ambiguity Log #13).
    Drawdown == 0 when t is a fresh `lookback`-day high. Negative below.
    NaN if any of the window's `lookback` sessions is NaN or if t lacks
    `lookback` prior history.
    """
    peak = close.rolling(window=lookback, min_periods=lookback).max()
    return (close / peak.where(peak > 0)) - 1.0


def compute_trend_v2_features(
    close: pd.Series,
    *,
    config: TrendDirectionV2Config,
) -> TrendDirectionV2Features:
    """Compute the seven v2 §1A trend-direction features from a close series.

    All parameters are sourced from ``TrendDirectionV2Config``; no magic
    numbers in the function body. Returns a frozen dataclass with each
    feature as a date-indexed ``pd.Series`` aligned to ``close.index``.
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        # Match the existing v1 trend_direction.classify_series convention —
        # callers are expected to provide a datetime-indexed close. Coerce
        # only when the index is convertible; do not silently re-shape.
        close = close.copy()
        close.index = pd.to_datetime(close.index)

    eff = _efficiency_ratio(close, config.efficiency_ratio_lookback_days).rename(
        "efficiency_ratio_20d"
    )
    hurst = _hurst_series(close, config.hurst_lookback_days).rename("hurst_250d")
    slope_short = _slope_of_sma(
        close,
        sma_period=config.sma_short_period,
        slope_lookback=config.slope_lookback_days,
    ).rename("slope_sma_50")
    slope_long = _slope_of_sma(
        close,
        sma_period=config.sma_long_period,
        slope_lookback=config.slope_lookback_days,
    ).rename("slope_sma_200")
    ret_short = (close / close.shift(config.return_short_period) - 1.0).rename(
        "return_63d"
    )
    ret_long = (close / close.shift(config.return_long_period) - 1.0).rename(
        "return_126d"
    )
    dd = _trailing_drawdown(close, config.drawdown_lookback_days).rename(
        "drawdown_252d"
    )

    return TrendDirectionV2Features(
        efficiency_ratio_20d=eff,
        hurst_250d=hurst,
        slope_sma_50=slope_short,
        slope_sma_200=slope_long,
        return_63d=ret_short,
        return_126d=ret_long,
        drawdown_252d=dd,
    )
