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
- ``sma_50``                 v2 §1A line 118 (recovery evidence: close > SMA_50)

Slice 2.5 lands the ``recovery`` label on top of these features. The
``euphoria`` / ``breakout_expansion`` / ``range_bound`` labels remain
deferred (see Ambiguity Log entries #32–#34).

Implementation choices that resolve ambiguities are documented in
``docs/regime_engine_v2_spec.md`` Implementation Ambiguity Log:

- Hurst estimator: classical Rescaled-Range (R/S) analysis applied to
  log-returns (Mandelbrot–Wallis). See Ambiguity Log entry #11.
- ``drawdown_252d`` peak window: trailing 252 sessions INCLUDING ``t``
  (so drawdown == 0 at a fresh 252d high), matching the
  ``_trailing_drawdown`` convention in ``network_fragility_rules.py``.
  See Ambiguity Log entry #13.
"""
# TODO(slice-2.x): v1 `trend_direction.compute_features` and `feature_store.build_feature_store`
# also compute sma_50, sma_200, and return_63d from the SPY close. Consolidate into a shared
# rolling-stats utility when the v2 trend labels slice lands (euphoria/recovery/breakout).
# Until then, the three computations are independent (acceptable per evidence-only slice scope).
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_detection.config import TrendDirectionV2Config, TrendDirectionV2RulesConfig
from regime_detection.volatility_state import realized_vol


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
    # v2 §1A line 118 — `recovery` rule input: close > SMA_50.
    # Exposed as a level (not a slope) so the slice-2.5 recovery predicate
    # has direct access without recomputing the 50d SMA.
    sma_50: pd.Series
    # v2 §1A line 161 — `euphoria` rule input: close > SMA_200. Exposed as
    # a level so the euphoria predicate compares against close[t] directly
    # without recomputing the 200d SMA (already computed for slope_sma_200).
    sma_200: pd.Series
    # v2 §1A line 163 — `euphoria` rule input: realized_vol_21d rising
    # (strict 5-session change per ADR 0004 Q2 / Log #68 §1D analogue).
    # 21-session annualized realized vol of SPY log-returns.
    realized_vol_21d: pd.Series
    # v2 §1A line 164 — `euphoria` rule input: sentiment_score >= threshold.
    # Optional because not every engine call wires AAII sentiment; when None
    # the euphoria predicate falsifies (V2 §10 "do not invent a proxy").
    # When supplied, the series is forward-filled from AAII weekly readings
    # onto the SPY session index per ADR 0004 Q4 (V1 §2.2 stateless replay).
    sentiment_score: pd.Series | None = None

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
            "sma_50",
            "sma_200",
            "realized_vol_21d",
        )

    def to_frame(self) -> pd.DataFrame:
        """All non-Optional features as a single date-indexed DataFrame.

        ``sentiment_score`` is excluded from the default frame view because
        it is Optional and routinely None when no AAII feed is wired. Use
        ``getattr(features, "sentiment_score")`` for direct access.
        """
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


def _sma(close: pd.Series, sma_period: int) -> pd.Series:
    """Rolling simple moving average with strict cold-start (NaN until
    `sma_period` observations are available)."""
    return close.rolling(window=sma_period, min_periods=sma_period).mean()


def _slope_of_sma(sma: pd.Series, slope_lookback: int) -> pd.Series:
    """v2 §1A line 106: (sma[t] - sma[t-N]) / sma[t-N].

    Accepts a pre-computed SMA series so callers can both expose the SMA
    level (slice 2.5 `recovery` predicate consumes ``sma_50``) and its
    slope in one pass.

    NaN propagates from the SMA (until t >= sma_period-1) and from the
    `slope_lookback` shift (until t >= sma_period-1+slope_lookback).
    """
    prior = sma.shift(slope_lookback)
    return (sma - prior) / prior.where(prior != 0)


def compute_trailing_drawdown(close: pd.Series, lookback: int) -> pd.Series:
    """v2 §1A line 116: (close[t] / max(close[t-N+1..t])) - 1.

    Peak window is inclusive of t (matching
    ``network_fragility_rules._trailing_drawdown`` — Ambiguity Log #13).
    Drawdown == 0 when t is a fresh `lookback`-day high. Negative below.
    NaN if any of the window's `lookback` sessions is NaN or if t lacks
    `lookback` prior history.

    Public name added in Slice 6: the HMM evidence layer (§6.1) reuses
    this formula for its `drawdown_63d` input — one home (AGENTS rule B).
    """
    peak = close.rolling(window=lookback, min_periods=lookback).max()
    return (close / peak.where(peak > 0)) - 1.0


# Internal alias preserved for in-module callers (slice 2.1 + 2.5 code paths).
_trailing_drawdown = compute_trailing_drawdown


def compute_trend_v2_features(
    close: pd.Series,
    *,
    config: TrendDirectionV2Config,
    sentiment_score: pd.Series | None = None,
) -> TrendDirectionV2Features:
    """Compute the nine v2 §1A trend-direction features from a close series.

    All parameters are sourced from ``TrendDirectionV2Config``; no magic
    numbers in the function body. Returns a frozen dataclass with each
    feature as a date-indexed ``pd.Series`` aligned to ``close.index``.

    ``sentiment_score`` is an Optional pre-aligned series carrying the
    forward-filled AAII bull-bear-spread 8w-MA on the close index (per
    ADR 0004 Q1 / Q4 — V1 §2.2 stateless replay; values originate from
    weekly publication dates ≤ as_of). When omitted, the v2 §1A
    `euphoria` rule falsifies on every session (V2 §10 "do not invent
    a sentiment proxy").
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
    sma_short = _sma(close, config.sma_short_period).rename("sma_50")
    sma_long = _sma(close, config.sma_long_period).rename("sma_200")
    slope_short = _slope_of_sma(
        sma_short,
        slope_lookback=config.slope_lookback_days,
    ).rename("slope_sma_50")
    slope_long = _slope_of_sma(
        sma_long,
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
    realized_vol_21d = _realized_vol_21d_for_euphoria(close).rename("realized_vol_21d")

    if sentiment_score is not None:
        sentiment_score = sentiment_score.reindex(close.index)
        sentiment_score = sentiment_score.rename("sentiment_score")

    return TrendDirectionV2Features(
        efficiency_ratio_20d=eff,
        hurst_250d=hurst,
        slope_sma_50=slope_short,
        slope_sma_200=slope_long,
        return_63d=ret_short,
        return_126d=ret_long,
        drawdown_252d=dd,
        sma_50=sma_short,
        sma_200=sma_long,
        realized_vol_21d=realized_vol_21d,
        sentiment_score=sentiment_score,
    )


def _realized_vol_21d_for_euphoria(close: pd.Series) -> pd.Series:
    """21-session annualized realized vol using the shared volatility helper.

    The euphoria predicate is computed inside the §1A trend pipeline before
    the volatility seam is necessarily lit, but the series definition must
    remain identical to ``regime_detection.volatility_state.realized_vol``.
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        close = close.copy()
        close.index = pd.to_datetime(close.index)
    return realized_vol(close, window=21)


# ---------------------------------------------------------------------------
# Slice 2.5 — v2 §1A `recovery` rule + precedence wrapper.
#
# Rule (v2 §1A lines 114-119, verbatim):
#     prior 252d drawdown <= -0.15
#     AND return_63d > 0.10
#     AND close > SMA_50
#
# Precedence (v2 §1A lines 132-134):
#     euphoria > bull > recovery > bear > sideways > transition > unknown
#
# `euphoria` is deferred (sentiment_score data source not ingested — see
# Implementation Ambiguity Log entry #32). The precedence slot stays
# defined so future authors can drop euphoria in without re-ordering;
# the rule predicate never fires today.
# ---------------------------------------------------------------------------


def evaluate_recovery(
    features: TrendDirectionV2Features,
    close: pd.Series,
    *,
    dt: pd.Timestamp,
    rules_config: TrendDirectionV2RulesConfig,
) -> bool:
    """v2 §1A line 114-119 `recovery` predicate at a single session.

    Returns False when any of the three inputs is NaN (cold-start
    contract — no silent "unknown → True" substitution). Spec citations:

    * line 116 — ``drawdown_252d <= recovery_drawdown_threshold`` (-0.15)
    * line 117 — ``return_63d   >  recovery_return_threshold``    ( 0.10)
    * line 118 — ``close        >  SMA_50``
    """
    if dt not in features.drawdown_252d.index:
        return False
    drawdown = features.drawdown_252d.loc[dt]
    return_63d = features.return_63d.loc[dt]
    sma_50 = features.sma_50.loc[dt]
    if dt not in close.index:
        return False
    close_t = close.loc[dt]

    # Cold-start / NaN propagation: any missing input falsifies the rule.
    if any(pd.isna(x) for x in (drawdown, return_63d, sma_50, close_t)):
        return False

    drawdown_ok = bool(drawdown <= rules_config.recovery_drawdown_threshold)  # line 116
    return_ok = bool(return_63d > rules_config.recovery_return_threshold)     # line 117
    above_sma = bool(close_t > sma_50)                                         # line 118
    return drawdown_ok and return_ok and above_sma


# v2 §1A line 132-134 ranking (lower index = higher precedence).
# `euphoria` (index 0) was reserved-but-inert before ADR 0004 / Log #32
# closure; it is now wired to a real predicate.
_V2_TREND_PRECEDENCE: tuple[str, ...] = (
    "euphoria",
    "bull",
    "recovery",
    "bear",
    "sideways",
    "transition",
    "unknown",
)


def evaluate_euphoria(
    features: TrendDirectionV2Features,
    close: pd.Series,
    *,
    dt: pd.Timestamp,
    rules_config: TrendDirectionV2RulesConfig,
) -> bool:
    """v2 §1A line 159-165 `euphoria` predicate at a single session.

    Returns False when any of the four inputs is NaN or when the
    Optional ``sentiment_score`` feature is absent (V2 §10 "do not
    invent a sentiment proxy" — Log #32 closure).

    Spec citations (post ADR 0004 amendment):

    * line 161 — ``close > SMA_200`` (strict)
    * line 162 — ``return_126d > euphoria_return_126d_threshold`` (0.20, strict)
    * line 163 — ``realized_vol_21d rising`` (strict 5-session change
      per Log #68 §1D analogue: ``vol[t] > vol[t - N]`` where
      ``N = euphoria_vol_rising_lookback_sessions``)
    * line 164 — ``sentiment_score >= euphoria_sentiment_threshold``
      (+20 default; non-strict at boundary)
    """
    sentiment_series = features.sentiment_score
    if sentiment_series is None:
        return False

    if dt not in features.return_126d.index or dt not in close.index:
        return False
    if dt not in features.sma_200.index or dt not in features.realized_vol_21d.index:
        return False
    if dt not in sentiment_series.index:
        return False

    lookback = rules_config.euphoria_vol_rising_lookback_sessions
    vol_index = features.realized_vol_21d.index
    try:
        pos_t = vol_index.get_loc(dt)
    except KeyError:
        return False
    pos_back = pos_t - lookback
    if pos_back < 0:
        return False
    vol_t = features.realized_vol_21d.iloc[pos_t]
    vol_back = features.realized_vol_21d.iloc[pos_back]

    close_t = close.loc[dt]
    sma_200_t = features.sma_200.loc[dt]
    return_126d_t = features.return_126d.loc[dt]
    sentiment_t = sentiment_series.loc[dt]

    # Cold-start / NaN propagation: any missing input falsifies the rule.
    if any(
        pd.isna(x)
        for x in (close_t, sma_200_t, return_126d_t, vol_t, vol_back, sentiment_t)
    ):
        return False

    close_above_sma = bool(close_t > sma_200_t)                                  # line 161
    return_ok = bool(return_126d_t > rules_config.euphoria_return_126d_threshold)  # line 162
    vol_rising = bool(vol_t > vol_back)                                          # line 163
    sentiment_ok = bool(sentiment_t >= rules_config.euphoria_sentiment_threshold)  # line 164
    return close_above_sma and return_ok and vol_rising and sentiment_ok


def evaluate_v2_trend_label(
    *,
    v1_label: str,
    features: TrendDirectionV2Features,
    close: pd.Series,
    dt: pd.Timestamp,
    rules_config: TrendDirectionV2RulesConfig,
) -> str | None:
    """Apply v2 §1A trend precedence on top of a v1 raw label.

    Returns the winning v2 label per the §1A line 132-134 ordering, or
    ``None`` when no v2 rule fires and the caller should keep ``v1_label``.

    Precedence (line 132-134): ``euphoria > bull > recovery > bear >
    sideways > transition > unknown``.

    Dispatch order: euphoria first (top of precedence — outranks every
    v1 label including bull); then recovery (only fires when v1 is
    strictly lower-ranked, i.e. ``bear`` / ``sideways`` / ``transition``
    / ``unknown``).
    """
    euphoria_fires = evaluate_euphoria(
        features, close, dt=dt, rules_config=rules_config
    )
    if euphoria_fires:
        return "euphoria"

    recovery_fires = evaluate_recovery(
        features, close, dt=dt, rules_config=rules_config
    )
    if not recovery_fires:
        return None

    try:
        v1_rank = _V2_TREND_PRECEDENCE.index(v1_label)
    except ValueError:
        # Unknown v1 label — treat as lowest precedence and let recovery win.
        v1_rank = len(_V2_TREND_PRECEDENCE)
    recovery_rank = _V2_TREND_PRECEDENCE.index("recovery")
    if v1_rank < recovery_rank:
        # v1 label outranks recovery (only possible value: bull).
        return None
    return "recovery"
