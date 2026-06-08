from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from regime_detection._rolling_stats import period_return, simple_moving_average
from regime_detection._rule_helpers import ev_float as _ev_float
from regime_detection.config import TrendDirectionV2Config, TrendDirectionV2RulesConfig
from regime_detection.volatility_state import realized_vol

# v2 §1A line 239 precedence: euphoria > bull > recovery > bear > sideways > transition > unknown.
TrendDirectionLabel = Literal[
    "euphoria",
    "bull",
    "bear",
    "sideways",
    "transition",
    "unknown",
    "recovery",
]


# `_RISK_RANK` encodes DOWNSIDE RISK for hysteresis decay, NOT spec
# precedence (which is in trend_direction_v2._V2_TREND_PRECEDENCE).
# The two orderings are intentionally different — see ADR 0016
# "trend_direction risk-rank vs precedence". Higher rank = more dangerous
# for risk management: euphoria=4 (speculative excess collapses fastest),
# bear=3 (established downtrend), unknown=2 (mid-rank — neither escalates
# past bear nor strands across cold-start), recovery/sideways=1, bull=0.
_RISK_RANK: dict[TrendDirectionLabel, int] = {
    "bull": 0,
    "sideways": 1,
    "recovery": 1,
    "transition": 2,
    "euphoria": 4,
    "bear": 3,
    "unknown": 2,
}

# V1-legacy trend-direction thresholds (sideways gate + close-to-SMA200 band).
# Not restated in v2 spec; v1 contract frozen by §2.2 stateless replay.
_SIDEWAYS_ABS_RETURN_63D = 0.05
_WITHIN_PCT_SMA200 = 0.05


@dataclass(frozen=True)
class TrendDirectionFeatures:
    close: pd.Series
    sma_50: pd.Series
    sma_200: pd.Series
    return_63d: pd.Series


def compute_features(close: pd.Series) -> TrendDirectionFeatures:
    sma_50 = simple_moving_average(close, window=50)
    sma_200 = simple_moving_average(close, window=200)
    return_63d = period_return(close, periods=63)
    return TrendDirectionFeatures(
        close=close, sma_50=sma_50, sma_200=sma_200, return_63d=return_63d
    )


def raw_label_for_day(
    f: TrendDirectionFeatures,
    dt: pd.Timestamp,
    *,
    trend_direction_v2_features: "TrendDirectionV2Features | None" = None,
    trend_direction_v2_rules: "TrendDirectionV2RulesConfig | None" = None,
) -> tuple[TrendDirectionLabel, dict[str, Any]]:
    """Per-day raw trend_direction label.

    When ``trend_direction_v2_features`` and ``trend_direction_v2_rules`` are
    both supplied, the v2 §1A precedence (line 239:
    ``euphoria > bull > recovery > bear > sideways > transition > unknown``)
    is layered ON TOP of the v1 label. When either is ``None`` the function
    returns the v1 label and evidence unchanged.

    F-043: this is a thin wrapper over :func:`build_raw_outputs` so the §3.5
    rule predicates, evidence shape, and v2 override have a single encoding.
    Slicing each feature to ``[dt]`` is safe because the vectorized builder and
    the v2 ``evaluate_recovery`` / ``evaluate_euphoria`` predicates only read
    ``close.loc[dt]`` and ``features.*.loc[dt]`` at the target session.
    """
    day_features = TrendDirectionFeatures(
        close=f.close.loc[[dt]],
        sma_50=f.sma_50.loc[[dt]],
        sma_200=f.sma_200.loc[[dt]],
        return_63d=f.return_63d.loc[[dt]],
    )
    labels, evidence = build_raw_outputs(
        day_features,
        trend_direction_v2_features=trend_direction_v2_features,
        trend_direction_v2_rules=trend_direction_v2_rules,
    )
    return labels[0], evidence[0]


def _v2_evidence_for_day(
    features: "TrendDirectionV2Features", dt: pd.Timestamp
) -> dict[str, Any]:
    evidence = {
        "efficiency_ratio_20d": _ev_float(features.efficiency_ratio_20d.loc[dt]),
        "hurst_250d": _ev_float(features.hurst_250d.loc[dt]),
        "slope_sma_50": _ev_float(features.slope_sma_50.loc[dt]),
        "slope_sma_200": _ev_float(features.slope_sma_200.loc[dt]),
        "return_126d": _ev_float(features.return_126d.loc[dt]),
        "drawdown_252d": _ev_float(features.drawdown_252d.loc[dt]),
        "realized_vol_21d": _ev_float(features.realized_vol_21d.loc[dt]),
    }
    if features.sentiment_score is not None:
        evidence["sentiment_score"] = _ev_float(features.sentiment_score.loc[dt])
    if features.news_sentiment_score is not None:
        evidence["news_sentiment_score"] = _ev_float(
            features.news_sentiment_score.loc[dt]
        )
    if features.sentiment_concordance is not None:
        evidence["sentiment_concordance"] = _ev_float(
            features.sentiment_concordance.loc[dt]
        )
    return evidence


def build_raw_outputs(
    f: TrendDirectionFeatures,
    *,
    trend_direction_v2_features: "TrendDirectionV2Features | None" = None,
    trend_direction_v2_rules: "TrendDirectionV2RulesConfig | None" = None,
) -> tuple[list[TrendDirectionLabel], list[dict[str, Any]]]:
    """Vectorized v1 raw labels + optional v2 §1A `recovery` / `euphoria` override.

    When both ``trend_direction_v2_features`` and ``trend_direction_v2_rules``
    are supplied, the v2 §1A precedence at line 239 is applied per-day AFTER
    the v1 pass: ``euphoria > bull > recovery > bear > sideways > transition
    > unknown``. ``recovery`` overrides v1 ``bear`` / ``sideways`` /
    ``transition`` / ``unknown`` (NOT ``bull`` or ``euphoria``, which both
    outrank ``recovery``). When either argument is None, output is
    byte-identical to v1.
    """
    close = f.close
    sma50 = f.sma_50
    sma200 = f.sma_200
    ret63 = f.return_63d

    valid = ~(close.isna() | sma50.isna() | sma200.isna() | ret63.isna())
    within_5pct_sma200 = valid & close.between(
        sma200 * (1 - _WITHIN_PCT_SMA200),
        sma200 * (1 + _WITHIN_PCT_SMA200),
    )
    bull = valid & close.gt(sma50) & close.gt(sma200) & sma50.gt(sma200)
    bear = valid & close.lt(sma50) & close.lt(sma200) & sma50.lt(sma200)
    sideways = valid & ret63.abs().lt(_SIDEWAYS_ABS_RETURN_63D) & within_5pct_sma200
    transition = valid & ~(bull | bear | sideways)
    close_gt_sma50 = valid & close.gt(sma50)
    close_gt_sma200 = valid & close.gt(sma200)
    sma50_gt_sma200 = valid & sma50.gt(sma200)

    labels = np.full(len(close), "unknown", dtype=object)
    labels[transition.to_numpy()] = "transition"
    labels[sideways.to_numpy()] = "sideways"
    labels[bear.to_numpy()] = "bear"
    labels[bull.to_numpy()] = "bull"

    evidence: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        if label == "unknown":
            evidence.append({"reason": "insufficient_history"})
            continue
        evidence.append(
            {
                "sma_50": _ev_float(sma50.iat[idx]),
                "sma_200": _ev_float(sma200.iat[idx]),
                "return_63d": _ev_float(ret63.iat[idx]),
                "close_gt_sma50": bool(close_gt_sma50.iat[idx]),
                "close_gt_sma200": bool(close_gt_sma200.iat[idx]),
                "sma50_gt_sma200": bool(sma50_gt_sma200.iat[idx]),
                "within_5pct_sma200": bool(within_5pct_sma200.iat[idx]),
            }
        )

    if trend_direction_v2_features is not None and trend_direction_v2_rules is not None:
        # v2 §1A line 239 precedence — applied per-day on top of v1.
        for idx, dt in enumerate(close.index):
            # Skip v2 evidence on "unknown" rows: v1 produced unknown only when
            # one of close/sma50/sma200/ret63 is NaN at cold-start, and the
            # evidence dict already carries `{"reason": "insufficient_history"}`.
            # Adding v2 features here would surface NaN floats and obscure the
            # cold-start signal. v2 rule eval below still runs on unknown rows
            # so recovery/euphoria can override unknown per spec §1A line 239
            # precedence — matches per-day `raw_label_for_day` behavior.
            if labels[idx] != "unknown":
                evidence[idx].update(
                    _v2_evidence_for_day(trend_direction_v2_features, dt)
                )
            v1_label = str(labels[idx])
            v2_label = evaluate_v2_trend_label(
                v1_label=v1_label,
                features=trend_direction_v2_features,
                close=close,
                dt=dt,
                rules_config=trend_direction_v2_rules,
            )
            if v2_label is None:
                continue
            evidence[idx]["v2_override"] = {
                "from": v1_label,
                "to": v2_label,
                "rule": v2_label,
            }
            labels[idx] = v2_label

    return list(labels), evidence


# Spec-fixed constants (not configurable — v2 §1A lines 105-107 define the
# directional-move and path-length sums over the same N=lookback window).
_PATH_DIFF_SHIFT = 1

# v2 §1A line 204 — realized_vol_21d window for the `euphoria` rule predicate.
# Pinned to 21 sessions by the spec name (`realized_vol_21d`); this constant
# exists to keep the lookback out of the function body, not to make it tunable.
_EUPHORIA_REALIZED_VOL_WINDOW = 21


@dataclass(frozen=True)
class TrendDirectionV2Features:
    """v2 §1A — per-session continuous features for the V2 trend
    classifier. Names match the spec line citations in the module docstring."""

    efficiency_ratio_20d: pd.Series
    hurst_250d: pd.Series
    slope_sma_50: pd.Series
    slope_sma_200: pd.Series
    return_63d: pd.Series
    return_126d: pd.Series
    drawdown_252d: pd.Series
    # v2 §1A line 197 — `recovery` rule input: close > SMA_50.
    # Exposed as a level (not a slope) so the recovery predicate
    # has direct access without recomputing the 50d SMA.
    sma_50: pd.Series
    # v2 §1A line 202 — `euphoria` rule input: close > SMA_200. Exposed as
    # a level so the euphoria predicate compares against close[t] directly
    # without recomputing the 200d SMA (already computed for slope_sma_200).
    sma_200: pd.Series
    # v2 §1A line 204 — `euphoria` rule input: realized_vol_21d rising
    # (strict 5-session change per ADR 0004 Q2 / implementation decision #68 §1D analogue).
    # 21-session annualized realized vol of SPY log-returns.
    realized_vol_21d: pd.Series
    # v2 §1A line 205 — `euphoria` rule input: sentiment_score >= threshold.
    # Optional because not every engine call wires AAII sentiment; when None
    # the euphoria predicate falsifies (V2 §10 no-hallucination rule).
    # When supplied, the series is forward-filled from AAII weekly readings
    # onto the SPY session index per ADR 0004 Q4 (V1 §2.2 stateless replay).
    sentiment_score: pd.Series | None = None
    # Engine-local second sentiment voice (NOT in V2 §1A — extension) — SF Fed Daily News Sentiment Index
    # (Shapiro, Sudhof, Wilson 2020). Smoothed onto the SPY session index.
    # EVIDENCE ONLY — never read by the `euphoria` rule predicate. See audit
    # follow-up: the source-data audit "news sentiment".
    # TODO(sentiment-calibration): decide, with walk-forward evidence, whether
    # news_sentiment_score or sentiment_concordance should become a configured
    # euphoria confidence gate or remain reporting-only evidence.
    news_sentiment_score: pd.Series | None = None
    # Derived concordance flag — True when AAII and news sentiment agree on
    # sign (both positive or both negative), False when they diverge, NaN
    # when either is NaN. Surfaces in evidence dicts so downstream
    # consumers (strategy_response, calibration review, dashboards) can
    # treat divergent euphoria firings as lower-conviction without the
    # label predicate itself changing.
    sentiment_concordance: pd.Series | None = None

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
        return pd.DataFrame({name: getattr(self, name) for name in self.feature_names})


def _efficiency_ratio(close: pd.Series, lookback: int) -> pd.Series:
    """v2 §1A lines 105–111.

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
    standard — implementation decision #12 R/S input-series pin). Returns NaN when the window
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
    return simple_moving_average(close, window=sma_period)


def _slope_of_sma(sma: pd.Series, slope_lookback: int) -> pd.Series:
    """v2 §1A line 178: (sma[t] - sma[t-N]) / sma[t-N].

    Accepts a pre-computed SMA series so callers can both expose the SMA
    level (the `recovery` predicate consumes ``sma_50``) and its
    slope in one pass.

    NaN propagates from the SMA (until t >= sma_period-1) and from the
    `slope_lookback` shift (until t >= sma_period-1+slope_lookback).
    """
    prior = sma.shift(slope_lookback)
    return (sma - prior) / prior.where(prior != 0)


def compute_trailing_drawdown(close: pd.Series, lookback: int) -> pd.Series:
    """v2 §1A line 195: (close[t] / max(close[t-N+1..t])) - 1.

    Peak window is inclusive of t (matching
    ``network_fragility_rules._trailing_drawdown`` — implementation decision #13 peak-inclusive pin).
    Drawdown == 0 when t is a fresh `lookback`-day high. Negative below.
    NaN if any of the window's `lookback` sessions is NaN or if t lacks
    `lookback` prior history.

    The HMM evidence layer (§6.1) reuses this formula for its
    `drawdown_63d` input — single home for trailing-drawdown math.
    """
    peak = close.rolling(window=lookback, min_periods=lookback).max()
    return (close / peak.where(peak > 0)) - 1.0


# Internal alias preserved for in-module callers.
_trailing_drawdown = compute_trailing_drawdown


def compute_trend_v2_features(
    close: pd.Series,
    *,
    config: TrendDirectionV2Config,
    sentiment_score: pd.Series | None = None,
    news_sentiment_score: pd.Series | None = None,
) -> TrendDirectionV2Features:
    """Compute the ten v2 §1A trend-direction features from a close series.

    All parameters are sourced from ``TrendDirectionV2Config``; no magic
    numbers in the function body. Returns a frozen dataclass with each
    feature as a date-indexed ``pd.Series`` aligned to ``close.index``.

    ``sentiment_score`` is an Optional pre-aligned series carrying the
    forward-filled AAII bull-bear-spread 8w-MA on the close index (per
    ADR 0004 Q1 / Q4 — V1 §2.2 stateless replay; values originate from
    weekly publication dates ≤ as_of). When omitted, the v2 §1A
    `euphoria` rule falsifies on every session (V2 §10 no-hallucination
    rule).
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
    ret_short = period_return(
        close, periods=config.return_short_period, output_name="return_63d"
    )
    ret_long = period_return(
        close, periods=config.return_long_period, output_name="return_126d"
    )
    dd = _trailing_drawdown(close, config.drawdown_lookback_days).rename(
        "drawdown_252d"
    )
    realized_vol_21d = _realized_vol_21d_for_euphoria(close).rename("realized_vol_21d")

    if sentiment_score is not None:
        sentiment_score = sentiment_score.reindex(close.index)
        sentiment_score = sentiment_score.rename("sentiment_score")

    # Engine-local second sentiment voice (SF Fed, NOT in V2 §1A). Evidence only; never read
    # by the `euphoria` rule. Reindex onto the close calendar so the
    # concordance flag can be computed pointwise against `sentiment_score`.
    sentiment_concordance: pd.Series | None = None
    if news_sentiment_score is not None:
        news_sentiment_score = news_sentiment_score.reindex(close.index).rename(
            "news_sentiment_score"
        )
        if sentiment_score is not None:
            sentiment_concordance = _compute_sentiment_concordance(
                aaii=sentiment_score, news=news_sentiment_score
            )

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
        news_sentiment_score=news_sentiment_score,
        sentiment_concordance=sentiment_concordance,
    )


def _compute_sentiment_concordance(*, aaii: pd.Series, news: pd.Series) -> pd.Series:
    """Pointwise concordance between AAII bull-bear and SF Fed news sentiment.

    Returns a per-session float Series:

        +1.0  — both signals positive
         0.0  — signals disagree on sign
        -1.0  — both signals negative
         NaN  — either signal NaN at that session

    NOTE: this is an EVIDENCE-only derivation. The §1A `euphoria` rule
    predicate is unchanged (it consumes only ``sentiment_score`` per
    spec line 164). Downstream consumers may treat
    `sentiment_concordance == 0` firings as lower-conviction.
    """
    aaii_aligned = aaii.reindex(news.index)
    out_index = news.index
    score = pd.Series(float("nan"), index=out_index, name="sentiment_concordance")
    aaii_sign = aaii_aligned.where(aaii_aligned.notna())
    news_sign = news.where(news.notna())
    both_valid = aaii_sign.notna() & news_sign.notna()
    both_pos = both_valid & (aaii_sign > 0) & (news_sign > 0)
    both_neg = both_valid & (aaii_sign < 0) & (news_sign < 0)
    disagree = both_valid & ~both_pos & ~both_neg
    score = score.mask(both_pos, 1.0)
    score = score.mask(both_neg, -1.0)
    score = score.mask(disagree, 0.0)
    return score


def _realized_vol_21d_for_euphoria(close: pd.Series) -> pd.Series:
    """21-session annualized realized vol using the shared volatility helper.

    The euphoria predicate is computed inside the §1A trend pipeline before
    the volatility seam is necessarily lit, but the series definition must
    remain identical to ``regime_detection.volatility_state.realized_vol``.
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        close = close.copy()
        close.index = pd.to_datetime(close.index)
    return realized_vol(close, window=_EUPHORIA_REALIZED_VOL_WINDOW)


# ---------------------------------------------------------------------------
# v2 §1A `recovery` rule + precedence wrapper.
#
# Rule (v2 §1A lines 193-197, verbatim):
#     prior 252d drawdown <= -0.15
#     AND return_63d > 0.10
#     AND close > SMA_50
#
# Precedence (v2 §1A line 239):
#     euphoria > bull > recovery > bear > sideways > transition > unknown
#
# `euphoria` is wired to a real predicate via `evaluate_euphoria` below
# (post ADR 0004 amendment). The predicate consumes the Optional
# `sentiment_score` (AAII) feature; when the feature is absent it
# falsifies per V2 §10 no-hallucination rule.
# ---------------------------------------------------------------------------


def evaluate_recovery(
    features: TrendDirectionV2Features,
    close: pd.Series,
    *,
    dt: pd.Timestamp,
    rules_config: TrendDirectionV2RulesConfig,
) -> bool:
    """v2 §1A lines 193-197 `recovery` predicate at a single session.

    Returns False when any of the three inputs is NaN (cold-start
    contract — no silent "unknown → True" substitution). Spec citations:

    * line 195 — ``drawdown_252d <= recovery_drawdown_threshold`` (-0.15)
    * line 196 — ``return_63d   >  recovery_return_threshold``    ( 0.10)
    * line 197 — ``close        >  SMA_50``
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

    drawdown_ok = bool(drawdown <= rules_config.recovery_drawdown_threshold)  # line 195
    return_ok = bool(return_63d > rules_config.recovery_return_threshold)  # line 196
    above_sma = bool(close_t > sma_50)  # line 197
    return drawdown_ok and return_ok and above_sma


# v2 §1A line 239 ranking (lower index = higher precedence).
# `euphoria` (index 0) was reserved-but-inert before ADR 0004 closure;
# it is now wired to a real predicate.
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
    """v2 §1A lines 200-205 `euphoria` predicate at a single session.

    Returns False when any of the four inputs is NaN or when the
    Optional ``sentiment_score`` feature is absent (V2 §10
    no-hallucination rule — ADR 0004 closure).

    Spec citations (post ADR 0004 amendment):

    * line 202 — ``close > SMA_200`` (strict)
    * line 203 — ``return_126d > euphoria_return_126d_threshold`` (0.20, strict)
    * line 204 — ``realized_vol_21d rising`` (strict 5-session change
      per implementation decision #68 §1D analogue: ``vol[t] > vol[t - N]`` where
      ``N = euphoria_vol_rising_lookback_sessions``)
    * line 205 — ``sentiment_score >= euphoria_sentiment_threshold``
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

    close_above_sma = bool(close_t > sma_200_t)  # line 202
    return_ok = bool(
        return_126d_t > rules_config.euphoria_return_126d_threshold
    )  # line 203
    vol_rising = bool(vol_t > vol_back)  # line 204
    sentiment_ok = bool(
        sentiment_t >= rules_config.euphoria_sentiment_threshold
    )  # line 205
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

    Returns the winning v2 label per the §1A line 239 ordering, or
    ``None`` when no v2 rule fires and the caller should keep ``v1_label``.

    Precedence (line 239): ``euphoria > bull > recovery > bear >
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
