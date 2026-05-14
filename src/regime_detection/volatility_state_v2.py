"""v2 §1C Layer 1 V2 Volatility features and labels.

Pure pandas/numpy implementation of the §1C continuous features and the
``rising_vol`` / ``vol_crush`` precedence overlay.

Features (all per-session series aligned to the input close index):

- ``atr_ratio``                       v2 §1C lines 140–143 (ATR_14 / ATR_50, Wilder)
- ``gap_frequency_20d``               v2 §1C lines 176–181
- ``intraday_range_percentile_252d``  v2 §1C lines 183–187

Optional external-data features:

- ``iv_rv_spread`` (§1C lines 151–155) — computed when FRED VIXCLS-derived
  ``implied_vol_30d`` is supplied.
- ``vol_crush`` (§1C lines 157–174) — computed when both implied-vol and
  event-window inputs are supplied; otherwise the predicate falsifies per
  v2 §10 "do not invent missing inputs".

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

from regime_detection.config import VolatilityV2Config, VolatilityV2RulesConfig
from regime_detection.volatility_state import realized_vol, wilders_atr


@dataclass(frozen=True)
class VolatilityV2Features:
    """v2 §1C — per-session continuous volatility features.

    Slice 2.2 fields: atr_ratio, gap_frequency_20d, intraday_range_percentile_252d.
    Slice 2.6 adds the two realized-vol windows used by the `rising_vol` rule
    (v2 §1C line 148): a short-window realised vol (default 10d) and a
    long-window realised vol (default 63d), both annualised via the shared
    ``regime_detection.volatility_state.realized_vol`` helper.
    """

    atr_ratio: pd.Series
    gap_frequency_20d: pd.Series
    # v2 §1E line 278 / Log #40 closure — 252d percentile rank of
    # `gap_frequency_20d`. Consumed by the §1E `liquidity_gap_behavior`
    # rule. Computed here (rather than at the rule layer) so the percentile
    # shares the volatility seam's session index and the rule layer reads
    # only scalars.
    gap_frequency_percentile_252d: pd.Series
    intraday_range_percentile_252d: pd.Series
    # v2 §1C line 148 — `rising_vol` rule inputs (slice 2.6).
    realized_vol_short: pd.Series
    realized_vol_long: pd.Series
    # v2 §1C `vol_crush` rule input (ADR 0005 / Log #20). 21-session
    # realized vol — the mid window for `realized_vol_10d < realized_vol_21d
    # * 0.75`. Always computable from close; never None.
    realized_vol_21d: pd.Series
    # v2 §1C IV features (ADR 0005 / Log #19+#20). Optional — populated
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
    # v2 §1E line 278 — 252d percentile rank of `gap_frequency_20d`. Same
    # rolling-rank shape as `intraday_range_percentile_252d` below and the
    # §1D `nh_nl_ratio` percentile pattern. Closes Log #40 by computing
    # the previously-missing percentile input for `liquidity_gap_behavior`.
    gap_freq_pct = (
        gap_freq.rolling(config.intraday_range_lookback_days, min_periods=config.intraday_range_lookback_days)
        .rank(pct=True)
        .rename("gap_frequency_percentile_252d")
    )
    intraday_pct = _intraday_range_percentile(
        high=high,
        low=low,
        close=close,
        lookback=config.intraday_range_lookback_days,
    )

    # v2 §1C line 148 — `rising_vol` rule inputs (slice 2.6). Computed via
    # the shared ``regime_detection.volatility_state.realized_vol`` helper
    # so v1 (slice 1, realized_vol_21d) and v2 (slice 2.6, rv_10d/rv_63d)
    # consume one annualisation path. When no rules_config is supplied,
    # default to spec windows so callers that read the feature seam without
    # explicit rule configuration still get a complete struct.
    if rules_config is not None:
        rv_short_window = rules_config.realized_vol_short_period
        rv_long_window = rules_config.realized_vol_long_period
    else:
        # Spec defaults — v2 §1C line 148 (realized_vol_10d / realized_vol_63d).
        # Hardcoded fallback values intentionally match VolatilityV2RulesConfig
        # defaults; both citations point at v2 §1C line 148.
        rv_short_window = 10
        rv_long_window = 63
    rv_short = realized_vol(close, window=rv_short_window).rename(
        "realized_vol_short"
    )
    rv_long = realized_vol(close, window=rv_long_window).rename(
        "realized_vol_long"
    )

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
            event_window_just_passed.reindex(close.index)
            .fillna(False)
            .astype(bool)
            .rename("event_window_just_passed")
        )

    return VolatilityV2Features(
        atr_ratio=atr_ratio,
        gap_frequency_20d=gap_freq,
        gap_frequency_percentile_252d=gap_freq_pct,
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
# Slice 2.6 — v2 §1C `rising_vol` rule + precedence wrapper.
#
# Rule (v2 §1C lines 146-148, verbatim):
#     ATR_ratio > 1.15
#     OR realized_vol_10d > realized_vol_63d * 1.25
#
# Precedence (v2 §1C line 191):
#     crisis_vol > vol_crush > high_vol > rising_vol > low_vol > normal_vol > unknown
#
# `vol_crush` is wired via ADR 0005 / Log #20 using FRED VIXCLS-derived
# implied_vol_30d plus event_window_just_passed.
# ---------------------------------------------------------------------------


def evaluate_rising_vol(
    features: VolatilityV2Features,
    *,
    dt: pd.Timestamp,
    rules_config: VolatilityV2RulesConfig,
) -> bool:
    """v2 §1C lines 146-148 `rising_vol` predicate at a single session.

    Returns False when ANY of the three inputs is NaN — strict cold-start
    contract (no silent "partial-input → True" substitution). Both limbs
    use strict ``>`` per spec text:

    * line 147 — ``atr_ratio > atr_ratio_threshold`` (1.15)
    * line 148 — ``realized_vol_short > realized_vol_long * realized_vol_ratio_threshold`` (1.25)
    * Combined: ATR limb OR realised-vol limb.

    The all-inputs-must-be-present contract is recorded in the
    Implementation Ambiguity Log entry #36 — spec §1C is silent on
    partial-NaN behavior so the conservative choice is "any NaN
    falsifies the rule" (matches slice 2.5 recovery cold-start).
    """
    if dt not in features.atr_ratio.index:
        return False
    atr = features.atr_ratio.loc[dt]
    rv_short = features.realized_vol_short.loc[dt]
    rv_long = features.realized_vol_long.loc[dt]

    # Strict cold-start: any missing input falsifies the rule
    # (Ambiguity Log #36 — partial-NaN handling).
    if any(pd.isna(x) for x in (atr, rv_short, rv_long)):
        return False

    atr_limb = bool(atr > rules_config.atr_ratio_threshold)            # line 147
    rv_limb = bool(rv_short > rv_long * rules_config.realized_vol_ratio_threshold)  # line 148
    return atr_limb or rv_limb


def evaluate_vol_crush(
    features: VolatilityV2Features,
    *,
    dt: pd.Timestamp,
    rules_config: VolatilityV2RulesConfig,
) -> bool:
    """v2 §1C `vol_crush` predicate at a single session (ADR 0005 / Log #20).

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


# v2 §1C line 191 ranking (lower index = higher precedence).
# `vol_crush` (index 1) was reserved-but-inert before ADR 0005 / Log #20
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

    Returns the winning v2 label per the §1C line 191 ordering, or
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
