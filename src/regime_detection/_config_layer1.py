from __future__ import annotations

from pydantic import Field

from regime_detection._config_core import StrictBaseModel


class AxisHysteresisConfig(StrictBaseModel):
    """Axis-level per-label hysteresis config shared by V1-origin and V2 axes."""

    deescalation_days_by_label: dict[str, int]
    default_deescalation_days: int = Field(default=0, ge=0)


class TrendDirectionV2RulesConfig(StrictBaseModel):
    """v2 §1A trend-direction rule thresholds.

    Each value cites its line in docs/regime_engine_v2_spec.md §1A.
    """

    # v2 §1A line 195 — "prior 252d drawdown <= -0.15". Must be < 0
    # because a drawdown is, by construction, in (-1.0, 0.0]; a non-negative
    # threshold would make the rule trivially true at any 252d-high.
    recovery_drawdown_threshold: float = Field(lt=0.0)

    # v2 §1A line 196 — "return_63d > 0.10". Must be > 0 because the rule
    # gates on a strictly-positive 63d return (a non-positive threshold would
    # admit drawdown days, defeating the rule's "rebound" intent).
    recovery_return_threshold: float = Field(gt=0.0)

    # v2 §1A line 203 — euphoria rule's `return_126d > 0.20`. Strict positive
    # required: a non-positive threshold would admit drawdown days, defeating
    # the rule's "strong long-horizon advance" intent.
    euphoria_return_126d_threshold: float = Field(gt=0.0, default=0.20)

    # v2 §1A line 205 — euphoria rule's `sentiment_score >= configured threshold`.
    # Default +20 anchors to historical top-decile of AAII bull-bear 8w-MA;
    # no Pydantic range bound because sentiment can go negative in bearish regimes.
    euphoria_sentiment_threshold: float = Field(default=20.0)

    # v2 §1A line 204 — euphoria rule's `realized_vol_21d rising`. 5-session
    # strict change (vol[t] > vol[t-5]) mirroring §1D `rising` / `falling` pin.
    # Must be > 0; a zero-lookback would make the rule self-comparing.
    euphoria_vol_rising_lookback_sessions: int = Field(gt=0, default=5)


class TrendDirectionV2Config(StrictBaseModel):
    """v2 §1A — Layer 1 V2 trend direction feature lookbacks."""

    # v2 §1A line 104 — Efficiency Ratio over 20 trading days.
    efficiency_ratio_lookback_days: int = Field(gt=0)

    # v2 §1A line 120 — Hurst exponent lookback ("250d minimum").
    hurst_lookback_days: int = Field(gt=0)

    # v2 §1A line 185 — slope_sma window: (sma[t] - sma[t-20]) / sma[t-20].
    slope_lookback_days: int = Field(gt=0)

    # v2 §1A line 185 — SMA_50 short window.
    sma_short_period: int = Field(gt=0)

    # v2 §1A line 186 — SMA_200 long window.
    sma_long_period: int = Field(gt=0)

    # v2 §1A line 196 — return_63d (recovery rule input).
    return_short_period: int = Field(gt=0)

    # v2 §1A line 203 — return_126d (euphoria rule input).
    return_long_period: int = Field(gt=0)

    # v2 §1A line 195 — prior 252d drawdown (recovery rule input).
    drawdown_lookback_days: int = Field(gt=0)

    # v2 §1A lines 193-198 `recovery` rule thresholds. Defaults to spec values
    # (drawdown <= -0.15, return > 0.10) when the yaml omits the sub-block.
    rules: TrendDirectionV2RulesConfig = Field(
        default_factory=lambda: TrendDirectionV2RulesConfig(
            recovery_drawdown_threshold=-0.15,  # v2 §1A line 195
            recovery_return_threshold=0.10,     # v2 §1A line 196
        )
    )

class VolatilityV2RulesConfig(StrictBaseModel):
    """v2 §1C `rising_vol` and `vol_crush` rule thresholds.

    Each value cites its line in docs/regime_engine_v2_spec.md §1C.
    `vol_crush` uses FRED VIXCLS-derived implied_vol_30d.
    """

    # v2 §1C line 253 — "ATR_ratio > 1.15". Must be > 0 because ATR_ratio
    # is a non-negative ratio (ATR_short / ATR_long, both >= 0); a non-
    # positive threshold would make the rule trivially true at any
    # non-trivial ratio.
    atr_ratio_threshold: float = Field(gt=0.0, default=1.15)

    # v2 §1C line 254 — "realized_vol_10d > realized_vol_63d * 1.25". Must
    # be > 0 because realised vols are non-negative; a non-positive
    # threshold would defang the "expansion" intent.
    realized_vol_ratio_threshold: float = Field(gt=0.0, default=1.25)

    # v2 §1C line 254 — short realised-vol window (10 sessions). Pinned at
    # 10 by spec text "realized_vol_10d"; exposed for v2 §9.1 calibration.
    realized_vol_short_period: int = Field(gt=0, default=10)

    # v2 §1C line 254 — long realised-vol window (63 sessions). Pinned at
    # 63 by spec text "realized_vol_63d"; exposed for v2 §9.1 calibration.
    realized_vol_long_period: int = Field(gt=0, default=63)

    # v2 §1C `vol_crush` rule. The rule:
    #   realized_vol_10d < realized_vol_21d * vol_crush_realized_vol_ratio_threshold
    #   AND implied_vol_5d_change <= vol_crush_implied_vol_change_threshold
    #   AND event_window_just_passed
    # `realized_vol_10d` reuses `realized_vol_short_period`; the 21d mid
    # window is its own knob.
    vol_crush_realized_vol_mid_period: int = Field(gt=0, default=21)
    # Spec line: "realized_vol_10d < realized_vol_21d * 0.75". Must be in
    # (0, 1) — the rule fires when the short-window vol has COLLAPSED below
    # a fraction of the mid-window vol.
    vol_crush_realized_vol_ratio_threshold: float = Field(gt=0.0, lt=1.0, default=0.75)
    # `implied_vol_5d_change <= -0.20`, a RELATIVE 5-session change. Must be < 0:
    # the rule gates on a strictly-negative IV move (a "crush").
    vol_crush_implied_vol_change_threshold: float = Field(lt=0.0, default=-0.20)
    # Lookback for the relative implied-vol change. Pinned at 5 sessions
    # (cross-axis "5-session memory" convention).
    vol_crush_implied_vol_change_lookback_sessions: int = Field(gt=0, default=5)
    # `event_window_just_passed` fires on the N NYSE sessions strictly AFTER
    # an event window-end. Spec pins N = 3.
    vol_crush_event_window_trailing_sessions: int = Field(gt=0, default=3)


class VolatilityV2Config(StrictBaseModel):
    """v2 §1C — Layer 1 V2 Volatility features.

    Ships ATR ratio, gap-frequency, intraday-range, realized-vol, IV/RV
    spread, and vol-crush inputs. IV-derived features are present when the
    context supplies FRED VIXCLS-derived ``implied_vol_30d``; otherwise
    those optional inputs stay absent and ``vol_crush`` falsifies per
    v2 §10.
    """

    # v2 §1C line 248 — short ATR window (ATR_14, Wilder smoothing).
    atr_short_period: int = Field(gt=0)

    # v2 §1C line 248 — long ATR window (ATR_50, Wilder smoothing).
    atr_long_period: int = Field(gt=0)

    # v2 §1C line 299 — gap_frequency lookback (20 sessions).
    gap_frequency_lookback_days: int = Field(gt=0)

    # v2 §1C line 301 — gap threshold (0.005 = 0.5%); US default, "configurable
    # per market". V2 universe is US-only so we pin a single 0.005 default.
    gap_threshold_pct: float = Field(gt=0.0, lt=1.0)

    # v2 §1C line 306 — intraday-range percentile lookback (252 sessions).
    intraday_range_lookback_days: int = Field(gt=0)

    # v2 §1C lines 252-255 `rising_vol` rule thresholds + RV windows. Defaults to
    # spec values (atr_ratio > 1.15, rv_10d > rv_63d * 1.25) when the yaml
    # omits the sub-block.
    rules: VolatilityV2RulesConfig = Field(
        default_factory=lambda: VolatilityV2RulesConfig(
            atr_ratio_threshold=1.15,                # v2 §1C line 253
            realized_vol_ratio_threshold=1.25,       # v2 §1C line 254
            realized_vol_short_period=10,            # v2 §1C line 254
            realized_vol_long_period=63,             # v2 §1C line 254
        )
    )

class VolumeLiquidityV2Config(StrictBaseModel):
    """v2 §1E — Layer 1 V2 Volume / Liquidity feature config.

    Ships ONLY ``volume_zscore_20d`` (v2 §1E line 395). The other two §1E
    features (``gap_frequency_20d``, ``intraday_range_percentile_252d``)
    already live on ``VolatilityV2Config`` / ``volatility_state_v2.py``
    and are read from the ``FeatureStore.volatility_state_v2`` seam by the
    §1E axis classifier — no recompute. The §1E labels
    (``normal_volume``, ``panic_volume``, ``liquidity_gap_behavior``),
    rule engine, risk-rank table, and hysteresis live in
    ``VolumeLiquidityConfig`` and ``volume_liquidity_rules``.
    """

    # v2 §1E line 395 — z-score lookback (20 sessions).
    volume_zscore_lookback_days: int = Field(gt=0, default=20)

    # v2 §1E is silent on population vs sample std. Pinned to pandas / numpy
    # default `ddof=1` (sample std) per the standard z-score convention for
    # financial time series.
    volume_zscore_ddof: int = Field(ge=0, default=1)


class VolumeLiquidityRulesConfig(StrictBaseModel):
    """v2 §1E rule-engine thresholds.

    Each threshold is cited to its line in
    ``docs/regime_engine_v2_spec.md`` §1E. The ``liquidity_gap_*``
    thresholds are live: the classifier receives
    ``gap_frequency_percentile_252d`` and
    ``intraday_range_percentile_252d`` from ``volatility_state_v2``.
    """

    # panic_volume — v2 §1E line 411. Must be > 0 because volume z-score
    # under any sane lookback has zero mean; a non-positive threshold
    # would fire on >50% of sessions, defanging the "abnormal volume" intent.
    panic_volume_zscore_threshold: float = Field(gt=0.0, default=2.0)

    # panic_volume — v2 §1E line 412. Must be < 0 because the rule gates
    # on a strictly NEGATIVE single-day return (a non-negative threshold
    # would admit up days, defeating the "selling pressure" intent).
    panic_volume_return_threshold: float = Field(lt=0.0, default=-0.02)

    # liquidity_gap_behavior — v2 §1E line 417.
    liquidity_gap_frequency_percentile_threshold: float = Field(
        ge=0.0, le=1.0, default=0.75
    )

    # liquidity_gap_behavior — v2 §1E line 418.
    liquidity_gap_intraday_range_percentile_threshold: float = Field(
        ge=0.0, le=1.0, default=0.75
    )


class VolumeLiquidityConfig(StrictBaseModel):
    """v2 §1E volume/liquidity axis classifier configuration.

    Holds the rule thresholds and per-label hysteresis days for the
    new ``volume_liquidity_state`` axis. The §1E spec is silent on
    hysteresis; defaults are pinned by risk_rank analogy with §3.7
    (panic_volume=2 after 2016-2026 walk-forward calibration; normal_volume=0;
    unknown=0).
    """

    rules: VolumeLiquidityRulesConfig

    # Per-label hysteresis: panic_volume=2 (high-risk hold retuned by the
    # 2016-2026 volume/liquidity walk-forward calibration),
    # normal_volume=0 (immediate de-escalation),
    # unknown=0 (absence-of-signal clears immediately on recovery),
    # liquidity_gap_behavior=2.
    deescalation_days_by_label: dict[str, int]

    # Default for labels NOT in `deescalation_days_by_label`.
    default_deescalation_days: int = Field(ge=0, default=0)


class BreadthV2Config(StrictBaseModel):
    """v2 §1D — Layer 1 V2 Breadth features.

    Ships sector breadth plus PIT-derived breadth features and labels when
    PIT constituent intervals and constituent OHLCV are supplied. The current
    free PIT source is bias-warning tagged; V2 PIT labels surface under
    ``pit_constituent_biased_research`` mode until a true vendor PIT feed
    replaces it.
    """

    # v2 §1D line 354 — % of 11 GICS sector ETFs with positive 21d return.
    sector_breadth_lookback_days: int = Field(gt=0, default=21)

    # v2 §1D line 329 — pct_above_50dma SMA window.
    sma_lookback_50: int = Field(default=50, ge=5)

    # v2 §1D line 334 — pct_above_200dma SMA window.
    sma_lookback_200: int = Field(default=200, ge=20)

    # v2 §1D line 343 — nh_nl_ratio 252-session lookback.
    nh_nl_lookback_sessions: int = Field(default=252, ge=20)

    # "rising"/"falling" = strict change over this many sessions. Used by the
    # narrowing_breadth + broadening_breadth rule predicates.
    label_rate_of_change_lookback_sessions: int = Field(default=5, ge=1)

    # v2 §1D line 381 — narrowing_breadth nh_nl_ratio threshold (< 0.4 fires).
    nh_nl_ratio_narrowing_threshold: float = Field(default=0.4, gt=0.0, lt=1.0)

class TrendCharacterV2Config(StrictBaseModel):
    """v2 §1A trend-character V2 axis configuration.

    Extends the existing V1 trend_character classifier with two new labels —
    ``breakout_expansion`` and ``range_bound``. All threshold defaults track
    the spec lines cited inline. Axis hysteresis lives on
    ``RegimeConfig.trend_character``, not in this V2 feature/rule config.
    """

    # v2 §1A line 131 + documented implementation decision. Must be in (0, 1] (a fraction).
    followthrough_rate_threshold: float = Field(default=0.60, gt=0.0, le=1.0)

    # v2 §1A line 152 — trailing-window cap on the followthrough walk.
    followthrough_lookback_sessions: int = Field(default=504, ge=20)

    # v2 §1A line 153 — collect up to N most-recent past breakouts.
    followthrough_window_count: int = Field(default=20, ge=1)

    # v2 §1A line 155 — sessions over which "held" is asserted.
    followthrough_hold_sessions: int = Field(default=5, ge=1)

    # v2 §1A line 146 — BB-width expansion lookback.
    bb_width_expanding_lookback: int = Field(default=5, ge=1)

    # v2 §1A line 142 — Bollinger Band period.
    bb_width_period: int = Field(default=20, ge=2)

    # v2 §1A line 142 — Bollinger Band multiplier.
    bb_width_multiplier: float = Field(default=2.0, gt=0.0)

    # v2 §1A line 168 — range_bound abs(return_63d) threshold.
    range_bound_return_63d_threshold: float = Field(default=0.05, gt=0.0)

    # v2 §1A line 169 — range_bound midpoint excursion threshold.
    range_bound_midpoint_excursion_threshold: float = Field(default=0.05, gt=0.0)

    # v2 §1A line 170 — range_bound ADX(14) threshold.
    range_bound_adx_threshold: float = Field(default=20.0, gt=0.0)
