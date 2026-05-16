from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from regime_detection import __version__


class HysteresisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trend_direction_deescalation_days: int = Field(ge=0)
    trend_character_deescalation_days: int = Field(ge=0)
    volatility_deescalation_days: int = Field(ge=0)
    breadth_deescalation_days: int = Field(ge=0)
    composite_deescalation_days: int = Field(ge=0)
    # NOTE: event_calendar has no hysteresis. Calendar windows are themselves
    # deterministic (you are inside a -2..+2 FOMC window or you are not), so a
    # debounce knob is meaningless. The previously-defined `event_calendar_days`
    # field was unused by every consumer and has been removed (V1 spec §2.10
    # updated). Adding it back requires a corresponding hysteresis call site
    # in event_calendar.py.


class DataQualityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Maximum allowed age (calendar days) of the newest row in each required series.
    max_freshness_days: int = Field(ge=0)

    # Minimum fraction of non-null values required in the lookback window for an axis to be "ok".
    min_completeness: float = Field(ge=0.0, le=1.0)


class EventCalendarConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: str


class ETFProxyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cap_weight_index: Literal["SPY"]
    equal_weight_proxy: Literal["RSP"]


class MonthlyOptionsExpiryRuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: Literal["third_friday_of_month"]
    window_trading_days: tuple[int, int]
    label: Literal["expiry_week"] = "expiry_week"


class ExpiryRulesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monthly_options: MonthlyOptionsExpiryRuleConfig


class EarningsSeasonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quarter: Literal["Q1", "Q2", "Q3", "Q4"]
    start_rule: Literal[
        "second_monday_of_january",
        "second_monday_of_april",
        "second_monday_of_july",
        "second_monday_of_october",
    ]
    end_offset_days: int = Field(ge=0)


# ---------------------------------------------------------------------------
# V2 sub-config classes (per docs/regime_engine_v2_spec.md).
#
# All V2 sub-configs default to None on RegimeConfig so the V2 spec can land
# slice-by-slice. extra="forbid" is enforced uniformly per V1 schema pattern.
# ---------------------------------------------------------------------------


class NetworkFragilityRulesConfig(BaseModel):
    """v2 §3.5 rule-engine thresholds.

    Each threshold is cited verbatim to its line in
    docs/regime_engine_v2_spec.md §3.5 (lines 617–657). The
    ``effective_rank_stability_threshold`` (0.05) encodes the spec-text
    "21d std < 5% of mean" inside the diversified_normal rule (line 620).
    """

    model_config = ConfigDict(extra="forbid")

    # diversified_normal — v2 §3.5 line 619
    diversified_normal_percentile_lo: float = Field(ge=0.0, le=1.0)
    diversified_normal_percentile_hi: float = Field(ge=0.0, le=1.0)
    # diversified_normal — v2 §3.5 line 620
    effective_rank_stability_threshold: float = Field(gt=0.0, le=1.0)
    # stock_picker_dispersion — v2 §3.5 lines 625–626
    stock_picker_percentile_max: float = Field(ge=0.0, le=1.0)
    stock_picker_dispersion_percentile_min: float = Field(ge=0.0, le=1.0)
    # correlation_concentration — v2 §3.5 lines 639–641
    concentration_corr_percentile_min: float = Field(ge=0.0, le=1.0)
    concentration_largest_eig_percentile_min: float = Field(ge=0.0, le=1.0)
    concentration_effective_rank_percentile_max: float = Field(ge=0.0, le=1.0)
    # correlation_to_one — v2 §3.5 lines 646–648
    corr_to_one_corr_percentile_min: float = Field(ge=0.0, le=1.0)
    corr_to_one_realized_vol_percentile_min: float = Field(ge=0.0, le=1.0)
    corr_to_one_drawdown_max: float
    # systemic_stress — v2 §3.5 lines 653–656
    systemic_stress_vix_percentile_min: float = Field(ge=0.0, le=1.0)


class NetworkFragilityConfig(BaseModel):
    """Network fragility axis configuration (v2 spec §3)."""

    model_config = ConfigDict(extra="forbid")

    # V2 §3.1 calls for >= 20 ETFs; default yaml ships the 22-ETF universe.
    universe: list[str] = Field(min_length=20)

    # V2 §3.2: "Average Pairwise Correlation (63d)".
    correlation_lookback_days: int = Field(ge=20)

    # V2 §3.2: percentile rank vs 504-day history.
    percentile_lookback_days: int = Field(ge=100)

    # V2 §3.2 dispersion_ratio uses 21d realised vol.
    realized_vol_lookback_days: int = Field(gt=0)

    # V2 §3.2 dispersion_ratio_percentile_252d lookback.
    dispersion_percentile_lookback_days: int = Field(gt=0)

    min_universe_size: int = Field(ge=20)

    # Aligns with V1 data_quality min_completeness (0.90 default).
    min_window_completeness: float = Field(ge=0.0, le=1.0)

    # V2 §3.7 per-label deescalation days.
    deescalation_days_by_label: dict[str, int]

    # V2 §3.7 ambiguity log entry #6: default for labels NOT in
    # deescalation_days_by_label (diversified_normal, stock_picker_dispersion,
    # unknown). 0 = immediate de-escalation, consistent with their low §3.6
    # risk rank. Exposed as config so it can be retuned in v2 §9.1 calibration.
    default_deescalation_days: int = Field(default=0, ge=0)

    # V2 §3.4–§3.5 rule engine thresholds (Slice 1.3).
    rules: NetworkFragilityRulesConfig


class TrendDirectionV2RulesConfig(BaseModel):
    """v2 §1A `recovery` rule thresholds (Slice 2.5).

    Each value cites its line in docs/regime_engine_v2_spec.md §1A. The
    `euphoria` / `breakout_expansion` / `range_bound` thresholds are
    deferred (see Implementation Ambiguity Log entries #32–#34) until
    their data inputs / spec ambiguities land.
    """

    model_config = ConfigDict(extra="forbid")

    # v2 §1A line 116 — "prior 252d drawdown <= -0.15". Must be < 0
    # because a drawdown is, by construction, in (-1.0, 0.0]; a non-negative
    # threshold would make the rule trivially true at any 252d-high.
    recovery_drawdown_threshold: float = Field(lt=0.0)

    # v2 §1A line 117 — "return_63d > 0.10". Must be > 0 because the rule
    # gates on a strictly-positive 63d return (a non-positive threshold would
    # admit drawdown days, defeating the rule's "rebound" intent).
    recovery_return_threshold: float = Field(gt=0.0)

    # v2 §1A line 162 — euphoria rule's `return_126d > 0.20`. Strict positive
    # required: a non-positive threshold would admit drawdown days, defeating
    # the rule's "strong long-horizon advance" intent. ADR 0004 + Log #32
    # closure pinned the default at +0.20 (spec verbatim).
    euphoria_return_126d_threshold: float = Field(gt=0.0, default=0.20)

    # v2 §1A line 164 — euphoria rule's `sentiment_score >= configured
    # threshold`. ADR 0004 Q3 picked `+20` (points of AAII bull-bear-spread
    # 8w-MA — historical top-decile / Yardeni-Stovall "high optimism"
    # anchor). V2 §9.1 walk-forward calibration may retune; no Pydantic
    # range bound because sentiment can go negative in the bearish regime.
    euphoria_sentiment_threshold: float = Field(default=20.0)

    # v2 §1A line 163 — euphoria rule's `realized_vol_21d rising`. ADR 0004
    # Q2 picked 5-session strict change (vol[t] > vol[t-5]) mirroring
    # Log #68's §1D `rising` / `falling` pin. Must be > 0; a zero-lookback
    # would make the rule self-comparing.
    euphoria_vol_rising_lookback_sessions: int = Field(gt=0, default=5)


class TrendDirectionV2Config(BaseModel):
    """v2 §1A — Layer 1 V2 trend direction feature lookbacks.

    Slice 2.1 ships the §1A continuous features as evidence-only.
    Slice 2.5 lands the ``recovery`` label + updated §1A precedence on
    top of those features (see ``rules`` sub-block). The other new V2
    trend labels (``euphoria`` / ``breakout_expansion`` / ``range_bound``)
    remain deferred — see Implementation Ambiguity Log entries #32–#34.
    """

    model_config = ConfigDict(extra="forbid")

    # v2 §1A line 66 — Efficiency Ratio over 20 trading days.
    efficiency_ratio_lookback_days: int = Field(gt=0)

    # v2 §1A line 79 — Hurst exponent lookback ("250d minimum").
    hurst_lookback_days: int = Field(gt=0)

    # v2 §1A line 106 — slope_sma window: (sma[t] - sma[t-20]) / sma[t-20].
    slope_lookback_days: int = Field(gt=0)

    # v2 §1A line 107 — SMA_50 short window.
    sma_short_period: int = Field(gt=0)

    # v2 §1A line 108 — SMA_200 long window.
    sma_long_period: int = Field(gt=0)

    # v2 §1A line 117 — return_63d (recovery rule input).
    return_short_period: int = Field(gt=0)

    # v2 §1A line 124 — return_126d (euphoria rule input).
    return_long_period: int = Field(gt=0)

    # v2 §1A line 116 — prior 252d drawdown (recovery rule input).
    drawdown_lookback_days: int = Field(gt=0)

    # v2 §1A line 114-119 `recovery` rule thresholds (Slice 2.5). Defaults
    # to spec values (drawdown <= -0.15, return > 0.10) when the yaml
    # omits the sub-block; v2 §9.1 calibration may retune via yaml.
    rules: TrendDirectionV2RulesConfig = Field(
        default_factory=lambda: TrendDirectionV2RulesConfig(
            recovery_drawdown_threshold=-0.15,  # v2 §1A line 116
            recovery_return_threshold=0.10,     # v2 §1A line 117
        )
    )


class VolatilityV2RulesConfig(BaseModel):
    """v2 §1C `rising_vol` and `vol_crush` rule thresholds.

    Each value cites its line in docs/regime_engine_v2_spec.md §1C.
    `vol_crush` uses FRED VIXCLS-derived implied_vol_30d and the
    event-window seam per ADR 0005 / Ambiguity Log #20.
    """

    model_config = ConfigDict(extra="forbid")

    # v2 §1C line 147 — "ATR_ratio > 1.15". Must be > 0 because ATR_ratio
    # is a non-negative ratio (ATR_short / ATR_long, both >= 0); a non-
    # positive threshold would make the rule trivially true at any
    # non-trivial ratio.
    atr_ratio_threshold: float = Field(gt=0.0, default=1.15)

    # v2 §1C line 148 — "realized_vol_10d > realized_vol_63d * 1.25". Must
    # be > 0 because realised vols are non-negative; a non-positive
    # threshold would defang the "expansion" intent.
    realized_vol_ratio_threshold: float = Field(gt=0.0, default=1.25)

    # v2 §1C line 148 — short realised-vol window (10 sessions). Pinned at
    # 10 by spec text "realized_vol_10d"; exposed for v2 §9.1 calibration.
    realized_vol_short_period: int = Field(gt=0, default=10)

    # v2 §1C line 148 — long realised-vol window (63 sessions). Pinned at
    # 63 by spec text "realized_vol_63d"; exposed for v2 §9.1 calibration.
    realized_vol_long_period: int = Field(gt=0, default=63)

    # v2 §1C `vol_crush` rule (ADR 0005 / Log #20 closure). The rule:
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
    # ADR 0005 Q1 — `implied_vol_5d_change <= -0.20`, a RELATIVE 5-session
    # change. Must be < 0: the rule gates on a strictly-negative IV move
    # (a "crush"); a non-negative threshold would defang the intent.
    vol_crush_implied_vol_change_threshold: float = Field(lt=0.0, default=-0.20)
    # ADR 0005 Q1 — lookback for the relative implied-vol change. Pinned at
    # 5 sessions (cross-axis "5-session memory" convention, Log #68 / ADR 0004).
    vol_crush_implied_vol_change_lookback_sessions: int = Field(gt=0, default=5)
    # ADR 0005 Q3 — `event_window_just_passed` fires on the N NYSE sessions
    # strictly AFTER an event window-end. Spec pins N = 3.
    vol_crush_event_window_trailing_sessions: int = Field(gt=0, default=3)


class VolatilityV2Config(BaseModel):
    """v2 §1C — Layer 1 V2 Volatility features.

    Ships ATR ratio, gap-frequency, intraday-range, realized-vol, IV/RV
    spread, and vol-crush inputs. IV-derived features are present when the
    context supplies FRED VIXCLS-derived ``implied_vol_30d``; otherwise
    those optional inputs stay absent and ``vol_crush`` falsifies per
    v2 §10.
    """

    model_config = ConfigDict(extra="forbid")

    # v2 §1C line 142 — short ATR window (ATR_14, Wilder smoothing).
    atr_short_period: int = Field(gt=0)

    # v2 §1C line 142 — long ATR window (ATR_50, Wilder smoothing).
    atr_long_period: int = Field(gt=0)

    # v2 §1C line 179 — gap_frequency lookback (20 sessions).
    gap_frequency_lookback_days: int = Field(gt=0)

    # v2 §1C line 181 — gap threshold (0.005 = 0.5%); US default, "configurable
    # per market". V2 universe is US-only so we pin a single 0.005 default.
    gap_threshold_pct: float = Field(gt=0.0, lt=1.0)

    # v2 §1C line 186 — intraday-range percentile lookback (252 sessions).
    intraday_range_lookback_days: int = Field(gt=0)

    # v2 §1C line 146-148 `rising_vol` rule thresholds + RV windows
    # (Slice 2.6). Defaults to spec values (atr_ratio > 1.15, rv_10d >
    # rv_63d * 1.25) when the yaml omits the sub-block; v2 §9.1
    # calibration may retune via yaml.
    rules: VolatilityV2RulesConfig = Field(
        default_factory=lambda: VolatilityV2RulesConfig(
            atr_ratio_threshold=1.15,                # v2 §1C line 147
            realized_vol_ratio_threshold=1.25,       # v2 §1C line 148
            realized_vol_short_period=10,            # v2 §1C line 148
            realized_vol_long_period=63,             # v2 §1C line 148
        )
    )


class VolumeLiquidityV2Config(BaseModel):
    """v2 §1E — Layer 1 V2 Volume / Liquidity feature config (Slice 2.4).

    Ships ONLY ``volume_zscore_20d`` (v2 §1E line 256). The other two §1E
    features (``gap_frequency_20d``, ``intraday_range_percentile_252d``)
    already live on ``VolatilityV2Config`` / ``volatility_state_v2.py``
    (Slice 2.2) and are read from the ``FeatureStore.volatility_state_v2``
    seam by the §1E axis classifier — no recompute. The §1E labels
    (``normal_volume``, ``panic_volume``, ``liquidity_gap_behavior``),
    rule engine, risk-rank table, and hysteresis live in
    ``VolumeLiquidityConfig`` and ``volume_liquidity_rules``.
    """

    model_config = ConfigDict(extra="forbid")

    # v2 §1E line 256 — z-score lookback (20 sessions).
    volume_zscore_lookback_days: int = Field(gt=0, default=20)

    # v2 §1E is silent on population vs sample std. Pinned to pandas /
    # numpy default `ddof=1` (sample std) per the standard z-score
    # convention for financial time series. See Ambiguity Log.
    volume_zscore_ddof: int = Field(ge=0, default=1)


class VolumeLiquidityRulesConfig(BaseModel):
    """v2 §1E rule-engine thresholds (Slice 2.7).

    Each threshold is cited to its line in
    ``docs/regime_engine_v2_spec.md`` §1E. The ``liquidity_gap_*``
    thresholds are live: the classifier receives
    ``gap_frequency_percentile_252d`` and
    ``intraday_range_percentile_252d`` from ``volatility_state_v2``.
    """

    model_config = ConfigDict(extra="forbid")

    # panic_volume — v2 §1E line 272. Must be > 0 because volume z-score
    # under any sane lookback has zero mean; a non-positive threshold
    # would fire on >50% of sessions, defanging the "abnormal volume" intent.
    panic_volume_zscore_threshold: float = Field(gt=0.0, default=2.0)

    # panic_volume — v2 §1E line 273. Must be < 0 because the rule gates
    # on a strictly NEGATIVE single-day return (a non-negative threshold
    # would admit up days, defeating the "selling pressure" intent).
    panic_volume_return_threshold: float = Field(lt=0.0, default=-0.02)

    # liquidity_gap_behavior — v2 §1E line 278.
    liquidity_gap_frequency_percentile_threshold: float = Field(
        ge=0.0, le=1.0, default=0.75
    )

    # liquidity_gap_behavior — v2 §1E line 279.
    liquidity_gap_intraday_range_percentile_threshold: float = Field(
        ge=0.0, le=1.0, default=0.75
    )


class VolumeLiquidityConfig(BaseModel):
    """v2 §1E volume/liquidity axis classifier configuration (Slice 2.7).

    Holds the rule thresholds and per-label hysteresis days for the
    new ``volume_liquidity_state`` axis. The §1E spec is silent on
    hysteresis (Implementation Ambiguity Log entry #41 pins defaults
    by risk_rank analogy with §3.7 — panic_volume = 3 like
    correlation_to_one's hold pattern; normal_volume = 0; unknown = 2).
    """

    model_config = ConfigDict(extra="forbid")

    # v2 §1E rule-engine thresholds (Slice 2.7).
    rules: VolumeLiquidityRulesConfig

    # v2 §1E is silent on per-label hysteresis days. Ambiguity Log entry
    # #41 pins panic_volume=3 (high-risk hold, analogous to §3.7
    # correlation_to_one=3-5), normal_volume=0 (immediate de-escalation),
    # unknown=2 (modest hold to absorb single-day NaN flickers without
    # stranding the axis), liquidity_gap_behavior=2 (same rank as unknown).
    deescalation_days_by_label: dict[str, int]

    # Default for labels NOT in `deescalation_days_by_label`. Matches the
    # §3.7 ambiguity #6 pattern.
    default_deescalation_days: int = Field(ge=0, default=0)


class BreadthV2Config(BaseModel):
    """v2 §1D — Layer 1 V2 Breadth features.

    Ships sector breadth plus PIT-derived breadth features and labels when
    PIT constituent intervals and constituent OHLCV are supplied. The current
    free PIT source is bias-warning tagged; V2 PIT labels surface under
    ``pit_constituent_biased_research`` mode until a true vendor PIT feed
    replaces it.
    """

    model_config = ConfigDict(extra="forbid")

    # v2 §1D line 229 — % of 11 GICS sector ETFs with positive 21d return.
    sector_breadth_lookback_days: int = Field(gt=0, default=21)

    # v2 §1D line 207 — pct_above_50dma SMA window (Slice 2.8c).
    sma_lookback_50: int = Field(default=50, ge=5)

    # v2 §1D line 209 — pct_above_200dma SMA window (Slice 2.8c).
    sma_lookback_200: int = Field(default=200, ge=20)

    # v2 §1D line 218 — nh_nl_ratio 252-session lookback (Slice 2.8c,
    # Ambiguity Log #55).
    nh_nl_lookback_sessions: int = Field(default=252, ge=20)

    # v2 §1D Ambiguity Log #68 — "rising"/"falling" = strict change over
    # this many sessions. Used by the narrowing_breadth + broadening_breadth
    # rule predicates (Slice 1D code).
    label_rate_of_change_lookback_sessions: int = Field(default=5, ge=1)

    # v2 §1D line 280 — narrowing_breadth nh_nl_ratio threshold (< 0.4 fires).
    # Exposed as config for v2 §9.1 calibration.
    nh_nl_ratio_narrowing_threshold: float = Field(default=0.4, gt=0.0, lt=1.0)


class TrendCharacterV2Config(BaseModel):
    """v2 §1B trend-character V2 axis configuration (Ambiguity Log #46/#47/#67).

    Extends the existing V1 trend_character classifier with two new labels —
    ``breakout_expansion`` and ``range_bound`` — plus the per-label
    asymmetric hysteresis days pinned in Log #67. All threshold defaults
    track the spec lines cited inline; v2 §9.1 walk-forward may retune.
    """

    model_config = ConfigDict(extra="forbid")

    # v2 §1B Ambiguity Log #67 per-label de-escalation days.
    deescalation_days_by_label: dict[str, int] = Field(
        default_factory=lambda: {
            "breakout_expansion": 3,
            "recovery_attempt": 3,
            "trending": 0,
            "range_bound": 3,
            "chop": 0,
            "transition": 2,
            "unknown": 2,
        }
    )
    # Default for labels not in `deescalation_days_by_label` (matches §3.7
    # Ambiguity Log #6 pattern).
    default_deescalation_days: int = Field(default=0, ge=0)

    # v2 §1B line 90 + Ambiguity Log #46. Must be in (0, 1] (a fraction).
    followthrough_rate_threshold: float = Field(default=0.60, gt=0.0, le=1.0)

    # v2 §1B line 111 — trailing-window cap on the followthrough walk.
    followthrough_lookback_sessions: int = Field(default=504, ge=20)

    # v2 §1B line 112 — collect up to N most-recent past breakouts.
    followthrough_window_count: int = Field(default=20, ge=1)

    # v2 §1B line 113 — sessions over which "held" is asserted.
    followthrough_hold_sessions: int = Field(default=5, ge=1)

    # v2 §1B line 105 — BB-width expansion lookback.
    bb_width_expanding_lookback: int = Field(default=5, ge=1)

    # v2 §1B line 102 — Bollinger Band period.
    bb_width_period: int = Field(default=20, ge=2)

    # v2 §1B line 102 — Bollinger Band multiplier.
    bb_width_multiplier: float = Field(default=2.0, gt=0.0)

    # v2 §1B line 127 — range_bound abs(return_63d) threshold.
    range_bound_return_63d_threshold: float = Field(default=0.05, gt=0.0)

    # v2 §1B line 128 — range_bound midpoint excursion threshold.
    range_bound_midpoint_excursion_threshold: float = Field(default=0.05, gt=0.0)

    # v2 §1B line 129 — range_bound ADX(14) threshold.
    range_bound_adx_threshold: float = Field(default=20.0, gt=0.0)


class TransitionScoreConfig(BaseModel):
    """Composite transition risk score configuration (v2 spec §4.3 / §4.4)."""

    model_config = ConfigDict(extra="forbid")

    # V2 §4.3 weights when HMM regime-probability shift is available.
    weights_with_hmm: dict[str, float]

    # V2 §4.3 weights when HMM is unavailable (5-component renormalization).
    weights_without_hmm: dict[str, float]

    # V2 §4.3 + Ambiguity Log #66 — weights when change_point evidence is
    # available but HMM is not (6 components).
    weights_with_change_point: dict[str, float]

    # V2 §4.3 + Ambiguity Log #66 — weights when both HMM and change_point
    # evidence are available (7 components, full V2 evidence stack).
    weights_with_hmm_with_change_point: dict[str, float]

    # V2 §4.4 interpretation bands: stable / weakening / transition_warning / high.
    bands: dict[str, tuple[float, float]]


class MonetaryPressureV2FeaturesConfig(BaseModel):
    """v2 §2A — Layer 2A Monetary/Liquidity V2 features (Slice 4.1, evidence-only).

    Ships ONLY the ONE feature formula spec-pinned at v2 §2A line 896::

        yield_change_zscore = (yield_change_63d - mean_5y) / std_5y

    applied to the two FRED yield series with explicit spec-given source
    contract (lines 887–889): ``DGS2`` (2y) and ``DGS10`` (10y).

    Per V2 §10 ABSOLUTE RULE the following are DEFERRED because the spec
    does not pin them (see Implementation Ambiguity Log entries #44 and
    #45):

    - ``broad_usd_index_zscore_63d`` (formula unspecified).
    - ``yield_change_zscore_21d_2y`` / ``yield_change_zscore_21d_10y``
      (21d variant: neither the change-window nor the mean/std window
      length is given).
    - The §2A label set (``tightening_pressure``, ``easing_pressure``,
      ``rate_shock``, neutral, unknown) — no Literal[...] declared in spec.
    - Precedence ordering, risk-rank table, per-label hysteresis days.
    - The ``MonetaryPressureSeriesClassifier`` axis classifier.

    The slice-2.4 precedent (Ambiguity Log #29) — shipping
    ``volume_zscore_20d`` as evidence-only before its §1E axis
    classifier landed in slice 2.7 — applies here: the two yield
    z-scores ship as evidence-only and become inputs to the future
    §2A axis classifier once the spec is amended.
    """

    model_config = ConfigDict(extra="forbid")

    # v2 §2A line 896 — `yield_change_63d[t] = yield[t] - yield[t-63]`.
    # Must be > 0 because the change is computed by `yield - yield.shift(N)`
    # with N >= 1; N == 0 would produce an identically-zero change series.
    yield_change_lookback_days: int = Field(gt=0, default=63)

    # v2 §2A line 896 — mean/std normalizer window ("5y"). 5y ≈ 1260
    # trading days under NYSE calendar conventions used throughout V2.
    # Must be > 0 (rolling mean/std requires at least one observation).
    zscore_normalizer_window_days: int = Field(gt=0, default=1260)

    # v2 §2A 21d-variant rate_shock predicate lookback per Ambiguity Log #46 (a).
    # Mechanical generalization of the line-896 template using a 21d change window.
    rate_shock_lookback_days: int = Field(gt=0, default=21)

    # v2 §2A broad_usd_index z-score lookback per Ambiguity Log #46 (a). Mechanical
    # generalization of the line-896 template applied to a USD-index level series.
    broad_usd_lookback_days: int = Field(gt=0, default=63)


class MonetaryPressureV2RulesConfig(BaseModel):
    """v2 §2A monetary-pressure rule thresholds (Ambiguity Log #46 b/c).

    Each value pins the verbatim §2A rule predicate threshold. Precedence
    is enforced in ``monetary_pressure.evaluate_rules`` per Log #46 (c).
    """

    model_config = ConfigDict(extra="forbid")

    # §2A tightening_pressure: yield_change_zscore_*_63d > +1.5 OR broad_usd > +1.5.
    tightening_pressure_zscore_threshold: float = Field(default=1.5, gt=0.0)
    # §2A easing_pressure: yield_change_zscore_*_63d < -1.5 on either tenor.
    easing_pressure_zscore_threshold: float = Field(default=-1.5, lt=0.0)
    # §2A rate_shock: abs(yield_change_zscore_21d_*) > 2.0.
    rate_shock_zscore_threshold: float = Field(default=2.0, gt=0.0)


class MonetaryPressureV2Config(BaseModel):
    """v2 §2A monetary-pressure axis classifier config (Ambiguity Log #46).

    Separate from ``MonetaryPressureV2FeaturesConfig`` (features vs
    classifier), mirroring the ``volume_liquidity_v2`` vs
    ``volume_liquidity_state`` split.
    """

    model_config = ConfigDict(extra="forbid")

    rules: MonetaryPressureV2RulesConfig = Field(
        default_factory=MonetaryPressureV2RulesConfig
    )
    # §2A per-label hysteresis days per Ambiguity Log #46 (e).
    deescalation_days_by_label: dict[str, int]
    # Default for labels NOT listed (matches §3.7 Ambiguity Log #6 pattern).
    default_deescalation_days: int = Field(default=0, ge=0)


class NewsSentimentConfig(BaseModel):
    """v2 §1A SF Fed Daily News Sentiment evidence config.

    Audit follow-up (post-#12). Pinned as an EVIDENCE-only second
    sentiment voice alongside the AAII bull-bear 8w-MA `sentiment_score`.
    The §1A `euphoria` rule predicate consumes only the AAII series per
    spec line 164; this config does NOT modify that rule. The news
    sentiment score and the derived `sentiment_concordance` flag surface
    in evidence dicts so downstream consumers can treat divergent
    euphoria firings as lower-conviction.

    Bias-warning code emitted in feature output:
    ``news_sentiment_sf_fed_daily_news_index``.
    """

    model_config = ConfigDict(extra="forbid")

    # Smoothing window over the daily SF Fed news sentiment. Default 21
    # NYSE sessions ≈ 1 month — short enough to react to material
    # narrative shifts, long enough to dampen single-day noise. v2 §9.1
    # walk-forward calibration placeholder.
    smoothing_window_sessions: int = Field(default=21, gt=0)


class CentralBankTextConfig(BaseModel):
    """v2 §2A central-bank-text classifier config (spec lines 2578-2586).

    Pinned as an approved deterministic-lexicon substitute for the
    spec's "LLM classifier" phrasing. The substitution preserves V1 §2.2
    stateless replay (LLM calls are non-deterministic; the lexicon is
    pure-function). The resulting score is fed into
    ``monetary_pressure.evidence`` — never a standalone label per spec.

    Bias-warning code emitted in the feature output:
    ``central_bank_text_deterministic_lexicon_substitute``.
    """

    model_config = ConfigDict(extra="forbid")

    # Smoothing window in NYSE sessions over the forward-filled per-release
    # net_score series. Default 30 sessions ≈ 6 weeks ≈ four FOMC-cycle
    # releases, mirrors the AAII 8w-MA smoothing pattern §1A uses for
    # ``sentiment_score``. v2 §9.1 walk-forward calibration may retune.
    smoothing_window_sessions: int = Field(default=30, gt=0)

    # Optional safety cap: drop releases older than this many calendar
    # days at score time. Default 365 keeps an entire policy cycle of
    # history while excluding stale rows that pre-date the OHLCV window.
    max_release_age_days: int = Field(default=365, gt=0)

    # Same-date collision strategy (audit follow-up #12). When FOMC
    # minutes and a Powell speech share a release date, this picks
    # which voice wins. Default `pick_longer` matches the audit M1
    # initial wiring (token-count is a rough proxy for material
    # content). `token_weighted_average` averages all same-date rows
    # by token weight. `fomc_priority` favours FOMC minutes
    # unconditionally. v2 §9.1 walk-forward calibration placeholder.
    same_date_aggregation: Literal[
        "pick_longer", "token_weighted_average", "fomc_priority"
    ] = Field(default="pick_longer")


class InflationGrowthRulesConfig(BaseModel):
    """v2 §2B inflation/growth rule thresholds (Slice 5).

    Defaults match the spec verbatim (§2B lines 2232-2270). v2 §9.1 walk-
    forward calibration may retune via yaml.
    """

    model_config = ConfigDict(extra="forbid")

    # §2B line 2234 — goldilocks "abs drift over 21d <= 0.005" (50bps).
    cpi_drift_threshold: float = Field(default=0.005, gt=0.0)
    # §2B line 2236 / 2259 — pmi > 50.
    pmi_goldilocks_threshold: float = Field(default=50.0, gt=0.0)
    pmi_recovery_threshold: float = Field(default=50.0, gt=0.0)
    # §2B line 2250 — disinflation "pmi > 45".
    pmi_disinflation_threshold: float = Field(default=45.0, gt=0.0)
    # §2B line 2242 — inflation_shock "commodity_return_63d > 0.15".
    commodity_return_threshold: float = Field(default=0.15, gt=0.0)
    # §2B line 2256 — recession_scare "spy_21d_return < -0.05".
    spy_recession_threshold: float = Field(default=-0.05, lt=0.0)
    # §2B lines 2197-2198 — CPI 3m / 6m percent-change lookbacks.
    cpi_lookback_3m_sessions: int = Field(default=63, ge=20)
    cpi_lookback_6m_sessions: int = Field(default=126, ge=20)
    # §2B line 2235 — 21d slope window on cpi_6m_change_pct.
    cpi_slope_lookback_sessions: int = Field(default=21, ge=5)
    # §2B line 2209 — 21d OLS slope on pmi_manufacturing.
    pmi_slope_lookback_sessions: int = Field(default=21, ge=5)
    # §2B line 2220 — DBC 63d return.
    commodity_return_lookback_sessions: int = Field(default=63, ge=5)
    # §2B line 2223 — DGS10 21d slope.
    treasury_slope_lookback_sessions: int = Field(default=21, ge=5)
    # §2B line 2227 — cyclical/defensive 21d slope.
    cyclical_defensive_slope_lookback_sessions: int = Field(default=21, ge=5)
    # §2B line 2237 — SPY 21d return.
    spy_return_lookback_sessions: int = Field(default=21, ge=5)
    # §2B line 2245 — TLT 21d return.
    tlt_return_lookback_sessions: int = Field(default=21, ge=5)
    # §2B line 2551 — inflation_shock single-signal limb threshold
    # (`inflation_surprise_zscore > +1.5`). ADR 0006. Must be > 0: the
    # limb gates on a strictly-positive (hotter-than-nowcast) surprise.
    inflation_surprise_zscore_threshold: float = Field(default=1.5, gt=0.0)
    # ADR 0006 — 5y rolling-std normalizer window for the inflation
    # surprise (1260 trading days, same convention as §2A yield z-scores).
    inflation_surprise_normalizer_window_sessions: int = Field(default=1260, ge=20)
    # ADR 0006 — lookback for the realized 1-month CPI inflation rate
    # (~21 trading days = 1 month, matches the Cleveland Fed nowcast cadence).
    inflation_surprise_realized_rate_lookback_sessions: int = Field(default=21, ge=5)
    # §2B line 2605 — earnings_expansion "aggregate_forward_eps_revision_
    # direction_4w > +0.02". Must be > 0 (a strictly-positive 4-week
    # forward-EPS revision).
    eps_revision_expansion_threshold: float = Field(default=0.02, gt=0.0)
    # §2B line 2609 — earnings_contraction "... < -0.02". Must be < 0
    # (a strictly-negative 4-week revision).
    eps_revision_contraction_threshold: float = Field(default=-0.02, lt=0.0)
    # §2A line 2587-2593 — first-release vs latest-revision CPI for replay.
    # When True AND ``MarketContext.cpi_first_release`` is supplied, the
    # realized inflation rate (and the cpi 3m/6m change series) read from
    # the first-release vintage so historical replay is PIT-accurate. When
    # False or when the vintage seam is absent, the existing revised
    # CPIAUCSL path is preserved unchanged. Default True per spec contract.
    use_first_release_cpi_when_available: bool = Field(default=True)


class InflationGrowthConfig(BaseModel):
    """v2 §2B Inflation/Growth axis configuration (Slice 5).

    Wires the rule thresholds, per-label hysteresis days, and the
    unknown-gate staleness thresholds (§2B lines 2308-2312).
    """

    model_config = ConfigDict(extra="forbid")

    series_ids: dict[str, str] = Field(default_factory=dict)
    rules: InflationGrowthRulesConfig = Field(default_factory=InflationGrowthRulesConfig)
    # §2B lines 2293-2303 — per-label asymmetric hysteresis days.
    deescalation_days_by_label: dict[str, int]
    default_deescalation_days: int = Field(default=0, ge=0)
    # §2B line 2309 — "CPI stale > 60 calendar days" (2× monthly cycle).
    cpi_stale_calendar_days: int = Field(default=60, ge=1)
    # §2B line 2310 — "PMI stale > 45 calendar days" (1.5× monthly cycle).
    pmi_stale_calendar_days: int = Field(default=45, ge=1)
    # §2B line 2311 — "DGS10 stale > 5 sessions".
    dgs10_stale_sessions: int = Field(default=5, ge=1)


class CreditFundingRulesConfig(BaseModel):
    """v2 §2C rule thresholds (Slice 4).

    Defaults match the spec verbatim (§2C lines 2064-2088). Calibration
    placeholders per spec line 2128 (v2 §9.1 may retune via yaml).
    """

    model_config = ConfigDict(extra="forbid")

    # §2C line 2066 — credit_calm "<0.50"
    hy_percentile_calm_max: float = Field(default=0.50, ge=0.0, le=1.0)
    # §2C line 2074 — credit_stress ">0.80"
    hy_percentile_stress_min: float = Field(default=0.80, ge=0.0, le=1.0)
    # §2C lines 2075/2083 — equities-falling threshold "< -0.05".
    spy_drop_threshold: float = Field(default=-0.05, le=0.0)
    # §2C line 2078 — funding_squeeze "> +1.5" (USD z-score 21d-change variant).
    broad_usd_zscore_funding_threshold: float = Field(default=1.5, gt=0.0)
    # §2C line 2085 — deleveraging "> 0" (USD z-score 21d-change variant).
    broad_usd_zscore_deleveraging_threshold: float = Field(default=0.0)
    # §2C line 2086 — deleveraging "realized_vol_21d_percentile_252d > 0.75".
    realized_vol_percentile_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    # §2C line 2087 — deleveraging "avg_pairwise_corr_percentile_504d > 0.75".
    correlation_percentile_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    # §2C line 2038 — 504d percentile lookback ("scale-invariant predicate").
    hy_percentile_504d_lookback: int = Field(default=504, ge=20)
    # §2C line 2041/2059 — 21d OLS slope window.
    slope_21d_lookback: int = Field(default=21, ge=5)
    # §2C line 2046 — KRE/SPY ratio 63d OLS slope window.
    slope_63d_lookback: int = Field(default=63, ge=5)
    # §2C line 2032 — 63d total-return lookback.
    total_return_lookback_days: int = Field(default=63, ge=5)
    # §2C line 2075 — spy_21d_return lookback.
    spy_return_lookback_days: int = Field(default=21, ge=5)
    # §2C line 2084 — tlt_21d_return lookback.
    tlt_return_lookback_days: int = Field(default=21, ge=5)
    # §2C lines 2052-2055 — 21d-change variant (vs §2A 63d-change variant).
    broad_usd_change_window_days: int = Field(default=21, ge=5)
    # §2C line 2055 normalizer window (~5y trading days, same as §2A).
    broad_usd_normalizer_window_days: int = Field(default=1260, ge=100)


class CreditFundingConfig(BaseModel):
    """v2 §2C Credit/Funding axis configuration (Slice 4).

    Wires the rule thresholds, per-label hysteresis days, and the
    unknown-gate staleness thresholds. The 8-symbol universe
    (HYG/LQD/TLT/KRE/SOFR/IORB/NFCI/broad_usd_index) is hard-pinned in
    code per spec §2C lines 2024-2030 — no yaml override.
    """

    model_config = ConfigDict(extra="forbid")

    rules: CreditFundingRulesConfig = Field(default_factory=CreditFundingRulesConfig)
    # §2C lines 2110-2117 — per-label asymmetric hysteresis days.
    deescalation_days_by_label: dict[str, int]
    # Labels not listed take this default (matches §3.7 Ambiguity Log #6 pattern).
    default_deescalation_days: int = Field(default=0, ge=0)
    # §2C line 2124 — NFCI weekly: "stale > 14 days (2× weekly release cycle)".
    nfci_stale_days: int = Field(default=14, ge=1)
    # §2C line 2123 — HYG/LQD/TLT stale > 5 sessions.
    etf_stale_sessions: int = Field(default=5, ge=1)


class EventCalendarV2Config(BaseModel):
    """Event calendar v2 configuration (v2 spec §2D). Stub."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


class HMMConfig(BaseModel):
    """Hidden Markov Model regime probability configuration (v2 spec §6.1)."""

    model_config = ConfigDict(extra="forbid")

    n_states: int = Field(ge=2)
    training_window_days: int = Field(ge=100)
    retrain_cadence_days: int = Field(ge=1)
    # Slice 6: deterministic seed for hmmlearn.GaussianHMM. Reproducibility
    # gate — same inputs + same seed → byte-identical posterior.
    random_state: int = Field(default=42, ge=0)
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    min_covar: float = Field(default=0.001, ge=0.0)
    standardize_inputs: bool = True
    random_seeds: tuple[int, ...] = Field(
        default=(42, 101, 202, 303, 404, 505, 606, 707, 808, 909),
        min_length=1,
    )


class ClusteringConfig(BaseModel):
    """v2 §6.2 K-Means/GMM clustering configuration (Slice 7).

    GMM is the V2 ship default; K-Means support deferred per spec line
    2835. Mapping cluster_id → economic_label is operator-side
    (cluster_label_map.yaml per spec line 2842); not part of this slice.
    """

    model_config = ConfigDict(extra="forbid")

    n_clusters: int = Field(default=8, ge=2)
    training_window_days: int = Field(default=1260, ge=100)
    random_state: int = Field(default=42, ge=0)
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    model_version: str = Field(default="gmm_8cluster_v1.0")


class ChangePointConfig(BaseModel):
    """v2 §6.3 BOCPD change-point detection (Slice 8, evidence-only).

    Implementation library: bayesian-changepoint-detection (Ambiguity Log #62).
    Observation series: realized_vol_21d (#63).
    Score = 5-session rolling max of posterior P(run_length=0) (#64).
    Break = posterior >= 0.5 threshold (#65).
    """

    model_config = ConfigDict(extra="forbid")

    hazard_lambda: float = Field(default=250.0, gt=0.0)  # spec line 2872: 1/250 → lambda=250
    score_window_days: int = Field(default=5, ge=1)  # Ambiguity Log #64
    break_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)  # Ambiguity Log #65
    training_window_days: int = Field(default=1260, ge=100)  # 5y, matches HMM/GMM
    # StudentT prior hyperparameters (Adams-MacKay defaults — conservative).
    student_t_alpha: float = Field(default=0.1, gt=0.0)
    student_t_beta: float = Field(default=0.01, gt=0.0)
    student_t_kappa: float = Field(default=1.0, gt=0.0)
    student_t_mu: float = Field(default=0.0)
    method: str = Field(default="BOCPD")


class VolCrushConfig(BaseModel):
    """Volatility crush detection configuration (v2 spec §5.3)."""

    model_config = ConfigDict(extra="forbid")

    # V2 §5.3: "as_of_date within 3 NYSE trading days AFTER configured event end".
    event_window_trading_days: int = Field(ge=0)

    implied_vol_5d_change_threshold: float

    realized_vol_ratio_threshold: float = Field(ge=0.0)


class NoFlipFlopConfig(BaseModel):
    """No-flip-flop stability guard configuration (v2 spec §5.4). Stub."""

    model_config = ConfigDict(extra="forbid")

    window_trading_days: int = Field(ge=0)


class CohortRoutingRulePredicate(BaseModel):
    """v2 §5.1 single-axis predicate (member-match against active label)."""

    model_config = ConfigDict(extra="forbid")

    axis: Literal[
        "network_fragility",
        "volatility_state",
        "trend_direction",
        "breadth_state",
        "monetary_pressure",
        "trend_character",
    ]
    values: list[str]


class CohortRoutingRule(BaseModel):
    """v2 §5.1 cohort routing rule.

    `any_of` predicates form an OR-match group; `all_of` predicates form an
    AND-match group. A rule fires when each non-empty group matches per its
    own quantifier. An empty rule (both lists empty) never fires —
    default_neutral is handled by the walker as the universal fallback.
    """

    model_config = ConfigDict(extra="forbid")

    any_of: list[CohortRoutingRulePredicate] = Field(default_factory=list)
    all_of: list[CohortRoutingRulePredicate] = Field(default_factory=list)


class CohortRoutingConfig(BaseModel):
    """v2 §5.1 Agent Cohort Routing configuration (Slice 5.1)."""

    model_config = ConfigDict(extra="forbid")

    routing_rules: dict[str, CohortRoutingRule]
    blocked_cohorts: dict[str, list[str]]


class FamilyOverride(BaseModel):
    """v2 §5.2 — one family's constraint values under one specialist cohort.

    All fields Optional so a specialist cohort can override just one
    dimension (e.g. just ``allowed``) and inherit the rest from
    ``default_neutral``. ``allowed`` is REQUIRED on the ``default_neutral``
    entry per the spec baseline contract (enforced in
    ``resolve_strategy_family_constraints``); on cohort overrides it stays
    Optional so a cohort can re-tune just a non-``allowed`` knob.
    """

    model_config = ConfigDict(extra="forbid")

    allowed: bool | None = None
    max_lookback_days: int | None = None
    max_holding_days: int | None = None
    max_position_pct: float | None = None
    min_adx: int | None = None
    require_breadth_confirmation: bool | None = None
    require_volume_confirmation: bool | None = None
    event_window_only: bool | None = None
    reason: str | None = None


class StrategyFamilyConstraintsConfig(BaseModel):
    """v2 §5.2 family constraints — override-on-default inheritance.

    ``default_neutral`` carries the baseline for every strategy family the
    engine constrains. Specialist cohorts declare ONLY the field-level
    overrides that diverge from the baseline; unspecified families inherit
    the ``default_neutral`` values verbatim.
    """

    model_config = ConfigDict(extra="forbid")

    # Keyed by family name (e.g. ``trend_following``). ``allowed`` is
    # REQUIRED on every default_neutral entry; the resolver enforces this.
    default_neutral: dict[str, FamilyOverride]
    # First key = cohort name, second key = family name.
    overrides: dict[str, dict[str, FamilyOverride]]


class RegimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: Literal["core3-v1.0.0", "core3-v2.0.0"]
    market: Literal["US"]
    trading_calendar: Literal["NYSE"]
    breadth_mode: Literal["etf_proxy"]
    etf_proxy: ETFProxyConfig
    event_calendar: EventCalendarConfig
    expiry_rules: ExpiryRulesConfig
    earnings_seasons: list[EarningsSeasonConfig]
    data_quality: DataQualityConfig
    hysteresis: HysteresisConfig

    # V2 optional sub-configs (default None so V2 slices can land independently).
    network_fragility: NetworkFragilityConfig | None = None
    trend_direction_v2: TrendDirectionV2Config | None = None
    volatility_state_v2: VolatilityV2Config | None = None
    breadth_state_v2: BreadthV2Config | None = None
    volume_liquidity_v2: VolumeLiquidityV2Config | None = None
    # v2 §1E axis classifier configuration (Slice 2.7).
    volume_liquidity_state: VolumeLiquidityConfig | None = None
    transition_score: TransitionScoreConfig | None = None
    # v2 §1B trend-character V2 axis configuration (Ambiguity Log #46/#47/#67).
    trend_character_v2: TrendCharacterV2Config | None = None
    monetary_pressure_v2: MonetaryPressureV2FeaturesConfig | None = None
    # v2 §2A axis classifier configuration (Ambiguity Log #46 pins).
    monetary_pressure_state: MonetaryPressureV2Config | None = None
    # v2 §2A central-bank-text evidence config (deterministic-lexicon
    # substitute for the spec's "LLM classifier" phrasing; see
    # docs/spec_code_data_audit_2026_05_15.md §3.1 / M1).
    central_bank_text: CentralBankTextConfig | None = None
    # v2 §1A SF Fed news sentiment evidence config (audit follow-up
    # post-#12). Evidence only — never read by the `euphoria` rule.
    news_sentiment: NewsSentimentConfig | None = None
    inflation_growth: InflationGrowthConfig | None = None
    credit_funding: CreditFundingConfig | None = None
    event_calendar_v2: EventCalendarV2Config | None = None
    hmm: HMMConfig | None = None
    # v2 §6.2 GMM clustering evidence layer (Slice 7).
    clustering: ClusteringConfig | None = None
    # v2 §6.3 BOCPD change-point evidence layer (Slice 8).
    change_point: ChangePointConfig | None = None
    vol_crush: VolCrushConfig | None = None
    no_flip_flop: NoFlipFlopConfig | None = None
    cohort_routing: CohortRoutingConfig | None = None  # v2 §5.1 (slice 5.1)
    strategy_family_constraints: StrategyFamilyConstraintsConfig | None = None


def load_regime_config(path: str | Path) -> RegimeConfig:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level")
    return RegimeConfig.model_validate(data)


def load_default_regime_config() -> RegimeConfig:
    """
    Load the packaged default config shipped with the library.

    Dispatches on package ``__version__``:
        - "2.x"  -> configs/core3-v2.0.0.yaml
        - "1.x"  -> configs/core3-v1.0.0.yaml

    NOTE: We load the resource content directly (instead of returning a filesystem
    Path) so this works even when the package is distributed as a zip/egg.
    """
    if __version__.startswith("2."):
        resource_name = "configs/core3-v2.0.0.yaml"
    elif __version__.startswith("1."):
        resource_name = "configs/core3-v1.0.0.yaml"
    else:
        raise ValueError(
            f"Unsupported package __version__ for default config dispatch: {__version__!r}"
        )

    pkg_file = importlib.resources.files("regime_detection").joinpath(resource_name)
    text = pkg_file.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Default config must contain a YAML mapping at the top level")
    return RegimeConfig.model_validate(data)
