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
    event_calendar_days: int = Field(ge=0)


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


class VolatilityV2Config(BaseModel):
    """v2 §1C — Layer 1 V2 Volatility features (Slice 2.2, evidence-only).

    Ships only the volatility V2 features that DO NOT require options data:
    ``atr_ratio``, ``gap_frequency_20d``, ``intraday_range_percentile_252d``.
    The IV/RV-spread and vol_crush features at v2 §1C lines 151–174 are
    deferred until an options-data ingestion + event-calendar slice lands
    (per v2 §10 absolute rule: do not invent missing inputs). See
    Implementation Ambiguity Log entries for the deferrals.
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


class VolumeLiquidityV2Config(BaseModel):
    """v2 §1E — Layer 1 V2 Volume / Liquidity features (Slice 2.4, evidence-only).

    Ships ONLY ``volume_zscore_20d`` (v2 §1E line 256). The other two §1E
    features (``gap_frequency_20d``, ``intraday_range_percentile_252d``)
    already live on ``VolatilityV2Config`` / ``volatility_state_v2.py``
    (Slice 2.2) and are read from the ``FeatureStore.volatility_state_v2``
    seam by the future §1E axis classifier — no recompute. The §1E labels
    (``normal_volume``, ``panic_volume``, ``liquidity_gap_behavior``),
    rule engine, risk-rank table, and hysteresis are all deferred to a
    follow-up volume-axis-classifier slice. See Implementation Ambiguity
    Log entries.
    """

    model_config = ConfigDict(extra="forbid")

    # v2 §1E line 256 — z-score lookback (20 sessions).
    volume_zscore_lookback_days: int = Field(gt=0, default=20)

    # v2 §1E is silent on population vs sample std. Pinned to pandas /
    # numpy default `ddof=1` (sample std) per the standard z-score
    # convention for financial time series. See Ambiguity Log.
    volume_zscore_ddof: int = Field(ge=0, default=1)


class BreadthV2Config(BaseModel):
    """v2 §1D — Layer 1 V2 Breadth features (Slice 2.3, evidence-only).

    Slice 2.3 ships ONLY the §1D feature that does not require a point-in-time
    (PIT) constituent-membership data pipeline: ``sector_breadth``. All other
    §1D features (`pct_above_200dma`, `ad_line` / `ad_line_slope_20d`,
    `nh_nl_ratio`, `upvol_downvol_ratio`, `breadth_thrust`) and the new V2
    breadth labels (`breadth_thrust`, `broadening_breadth`, `narrowing_breadth`)
    are deferred until the PIT membership pipeline lands (§1D lines 198–205).
    See Implementation Ambiguity Log entries #21–#27.
    """

    model_config = ConfigDict(extra="forbid")

    # v2 §1D line 229 — % of 11 GICS sector ETFs with positive 21d return.
    sector_breadth_lookback_days: int = Field(gt=0, default=21)


class TransitionScoreConfig(BaseModel):
    """Composite transition risk score configuration (v2 spec §4.3 / §4.4)."""

    model_config = ConfigDict(extra="forbid")

    # V2 §4.3 weights when HMM regime-probability shift is available.
    weights_with_hmm: dict[str, float]

    # V2 §4.3 weights when HMM is unavailable (5-component renormalization).
    weights_without_hmm: dict[str, float]

    # V2 §4.4 interpretation bands: stable / weakening / transition_warning / high.
    bands: dict[str, tuple[float, float]]


class MonetaryPressureV2Config(BaseModel):
    """Monetary pressure axis configuration (v2 spec §2A)."""

    model_config = ConfigDict(extra="forbid")

    # V2 §2A FRED series ids: 2y_yield=DGS2, 10y_yield=DGS10, broad_usd_index=DTWEXBGS.
    series_ids: dict[str, str]

    # Yield change lookback (V1 §7.3 _change_63d formula).
    yield_change_lookback_days: int = Field(ge=1)

    tightening_threshold_bps: int = Field(ge=0)

    easing_threshold_bps: int

    dxy_threshold_pct: float = Field(ge=0.0)


class InflationGrowthConfig(BaseModel):
    """Inflation/growth axis configuration (v2 spec §2B). Stub for slice 5."""

    model_config = ConfigDict(extra="forbid")

    series_ids: dict[str, str]


class CreditFundingConfig(BaseModel):
    """Credit/funding axis configuration (v2 spec §2C). Stub."""

    model_config = ConfigDict(extra="forbid")

    series_ids: dict[str, str]
    etf_universe: list[str]


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


class StrategyCohortConfig(BaseModel):
    """Strategy cohort configuration (v2 spec §5.1). Stub."""

    model_config = ConfigDict(extra="forbid")


class StrategyFamilyConstraintsConfig(BaseModel):
    """Strategy family constraints configuration (v2 spec §5.2). Stub."""

    model_config = ConfigDict(extra="forbid")


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
    transition_score: TransitionScoreConfig | None = None
    monetary_pressure_v2: MonetaryPressureV2Config | None = None
    inflation_growth: InflationGrowthConfig | None = None
    credit_funding: CreditFundingConfig | None = None
    event_calendar_v2: EventCalendarV2Config | None = None
    hmm: HMMConfig | None = None
    vol_crush: VolCrushConfig | None = None
    no_flip_flop: NoFlipFlopConfig | None = None
    strategy_cohort: StrategyCohortConfig | None = None
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
