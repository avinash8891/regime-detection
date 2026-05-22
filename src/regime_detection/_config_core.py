from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    """Base for all RegimeConfig sub-models — forbids unknown fields uniformly."""

    model_config = ConfigDict(extra="forbid")


AxisName = Literal[
    "network_fragility",
    "volatility_state",
    "trend_direction",
    "breadth_state",
    "monetary_pressure",
    "trend_character",
]


class DataQualityConfig(StrictBaseModel):
    # Maximum allowed age (calendar days) of the newest row in each required series.
    max_freshness_days: int = Field(ge=0)

    # Minimum fraction of non-null values required in the lookback window for an axis to be "ok".
    min_completeness: float = Field(ge=0.0, le=1.0)


class EventCalendarConfig(StrictBaseModel):
    market: str


class ETFProxyConfig(StrictBaseModel):
    cap_weight_index: Literal["SPY"]
    equal_weight_proxy: Literal["RSP"]


class MonthlyOptionsExpiryRuleConfig(StrictBaseModel):
    rule: Literal["third_friday_of_month"]
    window_trading_days: tuple[int, int]
    label: Literal["expiry_week"] = "expiry_week"


class ExpiryRulesConfig(StrictBaseModel):
    monthly_options: MonthlyOptionsExpiryRuleConfig


class EarningsSeasonConfig(StrictBaseModel):
    quarter: Literal["Q1", "Q2", "Q3", "Q4"]
    start_rule: Literal[
        "second_monday_of_january",
        "second_monday_of_april",
        "second_monday_of_july",
        "second_monday_of_october",
    ]
    end_offset_days: int = Field(ge=0)


# V2 sub-configs default to None on RegimeConfig so they can land slice-by-slice.


class NetworkFragilityRulesConfig(StrictBaseModel):
    """v2 §3.5 rule-engine thresholds.

    Each threshold is cited verbatim to its line in
    docs/regime_engine_v2_spec.md §3.5 (lines 3479–3524). The
    ``effective_rank_stability_threshold`` (0.05) encodes the spec-text
    "21d std < 5% of mean" inside the diversified_normal rule (line 3484).
    """

    # diversified_normal — v2 §3.5 line 3483
    diversified_normal_percentile_lo: float = Field(ge=0.0, le=1.0)
    diversified_normal_percentile_hi: float = Field(ge=0.0, le=1.0)
    # diversified_normal — v2 §3.5 line 3484
    effective_rank_stability_threshold: float = Field(gt=0.0, le=1.0)
    # stock_picker_dispersion — v2 §3.5 lines 3492–3494
    stock_picker_percentile_max: float = Field(ge=0.0, le=1.0)
    stock_picker_dispersion_percentile_min: float = Field(ge=0.0, le=1.0)
    # correlation_concentration — v2 §3.5 lines 3506–3508
    concentration_corr_percentile_min: float = Field(ge=0.0, le=1.0)
    concentration_largest_eig_percentile_min: float = Field(ge=0.0, le=1.0)
    concentration_effective_rank_percentile_max: float = Field(ge=0.0, le=1.0)
    # absorption_ratio_top3 > threshold → top-3 eigenvalue dominance.
    concentration_absorption_ratio_min: float = Field(default=0.90, ge=0.0, le=1.0)
    # correlation_to_one — v2 §3.5 lines 3513–3515
    corr_to_one_corr_percentile_min: float = Field(ge=0.0, le=1.0)
    corr_to_one_realized_vol_percentile_min: float = Field(ge=0.0, le=1.0)
    corr_to_one_drawdown_max: float
    # systemic_stress — v2 §3.5 lines 3520–3523
    systemic_stress_vix_percentile_min: float = Field(ge=0.0, le=1.0)
    # When True, diversified_normal fires on correlation in the inner band
    # (0.30-0.60) without requiring rank stability. Rank instability in
    # moderate-correlation regimes means factor rotation, not fragility.
    diversified_normal_relaxed_inner_band: bool = Field(default=True)
    diversified_normal_inner_band_lo: float = Field(default=0.30, ge=0.0, le=1.0)
    diversified_normal_inner_band_hi: float = Field(default=0.60, ge=0.0, le=1.0)


class NetworkFragilityConfig(StrictBaseModel):
    """Network fragility axis configuration (v2 spec §3)."""

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

    # V2 §3.4–§3.5 rule engine thresholds.
    rules: NetworkFragilityRulesConfig


