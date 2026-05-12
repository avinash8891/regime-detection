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
