from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


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


class RegimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: str
    market: Literal["US"]
    trading_calendar: str
    breadth_mode: Literal["etf_proxy"]
    cap_weight_index: Literal["SPY"]
    equal_weight_proxy: Literal["RSP"]
    event_calendar: EventCalendarConfig
    expiry_rules: ExpiryRulesConfig
    earnings_seasons: list[EarningsSeasonConfig]
    data_quality: DataQualityConfig
    hysteresis: HysteresisConfig


def load_regime_config(path: str | Path) -> RegimeConfig:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level")
    return RegimeConfig.model_validate(data)


def load_default_regime_config() -> RegimeConfig:
    """
    Load the packaged default config shipped with the library.

    NOTE: We load the resource content directly (instead of returning a filesystem Path)
    so this works even when the package is distributed as a zip/egg.
    """
    pkg_file = importlib.resources.files("regime_detection").joinpath("configs/core3-v1.0.0.yaml")
    text = pkg_file.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Default config must contain a YAML mapping at the top level")
    return RegimeConfig.model_validate(data)
