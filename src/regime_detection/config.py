from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any

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


class RegimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: str
    trading_calendar: str
    hysteresis: HysteresisConfig


def load_regime_config(path: str | Path) -> RegimeConfig:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level")
    return RegimeConfig.model_validate(data)


def default_config_path() -> Path:
    # Prefer an embedded package resource when installed; fall back to repo layout.
    try:
        pkg_file = importlib.resources.files("regime_detection").joinpath(
            "configs/core3-v1.0.0.yaml"
        )
        # as_file() materializes to a real filesystem Path if needed.
        with importlib.resources.as_file(pkg_file) as p:
            if p.exists():
                return p
    except Exception:
        pass

    return Path(__file__).resolve().parents[2] / "configs" / "core3-v1.0.0.yaml"
