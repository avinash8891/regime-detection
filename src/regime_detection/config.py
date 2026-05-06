from __future__ import annotations

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
    """
    Default config resolution:
    1. If running from a repo checkout with a top-level `configs/core3-v1.0.0.yaml`, prefer it.
       This is the human-edited config in this repository.
    2. Otherwise fall back to the packaged config shipped with the library.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    repo_cfg = repo_root / "configs" / "core3-v1.0.0.yaml"
    if repo_cfg.exists():
        return repo_cfg
    return here.parent / "configs" / "core3-v1.0.0.yaml"
