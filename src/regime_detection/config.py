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


class DataQualityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Maximum allowed age (calendar days) of the newest row in each required series.
    max_freshness_days: int = Field(ge=0)

    # Minimum fraction of non-null values required in the lookback window for an axis to be "ok".
    min_completeness: float = Field(ge=0.0, le=1.0)


class EventCalendarConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market: str


class RegimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: str
    market: str
    trading_calendar: str
    breadth_mode: str
    cap_weight_index: str
    equal_weight_proxy: str
    event_calendar: EventCalendarConfig
    data_quality: DataQualityConfig
    hysteresis: HysteresisConfig


def load_regime_config(path: str | Path) -> RegimeConfig:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level")
    return RegimeConfig.model_validate(data)


def default_config_path() -> Path:
    """
    Default config resolution:
    V1 requires a repo-local config at `configs/core3-v1.0.0.yaml`.
    """
    here = Path(__file__).resolve()

    # Best-effort repo-root detection (works in a source checkout; gracefully falls back when installed).
    repo_root: Path | None = None
    for p in here.parents:
        if (p / "pyproject.toml").exists():
            repo_root = p
            break

    if repo_root is not None:
        repo_cfg = repo_root / "configs" / "core3-v1.0.0.yaml"
        if repo_cfg.exists():
            return repo_cfg

    raise FileNotFoundError(
        "Default config not found. Expected repo-local configs/core3-v1.0.0.yaml. "
        "Pass config_path explicitly if running outside the repository."
    )
