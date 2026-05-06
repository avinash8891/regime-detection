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


class RegimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: str
    market: Literal["US"]
    trading_calendar: str
    breadth_mode: Literal["etf_proxy"]
    cap_weight_index: Literal["SPY"]
    equal_weight_proxy: Literal["RSP"]
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
    - Prefer the packaged config shipped with the library.
    - If the repo-level `configs/core3-v1.0.0.yaml` exists (development override), prefer it.
    """
    here = Path(__file__).resolve()

    # Development override: use repo-level config when present.
    repo_root: Path | None = None
    for p in here.parents:
        if (p / "pyproject.toml").exists():
            repo_root = p
            break
    if repo_root is not None:
        repo_cfg = repo_root / "configs" / "core3-v1.0.0.yaml"
        if repo_cfg.exists():
            return repo_cfg

    # Installed/default: packaged config.
    pkg_file = importlib.resources.files("regime_detection").joinpath("configs/core3-v1.0.0.yaml")
    with importlib.resources.as_file(pkg_file) as p:
        if p.exists():
            return p

    raise FileNotFoundError("Packaged default config not found: regime_detection/configs/core3-v1.0.0.yaml")
