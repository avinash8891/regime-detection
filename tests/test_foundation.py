from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from regime_detection.calendar import is_nyse_trading_day
from regime_detection.config import RegimeConfig
from regime_detection.engine import RegimeEngine
from regime_detection.versioning import engine_version


def test_engine_version_matches_spec_prefix() -> None:
    assert engine_version().startswith("regime-engine-v")


def test_version_coupling_pyproject_matches_engine_version() -> None:
    # Spec lock: package version and emitted engine_version must stay aligned.
    import tomllib

    repo_root = Path(__file__).resolve().parents[1]
    pyproject = (repo_root / "pyproject.toml").read_bytes()
    version = tomllib.loads(pyproject.decode("utf-8"))["project"]["version"]
    assert engine_version() == f"regime-engine-v{version}"


def test_classify_requires_nyse_trading_day(market_df_for_asof) -> None:
    as_of = date(2017, 1, 1)  # Sunday
    assert not is_nyse_trading_day(as_of)
    engine = RegimeEngine()
    df = market_df_for_asof(date(2017, 1, 3))
    with pytest.raises(ValueError) as excinfo:
        engine.classify(as_of_date=as_of, market_data=df)
    msg = str(excinfo.value)
    assert "Nearest prior trading day" in msg
    assert "Nearest next trading day" in msg


def test_market_data_contract_requires_spy(market_df_for_asof) -> None:
    engine = RegimeEngine()
    as_of = date(2026, 5, 5)
    assert is_nyse_trading_day(as_of)
    df = market_df_for_asof(as_of)
    df = df[df["symbol"] != "SPY"].copy()
    with pytest.raises(ValueError) as excinfo:
        engine.classify(as_of_date=as_of, market_data=df)
    assert "must contain SPY" in str(excinfo.value)


def test_regime_config_forbids_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        RegimeConfig.model_validate(
            {
                "config_version": "core3-v1.0.0",
                "trading_calendar": "NYSE",
                "hysteresis": {
                    "trend_direction_escalation_days": 1,
                    "trend_direction_deescalation_days": 3,
                    "trend_character_escalation_days": 1,
                    "trend_character_deescalation_days": 3,
                    "volatility_escalation_days": 1,
                    "volatility_deescalation_days": 2,
                    "breadth_escalation_days": 1,
                    "breadth_deescalation_days": 2,
                    "composite_deescalation_days": 3,
                },
                "unknown_key": True,
            }
        )


def test_default_config_is_packaged_and_loadable() -> None:
    # Guard against config drift: default config must be loadable from packaged resources.
    from regime_detection.config import load_default_regime_config

    cfg = load_default_regime_config()
    # Default dispatch is keyed on package __version__; package is 2.x → v2 yaml.
    assert cfg.config_version == "core3-v2.0.0"
    assert cfg.hysteresis.trend_direction_escalation_days == 1
    assert cfg.hysteresis.trend_character_escalation_days == 1
    assert cfg.hysteresis.volatility_escalation_days == 1
    assert cfg.hysteresis.breadth_escalation_days == 1


def test_classify_emits_regime_output_shape(market_df_for_asof) -> None:
    as_of = date(2026, 5, 5)
    assert is_nyse_trading_day(as_of)
    engine = RegimeEngine()
    df = market_df_for_asof(as_of)
    out = engine.classify(as_of_date=as_of, market_data=df)
    assert out.engine_version == engine_version()
    assert out.config_version == engine.config.config_version
    assert out.as_of_date == as_of
    assert out.market == "SPY"


def test_classify_accepts_timestamp_as_of_date(market_df_for_asof) -> None:
    engine = RegimeEngine()
    df = market_df_for_asof(date(2026, 5, 5))
    # Common caller input: pandas Timestamp. Must be accepted and normalized.
    out = engine.classify(as_of_date=pd.Timestamp("2026-05-05", tz="America/New_York"), market_data=df)
    assert out.as_of_date == date(2026, 5, 5)


def test_classify_accepts_market_data_with_string_dates(market_df_for_asof) -> None:
    engine = RegimeEngine()
    df = market_df_for_asof(date(2026, 5, 5)).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    out = engine.classify(as_of_date=date(2026, 5, 5), market_data=df)

    assert out.as_of_date == date(2026, 5, 5)


def test_engine_rejects_path_based_event_calendar_input(market_df_for_asof) -> None:
    engine = RegimeEngine()
    df = market_df_for_asof(date(2023, 12, 14))
    event_path = Path(__file__).resolve().parent / "fixtures" / "events" / "us_events.yaml"

    with pytest.raises(TypeError, match="event_calendar must be a pandas DataFrame"):
        engine.classify(
            as_of_date=date(2023, 12, 14),
            market_data=df,
            event_calendar=event_path,
        )
