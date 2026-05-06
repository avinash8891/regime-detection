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


def _load_symbol(symbol: str) -> pd.DataFrame:
    csv_path = Path(__file__).resolve().parent / "fixtures" / "raw" / f"{symbol}.csv"
    assert csv_path.exists(), f"Missing fixture CSV for {symbol}: {csv_path}"
    return pd.read_csv(csv_path)


def _market_df_for_asof(as_of: date) -> pd.DataFrame:
    spy = _load_symbol("SPY")
    rsp = _load_symbol("RSP")
    vixy = _load_symbol("VIXY")
    df = pd.concat([spy, rsp, vixy], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] <= as_of].copy()

    # Engine contract requires these columns. Some feeds include extras; drop them.
    keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
    return df[keep]


def test_engine_version_matches_spec_prefix() -> None:
    assert engine_version().startswith("regime-engine-v")


def test_version_coupling_pyproject_matches_engine_version() -> None:
    # Spec lock: package version and emitted engine_version must stay aligned.
    import tomllib

    pyproject = Path("pyproject.toml").read_bytes()
    version = tomllib.loads(pyproject.decode("utf-8"))["project"]["version"]
    assert engine_version() == f"regime-engine-v{version}"


def test_classify_requires_nyse_trading_day() -> None:
    as_of = date(2017, 1, 1)  # Sunday
    assert not is_nyse_trading_day(as_of)
    engine = RegimeEngine()
    df = _market_df_for_asof(date(2017, 1, 3))
    with pytest.raises(ValueError) as excinfo:
        engine.classify(as_of_date=as_of, market_data=df)
    msg = str(excinfo.value)
    assert "Nearest prior trading day" in msg
    assert "Nearest next trading day" in msg


def test_market_data_contract_requires_spy() -> None:
    engine = RegimeEngine()
    as_of = date(2026, 5, 5)
    assert is_nyse_trading_day(as_of)
    df = _market_df_for_asof(as_of)
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
                    "trend_direction_deescalation_days": 3,
                    "trend_character_deescalation_days": 3,
                    "volatility_deescalation_days": 2,
                    "breadth_deescalation_days": 2,
                    "composite_deescalation_days": 3,
                    "event_calendar_days": 1,
                },
                "unknown_key": True,
            }
        )


def test_classify_emits_regime_output_shape() -> None:
    as_of = date(2026, 5, 5)
    assert is_nyse_trading_day(as_of)
    engine = RegimeEngine()
    df = _market_df_for_asof(as_of)
    out = engine.classify(as_of_date=as_of, market_data=df)
    assert out.engine_version == engine_version()
    assert out.config_version == engine.config.config_version
    assert out.as_of_date == as_of
    assert out.market == "SPY"


def test_classify_accepts_timestamp_as_of_date() -> None:
    engine = RegimeEngine()
    df = _market_df_for_asof(date(2026, 5, 5))
    # Common caller input: pandas Timestamp. Must be accepted and normalized.
    out = engine.classify(as_of_date=pd.Timestamp("2026-05-05"), market_data=df)
    assert out.as_of_date == date(2026, 5, 5)
