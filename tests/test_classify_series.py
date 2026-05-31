"""V1GEC-017 — RegimeEngine.classify_series(...) V1.1 helper.

Flattens a [start_date, end_date] window into a per-session DataFrame of active
axis labels. Thin wrapper over classify_window — must agree with the pointwise
classify() on every emitted session.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from regime_detection.calendar import nyse_sessions_between
from regime_detection.config import load_regime_config
from regime_detection.engine import RegimeEngine

_REPO_ROOT = Path(__file__).resolve().parents[1]
_V1_CONFIG_PATH = (
    _REPO_ROOT / "src" / "regime_detection" / "configs" / "core3-v1.0.0.yaml"
)


def _v1_config():
    return load_regime_config(_V1_CONFIG_PATH)


_START = date(2024, 1, 3)
_END = date(2024, 1, 12)
_AXIS_COLUMNS = (
    "trend_direction",
    "trend_character",
    "volatility_state",
    "breadth_state",
    "transition_risk",
)


def test_classify_series_emits_one_row_per_nyse_session(
    raw_market_data: pd.DataFrame, event_calendar_df: pd.DataFrame
) -> None:
    df = RegimeEngine().classify_series(
        start_date=_START,
        end_date=_END,
        market_data=raw_market_data,
        event_calendar=event_calendar_df,
        config=_v1_config(),
    )
    expected_sessions = list(nyse_sessions_between(_START, _END))
    assert isinstance(df, pd.DataFrame)
    # One row per inclusive NYSE session, ascending by as_of_date.
    assert list(df["as_of_date"]) == expected_sessions
    for col in (*_AXIS_COLUMNS, "engine_version", "config_version", "market"):
        assert col in df.columns
    for col in _AXIS_COLUMNS:
        assert df[col].notna().all(), f"{col} has nulls"


def test_classify_series_agrees_with_pointwise_classify(
    raw_market_data: pd.DataFrame, event_calendar_df: pd.DataFrame
) -> None:
    as_of = date(2024, 1, 10)
    series_df = RegimeEngine().classify_series(
        start_date=as_of,
        end_date=as_of,
        market_data=raw_market_data,
        event_calendar=event_calendar_df,
        config=_v1_config(),
    )
    point = RegimeEngine().classify(
        as_of_date=as_of,
        market_data=raw_market_data,
        event_calendar=event_calendar_df,
        config=_v1_config(),
    )
    assert len(series_df) == 1
    row = series_df.iloc[-1]
    assert row["as_of_date"] == as_of
    assert row["trend_direction"] == point.trend_direction.active_label
    assert row["trend_character"] == point.trend_character.active_label
    assert row["volatility_state"] == point.volatility_state.active_label
    assert row["breadth_state"] == point.breadth_state.active_label
    assert row["transition_risk"] == point.transition_risk.state


def test_classify_series_rejects_reversed_window(
    raw_market_data: pd.DataFrame, event_calendar_df: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="start_date"):
        RegimeEngine().classify_series(
            start_date=_END,
            end_date=_START,
            market_data=raw_market_data,
            event_calendar=event_calendar_df,
            config=_v1_config(),
        )
