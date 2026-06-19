"""V1GEC-017 — RegimeEngine.classify_series(...) V1.1 helper.

Flattens a [start_date, end_date] window into a per-session DataFrame of active
axis labels. Thin wrapper over classify_window — must agree with the pointwise
classify() on every emitted session.
"""

from __future__ import annotations

import json
import logging
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


def test_classify_logs_completed_run(
    raw_market_data: pd.DataFrame,
    event_calendar_df: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="regime_detection.engine"):
        RegimeEngine().classify(
            as_of_date=date(2024, 1, 10),
            market_data=raw_market_data,
            event_calendar=event_calendar_df,
            config=_v1_config(),
        )

    payload = json.loads(caplog.records[-1].message)
    assert payload["event"] == "classify_request_completed"
    assert payload["end_date"] == "2024-01-10"
    assert payload["outputs"] == 1
    assert payload["config_version"] == "core3-v1.0.0"


def test_classify_series_agrees_with_pointwise_classify_multi_day(
    raw_market_data: pd.DataFrame, event_calendar_df: pd.DataFrame
) -> None:
    # F-025: the byte-identity contract is load-bearing for a MULTI-day window
    # (lookback_days=N>1), where classify_series builds the feature store / model
    # windows once over a window ending at _END while pointwise classify(d) anchors
    # at each interior day d. The prior agreement test used a single-session window
    # (both sides lookback_days=1) and passed trivially. Assert every emitted session
    # equals its pointwise classify.
    series_df = RegimeEngine().classify_series(
        start_date=_START,
        end_date=_END,
        market_data=raw_market_data,
        event_calendar=event_calendar_df,
        config=_v1_config(),
    )
    assert len(series_df) >= 5  # genuinely multi-session

    for _, row in series_df.iterrows():
        as_of = row["as_of_date"]
        point = RegimeEngine().classify(
            as_of_date=as_of,
            market_data=raw_market_data,
            event_calendar=event_calendar_df,
            config=_v1_config(),
        )
        assert row["trend_direction"] == point.trend_direction.active_label, as_of
        assert row["trend_character"] == point.trend_character.active_label, as_of
        assert row["volatility_state"] == point.volatility_state.active_label, as_of
        assert row["breadth_state"] == point.breadth_state.active_label, as_of
        assert row["transition_risk"] == point.transition_risk.state, as_of


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
