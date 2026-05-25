from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

import regime_detection.calendar as calendar
from regime_detection.calendar import (
    as_date,
    is_nyse_trading_day,
    nyse_neighbors,
    nyse_sessions_between,
    require_nyse_trading_day,
)


def test_nyse_trading_day_distinguishes_session_weekend_and_holiday() -> None:
    assert is_nyse_trading_day(date(2024, 1, 2)) is True
    assert is_nyse_trading_day(date(2024, 1, 6)) is False
    assert is_nyse_trading_day(date(2024, 1, 1)) is False


def test_nyse_sessions_between_cache_is_bounded() -> None:
    assert nyse_sessions_between.cache_info().maxsize == 1024


def test_nyse_neighbors_for_monday_new_year_holiday() -> None:
    neighbors = nyse_neighbors(date(2024, 1, 1))

    assert neighbors.prev_trading_day == date(2023, 12, 29)
    assert neighbors.next_trading_day == date(2024, 1, 2)


def test_require_nyse_trading_day_reports_nearest_neighbors_for_holiday() -> None:
    with pytest.raises(ValueError) as exc:
        require_nyse_trading_day(date(2024, 1, 1))

    message = str(exc.value)
    assert "2024-01-01" in message
    assert "Nearest prior trading day: 2023-12-29" in message
    assert "Nearest next trading day: 2024-01-02" in message


def test_as_date_rejects_tz_naive_datetime() -> None:
    with pytest.raises(TypeError, match="tz-naive datetime is ambiguous"):
        as_date(datetime(2024, 1, 2, 12, 0, 0))


def test_as_date_rejects_tz_naive_pandas_timestamp() -> None:
    with pytest.raises(TypeError, match="tz-naive pandas Timestamp is ambiguous"):
        as_date(pd.Timestamp("2024-01-02 12:00:00"))


def test_as_date_rejects_non_date_like_value() -> None:
    with pytest.raises(TypeError, match="Expected date-like value, got str"):
        as_date("2024-01-02")


def test_as_date_converts_tz_aware_datetime_to_new_york_date() -> None:
    value = datetime(2024, 1, 2, 2, 30, tzinfo=timezone.utc)
    assert as_date(value) == date(2024, 1, 1)


def test_nyse_neighbors_raises_when_schedule_window_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(calendar, "nyse_sessions_between", lambda start, end: ())

    with pytest.raises(RuntimeError, match="empty schedule window"):
        nyse_neighbors(date(2024, 1, 1))


def test_nyse_neighbors_raises_when_neighbor_session_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        calendar,
        "nyse_sessions_between",
        lambda start, end: (date(2024, 1, 1),),
    )

    with pytest.raises(RuntimeError, match="Unable to find NYSE neighbor sessions"):
        nyse_neighbors(date(2024, 1, 1))
