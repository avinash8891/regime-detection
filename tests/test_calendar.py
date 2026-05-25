from __future__ import annotations

from datetime import date

import pytest

from regime_detection.calendar import (
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
