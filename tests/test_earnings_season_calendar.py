from __future__ import annotations

import datetime as dt

from regime_data_fetch.earnings_season_calendar import (
    build_earnings_season_windows,
    compute_earnings_season_window,
    is_in_earnings_season,
)


def test_compute_earnings_season_window_uses_second_monday_anchor() -> None:
    start_date, end_date = compute_earnings_season_window(
        year=2026, quarter_start_month=1
    )
    assert start_date == dt.date(2026, 1, 12)
    assert end_date == dt.date(2026, 2, 16)


def test_compute_earnings_season_window_for_april_quarter() -> None:
    start_date, end_date = compute_earnings_season_window(
        year=2025, quarter_start_month=4
    )
    assert start_date == dt.date(2025, 4, 14)
    assert end_date == dt.date(2025, 5, 19)


def test_is_in_earnings_season_includes_start_and_end_dates() -> None:
    assert is_in_earnings_season(as_of_date=dt.date(2026, 1, 12)) is True
    assert is_in_earnings_season(as_of_date=dt.date(2026, 2, 16)) is True
    assert is_in_earnings_season(as_of_date=dt.date(2026, 2, 17)) is False


def test_build_earnings_season_windows_covers_2015_to_current_date() -> None:
    windows = build_earnings_season_windows(
        start_date=dt.date(2015, 1, 1),
        end_date=dt.date(2026, 5, 7),
    )

    assert len(windows) == 46
    assert windows[0] == (dt.date(2015, 1, 12), dt.date(2015, 2, 16))
    assert windows[-1] == (dt.date(2026, 4, 13), dt.date(2026, 5, 18))
