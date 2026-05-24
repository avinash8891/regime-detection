from __future__ import annotations

import datetime as dt

import pytest

from regime_data_fetch.event_calendar_global_rate_parsers import (
    global_rate_event,
    parse_global_rate_decision_events,
)
from regime_data_fetch.event_calendar_models import EventCalendarFetchError
from regime_data_fetch.event_calendar_windows import (
    expand_nyse_window_for_scheduled_event,
    us_general_election_date,
)


def test_us_general_election_date_returns_first_tuesday_after_first_monday() -> None:
    assert us_general_election_date(2024) == dt.date(2024, 11, 5)
    assert us_general_election_date(2026) == dt.date(2026, 11, 3)
    assert us_general_election_date(2032) == dt.date(2032, 11, 2)


def test_expand_nyse_window_for_scheduled_event_uses_trading_days() -> None:
    assert expand_nyse_window_for_scheduled_event(
        anchor_date=dt.date(2026, 11, 3),
        lookback_trading_days=2,
        lookahead_trading_days=2,
    ) == [
        dt.date(2026, 10, 30),
        dt.date(2026, 11, 2),
        dt.date(2026, 11, 3),
        dt.date(2026, 11, 4),
        dt.date(2026, 11, 5),
    ]


def test_expand_nyse_window_for_scheduled_event_rejects_non_trading_anchor() -> None:
    with pytest.raises(
        EventCalendarFetchError,
        match="Scheduled event date 2026-01-01 is not an NYSE trading day",
    ):
        expand_nyse_window_for_scheduled_event(
            anchor_date=dt.date(2026, 1, 1),
            lookback_trading_days=1,
            lookahead_trading_days=1,
        )


def test_global_rate_parser_wraps_events_as_global_midnight_et() -> None:
    events = parse_global_rate_decision_events(
        source_key="boe",
        text="2026 confirmed dates Thursday 7 May Monetary Policy Committee",
    )

    assert len(events) == 1
    event = events[0]
    assert event.date == dt.date(2026, 5, 7)
    assert event.release_timestamp_et == dt.datetime(
        2026,
        5,
        7,
        0,
        0,
        tzinfo=dt.timezone(dt.timedelta(hours=-5)),
    )
    assert event.market == "GLOBAL"
    assert event.type == "BOE_decision"
    assert event.importance == "high"
    assert event.source == "bankofengland.co.uk:mpc-dates"


def test_global_rate_parser_converts_unsupported_source_to_fetch_error() -> None:
    with pytest.raises(
        EventCalendarFetchError,
        match="Unsupported global rate calendar source: rba",
    ):
        parse_global_rate_decision_events(source_key="rba", text="")


def test_global_rate_event_preserves_source_specific_decision_type() -> None:
    event = global_rate_event(
        dt.date(2026, 6, 11),
        "ECB_decision",
        "ecb.europa.eu:governing-council-calendar",
    )

    assert event.date == dt.date(2026, 6, 11)
    assert event.release_timestamp_et.isoformat() == "2026-06-11T00:00:00-05:00"
    assert event.market == "GLOBAL"
    assert event.type == "ECB_decision"
    assert event.importance == "high"
    assert event.source == "ecb.europa.eu:governing-council-calendar"
