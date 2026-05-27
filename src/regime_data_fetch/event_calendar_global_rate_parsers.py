from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false

import datetime as dt

from regime_data_fetch import event_calendar_global_rates as _global_rates
from regime_data_fetch.event_calendar_models import (
    EventCalendarFetchError,
    ScheduledEvent,
    US_EASTERN,
)


def parse_global_rate_decision_events(
    *, source_key: str, text: str
) -> list[ScheduledEvent]:
    try:
        decisions = _global_rates.parse_global_rate_decision_events(
            source_key=source_key, text=text
        )
    except _global_rates.UnsupportedGlobalRateSource as exc:
        raise EventCalendarFetchError(str(exc)) from None
    return [
        global_rate_event(decision.date, decision.event_type, decision.source)
        for decision in decisions
    ]


def parse_ecb_decision_events(text: str) -> list[ScheduledEvent]:
    return [
        global_rate_event(decision.date, decision.event_type, decision.source)
        for decision in _global_rates.parse_ecb_decision_events(text)
    ]


def parse_boe_decision_events(text: str) -> list[ScheduledEvent]:
    return [
        global_rate_event(decision.date, decision.event_type, decision.source)
        for decision in _global_rates.parse_boe_decision_events(text)
    ]


def parse_boj_decision_events(text: str) -> list[ScheduledEvent]:
    return [
        global_rate_event(decision.date, decision.event_type, decision.source)
        for decision in _global_rates.parse_boj_decision_events(text)
    ]


def global_rate_event(
    event_date: dt.date, event_type: str, source: str
) -> ScheduledEvent:
    return ScheduledEvent(
        date=event_date,
        release_timestamp_et=midnight_et(event_date),
        market="GLOBAL",
        type=event_type,
        importance="high",
        source=source,
    )


def global_rate_source_name(source_key: str) -> str:
    return _global_rates.global_rate_source_name(source_key)


def midnight_et(value: dt.date) -> dt.datetime:
    return dt.datetime(value.year, value.month, value.day, 0, 0, tzinfo=US_EASTERN)
