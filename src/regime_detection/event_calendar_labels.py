from __future__ import annotations

from typing import Literal

EventCalendarLabel = Literal[
    "geopolitical_event",
    "election_window",
    "fed_week",
    "global_rate_decision",
    "budget_week",
    "cpi_week",
    "nfp_week",
    "expiry_week",
    "earnings_season",
    "normal_calendar",
    "unknown",
]

EVENT_CALENDAR_LABELS: tuple[EventCalendarLabel, ...] = (
    "geopolitical_event",
    "election_window",
    "fed_week",
    "global_rate_decision",
    "budget_week",
    "cpi_week",
    "nfp_week",
    "expiry_week",
    "earnings_season",
    "normal_calendar",
    "unknown",
)

EVENT_CALENDAR_LABEL_SET: frozenset[EventCalendarLabel] = frozenset(
    EVENT_CALENDAR_LABELS
)
