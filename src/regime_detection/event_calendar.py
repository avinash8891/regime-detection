from __future__ import annotations

from functools import lru_cache
import logging
from datetime import date, timedelta
from typing import Literal

import pandas as pd

from regime_detection.calendar import nyse_calendar
from regime_detection.config import RegimeConfig
from regime_detection.loaders import load_event_calendar
from regime_detection.models import EventCalendarOutput


LOG = logging.getLogger(__name__)

EventCalendarLabel = Literal[
    "fed_week",
    "cpi_week",
    "nfp_week",
    "expiry_week",
    "earnings_season",
    "normal_calendar",
    "unknown",
]

_RISK_RANK: dict[EventCalendarLabel, int] = {
    "unknown": 0,
    "normal_calendar": 1,
    "earnings_season": 2,
    "expiry_week": 3,
    "nfp_week": 4,
    "cpi_week": 5,
    "fed_week": 6,
}
_PRECEDENCE: list[EventCalendarLabel] = [
    "fed_week",
    "cpi_week",
    "nfp_week",
    "expiry_week",
    "earnings_season",
    "normal_calendar",
    "unknown",
]
_TYPE_TO_LABEL = {"FOMC": "fed_week", "CPI": "cpi_week", "NFP": "nfp_week"}
_WINDOWS = {"fed_week": (-2, 2), "cpi_week": (-1, 1), "nfp_week": (-1, 1)}


def classify_event_calendar(
    *,
    as_of_date: date,
    event_calendar: pd.DataFrame | None,
    config: RegimeConfig,
) -> EventCalendarOutput:
    event_df = _normalized_events(event_calendar, market=config.event_calendar.market)
    raw_label, evidence = _raw_event_calendar_label(
        as_of_date=as_of_date,
        event_calendar=event_df,
        config=config,
    )
    return EventCalendarOutput(
        raw_label=raw_label,
        stable_label=raw_label,
        active_label=raw_label,
        evidence=evidence,
    )


def _normalized_events(event_calendar: pd.DataFrame | None, *, market: str) -> pd.DataFrame:
    if event_calendar is None:
        return pd.DataFrame(columns=["date", "market", "type", "importance", "publication_date"])
    return load_event_calendar(event_calendar, market=market)


def _raw_event_calendar_label(
    *,
    as_of_date: date,
    event_calendar: pd.DataFrame,
    config: RegimeConfig,
) -> tuple[EventCalendarLabel, dict[str, object]]:
    matches: list[EventCalendarLabel] = []

    if not event_calendar.empty:
        if any((row["date"] - as_of_date).days > 90 for _, row in event_calendar.iterrows()):
            LOG.warning("Event calendar contains row more than 90 calendar days after as_of_date=%s", as_of_date)

    for _, row in event_calendar.iterrows():
        event_type = str(row["type"])
        label = _TYPE_TO_LABEL.get(event_type)
        if label is None:
            continue
        publication_date = row["publication_date"]
        if publication_date > as_of_date:
            continue
        if _is_within_trading_window(
            as_of_date=as_of_date,
            event_date=row["date"],
            start=_WINDOWS[label][0],
            end=_WINDOWS[label][1],
        ):
            matches.append(label)

    if _is_expiry_week(as_of_date=as_of_date, config=config):
        matches.append("expiry_week")
    if _is_earnings_season(as_of_date=as_of_date, config=config):
        matches.append("earnings_season")

    ordered = [label for label in _PRECEDENCE if label in set(matches)]
    if ordered:
        selected = ordered[0]
    else:
        selected = "normal_calendar"

    return selected, {
        "all_matching_events": ordered,
        "selected_via_precedence": selected,
    }


def _is_within_trading_window(*, as_of_date: date, event_date: date, start: int, end: int) -> bool:
    start_date = min(as_of_date, event_date) - timedelta(days=20)
    end_date = max(as_of_date, event_date) + timedelta(days=20)
    sessions = list(_sessions_between(start_date, end_date))
    if as_of_date not in sessions or event_date not in sessions:
        return False
    delta = sessions.index(as_of_date) - sessions.index(event_date)
    return start <= delta <= end


@lru_cache(maxsize=None)
def _sessions_between(start_date: date, end_date: date) -> tuple[date, ...]:
    return tuple(nyse_calendar().schedule(start_date=start_date, end_date=end_date).index.date)


@lru_cache(maxsize=None)
def _month_expiry_date(year: int, month: int) -> date:
    first = date(year, month, 1)
    offset = (4 - first.weekday()) % 7  # Friday
    first_friday = first + timedelta(days=offset)
    candidate = first_friday + timedelta(days=14)
    sessions = list(_sessions_between(candidate - timedelta(days=7), candidate))
    sessions_before = [session for session in sessions if session <= candidate]
    if not sessions_before:
        raise RuntimeError(f"No NYSE sessions available on or before candidate expiry {candidate}")
    return sessions_before[-1]


def _is_expiry_week(*, as_of_date: date, config: RegimeConfig) -> bool:
    monthly = config.expiry_rules.monthly_options
    expiry_date = _month_expiry_date(as_of_date.year, as_of_date.month)
    start, end = monthly.window_trading_days
    return _is_within_trading_window(
        as_of_date=as_of_date,
        event_date=expiry_date,
        start=start,
        end=end,
    )


def _second_weekday_of_month(*, year: int, month: int, weekday: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    first_match = first + timedelta(days=offset)
    return first_match + timedelta(days=7)


def _is_earnings_season(*, as_of_date: date, config: RegimeConfig) -> bool:
    month_lookup = {
        "second_monday_of_january": 1,
        "second_monday_of_april": 4,
        "second_monday_of_july": 7,
        "second_monday_of_october": 10,
    }
    for season in config.earnings_seasons:
        start_month = month_lookup[season.start_rule]
        start = _second_weekday_of_month(year=as_of_date.year, month=start_month, weekday=0)
        end = start + timedelta(days=season.end_offset_days)
        if start <= as_of_date <= end:
            return True
    return False
