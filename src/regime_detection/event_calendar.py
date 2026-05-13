from __future__ import annotations

import calendar
from bisect import bisect_left, bisect_right
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
    """Point-classifier wrapper around :func:`compute_event_calendar_outputs`.

    Kept as a public symbol for callers that want a single-session label
    without constructing a ``MarketContext``. Delegates to the unified
    bulk algorithm so the two surfaces cannot drift.
    """
    normalized = _normalized_events(event_calendar, market=config.event_calendar.market)
    if not normalized.empty:
        if (normalized["date"] - as_of_date > timedelta(days=90)).any():
            LOG.warning(
                "Event calendar contains row more than 90 calendar days after as_of_date=%s",
                as_of_date,
            )
    outputs = compute_event_calendar_outputs(
        sessions=(as_of_date,),
        normalized_event_calendar=normalized,
        config=config,
    )
    return outputs[as_of_date]


def compute_event_calendar_outputs(
    *,
    sessions: tuple[date, ...] | list[date],
    normalized_event_calendar: pd.DataFrame | None,
    config: RegimeConfig,
) -> dict[date, EventCalendarOutput]:
    """Compute event-calendar labels for every session in ``sessions``.

    Pure compute, shared by:
      - :func:`classify_event_calendar` (point API)
      - :func:`regime_detection.axis_series.build_event_calendar_series`
        (bulk wrapper that pulls ``sessions`` + ``normalized_event_calendar``
        off a :class:`MarketContext`).

    Algorithm: build a global NYSE session list spanning the union of
    requested ``sessions`` and any event-date / publication-date in the
    calendar, then for each event/expiry/earnings rule paint a bit-mask
    across the relevant trading window. Resolve each session's mask via
    the :data:`_PRECEDENCE` order. O(events + |sessions|) instead of the
    O(|sessions| × events) iterrows-per-session pattern the old point
    classifier used.
    """
    sessions_tuple = tuple(sessions)
    if not sessions_tuple:
        return {}
    session_list = list(sessions_tuple)
    session_pos = {day: idx for idx, day in enumerate(sessions_tuple)}
    label_bits = {label: 1 << idx for idx, label in enumerate(_PRECEDENCE)}
    match_masks = [0] * len(sessions_tuple)

    event_rows = (
        []
        if normalized_event_calendar is None or normalized_event_calendar.empty
        else list(normalized_event_calendar.itertuples(index=False))
    )
    first_session = sessions_tuple[0]
    last_session = sessions_tuple[-1]
    global_start = first_session.replace(day=1) - timedelta(days=31)
    global_end = last_session.replace(
        day=calendar.monthrange(last_session.year, last_session.month)[1]
    ) + timedelta(days=31)
    if event_rows:
        min_event_day = min(
            min(row.publication_date, row.date) for row in event_rows
        )
        max_event_day = max(
            max(row.publication_date, row.date) for row in event_rows
        )
        global_start = min(global_start, min_event_day) - timedelta(days=20)
        global_end = max(global_end, max_event_day) + timedelta(days=20)
    global_sessions = _sessions_between(global_start, global_end)
    global_session_list = list(global_sessions)
    global_pos = {day: idx for idx, day in enumerate(global_session_list)}

    for row in event_rows:
        label = _TYPE_TO_LABEL.get(str(row.type))
        if label is None:
            continue
        event_date = row.date
        event_idx = global_pos.get(event_date)
        if event_idx is None:
            continue
        publication_date = row.publication_date
        start_offset, end_offset = _WINDOWS[label]
        start_idx = max(0, event_idx + start_offset)
        end_idx = min(len(global_sessions) - 1, event_idx + end_offset)
        bit = label_bits[label]
        for day in global_session_list[start_idx : end_idx + 1]:
            if day < publication_date:
                continue
            context_idx = session_pos.get(day)
            if context_idx is not None:
                match_masks[context_idx] |= bit

    expiry_start, expiry_end = config.expiry_rules.monthly_options.window_trading_days
    expiry_bit = label_bits["expiry_week"]
    for year, month in sorted({(day.year, day.month) for day in session_list}):
        third_friday = _third_friday_of_month(year=year, month=month)
        expiry_idx = bisect_right(global_session_list, third_friday) - 1
        if expiry_idx < 0:
            continue
        if global_session_list[expiry_idx].month != month:
            continue
        start_idx = max(0, expiry_idx + expiry_start)
        end_idx = min(len(global_session_list) - 1, expiry_idx + expiry_end)
        for day in global_session_list[start_idx : end_idx + 1]:
            context_idx = session_pos.get(day)
            if context_idx is not None:
                match_masks[context_idx] |= expiry_bit

    month_lookup = {
        "second_monday_of_january": 1,
        "second_monday_of_april": 4,
        "second_monday_of_july": 7,
        "second_monday_of_october": 10,
    }
    earnings_bit = label_bits["earnings_season"]
    for year in sorted({day.year for day in sessions_tuple}):
        for season in config.earnings_seasons:
            start = _second_weekday_of_month(
                year=year,
                month=month_lookup[season.start_rule],
                weekday=0,
            )
            end = start + timedelta(days=season.end_offset_days)
            start_idx = bisect_left(session_list, start)
            end_idx = bisect_right(session_list, end)
            for idx in range(start_idx, end_idx):
                match_masks[idx] |= earnings_bit

    outputs: dict[date, EventCalendarOutput] = {}
    for idx, day in enumerate(sessions_tuple):
        mask = match_masks[idx]
        ordered = [label for label in _PRECEDENCE if mask & label_bits[label]]
        selected = ordered[0] if ordered else "normal_calendar"
        outputs[day] = EventCalendarOutput(
            raw_label=selected,
            stable_label=selected,
            active_label=selected,
            evidence={
                "all_matching_events": ordered,
                "selected_via_precedence": selected,
            },
        )
    return outputs


def _normalized_events(event_calendar: pd.DataFrame | None, *, market: str) -> pd.DataFrame:
    if event_calendar is None:
        return pd.DataFrame(columns=["date", "market", "type", "importance", "publication_date"])
    return load_event_calendar(event_calendar, market=market)


def _third_friday_of_month(*, year: int, month: int) -> date:
    month_weeks = calendar.monthcalendar(year, month)
    friday_days = [week[calendar.FRIDAY] for week in month_weeks if week[calendar.FRIDAY] != 0]
    return date(year, month, friday_days[2])


@lru_cache(maxsize=None)
def _sessions_between(start_date: date, end_date: date) -> tuple[date, ...]:
    return tuple(nyse_calendar().schedule(start_date=start_date, end_date=end_date).index.date)


def _second_weekday_of_month(*, year: int, month: int, weekday: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    first_match = first + timedelta(days=offset)
    return first_match + timedelta(days=7)
