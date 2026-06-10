from __future__ import annotations

import calendar
from bisect import bisect_left, bisect_right
from functools import lru_cache
import logging
from datetime import date, timedelta

import pandas as pd

from regime_detection.calendar import nyse_calendar
from regime_detection.config import RegimeConfig
from regime_detection.event_calendar_labels import (
    EVENT_CALENDAR_LABELS,
    EventCalendarLabel,
)
from regime_detection.loaders import load_event_calendar
from regime_detection.models import EventCalendarOutput

LOGGER = logging.getLogger(__name__)

# ADR 0014 R1 — event-calendar precedence (V1 + V2 unified ordering).
# V1 sub-sequence (`fed_week > cpi_week > nfp_week > expiry_week >
# earnings_season > normal_calendar > unknown`) preserved verbatim from
# ADR 0002 §63-64. V2 additions slotted by approval-gate / rarity /
# breadth: `geopolitical_event` first (overlay-promoted, never spurious),
# `election_window` second (widest window [-5,+10], rarest cadence),
# `global_rate_decision` between `fed_week` and `cpi_week` (foreign-CB
# meetings outrank CPI/NFP releases as cross-axis macro events), and
# `budget_week` between `global_rate_decision` and `cpi_week`. See
# docs/decisions/0014-event-calendar-v2-precedence-and-windows.md.
_PRECEDENCE: list[EventCalendarLabel] = list(EVENT_CALENDAR_LABELS)
_TYPE_TO_LABEL: dict[str, EventCalendarLabel] = {
    "FOMC": "fed_week",
    "CPI": "cpi_week",
    "NFP": "nfp_week",
}
_V2_TYPE_TO_LABEL: dict[str, EventCalendarLabel] = {
    "budget": "budget_week",
    "election": "election_window",
    "geopolitical_event": "geopolitical_event",
    "global_rate_decision": "global_rate_decision",
    "ECB_decision": "global_rate_decision",
    "BOE_decision": "global_rate_decision",
    "BOJ_decision": "global_rate_decision",
}
# Default trading-day windows around each event date. V1 entries
# (fed/cpi/nfp) inherited verbatim from V1 §7.2 lines 757-759. V2 entries
# pinned by ADR 0014 R2. Row-level ``window_days`` always takes precedence
# over these defaults.
#   - election_window [-5, +10] per spec §2D line 3366.
#   - geopolitical_event (0, 0) — manual no-window fallback only. Generated
#     approved candidates may carry GPR-derived row-level ``window_days``.
#   - budget_week (0, 0) — fires on the deterministic deadline day only;
#     budget runup behavior is not in spec §2D.
#   - global_rate_decision (0, 0) — known asymmetry vs fed_week (-2, +2);
#     ADR 0014 R2 records the rationale (foreign-CB events do not dominate
#     US session structure the way the Fed does). Open for revision if the
#     calibration §9.1 study finds a different empirically-correct window.
_WINDOWS = {
    "fed_week": (-2, 2),
    "cpi_week": (-1, 1),
    "nfp_week": (-1, 1),
    "budget_week": (0, 0),
    "election_window": (-5, 10),
    "geopolitical_event": (0, 0),
    "global_rate_decision": (0, 0),
}

# Forward-event logger warning: ADR 0002 §"Optional operator guard"
# (decisions/0002:57) — warn when an event row is more than this many
# calendar days after as_of_date. Warning-only; does not fail classification.
_FORWARD_EVENT_WARNING_DAYS = 90

# Global-session window padding for compute_event_window_just_passed.
# Covers the max _WINDOWS end_offset (election_window: +10 trading days)
# plus the largest realistic trailing_sessions caller-value, converted to
# calendar days with a 2x safety margin for weekend/holiday slack.
_SESSION_PADDING_DAYS = 40


def _window_offsets_for_row(
    *, label: EventCalendarLabel, row: object
) -> tuple[int, int]:
    row_window = getattr(row, "window_days", None)
    if isinstance(row_window, (list, tuple)) and len(row_window) == 2:
        return int(row_window[0]), int(row_window[1])
    return _WINDOWS[label]


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
        if (
            normalized["date"] - as_of_date
            > timedelta(days=_FORWARD_EVENT_WARNING_DAYS)
        ).any():
            LOGGER.warning(
                "Event calendar contains row more than %d calendar days after as_of_date=%s",
                _FORWARD_EVENT_WARNING_DAYS,
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

    if normalized_event_calendar is None:
        raise ValueError("event_calendar is required and must not be None")
    event_rows = (
        []
        if normalized_event_calendar.empty
        else list(normalized_event_calendar.itertuples(index=False))
    )
    first_session = sessions_tuple[0]
    last_session = sessions_tuple[-1]
    allow_v2_event_labels = config.config_version != "core3-v1.0.0"
    # Align to year boundaries so the lru_cached `_sessions_between` hits
    # whenever a tight-loop caller (e.g. the bulk-matches-point test) walks
    # sessions one at a time. Month-boundary alignment cycled the cache key
    # every month, causing N×O(global_days) per-call rebuilds. The year-
    # boundary range is a slight superset; bitmask painting only writes to
    # in-range sessions, so the wider window doesn't affect output.
    global_start = date(first_session.year, 1, 1)
    global_end = date(last_session.year, 12, 31)
    if event_rows:
        min_event_day = min(min(row.publication_date, row.date) for row in event_rows)
        max_event_day = max(max(row.publication_date, row.date) for row in event_rows)
        global_start = min(global_start, date(min_event_day.year, 1, 1))
        global_end = max(global_end, date(max_event_day.year, 12, 31))
    global_sessions = _sessions_between(global_start, global_end)
    global_session_list = list(global_sessions)
    global_pos = {day: idx for idx, day in enumerate(global_session_list)}

    for row in event_rows:
        label = _label_for_event_type(
            str(row.type), allow_v2_event_labels=allow_v2_event_labels
        )
        if label is None:
            continue
        if (
            label == "geopolitical_event"
            and getattr(row, "approved_label", None) != "geopolitical_event"
        ):
            continue
        event_date = row.date
        event_idx = global_pos.get(event_date)
        if event_idx is None:
            continue
        publication_date = row.publication_date
        start_offset, end_offset = _window_offsets_for_row(label=label, row=row)
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
        primary = ordered[0] if ordered else "normal_calendar"
        matching = tuple(ordered) if ordered else ("normal_calendar",)
        outputs[day] = EventCalendarOutput(
            primary_label=primary,
            matching_labels=matching,
            evidence={
                "selection_method": "precedence",
            },
        )
    return outputs


def _label_for_event_type(
    event_type: str, *, allow_v2_event_labels: bool
) -> EventCalendarLabel | None:
    label = _TYPE_TO_LABEL.get(event_type)
    if label is not None:
        return label
    if not allow_v2_event_labels:
        return None
    return _V2_TYPE_TO_LABEL.get(event_type)


def _normalized_events(
    event_calendar: pd.DataFrame | None, *, market: str
) -> pd.DataFrame:
    if event_calendar is None:
        raise ValueError("event_calendar is required and must not be None")
    return load_event_calendar(event_calendar, market=market)


def compute_event_window_just_passed(
    *,
    normalized_event_calendar: pd.DataFrame | None,
    sessions: tuple[date, ...] | list[date],
    trailing_sessions: int,
) -> pd.Series:
    """v2 §1C `event_window_just_passed` — per-session boolean (ADR 0005 Q3).

    Fires at session ``t`` iff there EXISTS a calendar event whose window
    closes at NYSE session ``E`` such that
    ``1 <= trading_days_between(E, t) <= trailing_sessions`` — i.e. ``t``
    is one of the ``trailing_sessions`` NYSE sessions strictly AFTER an
    event window closed. ``t == E`` does NOT fire (still inside the
    window).

    Window-end ``E`` = the NYSE session ``end_offset`` trading days after
    the event date, using the §1C per-type windows (``_WINDOWS``:
    fed_week +2, cpi_week +1, nfp_week +1). Event rows whose type has no
    recognized window (or whose publication_date is after ``t``) are
    skipped.

    ``normalized_event_calendar`` must be non-empty; absent event data is a
    broken dependency and must fail loudly instead of suppressing ``vol_crush``.

    Only V1 window types (``FOMC``/``CPI``/``NFP``) drive
    ``event_window_just_passed`` per ADR 0005 Q3. V2 event types
    (``ECB_decision``/``BOE_decision``/``BOJ_decision``/``election``/
    ``budget``/``geopolitical_event``) are intentionally excluded from
    this output to preserve ``vol_crush`` V1-byte-identity semantics —
    adding them would silently shift which sessions fire the rule.
    """
    session_tuple = tuple(sessions)
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in session_tuple])
    result = pd.Series(False, index=index, dtype=bool)
    if not session_tuple:
        return result
    if normalized_event_calendar is None or normalized_event_calendar.empty:
        raise ValueError("event_calendar is required for event_window_just_passed")

    event_rows = list(normalized_event_calendar.itertuples(index=False))
    if not event_rows:
        return result

    # Global NYSE session list spanning the union of session calendar and
    # all event/publication dates, padded so window arithmetic never runs
    # off the end.
    first_session = session_tuple[0]
    last_session = session_tuple[-1]
    min_event = min(min(r.publication_date, r.date) for r in event_rows)
    max_event = max(max(r.publication_date, r.date) for r in event_rows)
    global_start = min(first_session, min_event) - timedelta(days=_SESSION_PADDING_DAYS)
    global_end = max(last_session, max_event) + timedelta(days=_SESSION_PADDING_DAYS)
    global_sessions = _sessions_between(global_start, global_end)
    global_pos = {day: idx for idx, day in enumerate(global_sessions)}
    n_global = len(global_sessions)
    session_pos = {day: idx for idx, day in enumerate(session_tuple)}

    for row in event_rows:
        label = _TYPE_TO_LABEL.get(str(row.type))
        if label is None or label not in _WINDOWS:
            continue
        end_offset = _WINDOWS[label][1]
        event_idx = global_pos.get(row.date)
        if event_idx is None:
            continue
        window_end_idx = event_idx + end_offset
        if window_end_idx < 0 or window_end_idx >= n_global:
            continue
        # Sessions [E+1, E+trailing_sessions] (trading days) just-passed.
        for offset in range(1, trailing_sessions + 1):
            trailing_idx = window_end_idx + offset
            if trailing_idx >= n_global:
                break
            trailing_day = global_sessions[trailing_idx]
            # V1 §2.2 stateless replay: only consult events whose
            # publication_date is on or before the firing session.
            if trailing_day < row.publication_date:
                continue
            pos = session_pos.get(trailing_day)
            if pos is not None:
                result.iloc[pos] = True
    return result


def _third_friday_of_month(*, year: int, month: int) -> date:
    month_weeks = calendar.monthcalendar(year, month)
    friday_days = [
        week[calendar.FRIDAY] for week in month_weeks if week[calendar.FRIDAY] != 0
    ]
    return date(year, month, friday_days[2])


@lru_cache(maxsize=None)
def _sessions_between(start_date: date, end_date: date) -> tuple[date, ...]:
    return tuple(
        nyse_calendar().schedule(start_date=start_date, end_date=end_date).index.date
    )


def _second_weekday_of_month(*, year: int, month: int, weekday: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    first_match = first + timedelta(days=offset)
    return first_match + timedelta(days=7)
