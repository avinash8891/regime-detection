from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from regime_detection.calendar import nyse_calendar
from regime_detection.models import EventCalendarOutput


@dataclass(frozen=True)
class _Event:
    d: date
    market: str
    typ: str
    importance: str


_LABELS = [
    "fed_week",
    "cpi_week",
    "nfp_week",
    "expiry_week",
    "earnings_season",
    "normal_calendar",
    "unknown",
]

_PRECEDENCE = [
    "fed_week",
    "cpi_week",
    "nfp_week",
    "expiry_week",
    "earnings_season",
    "normal_calendar",
    "unknown",
]


def classify_event_calendar(
    *,
    as_of_date: date,
    event_calendar: pd.DataFrame | None,
) -> EventCalendarOutput:
    """
    V1 event calendar classification.

    Input contract (V1):
    - DataFrame with columns: date, market, type, importance
    - market must be "US" (other markets ignored in US V1)

    Matching:
    - fed_week: within [-2, +2] NYSE trading days of FOMC
    - cpi_week: within [-1, +1] NYSE trading days of CPI
    - nfp_week: within [-1, +1] NYSE trading days of NFP

    Configured labels:
    - expiry_week and earnings_season are supported via explicit daily markers in the input
      event_calendar (types "EXPIRY_WEEK" / "EARNINGS_SEASON"). This keeps V1 deterministic
      without embedding hardcoded assumptions about those season/window definitions.
    """
    if event_calendar is None:
        return EventCalendarOutput(
            raw_label="unknown",
            stable_label="unknown",
            active_label="unknown",
            evidence={"all_matching_events": [], "selected_via_precedence": "unknown"},
        )

    events = _load_events(event_calendar)
    if not events:
        return EventCalendarOutput(
            raw_label="normal_calendar",
            stable_label="normal_calendar",
            active_label="normal_calendar",
            evidence={"all_matching_events": [], "selected_via_precedence": "normal_calendar"},
        )

    matches: list[str] = []

    # Window-based labels from dated macro events.
    if _matches_trading_window(as_of_date, events, typ="FOMC", before=2, after=2):
        matches.append("fed_week")
    if _matches_trading_window(as_of_date, events, typ="CPI", before=1, after=1):
        matches.append("cpi_week")
    if _matches_trading_window(as_of_date, events, typ="NFP", before=1, after=1):
        matches.append("nfp_week")

    # Explicit daily markers for configured windows.
    if _has_exact_day_marker(as_of_date, events, typ="EXPIRY_WEEK"):
        matches.append("expiry_week")
    if _has_exact_day_marker(as_of_date, events, typ="EARNINGS_SEASON"):
        matches.append("earnings_season")

    if not matches:
        label = "normal_calendar"
    else:
        label = _pick_by_precedence(matches)

    all_matching = [l for l in _PRECEDENCE if l in set(matches)]
    evidence: dict[str, object] = {
        "all_matching_events": all_matching,
        "selected_via_precedence": label,
    }

    return EventCalendarOutput(
        raw_label=label,
        stable_label=label,
        active_label=label,
        evidence=evidence,
    )


def _pick_by_precedence(labels: list[str]) -> str:
    s = set(labels)
    for lab in _PRECEDENCE:
        if lab in s:
            return lab
    return "unknown"


def _load_events(df: pd.DataFrame) -> list[_Event]:
    cols = set(df.columns)
    required = {"date", "market", "type", "importance"}
    missing = sorted(required - cols)
    if missing:
        raise ValueError(f"event_calendar missing required columns: {missing}")
    if df.empty:
        return []

    out: list[_Event] = []
    d = pd.to_datetime(df["date"], errors="coerce")
    if d.isna().any():
        raise ValueError("event_calendar contains unparseable date values")

    for row in df.assign(date=d.dt.date).to_dict(orient="records"):
        market = str(row["market"])
        if market != "US":
            continue
        out.append(
            _Event(
                d=row["date"],
                market=market,
                typ=str(row["type"]),
                importance=str(row["importance"]),
            )
        )
    return out


def _has_exact_day_marker(as_of_date: date, events: list[_Event], *, typ: str) -> bool:
    return any(e.typ == typ and e.d == as_of_date for e in events)


def _matches_trading_window(
    as_of_date: date,
    events: list[_Event],
    *,
    typ: str,
    before: int,
    after: int,
) -> bool:
    for e in events:
        if e.typ != typ:
            continue
        if _in_nyse_trading_window(as_of_date, center=e.d, before=before, after=after):
            return True
    return False


def _in_nyse_trading_window(target: date, *, center: date, before: int, after: int) -> bool:
    # Build a small schedule around the center date and measure offsets in *trading days*.
    cal = nyse_calendar()
    center_ts = pd.Timestamp(center)
    target_ts = pd.Timestamp(target)
    start = (center_ts - pd.Timedelta(days=21)).date()
    end = (center_ts + pd.Timedelta(days=21)).date()
    schedule = cal.schedule(start_date=start, end_date=end)
    sessions = schedule.index
    if sessions.empty:
        return False
    if center_ts not in sessions:
        # If the event date is not a trading day, treat the nearest prior session as the anchor.
        anchor = sessions[sessions <= center_ts].max()
    else:
        anchor = center_ts

    anchor_pos = sessions.get_loc(anchor)
    if target_ts not in sessions:
        return False
    target_pos = sessions.get_loc(target_ts)

    delta = target_pos - anchor_pos
    return -before <= delta <= after

