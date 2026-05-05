from __future__ import annotations

from datetime import date

import pandas as pd

from regime_detection.event_calendar import classify_event_calendar


def _events_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "market", "type", "importance"])
    if not df.empty:
        # Keep the test fixtures explicit: date strings are acceptable input, but we normalize here.
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def test_event_calendar_empty_calendar_is_normal_calendar() -> None:
    out = classify_event_calendar(as_of_date=date(2020, 8, 11), event_calendar=_events_df([]))
    assert out.raw_label == "normal_calendar"
    assert out.stable_label == "normal_calendar"
    assert out.active_label == "normal_calendar"
    assert out.evidence["all_matching_events"] == []
    assert out.evidence["selected_via_precedence"] == "normal_calendar"


def test_event_calendar_fed_week_uses_trading_day_window() -> None:
    # 2020-04-29 was an FOMC statement date (used here only as a deterministic test fixture).
    events = _events_df(
        [
            {"date": "2020-04-29", "market": "US", "type": "FOMC", "importance": "high"},
        ]
    )
    # Two NYSE trading days before 2020-04-29 is 2020-04-27.
    out = classify_event_calendar(as_of_date=date(2020, 4, 27), event_calendar=events)
    assert out.active_label == "fed_week"
    assert "fed_week" in out.evidence["all_matching_events"]


def test_event_calendar_precedence_fed_over_cpi() -> None:
    events = _events_df(
        [
            {"date": "2020-04-29", "market": "US", "type": "FOMC", "importance": "high"},
            {"date": "2020-04-28", "market": "US", "type": "CPI", "importance": "high"},
        ]
    )
    out = classify_event_calendar(as_of_date=date(2020, 4, 29), event_calendar=events)
    assert set(out.evidence["all_matching_events"]) == {"fed_week", "cpi_week"}
    assert out.active_label == "fed_week"
