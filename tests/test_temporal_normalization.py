from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from regime_detection.loaders import (
    load_cpi_nowcast_series,
    load_event_calendar,
    load_news_sentiment_series,
)
from regime_detection.calendar import as_date
from regime_detection.temporal import parse_date_series, parse_datetime_index


def test_parse_date_series_rejects_malformed_date_with_context() -> None:
    with pytest.raises(
        ValueError,
        match=r"event_calendar contains malformed publication_date values.*not-a-date",
    ):
        parse_date_series(
            ["2026-05-15", "not-a-date"],
            field_name="publication_date",
            context="event_calendar",
            nullable=True,
        )


def test_parse_datetime_index_rejects_missing_values() -> None:
    with pytest.raises(ValueError, match="news_sentiment source contains missing date"):
        parse_datetime_index(
            [pd.Timestamp("2026-05-15"), None],
            field_name="date",
            context="news_sentiment source",
        )


def test_parse_datetime_index_outputs_nyse_timestamps_accepted_by_as_date() -> None:
    index = parse_datetime_index(
        ["2024-03-11", pd.Timestamp("2024-03-12 15:00:00", tz="UTC")],
        field_name="date",
        context="mixed source",
    )

    assert index.tz is None
    assert [as_date(value) for value in index] == [
        dt.date(2024, 3, 11),
        dt.date(2024, 3, 12),
    ]


def test_parse_datetime_index_normalizes_mixed_timezone_values_deterministically() -> (
    None
):
    index = parse_datetime_index(
        [
            pd.Timestamp("2024-03-12 00:30:00", tz="UTC"),
            pd.Timestamp("2024-03-12"),
            pd.Timestamp("2024-03-12 09:30:00", tz="America/New_York"),
        ],
        field_name="date",
        context="mixed timezone source",
    )

    assert index.tz is None
    assert index.tolist() == [
        pd.Timestamp("2024-03-11"),
        pd.Timestamp("2024-03-12"),
        pd.Timestamp("2024-03-12"),
    ]


def test_parse_datetime_index_handles_dst_boundary_as_nyse_session_date() -> None:
    index = parse_datetime_index(
        ["2024-03-08", "2024-03-11"],
        field_name="date",
        context="dst source",
    )

    assert [as_date(value) for value in index] == [
        dt.date(2024, 3, 8),
        dt.date(2024, 3, 11),
    ]


def test_event_calendar_uses_shared_nullable_publication_date_parser() -> None:
    events = pd.DataFrame(
        [
            {
                "date": "2026-06-17",
                "market": "US",
                "type": "FOMC",
                "importance": "high",
                "publication_date": None,
            },
            {
                "date": "2026-06-18",
                "market": "US",
                "type": "ad_hoc",
                "importance": "medium",
                "publication_date": "2026-06-17",
            },
        ]
    )

    out = load_event_calendar(events)

    assert out.loc[0, "date"] == dt.date(2026, 6, 17)
    assert out.loc[0, "publication_date"] == dt.date(2026, 3, 19)
    assert out.loc[1, "publication_date"] == dt.date(2026, 6, 17)


def test_date_indexed_loaders_use_shared_malformed_date_errors() -> None:
    with pytest.raises(
        ValueError,
        match=r"news_sentiment source contains malformed date values.*bad",
    ):
        load_news_sentiment_series(
            pd.DataFrame({"date": ["2026-05-15", "bad"], "news_sentiment": [0.1, 0.2]})
        )

    with pytest.raises(
        ValueError,
        match=r"cpi_nowcast source contains malformed date values.*bad",
    ):
        load_cpi_nowcast_series(
            pd.DataFrame({"date": ["2026-05-15", "bad"], "cpi_nowcast": [0.1, 0.2]})
        )
