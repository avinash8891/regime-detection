from __future__ import annotations

import datetime as dt
from pathlib import Path

from regime_data_fetch.event_calendar import (
    EventLabelResolution,
    ScheduledEvent,
    load_scheduled_events_yaml,
    resolve_event_label,
)


def test_load_scheduled_events_yaml_reads_generated_shape(tmp_path: Path) -> None:
    path = tmp_path / "events.yaml"
    path.write_text(
        "\n".join(
            [
                "events:",
                '  - date: "2026-01-28"',
                '    release_timestamp_et: "2026-02-18T14:00:00-05:00"',
                '    market: "US"',
                '    type: "FOMC"',
                '    importance: "high"',
                '    source: "federalreserve.gov:fomccalendars"',
            ]
        )
        + "\n"
    )

    events = load_scheduled_events_yaml(path)

    assert events == [
        ScheduledEvent(
            date=dt.date(2026, 1, 28),
            release_timestamp_et=dt.datetime.fromisoformat("2026-02-18T14:00:00-05:00"),
            market="US",
            type="FOMC",
            importance="high",
            source="federalreserve.gov:fomccalendars",
        )
    ]


def test_resolve_event_label_uses_precedence_over_earnings_season() -> None:
    events = [
        ScheduledEvent(
            date=dt.date(2026, 1, 28),
            release_timestamp_et=dt.datetime.fromisoformat("2026-02-18T14:00:00-05:00"),
            market="US",
            type="FOMC",
            importance="high",
            source="federalreserve.gov:fomccalendars",
        )
    ]

    resolution = resolve_event_label(as_of_date=dt.date(2026, 1, 28), scheduled_events=events)

    assert resolution == EventLabelResolution(
        all_matching_events=["fed_week", "earnings_season"],
        selected_via_precedence="fed_week",
    )


def test_resolve_event_label_matches_cpi_week_trading_day_window() -> None:
    events = [
        ScheduledEvent(
            date=dt.date(2026, 2, 12),
            release_timestamp_et=dt.datetime.fromisoformat("2026-02-12T08:30:00-05:00"),
            market="US",
            type="CPI",
            importance="high",
            source="bls.gov:schedule:consumer-price-index",
        )
    ]

    resolution = resolve_event_label(as_of_date=dt.date(2026, 2, 13), scheduled_events=events)

    assert resolution == EventLabelResolution(
        all_matching_events=["cpi_week", "earnings_season"],
        selected_via_precedence="cpi_week",
    )


def test_resolve_event_label_selects_expiry_week_when_no_higher_priority_event_matches() -> None:
    resolution = resolve_event_label(as_of_date=dt.date(2026, 6, 17), scheduled_events=[])

    assert resolution == EventLabelResolution(
        all_matching_events=["expiry_week"],
        selected_via_precedence="expiry_week",
    )


def test_resolve_event_label_selects_earnings_season_when_only_rule_matches() -> None:
    resolution = resolve_event_label(as_of_date=dt.date(2026, 1, 20), scheduled_events=[])

    assert resolution == EventLabelResolution(
        all_matching_events=["earnings_season"],
        selected_via_precedence="earnings_season",
    )


def test_resolve_event_label_returns_normal_calendar_when_no_event_matches() -> None:
    resolution = resolve_event_label(as_of_date=dt.date(2026, 2, 24), scheduled_events=[])

    assert resolution == EventLabelResolution(
        all_matching_events=[],
        selected_via_precedence="normal_calendar",
    )
