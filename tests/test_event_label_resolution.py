from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import get_args, get_type_hints

import pytest

from regime_data_fetch.event_calendar import (
    EventLabelResolution,
    EventImportance,
    EventMarket,
    EventSource,
    EventType,
    ScheduledEvent,
    load_scheduled_events_yaml,
    _parse_global_rate_decision_events,
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


def test_scheduled_event_closed_vocabularies_are_typed_and_validated() -> None:
    hints = get_type_hints(ScheduledEvent)
    assert set(get_args(EventMarket)) == {"GLOBAL", "US"}
    assert {"FOMC", "CPI", "NFP", "budget", "election"}.issubset(
        set(get_args(EventType))
    )
    assert set(get_args(EventImportance)) == {"high", "medium"}
    assert "federalreserve.gov:fomccalendars" in set(get_args(EventSource))
    assert hints["market"] == EventMarket
    assert hints["type"] == EventType
    assert hints["importance"] == EventImportance
    assert hints["source"] == EventSource

    with pytest.raises(ValueError, match="unknown scheduled event type"):
        ScheduledEvent(
            date=dt.date(2026, 1, 28),
            release_timestamp_et=dt.datetime.fromisoformat("2026-02-18T14:00:00-05:00"),
            market="US",
            type="vendor_changed_name",  # type: ignore[arg-type]
            importance="high",
            source="federalreserve.gov:fomccalendars",
        )


def test_load_scheduled_events_yaml_reads_v2_manual_window_days(tmp_path: Path) -> None:
    path = tmp_path / "events.yaml"
    path.write_text(
        "\n".join(
            [
                "events:",
                '  - date: "2026-11-03"',
                '    market: "US"',
                '    type: "election"',
                '    importance: "high"',
                '    source: "operator:event_calendar_v2"',
                "    window_days: [-5, +10]",
            ]
        )
        + "\n"
    )

    events = load_scheduled_events_yaml(path)

    assert events[0].type == "election"
    assert events[0].window_days == (-5, 10)


def test_parse_boj_calendar_reads_html_table_year_heading() -> None:
    events = _parse_global_rate_decision_events(
        source_key="boj",
        text="""
        <h2 id="p2026">2026</h2>
        <table>
          <tr>
            <td>June 15 (Mon.), 16 (Tues.)</td>
            <td>-</td>
            <td>June 24 (Wed.)</td>
          </tr>
          <tr>
            <td>Dec. 17 (Thurs.), 18 (Fri.)</td>
            <td>-</td>
            <td>Dec. 28 (Mon.)</td>
          </tr>
        </table>
        """,
    )

    assert [(event.date.isoformat(), event.type) for event in events] == [
        ("2026-06-16", "BOJ_decision"),
        ("2026-12-18", "BOJ_decision"),
    ]


def test_parse_ecb_calendar_uses_day_2_monetary_policy_rows() -> None:
    events = _parse_global_rate_decision_events(
        source_key="ecb",
        text="""
        <dt>20/05/2026</dt>
        <dd>Governing Council of the ECB: non-monetary policy meeting in Frankfurt</dd>
        <dt>10/06/2026</dt>
        <dd>Governing Council of the ECB: monetary policy meeting in Frankfurt (Day 1)</dd>
        <dt>11/06/2026</dt>
        <dd>Governing Council of the ECB: monetary policy meeting in Frankfurt (Day 2), followed by press conference</dd>
        <dt>25/06/2026</dt>
        <dd>General Council meeting of the ECB (virtual)</dd>
        """,
    )

    assert [(event.date.isoformat(), event.type) for event in events] == [
        ("2026-06-11", "ECB_decision"),
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


def test_resolve_event_label_matches_election_window_default() -> None:
    events = [
        ScheduledEvent(
            date=dt.date(2026, 11, 3),
            release_timestamp_et=dt.datetime.fromisoformat("2026-11-03T00:00:00-05:00"),
            market="US",
            type="election",
            importance="high",
            source="operator:event_calendar_v2",
        )
    ]

    resolution = resolve_event_label(as_of_date=dt.date(2026, 10, 27), scheduled_events=events)

    assert resolution == EventLabelResolution(
        all_matching_events=["election_window", "earnings_season"],
        selected_via_precedence="election_window",
    )


def test_resolve_event_label_geopolitical_outranks_election() -> None:
    events = [
        ScheduledEvent(
            date=dt.date(2026, 11, 3),
            release_timestamp_et=dt.datetime.fromisoformat("2026-11-03T00:00:00-05:00"),
            market="US",
            type="election",
            importance="high",
            source="operator:event_calendar_v2",
        ),
        ScheduledEvent(
            date=dt.date(2026, 11, 3),
            release_timestamp_et=dt.datetime.fromisoformat("2026-11-03T00:00:00-05:00"),
            market="US",
            type="geopolitical_event",
            importance="high",
            source="operator:event_calendar_v2",
        ),
    ]

    resolution = resolve_event_label(as_of_date=dt.date(2026, 11, 3), scheduled_events=events)

    assert resolution == EventLabelResolution(
        all_matching_events=["geopolitical_event", "election_window", "earnings_season"],
        selected_via_precedence="geopolitical_event",
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
