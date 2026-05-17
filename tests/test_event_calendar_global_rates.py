from __future__ import annotations

import datetime as dt

import pytest

from regime_data_fetch.event_calendar_global_rates import (
    SOURCE_BOE,
    SOURCE_ECB,
    SOURCE_BOJ,
    UnsupportedGlobalRateSource,
    global_rate_source_name,
    parse_global_rate_decision_events,
)


def test_parse_global_rate_decision_events_dispatches_ecb_day_two_rows() -> None:
    text = """
    <dt>30/04/2026</dt><dd>Governing Council monetary policy meeting day 1</dd>
    <dt>01/05/2026</dt><dd>Governing Council monetary policy meeting day 2</dd>
    <dt>02/05/2026</dt><dd>Non-monetary policy meeting</dd>
    """

    events = parse_global_rate_decision_events(source_key="ecb", text=text)

    assert [(event.date, event.event_type, event.source) for event in events] == [
        (dt.date(2026, 5, 1), "ECB_decision", SOURCE_ECB)
    ]


def test_parse_global_rate_decision_events_dispatches_boe_dates() -> None:
    text = "2026 confirmed dates Thursday 7 May Monetary Policy Committee"

    events = parse_global_rate_decision_events(source_key="boe", text=text)

    assert [(event.date, event.event_type, event.source) for event in events] == [
        (dt.date(2026, 5, 7), "BOE_decision", SOURCE_BOE)
    ]


def test_parse_global_rate_decision_events_dispatches_boj_table_end_date() -> None:
    text = """
    <h2>2026</h2>
    <table><tr><td>Apr. 27, 28</td><td>Monetary Policy Meeting</td></tr></table>
    """

    events = parse_global_rate_decision_events(source_key="boj", text=text)

    assert [(event.date, event.event_type, event.source) for event in events] == [
        (dt.date(2026, 4, 28), "BOJ_decision", SOURCE_BOJ)
    ]


def test_parse_global_rate_decision_events_rejects_unknown_source() -> None:
    with pytest.raises(UnsupportedGlobalRateSource, match="Unsupported"):
        parse_global_rate_decision_events(source_key="rba", text="")


def test_global_rate_source_name_falls_back_to_source_key() -> None:
    assert global_rate_source_name("ecb") == SOURCE_ECB
    assert global_rate_source_name("rba") == "rba"
