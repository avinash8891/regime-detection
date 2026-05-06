from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from regime_detection.config import load_default_regime_config
from regime_detection.event_calendar import classify_event_calendar
from regime_detection.loaders import load_event_calendar


def test_load_event_calendar_yaml_defaults_publication_date() -> None:
    path = Path(__file__).resolve().parent / "fixtures" / "events" / "us_events.yaml"
    df = load_event_calendar(path)

    assert set(df["type"]) == {"FOMC", "CPI", "NFP", "ad_hoc"}
    fomc_row = df[df["type"] == "FOMC"].iloc[0]
    assert fomc_row["publication_date"] == date(2023, 10, 21)


def test_load_event_calendar_csv_defaults_publication_date() -> None:
    path = Path(__file__).resolve().parent / "fixtures" / "events" / "us_events.csv"
    df = load_event_calendar(path)

    nfp_row = df[df["type"] == "NFP"].iloc[0]
    assert nfp_row["publication_date"] == date(2023, 10, 22)


def test_event_calendar_uses_publication_date_and_precedence() -> None:
    cfg = load_default_regime_config()
    events = pd.DataFrame(
        [
            {
                "date": date(2024, 1, 19),
                "market": "US",
                "type": "FOMC",
                "importance": "high",
                "publication_date": date(2023, 12, 1),
            },
            {
                "date": date(2024, 1, 18),
                "market": "US",
                "type": "CPI",
                "importance": "high",
                "publication_date": date(2023, 12, 1),
            },
        ]
    )

    out = classify_event_calendar(
        as_of_date=date(2024, 1, 17),
        event_calendar=events,
        config=cfg,
    )

    assert out.active_label == "fed_week"
    assert out.evidence["selected_via_precedence"] == "fed_week"
    assert set(out.evidence["all_matching_events"]) >= {"fed_week", "cpi_week", "expiry_week", "earnings_season"}


def test_event_calendar_blocks_unpublished_future_scheduled_event() -> None:
    cfg = load_default_regime_config()
    events = pd.DataFrame(
        [
            {
                "date": date(2024, 1, 19),
                "market": "US",
                "type": "FOMC",
                "importance": "high",
                "publication_date": date(2024, 1, 18),
            },
        ]
    )

    out = classify_event_calendar(
        as_of_date=date(2024, 1, 17),
        event_calendar=events,
        config=cfg,
    )

    assert "fed_week" not in out.evidence["all_matching_events"]
    assert out.active_label == "expiry_week"


def test_event_calendar_uses_holiday_adjusted_monthly_expiry_rule() -> None:
    cfg = load_default_regime_config()

    out = classify_event_calendar(
        as_of_date=date(2025, 4, 17),
        event_calendar=None,
        config=cfg,
    )

    assert out.active_label == "expiry_week"

