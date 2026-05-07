from __future__ import annotations

import datetime as dt
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from regime_detection.config import load_default_regime_config
from regime_detection.event_calendar import classify_event_calendar
from regime_detection.loaders import load_event_calendar
from regime_data_fetch.event_calendar import (
    EventCalendarFetchError,
    ScheduledEvent,
    run_us_event_calendar_fetch,
    validate_fomc_listing_integrity,
)


FOMC_FIXTURES = Path("tests/fixtures/raw/fomc")


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


def test_load_event_calendar_rejects_malformed_publication_date() -> None:
    df = pd.DataFrame(
        [
            {
                "date": "2024-01-19",
                "market": "US",
                "type": "FOMC",
                "importance": "high",
                "publication_date": "not-a-date",
            }
        ]
    )

    with pytest.raises(ValueError):
        load_event_calendar(df)


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
                "date": date(2024, 1, 31),
                "market": "US",
                "type": "FOMC",
                "importance": "high",
                "publication_date": date(2024, 1, 25),
            }
        ]
    )

    out = classify_event_calendar(
        as_of_date=date(2024, 1, 17),
        event_calendar=events,
        config=cfg,
    )

    assert out.active_label == "expiry_week"
    assert out.evidence["all_matching_events"] == ["expiry_week", "earnings_season"]
    assert "fed_week" not in out.evidence["all_matching_events"]


def test_validate_fomc_listing_integrity_detects_missing_structured_dates() -> None:
    html = """
    <a href="/monetarypolicy/fomcminutes20230201.htm">HTML</a>
    <a href="/monetarypolicy/fomcminutes20230322.htm">HTML</a>
    """
    parsed_entries = [
        ScheduledEvent(
            date=dt.date(2023, 3, 22),
            release_timestamp_et=dt.datetime(2023, 4, 12, 14, 0, tzinfo=dt.timezone.utc),
            market="US",
            type="FOMC",
            importance="high",
            source="federalreserve.gov:fomccalendars",
        )
    ]

    try:
        validate_fomc_listing_integrity(html=html, parsed_entries=parsed_entries, min_year=2023)
    except EventCalendarFetchError as exc:
        assert "mismatch" in str(exc).lower()
        assert "2023-02-01" in str(exc)
    else:
        raise AssertionError("Expected EventCalendarFetchError for missing FOMC dates")


def test_run_us_event_calendar_fetch_writes_yaml_and_report(tmp_path: Path) -> None:
    fomc_listing_html = (FOMC_FIXTURES / "fomc_calendars_snippet.html").read_text()
    historical_2019_html = (FOMC_FIXTURES / "fomchistorical2019_snippet.html").read_text()

    def fake_fomc_listing_fetcher() -> str:
        return fomc_listing_html

    def fake_fomc_historical_index_fetcher() -> str:
        return '<a href="/monetarypolicy/fomchistorical2019.htm">2019</a>'

    def fake_fomc_historical_page_fetcher(url: str) -> str:
        if url.endswith("fomchistorical2019.htm"):
            return historical_2019_html
        raise AssertionError(f"Unexpected FOMC history URL: {url}")

    def fake_bls_page_fetcher(url: str) -> str:
        if url.endswith("/2026/"):
            return """
            Friday, February 07, 2026
            08:30 AM
            Employment Situation for January 2026
            Thursday, February 12, 2026
            08:30 AM
            Consumer Price Index for January 2026
            """
        raise AssertionError(f"Unexpected BLS schedule URL: {url}")

    report_path = run_us_event_calendar_fetch(
        repo_root=tmp_path,
        fred_api_key="test-key",
        fomc_listing_fetcher=fake_fomc_listing_fetcher,
        fomc_historical_index_fetcher=fake_fomc_historical_index_fetcher,
        fomc_historical_page_fetcher=fake_fomc_historical_page_fetcher,
        bls_page_fetcher=fake_bls_page_fetcher,
        bls_start_year=2026,
        bls_end_year=2026,
    )

    report = json.loads(report_path.read_text())
    yaml_path = tmp_path / "configs" / "events" / "us_events.yaml"
    contents = yaml_path.read_text()

    assert report["counts"]["total_events"] == 6
    assert report["counts"]["by_type"] == {"CPI": 1, "FOMC": 4, "NFP": 1}
    assert report["paths"]["event_calendar_yaml"] == str(yaml_path)
    assert 'type: "FOMC"' in contents
    assert 'type: "CPI"' in contents
    assert 'type: "NFP"' in contents
    assert 'release_timestamp_et: "2026-02-12T08:30:00-05:00"' in contents
    assert 'source: "federalreserve.gov:fomccalendars"' in contents
    assert 'source: "bls.gov:schedule:consumer-price-index"' in contents


def test_run_us_event_calendar_fetch_sorts_events_by_release_timestamp(tmp_path: Path) -> None:
    def fake_fomc_listing_fetcher() -> str:
        return (FOMC_FIXTURES / "fomc_calendars_snippet.html").read_text()

    def fake_fomc_historical_index_fetcher() -> str:
        return ""

    def fake_fomc_historical_page_fetcher(url: str) -> str:
        raise AssertionError(f"Unexpected historical URL: {url}")

    def fake_bls_page_fetcher(url: str) -> str:
        if url.endswith("/2026/"):
            return """
            Friday, January 09, 2026
            08:30 AM
            Employment Situation for December 2025
            Wednesday, January 14, 2026
            08:30 AM
            Consumer Price Index for December 2025
            """
        raise AssertionError(f"Unexpected BLS schedule URL: {url}")

    run_us_event_calendar_fetch(
        repo_root=tmp_path,
        fred_api_key="test-key",
        fomc_listing_fetcher=fake_fomc_listing_fetcher,
        fomc_historical_index_fetcher=fake_fomc_historical_index_fetcher,
        fomc_historical_page_fetcher=fake_fomc_historical_page_fetcher,
        bls_page_fetcher=fake_bls_page_fetcher,
        bls_start_year=2026,
        bls_end_year=2026,
    )

    contents = (tmp_path / "configs" / "events" / "us_events.yaml").read_text().splitlines()
    first_event_type_line = next(line for line in contents if 'type: "' in line)
    assert first_event_type_line == '    type: "NFP"'


def test_run_us_event_calendar_fetch_does_not_require_fred_api_key(tmp_path: Path) -> None:
    def fake_fomc_listing_fetcher() -> str:
        return (FOMC_FIXTURES / "fomc_calendars_snippet.html").read_text()

    def fake_fomc_historical_index_fetcher() -> str:
        return ""

    def fake_fomc_historical_page_fetcher(url: str) -> str:
        raise AssertionError(f"Unexpected historical URL: {url}")

    def fake_bls_page_fetcher(url: str) -> str:
        if url.endswith("/2026/"):
            return """
            Friday, January 09, 2026
            08:30 AM
            Employment Situation for December 2025
            Wednesday, January 14, 2026
            08:30 AM
            Consumer Price Index for December 2025
            """
        raise AssertionError(f"Unexpected BLS schedule URL: {url}")

    run_us_event_calendar_fetch(
        repo_root=tmp_path,
        fred_api_key=None,
        fomc_listing_fetcher=fake_fomc_listing_fetcher,
        fomc_historical_index_fetcher=fake_fomc_historical_index_fetcher,
        fomc_historical_page_fetcher=fake_fomc_historical_page_fetcher,
        bls_page_fetcher=fake_bls_page_fetcher,
        bls_start_year=2026,
        bls_end_year=2026,
    )


def test_run_us_event_calendar_fetch_records_raw_artifacts_in_sqlite(tmp_path: Path) -> None:
    acquisition_db = tmp_path / "acquisition.db"

    def fake_fomc_listing_fetcher() -> str:
        return (FOMC_FIXTURES / "fomc_calendars_snippet.html").read_text()

    def fake_fomc_historical_index_fetcher() -> str:
        return ""

    def fake_fomc_historical_page_fetcher(url: str) -> str:
        raise AssertionError(f"Unexpected historical URL: {url}")

    def fake_bls_page_fetcher(url: str) -> str:
        if url.endswith("/2026/"):
            return """
            Friday, January 09, 2026
            08:30 AM
            Employment Situation for December 2025
            Wednesday, January 14, 2026
            08:30 AM
            Consumer Price Index for December 2025
            """
        raise AssertionError(f"Unexpected BLS schedule URL: {url}")

    report_path = run_us_event_calendar_fetch(
        repo_root=tmp_path,
        fred_api_key=None,
        fomc_listing_fetcher=fake_fomc_listing_fetcher,
        fomc_historical_index_fetcher=fake_fomc_historical_index_fetcher,
        fomc_historical_page_fetcher=fake_fomc_historical_page_fetcher,
        bls_page_fetcher=fake_bls_page_fetcher,
        acquisition_db_path=acquisition_db,
        bls_start_year=2026,
        bls_end_year=2026,
    )

    report = json.loads(report_path.read_text())
    assert report["paths"]["acquisition_db"] == str(acquisition_db)

    import sqlite3

    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute("SELECT fetch_type, status FROM fetch_runs").fetchall()
        artifacts = conn.execute("SELECT source_name, artifact_kind, source_identifier FROM artifacts ORDER BY artifact_id").fetchall()
        outputs = conn.execute("SELECT output_kind FROM derived_outputs ORDER BY output_id").fetchall()

    assert fetch_runs == [("events", "ok")]
    assert artifacts == [
        ("federalreserve.gov:fomccalendars", "html", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"),
        ("bls.gov:schedule", "html", "https://www.bls.gov/schedule/2026/"),
    ]
    assert outputs == [("event_calendar_yaml",), ("event_calendar_report",)]
