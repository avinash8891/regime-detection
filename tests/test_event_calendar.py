from __future__ import annotations

import datetime as dt
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from regime_detection.config import load_default_regime_config
from regime_detection.axis_series import build_event_calendar_series
from regime_detection.event_calendar import classify_event_calendar
from regime_detection.market_context import build_market_context, slice_context_to_recent_sessions
from regime_detection.loaders import load_event_calendar
from regime_data_fetch.event_calendar import (
    EventCalendarFetchError,
    ScheduledEvent,
    run_us_event_calendar_fetch,
    validate_fomc_listing_integrity,
    _validate_bls_events,
)


FOMC_FIXTURES = Path("tests/fixtures/raw/fomc")


def test_load_event_calendar_yaml_defaults_publication_date() -> None:
    path = Path(__file__).resolve().parent / "fixtures" / "events" / "us_events.yaml"
    df = load_event_calendar(path)

    assert set(df["type"]) == {"FOMC", "CPI", "NFP", "ad_hoc"}
    fomc_row = df[df["type"] == "FOMC"].iloc[0]
    assert fomc_row["publication_date"] == date(2023, 10, 21)


def test_load_event_calendar_yaml_accepts_v2_manual_types_and_window_days(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.yaml"
    path.write_text(
        "\n".join(
            [
                "events:",
                '  - date: "2026-11-03"',
                '    market: "US"',
                '    type: "election"',
                '    importance: "high"',
                "    window_days: [-5, +10]",
                '  - date: "2026-12-10"',
                '    market: "GLOBAL"',
                '    type: "global_rate_decision"',
                '    importance: "medium"',
                '  - date: "2026-06-15"',
                '    market: "US"',
                '    type: "geopolitical_event"',
                '    importance: "high"',
            ]
        )
        + "\n"
    )

    us = load_event_calendar(path, market="US")
    assert set(us["type"]) == {
        "election",
        "geopolitical_event",
        "global_rate_decision",
    }
    election = us[us["type"] == "election"].iloc[0]
    assert election["window_days"] == [-5, 10]
    assert election["publication_date"] == date(2026, 8, 5)

    global_events = load_event_calendar(path, market="GLOBAL")
    assert list(global_events["type"]) == ["global_rate_decision"]


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


def test_event_calendar_v2_election_uses_default_trading_day_window() -> None:
    cfg = load_default_regime_config()
    events = pd.DataFrame(
        [
            {
                "date": date(2026, 11, 3),
                "market": "US",
                "type": "election",
                "importance": "high",
            }
        ]
    )

    start_out = classify_event_calendar(
        as_of_date=date(2026, 10, 27),
        event_calendar=events,
        config=cfg,
    )
    end_out = classify_event_calendar(
        as_of_date=date(2026, 11, 17),
        event_calendar=events,
        config=cfg,
    )
    before_out = classify_event_calendar(
        as_of_date=date(2026, 10, 26),
        event_calendar=events,
        config=cfg,
    )

    assert start_out.active_label == "election_window"
    assert end_out.active_label == "election_window"
    assert before_out.active_label != "election_window"


def test_event_calendar_v2_precedence_and_labels() -> None:
    cfg = load_default_regime_config()
    events = pd.DataFrame(
        [
            {
                "date": date(2026, 11, 3),
                "market": "US",
                "type": "election",
                "importance": "high",
                "publication_date": date(2026, 8, 1),
            },
            {
                "date": date(2026, 11, 3),
                "market": "US",
                "type": "geopolitical_event",
                "importance": "high",
                "publication_date": date(2026, 11, 3),
                "approved_label": "geopolitical_event",
            },
            {
                "date": date(2026, 11, 3),
                "market": "US",
                "type": "budget",
                "importance": "medium",
                "publication_date": date(2026, 8, 1),
            },
        ]
    )

    out = classify_event_calendar(
        as_of_date=date(2026, 11, 3),
        event_calendar=events,
        config=cfg,
    )

    assert out.active_label == "geopolitical_event"
    assert out.evidence["selected_via_precedence"] == "geopolitical_event"
    assert set(out.evidence["all_matching_events"]) >= {
        "geopolitical_event",
        "election_window",
        "budget_week",
    }


def test_build_event_calendar_series_matches_point_classifier(market_df_for_asof) -> None:
    cfg = load_default_regime_config()
    end_date = date(2024, 1, 17)
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

    context = build_market_context(
        end_date=end_date,
        market_data=market_df_for_asof(end_date),
        config=cfg,
        event_calendar=events,
    )
    outputs = build_event_calendar_series(context)

    for day in context.sessions:
        expected = classify_event_calendar(
            as_of_date=day,
            event_calendar=events,
            config=cfg,
        )
        assert outputs[day].model_dump() == expected.model_dump()


def test_build_event_calendar_series_matches_point_classifier_for_holiday_shifted_expiry(
    market_df_for_asof,
) -> None:
    cfg = load_default_regime_config()
    end_date = date(2022, 4, 14)
    context = build_market_context(
        end_date=end_date,
        market_data=market_df_for_asof(end_date),
        config=cfg,
    )
    context = slice_context_to_recent_sessions(context=context, required_sessions=10)
    outputs = build_event_calendar_series(context)

    for day in context.sessions:
        expected = classify_event_calendar(
            as_of_date=day,
            event_calendar=None,
            config=cfg,
        )
        assert outputs[day].model_dump() == expected.model_dump()


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
    assert report["paths"]["event_calendar_yaml"] == "configs/events/us_events.yaml"
    assert 'type: "FOMC"' in contents
    assert 'type: "CPI"' in contents
    assert 'type: "NFP"' in contents
    assert 'release_timestamp_et: "2026-02-12T08:30:00-05:00"' in contents
    assert 'source: "federalreserve.gov:fomccalendars"' in contents
    assert 'source: "bls.gov:schedule:consumer-price-index"' in contents


def test_run_us_event_calendar_fetch_adds_v2_curated_candidates(tmp_path: Path) -> None:
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
        bls_start_year=2026,
        bls_end_year=2026,
        include_v2_curated_candidates=True,
        global_rate_calendar_text_fetchers={
            "ecb": lambda: """
                <dt>10/06/2026</dt>
                <dd>Governing Council of the ECB: monetary policy meeting in Frankfurt (Day 1)</dd>
                <dt>11/06/2026</dt>
                <dd>Governing Council of the ECB: monetary policy meeting in Frankfurt (Day 2), followed by press conference</dd>
            """,
            "boe": lambda: """
                2026 confirmed dates
                Thursday 5 February | February MPC Summary and minutes
            """,
            "boj": lambda: "Next Monetary Policy Meeting Date June 15 and 16, 2026",
        },
    )

    report = json.loads(report_path.read_text())
    contents = (tmp_path / "configs" / "events" / "us_events.yaml").read_text()

    assert report["counts"]["by_type"]["election"] == 1
    assert report["counts"]["by_type"]["budget"] == 1
    assert report["counts"]["by_type"]["ECB_decision"] == 1
    assert report["counts"]["by_type"]["BOE_decision"] == 1
    assert report["counts"]["by_type"]["BOJ_decision"] == 1
    assert 'date: "2026-11-03"' in contents
    assert 'type: "election"' in contents
    assert "window_days: [-5, 10]" in contents
    assert 'date: "2026-09-30"' in contents
    assert 'type: "budget"' in contents
    assert 'type: "ECB_decision"' in contents
    assert 'type: "BOE_decision"' in contents
    assert 'type: "BOJ_decision"' in contents


def test_run_us_event_calendar_fetch_uses_supplied_as_of_date_for_v2_candidates(tmp_path: Path) -> None:
    def fake_fomc_listing_fetcher() -> str:
        return (FOMC_FIXTURES / "fomc_calendars_snippet.html").read_text()

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
        fomc_historical_index_fetcher=lambda: "",
        fomc_historical_page_fetcher=lambda url: "",
        bls_page_fetcher=fake_bls_page_fetcher,
        bls_start_year=2026,
        bls_end_year=2026,
        include_v2_curated_candidates=True,
        as_of_date=dt.date(2025, 1, 1),
        global_rate_calendar_text_fetchers={
            "ecb": lambda: "",
            "boe": lambda: """
                2026 confirmed dates
                Thursday 5 February | February MPC Summary and minutes
            """,
            "boj": lambda: "",
        },
    )

    candidates = pd.read_parquet(tmp_path / "data" / "raw" / "event_calendar" / "candidates" / "event_candidates.parquet")
    boe = candidates[candidates["event_type"] == "BOE_decision"].iloc[0]
    assert boe["date"] == "2026-02-05"
    assert bool(boe["is_future_scheduled"]) is True


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
    assert report["paths"]["acquisition_db"] == "acquisition.db"

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


def test_run_us_event_calendar_fetch_wires_group_a_candidate_artifacts(tmp_path: Path) -> None:
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

    def group_a_text_fetcher(url: str) -> str:
        if url.endswith("/press/govcdec/mopo/html/index.en.html"):
            return "data-snippets='../2026/html/index_include.en.html'"
        if url.endswith("/press/govcdec/mopo/2026/html/index_include.en.html"):
            return """
            <dt isoDate="2026-04-30"><div class="date">30 April 2026</div></dt>
            <dd><div class="title"><a href="/press/pr/date/2026/html/ecb.mp260430~81b7179e6f.en.html">Monetary policy decisions</a></div></dd>
            """
        if url.endswith("/press/calendars/mgcgc/html/index.en.html"):
            return """
            <dt>11/06/2026</dt>
            <dd>Governing Council of the ECB: monetary policy meeting in Frankfurt (Day 2), followed by press conference</dd>
            """
        if url.endswith("/monetary-policy/upcoming-mpc-dates"):
            return """
            <h2>2026 confirmed dates</h2>
            <table><tbody>
            <tr><td>Thursday 5 February</td><td><a href="/monetary-policy-summary-and-minutes/2026/february-2026">February MPC Summary and minutes</a></td></tr>
            </tbody></table>
            """
        if url.endswith("/sitemap/news"):
            return ""
        if url.endswith("/en/mopo/mpmsche_minu/index.htm"):
            return """
            <h2 id="p2026">2026</h2><table><tbody>
            <tr><td>June 15 (Mon.), 16 (Tues.)</td></tr>
            </tbody></table>
            """
        if url.endswith("/en/mopo/mpmsche_minu/past.htm"):
            return ""
        raise AssertionError(f"Unexpected Group A URL: {url}")

    def boe_news_fetcher(page: int) -> str:
        assert page == 1
        return '{"Results": ""}'

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
        include_v2_curated_candidates=True,
        group_a_text_fetcher=group_a_text_fetcher,
        group_a_boe_news_fetcher=boe_news_fetcher,
        group_a_hf_parquet_fetcher=lambda: b"",
    )

    report = json.loads(report_path.read_text())
    assert report["group_a"]["candidates"]["ECB_decision"] == 2
    assert report["group_a"]["promoted"]["election"] == 1
    assert report["group_b"]["candidates"]["budget"] == 1
    assert report["group_b"]["promoted"]["budget"] == 1

    candidate_path = tmp_path / "data" / "raw" / "event_calendar" / "candidates" / "event_candidates.parquet"
    validation_path = tmp_path / "data" / "raw" / "event_calendar" / "candidates" / "event_validations.parquet"
    quarantine_path = tmp_path / "data" / "raw" / "event_calendar" / "candidates" / "quarantine.parquet"
    assert candidate_path.exists()
    assert validation_path.exists()
    assert quarantine_path.exists()
    candidate_columns = list(pd.read_parquet(candidate_path).columns)
    assert list(pd.read_parquet(quarantine_path).columns) == candidate_columns
    candidates_df = pd.read_parquet(candidate_path)
    budget_rows = candidates_df[candidates_df["event_type"] == "budget"]
    assert budget_rows[["date", "event_subtype", "source_id"]].to_dict("records") == [
        {
            "date": "2026-09-30",
            "event_subtype": "fy_deadline",
            "source_id": "usa.gov:federal-budget-process",
        }
    ]
    assert budget_rows["candidate_id"].iloc[0]

    import sqlite3

    with sqlite3.connect(acquisition_db) as conn:
        outputs = conn.execute("SELECT output_kind FROM derived_outputs ORDER BY output_id").fetchall()

    assert ("event_group_a_candidates",) in outputs
    assert ("event_group_a_validations",) in outputs
    assert ("event_group_a_quarantine",) in outputs
    contents = (tmp_path / "configs" / "events" / "us_events.yaml").read_text()
    assert 'type: "budget"' in contents
    assert 'type: "ECB_decision"' in contents
    assert 'source: "ecb.europa.eu:monetary-policy-decisions"' in contents


def test_validate_bls_events_allows_official_2025_lapse_cancellations() -> None:
    events = []
    for month in range(1, 12):
        release_date = dt.date(2025, month, min(month, 28))
        events.extend(
            [
                ScheduledEvent(
                    date=release_date,
                    release_timestamp_et=dt.datetime(2025, month, min(month, 28), 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))),
                    market="US",
                    type="CPI",
                    importance="high",
                    source="bls.gov:schedule:consumer-price-index",
                ),
                ScheduledEvent(
                    date=release_date,
                    release_timestamp_et=dt.datetime(2025, month, min(month, 28), 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))),
                    market="US",
                    type="NFP",
                    importance="high",
                    source="bls.gov:schedule:employment-situation",
                ),
            ]
        )

    _validate_bls_events(events=events, start_year=2025, end_year=2026)


def test_validate_bls_events_rejects_unexplained_11_row_completed_year() -> None:
    events = []
    for month in range(1, 12):
        release_date = dt.date(2024, month, min(month, 28))
        events.extend(
            [
                ScheduledEvent(
                    date=release_date,
                    release_timestamp_et=dt.datetime(2024, month, min(month, 28), 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))),
                    market="US",
                    type="CPI",
                    importance="high",
                    source="bls.gov:schedule:consumer-price-index",
                ),
                ScheduledEvent(
                    date=release_date,
                    release_timestamp_et=dt.datetime(2024, month, min(month, 28), 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))),
                    market="US",
                    type="NFP",
                    importance="high",
                    source="bls.gov:schedule:employment-situation",
                ),
            ]
        )

    with pytest.raises(EventCalendarFetchError, match="BLS CPI year 2024 had 11 release dates; expected 12"):
        _validate_bls_events(events=events, start_year=2024, end_year=2025)
