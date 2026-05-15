from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from regime_data_fetch.investing_live import (
    CALENDAR_BASE,
    EARNINGS_BASE,
    SOURCE_CALENDAR_URL,
    SOURCE_EARNINGS_URL,
    run_investing_live_fetch,
)


def test_run_investing_live_fetch_materializes_archive_and_records_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "data" / "raw"
    db_path = out_dir / "acquisition" / "acquisition.db"

    def page_fetcher(url: str) -> str:
        if url == SOURCE_CALENDAR_URL:
            return _next_data_html({"eventAndHolidayCountries": [{"id": 5, "name": "United States", "country_code": "US"}]})
        if url == SOURCE_EARNINGS_URL:
            return _next_data_html({"stockCountries": [{"id": 5, "name": "United States", "country_code": "US"}]}, access_token="fixture-token")
        raise AssertionError(url)

    def json_fetcher(url: str, params: dict[str, str], headers: dict[str, str]) -> object:
        if url == f"{CALENDAR_BASE}/v1/calendars/economic/events/occurrences":
            assert params["country_ids"] == "5"
            return {
                "events": [
                    {
                        "id": 1,
                        "country_id": 5,
                        "currency": "USD",
                        "category": "employment",
                        "importance": "high",
                        "event_translated": "Payrolls",
                        "short_name": "Payrolls",
                        "long_name": "U.S. Payrolls",
                        "source": "BLS",
                        "source_url": "https://www.bls.gov/",
                        "page_link": "/economic-calendar/payrolls-1",
                    }
                ],
                "occurrences": [
                    {
                        "occurrence_id": 10,
                        "event_id": 1,
                        "occurrence_time": "2026-05-01T12:30:00Z",
                        "actual": 1,
                        "forecast": 2,
                    }
                ],
            }
        if url == f"{CALENDAR_BASE}/v1/calendars/holidays":
            return {
                "holidays": [
                    {
                        "holiday_id": 20,
                        "holiday_start": "2026-05-01T00:00:00Z",
                        "holiday_end": "2026-05-01T23:59:59Z",
                        "holiday_name": "Market Holiday",
                        "exchange_id": 1,
                        "exchange_closed": True,
                        "exchange": {"country_id": 5, "country": "United States", "short_name": "NYSE"},
                    }
                ]
            }
        if url == f"{EARNINGS_BASE}/v1/instruments/earnings":
            assert headers["Authorization"] == "Bearer fixture-token"
            return {
                "earnings": [
                    {
                        "date": "2026-05-01",
                        "instrument_id": 100,
                        "country_id": 5,
                        "company": "Fixture Co",
                        "symbol": "FIX",
                        "eps_actual": 1.2,
                    }
                ]
            }
        raise AssertionError(url)

    report_path = run_investing_live_fetch(
        out_dir=out_dir,
        start=pd.Timestamp("2026-05-01").date(),
        end=pd.Timestamp("2026-05-01").date(),
        acquisition_db_path=db_path,
        artifact_store_root=tmp_path / "store",
        page_fetcher=page_fetcher,
        json_fetcher=json_fetcher,
        calendar_country_ids=[5],
        earnings_country_ids=[5],
    )

    report = json.loads(report_path.read_text())
    assert report["counts"] == {
        "economic_events_rows": 1,
        "holiday_rows": 1,
        "earnings_rows": 1,
        "raw_files": 7,
    }
    assert pd.read_parquet(out_dir / "investing" / "economic_events.parquet").shape[0] == 1
    assert pd.read_parquet(out_dir / "investing" / "holidays.parquet").shape[0] == 1
    assert pd.read_parquet(out_dir / "investing" / "earnings.parquet").shape[0] == 1
    assert (out_dir / "investing_live_archive").exists()

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT fetch_type, status FROM fetch_runs").fetchall() == [
            ("investing_archive_local", "ok")
        ]
        assert conn.execute(
            "SELECT count(*) FROM artifact_records WHERE source_name='investing.com'"
        ).fetchone() == (3,)


def test_run_investing_live_fetch_skips_earnings_without_token(tmp_path: Path) -> None:
    out_dir = tmp_path / "data" / "raw"
    db_path = out_dir / "acquisition" / "acquisition.db"

    def json_fetcher(url: str, params: dict[str, str], headers: dict[str, str]) -> object:
        if url == f"{CALENDAR_BASE}/v1/calendars/economic/events/occurrences":
            return {
                "events": [{"id": 1, "country_id": 5, "event_translated": "Payrolls"}],
                "occurrences": [{"occurrence_id": 10, "event_id": 1, "occurrence_time": "2026-05-01T12:30:00Z"}],
            }
        if url == f"{CALENDAR_BASE}/v1/calendars/holidays":
            return {
                "holidays": [
                    {
                        "holiday_id": 20,
                        "holiday_start": "2026-05-01T00:00:00Z",
                        "exchange": {"country_id": 5, "country": "United States"},
                    }
                ]
            }
        if url == f"{EARNINGS_BASE}/v1/instruments/earnings":
            raise AssertionError("earnings endpoint should not be called without token")
        raise AssertionError(url)

    report_path = run_investing_live_fetch(
        out_dir=out_dir,
        start=pd.Timestamp("2026-05-01").date(),
        end=pd.Timestamp("2026-05-01").date(),
        acquisition_db_path=db_path,
        artifact_store_root=tmp_path / "store",
        json_fetcher=json_fetcher,
        calendar_country_ids=[5],
        earnings_country_ids=[5],
    )

    report = json.loads(report_path.read_text())
    assert report["counts"] == {
        "economic_events_rows": 1,
        "holiday_rows": 1,
        "earnings_rows": 0,
        "raw_files": 7,
    }
    earnings_report = json.loads(
        (out_dir / "investing_live_archive" / "investing_earnings_2026-05-01_2026-05-01" / "fetch_report.json").read_text()
    )
    assert earnings_report["chunk_reports"] == [
        {
            "date_from": "2026-05-01",
            "date_to": "2026-05-01",
            "status": "skipped",
            "reason": "missing_INVESTING_EARNINGS_ACCESS_TOKEN",
        }
    ]
    assert pd.read_parquet(out_dir / "investing" / "earnings.parquet").empty


def _next_data_html(countries: dict[str, list[dict[str, object]]], *, access_token: str = "") -> str:
    payload = {
        "props": {
            "pageProps": {
                "accessToken": access_token,
                "state": {
                    "countryStore": {
                        key: [{"countries": value}]
                        for key, value in countries.items()
                    }
                },
            }
        }
    }
    return f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script>'
