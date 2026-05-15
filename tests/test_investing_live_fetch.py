from __future__ import annotations

import base64
import json
import sqlite3
import time
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
        if url == f"{CALENDAR_BASE}/v1/instruments":
            assert params == {"instrument_ids": "100", "domain_id": "56"}
            return [
                {
                    "id": 100,
                    "long_name": "Fixture Co",
                    "short_name": "Fixture",
                    "symbol": "FIX",
                    "display_symbol": "FIX",
                    "country_id": 5,
                    "country": "United States",
                    "exchange_id": 1,
                    "exchange_short_name": "NYSE",
                    "currency_code": "USD",
                    "attributes": {"sector_id": 9, "importance": "high"},
                    "price": {"last": 10.0, "change": 0.1, "change_percent": 1.0},
                    "active": True,
                }
            ]
        if url == f"{CALENDAR_BASE}/v1/instruments/key-metrics":
            assert params == {"instrument_ids": "100", "domain_id": "56"}
            return [{"instrument_id": 100, "key_metrics": {"market_cap": 123, "instrument_type": "Stock"}}]
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
        "raw_files": 9,
    }
    assert pd.read_parquet(out_dir / "investing" / "economic_events.parquet").shape[0] == 1
    assert pd.read_parquet(out_dir / "investing" / "holidays.parquet").shape[0] == 1
    earnings = pd.read_parquet(out_dir / "investing" / "earnings.parquet")
    assert earnings.shape[0] == 1
    assert earnings[["company", "country_code", "market_cap", "importance"]].iloc[0].to_dict() == {
        "company": "Fixture Co",
        "country_code": "US",
        "market_cap": 123,
        "importance": "high",
    }
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
        earnings_browser_capture=False,
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


def test_run_investing_live_fetch_captures_browser_page_when_missing_token(tmp_path: Path) -> None:
    out_dir = tmp_path / "data" / "raw"
    db_path = out_dir / "acquisition" / "acquisition.db"

    def earnings_page_capturer(output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _next_data_html({"stockCountries": [{"id": 5, "name": "United States", "country_code": "US"}]}, access_token=_future_jwt())
        )
        return output_path

    def json_fetcher(url: str, params: dict[str, str], headers: dict[str, str]) -> object:
        if url == f"{CALENDAR_BASE}/v1/calendars/economic/events/occurrences":
            return {"events": [], "occurrences": []}
        if url == f"{CALENDAR_BASE}/v1/calendars/holidays":
            return {"holidays": []}
        if url == f"{EARNINGS_BASE}/v1/instruments/earnings":
            assert headers["Authorization"].startswith("Bearer ")
            return {"earnings": [{"date": "2026-05-01", "instrument_id": 100}]}
        if url == f"{CALENDAR_BASE}/v1/instruments":
            return [{"id": 100, "long_name": "Fixture Co", "country_id": 5}]
        if url == f"{CALENDAR_BASE}/v1/instruments/key-metrics":
            return [{"instrument_id": 100, "key_metrics": {"market_cap": 123}}]
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
        earnings_page_capturer=earnings_page_capturer,
    )

    report = json.loads(report_path.read_text())
    assert report["counts"] == {
        "economic_events_rows": 0,
        "holiday_rows": 0,
        "earnings_rows": 1,
        "raw_files": 10,
    }
    assert (
        out_dir
        / "investing"
        / "raw_archive"
        / "investing_earnings_2026-05-01_2026-05-01"
        / "browser_pages"
        / "investing_earnings_calendar_loaded_page.html"
    ).exists()


def test_run_investing_live_fetch_reads_token_from_loaded_earnings_page(tmp_path: Path) -> None:
    out_dir = tmp_path / "data" / "raw"
    db_path = out_dir / "acquisition" / "acquisition.db"
    loaded_page = tmp_path / "earnings_page.html"
    loaded_page.write_text(_next_data_html({"stockCountries": [{"id": 5, "name": "United States", "country_code": "US"}]}, access_token=_future_jwt()))

    def json_fetcher(url: str, params: dict[str, str], headers: dict[str, str]) -> object:
        if url == f"{CALENDAR_BASE}/v1/calendars/economic/events/occurrences":
            return {"events": [], "occurrences": []}
        if url == f"{CALENDAR_BASE}/v1/calendars/holidays":
            return {"holidays": []}
        if url == f"{EARNINGS_BASE}/v1/instruments/earnings":
            assert headers["Authorization"].startswith("Bearer ")
            return {"earnings": [{"date": "2026-05-01", "instrument_id": 100, "eps_actual": 1.2}]}
        if url == f"{CALENDAR_BASE}/v1/instruments":
            return [{"id": 100, "long_name": "Fixture Co", "country_id": 5}]
        if url == f"{CALENDAR_BASE}/v1/instruments/key-metrics":
            return [{"instrument_id": 100, "key_metrics": {"market_cap": 123}}]
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
        earnings_loaded_page_path=loaded_page,
    )

    report = json.loads(report_path.read_text())
    assert report["counts"]["earnings_rows"] == 1
    earnings = pd.read_parquet(out_dir / "investing" / "earnings.parquet")
    assert earnings[["company", "country_code", "market_cap"]].iloc[0].to_dict() == {
        "company": "Fixture Co",
        "country_code": "US",
        "market_cap": 123,
    }


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


def _future_jwt() -> str:
    header = _b64({"alg": "HS256", "typ": "JWT"})
    payload = _b64({"exp": int(time.time()) + 3600})
    return f"{header}.{payload}.signature"


def _b64(payload: dict[str, object]) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    return encoded.rstrip("=")
