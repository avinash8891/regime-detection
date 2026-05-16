from __future__ import annotations

import base64
import json
import sqlite3
import sys
import time
import types
from pathlib import Path

import pandas as pd
import pytest

from regime_data_fetch.investing_live import (
    CALENDAR_BASE,
    EARNINGS_BASE,
    SOURCE_CALENDAR_URL,
    SOURCE_EARNINGS_URL,
    capture_investing_earnings_loaded_page,
    _validate_token_not_expired,
    run_investing_live_fetch,
)

APPLE_INSTRUMENT_ID = 6408
APPLE_SYMBOL = "AAPL"
APPLE_COMPANY = "Apple Inc"
US_COUNTRY_ID = 5


def test_run_investing_live_fetch_materializes_archive_and_records_outputs(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "data" / "raw"
    db_path = out_dir / "acquisition" / "acquisition.db"
    earnings_params: list[dict[str, str]] = []

    def page_fetcher(url: str) -> str:
        if url == SOURCE_CALENDAR_URL:
            return _next_data_html(
                {
                    "eventAndHolidayCountries": [
                        {
                            "id": US_COUNTRY_ID,
                            "name": "United States",
                            "country_code": "US",
                        }
                    ]
                }
            )
        if url == SOURCE_EARNINGS_URL:
            return _next_data_html(
                {
                    "stockCountries": [
                        {
                            "id": US_COUNTRY_ID,
                            "name": "United States",
                            "country_code": "US",
                        }
                    ]
                },
                access_token="fixture-token",
            )
        raise AssertionError(url)

    def json_fetcher(
        url: str, params: dict[str, str], headers: dict[str, str]
    ) -> object:
        if url == f"{CALENDAR_BASE}/v1/calendars/economic/events/occurrences":
            assert params["country_ids"] == str(US_COUNTRY_ID)
            return {
                "events": [
                    {
                        "id": 1,
                        "country_id": US_COUNTRY_ID,
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
                        "exchange": {
                            "country_id": US_COUNTRY_ID,
                            "country": "United States",
                            "short_name": "NYSE",
                        },
                    }
                ]
            }
        if url == f"{EARNINGS_BASE}/v1/instruments/earnings":
            assert headers["Authorization"] == "Bearer fixture-token"
            earnings_params.append(params)
            return {
                "earnings": [
                    {
                        "date": "2026-05-01",
                        "instrument_id": APPLE_INSTRUMENT_ID,
                        "country_id": US_COUNTRY_ID,
                        "company": APPLE_COMPANY,
                        "symbol": APPLE_SYMBOL,
                        "eps_actual": 1.2,
                    }
                ]
            }
        if url == f"{CALENDAR_BASE}/v1/instruments":
            assert params == {
                "instrument_ids": str(APPLE_INSTRUMENT_ID),
                "domain_id": "56",
            }
            return [
                {
                    "id": APPLE_INSTRUMENT_ID,
                    "long_name": APPLE_COMPANY,
                    "short_name": "Apple",
                    "symbol": APPLE_SYMBOL,
                    "display_symbol": APPLE_SYMBOL,
                    "country_id": US_COUNTRY_ID,
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
            assert params == {
                "instrument_ids": str(APPLE_INSTRUMENT_ID),
                "domain_id": "56",
            }
            return [
                {
                    "instrument_id": APPLE_INSTRUMENT_ID,
                    "key_metrics": {"market_cap": 123, "instrument_type": "Stock"},
                }
            ]
        raise AssertionError(url)

    report_path = run_investing_live_fetch(
        out_dir=out_dir,
        start=pd.Timestamp("2026-05-01").date(),
        end=pd.Timestamp("2026-05-01").date(),
        acquisition_db_path=db_path,
        artifact_store_root=tmp_path / "store",
        page_fetcher=page_fetcher,
        json_fetcher=json_fetcher,
        calendar_country_ids=[US_COUNTRY_ID],
        earnings_country_ids=[US_COUNTRY_ID],
    )

    report = json.loads(report_path.read_text())
    assert report["counts"] == {
        "economic_events_rows": 1,
        "holiday_rows": 1,
        "earnings_rows": 1,
        "raw_files": 9,
    }
    assert (
        pd.read_parquet(out_dir / "investing" / "economic_events.parquet").shape[0] == 1
    )
    assert pd.read_parquet(out_dir / "investing" / "holidays.parquet").shape[0] == 1
    earnings = pd.read_parquet(out_dir / "investing" / "earnings.parquet")
    assert earnings.shape[0] == 1
    assert earnings[["company", "country_code", "market_cap", "importance"]].iloc[
        0
    ].to_dict() == {
        "company": APPLE_COMPANY,
        "country_code": "US",
        "market_cap": 123,
        "importance": "high",
    }
    assert earnings_params[0]["start_date"] == "2026-05-01T00:00:00.000Z"
    assert earnings_params[0]["end_date"] == "2026-05-01T23:59:59.999Z"
    assert (out_dir / "investing_live_archive").exists()

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT fetch_type, status FROM fetch_runs").fetchall() == [
            ("investing_archive_local", "ok")
        ]
        assert conn.execute(
            "SELECT count(*) FROM artifact_records WHERE source_name='investing.com'"
        ).fetchone() == (3,)


def test_run_investing_live_fetch_fails_loudly_without_earnings_token(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "data" / "raw"
    db_path = out_dir / "acquisition" / "acquisition.db"

    def json_fetcher(
        url: str, params: dict[str, str], headers: dict[str, str]
    ) -> object:
        if url == f"{CALENDAR_BASE}/v1/calendars/economic/events/occurrences":
            return {
                "events": [{"id": 1, "country_id": US_COUNTRY_ID, "event_translated": "Payrolls"}],
                "occurrences": [
                    {
                        "occurrence_id": 10,
                        "event_id": 1,
                        "occurrence_time": "2026-05-01T12:30:00Z",
                    }
                ],
            }
        if url == f"{CALENDAR_BASE}/v1/calendars/holidays":
            return {
                "holidays": [
                    {
                        "holiday_id": 20,
                        "holiday_start": "2026-05-01T00:00:00Z",
                        "exchange": {"country_id": US_COUNTRY_ID, "country": "United States"},
                    }
                ]
            }
        if url == f"{EARNINGS_BASE}/v1/instruments/earnings":
            raise AssertionError("earnings endpoint should not be called without token")
        raise AssertionError(url)

    with pytest.raises(
        RuntimeError, match="Investing.com earnings access token unavailable"
    ):
        run_investing_live_fetch(
            out_dir=out_dir,
            start=pd.Timestamp("2026-05-01").date(),
            end=pd.Timestamp("2026-05-01").date(),
            acquisition_db_path=db_path,
            artifact_store_root=tmp_path / "store",
            json_fetcher=json_fetcher,
            calendar_country_ids=[US_COUNTRY_ID],
            earnings_country_ids=[US_COUNTRY_ID],
            earnings_browser_capture=False,
        )

    assert not (out_dir / "investing" / "earnings.parquet").exists()


def test_run_investing_live_fetch_captures_browser_page_when_missing_token(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "data" / "raw"
    db_path = out_dir / "acquisition" / "acquisition.db"
    token = _future_jwt()

    def earnings_page_capturer(output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            _next_data_html(
                {
                    "stockCountries": [
                        {
                            "id": US_COUNTRY_ID,
                            "name": "United States",
                            "country_code": "US",
                        }
                    ]
                },
                access_token=token,
            )
        )
        return output_path

    def json_fetcher(
        url: str, params: dict[str, str], headers: dict[str, str]
    ) -> object:
        if url == f"{CALENDAR_BASE}/v1/calendars/economic/events/occurrences":
            return {"events": [], "occurrences": []}
        if url == f"{CALENDAR_BASE}/v1/calendars/holidays":
            return {"holidays": []}
        if url == f"{EARNINGS_BASE}/v1/instruments/earnings":
            assert headers["Authorization"].startswith("Bearer ")
            return {
                "earnings": [
                    {"date": "2026-05-01", "instrument_id": APPLE_INSTRUMENT_ID}
                ]
            }
        if url == f"{CALENDAR_BASE}/v1/instruments":
            return [
                {
                    "id": APPLE_INSTRUMENT_ID,
                    "long_name": APPLE_COMPANY,
                    "symbol": APPLE_SYMBOL,
                    "country_id": US_COUNTRY_ID,
                }
            ]
        if url == f"{CALENDAR_BASE}/v1/instruments/key-metrics":
            return [
                {
                    "instrument_id": APPLE_INSTRUMENT_ID,
                    "key_metrics": {"market_cap": 123},
                }
            ]
        raise AssertionError(url)

    report_path = run_investing_live_fetch(
        out_dir=out_dir,
        start=pd.Timestamp("2026-05-01").date(),
        end=pd.Timestamp("2026-05-01").date(),
        acquisition_db_path=db_path,
        artifact_store_root=tmp_path / "store",
        json_fetcher=json_fetcher,
        calendar_country_ids=[US_COUNTRY_ID],
        earnings_country_ids=[US_COUNTRY_ID],
        earnings_page_capturer=earnings_page_capturer,
    )

    report = json.loads(report_path.read_text())
    assert report["counts"] == {
        "economic_events_rows": 0,
        "holiday_rows": 0,
        "earnings_rows": 1,
        "raw_files": 10,
    }
    live_page = (
        out_dir
        / "investing_live_archive"
        / "investing_earnings_2026-05-01_2026-05-01"
        / "browser_pages"
        / "investing_earnings_calendar_loaded_page.html"
    )
    raw_page = (
        out_dir
        / "investing"
        / "raw_archive"
        / "investing_earnings_2026-05-01_2026-05-01"
        / "browser_pages"
        / "investing_earnings_calendar_loaded_page.html"
    )
    assert live_page.exists()
    assert raw_page.exists()
    live_page_html = live_page.read_text()
    raw_page_html = raw_page.read_text()
    assert "accessToken" in live_page_html
    assert "accessToken" in raw_page_html
    assert "[redacted]" in live_page_html
    assert "[redacted]" in raw_page_html
    assert token not in live_page_html
    assert token not in raw_page_html


def test_run_investing_live_fetch_reads_token_from_loaded_earnings_page(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "data" / "raw"
    db_path = out_dir / "acquisition" / "acquisition.db"
    loaded_page = tmp_path / "earnings_page.html"
    loaded_page.write_text(
        _next_data_html(
            {
                "stockCountries": [
                    {"id": US_COUNTRY_ID, "name": "United States", "country_code": "US"}
                ]
            },
            access_token=_future_jwt(),
        )
    )

    def json_fetcher(
        url: str, params: dict[str, str], headers: dict[str, str]
    ) -> object:
        if url == f"{CALENDAR_BASE}/v1/calendars/economic/events/occurrences":
            return {"events": [], "occurrences": []}
        if url == f"{CALENDAR_BASE}/v1/calendars/holidays":
            return {"holidays": []}
        if url == f"{EARNINGS_BASE}/v1/instruments/earnings":
            assert headers["Authorization"].startswith("Bearer ")
            return {
                "earnings": [
                    {
                        "date": "2026-05-01",
                        "instrument_id": APPLE_INSTRUMENT_ID,
                        "eps_actual": 1.2,
                    }
                ]
            }
        if url == f"{CALENDAR_BASE}/v1/instruments":
            return [
                {
                    "id": APPLE_INSTRUMENT_ID,
                    "long_name": APPLE_COMPANY,
                    "symbol": APPLE_SYMBOL,
                    "country_id": US_COUNTRY_ID,
                }
            ]
        if url == f"{CALENDAR_BASE}/v1/instruments/key-metrics":
            return [
                {
                    "instrument_id": APPLE_INSTRUMENT_ID,
                    "key_metrics": {"market_cap": 123},
                }
            ]
        raise AssertionError(url)

    report_path = run_investing_live_fetch(
        out_dir=out_dir,
        start=pd.Timestamp("2026-05-01").date(),
        end=pd.Timestamp("2026-05-01").date(),
        acquisition_db_path=db_path,
        artifact_store_root=tmp_path / "store",
        json_fetcher=json_fetcher,
        calendar_country_ids=[US_COUNTRY_ID],
        earnings_country_ids=[US_COUNTRY_ID],
        earnings_loaded_page_path=loaded_page,
    )

    report = json.loads(report_path.read_text())
    assert report["counts"]["earnings_rows"] == 1
    earnings = pd.read_parquet(out_dir / "investing" / "earnings.parquet")
    assert earnings[["company", "country_code", "market_cap"]].iloc[0].to_dict() == {
        "company": APPLE_COMPANY,
        "country_code": "US",
        "market_cap": 123,
    }


def test_validate_token_rejects_malformed_jwt_payload() -> None:
    with pytest.raises(RuntimeError, match="malformed"):
        _validate_token_not_expired("header.not-base64.signature")


def test_validate_token_rejects_expired_jwt_payload() -> None:
    header = _b64({"alg": "HS256", "typ": "JWT"})
    payload = _b64({"exp": int(time.time()) - 60})

    with pytest.raises(RuntimeError, match="expired"):
        _validate_token_not_expired(f"{header}.{payload}.signature")


def test_run_investing_live_fetch_slices_earnings_by_inclusive_month(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "data" / "raw"
    db_path = out_dir / "acquisition" / "acquisition.db"
    earnings_windows: list[tuple[str, str]] = []

    def json_fetcher(
        url: str, params: dict[str, str], headers: dict[str, str]
    ) -> object:
        del headers
        if url == f"{CALENDAR_BASE}/v1/calendars/economic/events/occurrences":
            return {"events": [], "occurrences": []}
        if url == f"{CALENDAR_BASE}/v1/calendars/holidays":
            return {"holidays": []}
        if url == f"{EARNINGS_BASE}/v1/instruments/earnings":
            earnings_windows.append((params["start_date"], params["end_date"]))
            return {"earnings": []}
        raise AssertionError(url)

    run_investing_live_fetch(
        out_dir=out_dir,
        start=pd.Timestamp("2026-01-31").date(),
        end=pd.Timestamp("2026-02-01").date(),
        acquisition_db_path=db_path,
        artifact_store_root=tmp_path / "store",
        json_fetcher=json_fetcher,
        calendar_country_ids=[US_COUNTRY_ID],
        earnings_country_ids=[US_COUNTRY_ID],
        earnings_access_token=_future_jwt(),
    )

    assert earnings_windows == [
        ("2026-01-31T00:00:00.000Z", "2026-01-31T23:59:59.999Z"),
        ("2026-02-01T00:00:00.000Z", "2026-02-01T23:59:59.999Z"),
    ]


def test_browser_capture_writes_redacted_page_without_access_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = _future_jwt()
    html = _next_data_html({"stockCountries": []}, access_token=token)

    class FakePage:
        def goto(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def wait_for_function(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def content(self) -> str:
            return html

    class FakeContext:
        pages = [FakePage()]

        def new_page(self) -> FakePage:
            return FakePage()

        def close(self) -> None:
            pass

    class FakeChromium:
        def launch_persistent_context(self, **kwargs: object) -> FakeContext:
            del kwargs
            return FakeContext()

    class FakePlaywright:
        chromium = FakeChromium()

        def __enter__(self) -> "FakePlaywright":
            return self

        def __exit__(self, *args: object) -> None:
            del args

    fake_sync_api = types.SimpleNamespace(
        TimeoutError=TimeoutError,
        sync_playwright=lambda: FakePlaywright(),
    )
    monkeypatch.setitem(sys.modules, "playwright", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync_api)
    output_path = (
        tmp_path / "data" / "raw" / "investing_live_archive" / "loaded_page.html"
    )

    captured_path = capture_investing_earnings_loaded_page(
        output_path=output_path,
        user_data_dir=tmp_path / "profile",
        headless=True,
        timeout_ms=1000,
    )

    persisted_html = captured_path.read_text()
    assert token not in persisted_html
    assert "accessToken" in persisted_html


def _next_data_html(
    countries: dict[str, list[dict[str, object]]], *, access_token: str = ""
) -> str:
    payload = {
        "props": {
            "pageProps": {
                "accessToken": access_token,
                "state": {
                    "countryStore": {
                        key: [{"countries": value}] for key, value in countries.items()
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
