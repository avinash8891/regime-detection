from __future__ import annotations

import csv
import datetime as dt
import base64
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from collections.abc import Iterable
from pathlib import Path

from regime_data_fetch.investing_archive import run_local_investing_archive_import

DOMAIN_ID = "56"
SUBDOMAIN = "in"
TIMEZONE_OFFSET = "+05:30"
CALENDAR_BASE = "https://endpoints.investing.com/pd-instruments"
EARNINGS_BASE = "https://endpoints.investing.com/earnings"
SOURCE_CALENDAR_URL = "https://in.investing.com/economic-calendar"
SOURCE_EARNINGS_URL = "https://in.investing.com/earnings-calendar"
DEFAULT_CALENDAR_COUNTRY_IDS = [
    25, 32, 6, 37, 72, 22, 17, 39, 14, 10, 35, 43, 36, 110, 11,
    26, 12, 4, 5, 9, 33, 34, 38, 51, 71, 42, 41, 178, 45, 56,
]
DEFAULT_EARNINGS_COUNTRY_IDS = [5, 4, 14]
JsonFetcher = Callable[[str, dict[str, str], dict[str, str]], object]
PageFetcher = Callable[[str], str]
EarningsPageCapturer = Callable[[Path], Path]


def run_investing_live_fetch(
    *,
    out_dir: Path,
    start: dt.date,
    end: dt.date,
    acquisition_db_path: Path,
    artifact_store_root: str | Path | None = None,
    page_fetcher: PageFetcher | None = None,
    json_fetcher: JsonFetcher | None = None,
    calendar_country_ids: list[int] | None = None,
    earnings_country_ids: list[int] | None = None,
    earnings_access_token: str | None = None,
    earnings_loaded_page_path: Path | None = None,
    earnings_browser_capture: bool = True,
    earnings_browser_user_data_dir: Path | None = None,
    earnings_browser_executable: Path | None = None,
    earnings_browser_headless: bool | None = None,
    earnings_browser_timeout_ms: int | None = None,
    earnings_page_capturer: EarningsPageCapturer | None = None,
) -> Path:
    archive_root = out_dir / "investing_live_archive"
    capture_investing_live_archive(
        archive_root=archive_root,
        start=start,
        end=end,
        page_fetcher=page_fetcher,
        json_fetcher=json_fetcher,
        calendar_country_ids=calendar_country_ids or DEFAULT_CALENDAR_COUNTRY_IDS,
        earnings_country_ids=earnings_country_ids or DEFAULT_EARNINGS_COUNTRY_IDS,
        earnings_access_token=earnings_access_token,
        earnings_loaded_page_path=earnings_loaded_page_path,
        earnings_browser_capture=earnings_browser_capture,
        earnings_browser_user_data_dir=earnings_browser_user_data_dir,
        earnings_browser_executable=earnings_browser_executable,
        earnings_browser_headless=earnings_browser_headless,
        earnings_browser_timeout_ms=earnings_browser_timeout_ms,
        earnings_page_capturer=earnings_page_capturer,
    )
    return run_local_investing_archive_import(
        out_dir=out_dir,
        archive_root=archive_root,
        acquisition_db_path=acquisition_db_path,
        artifact_store_root=artifact_store_root,
    )


def capture_investing_live_archive(
    *,
    archive_root: Path,
    start: dt.date,
    end: dt.date,
    page_fetcher: PageFetcher | None = None,
    json_fetcher: JsonFetcher | None = None,
    calendar_country_ids: list[int],
    earnings_country_ids: list[int],
    earnings_access_token: str | None = None,
    earnings_loaded_page_path: Path | None = None,
    earnings_browser_capture: bool = True,
    earnings_browser_user_data_dir: Path | None = None,
    earnings_browser_executable: Path | None = None,
    earnings_browser_headless: bool | None = None,
    earnings_browser_timeout_ms: int | None = None,
    earnings_page_capturer: EarningsPageCapturer | None = None,
) -> None:
    if end < start:
        raise ValueError("end must be >= start")
    json_fetcher = json_fetcher or _request_json
    calendar_dir = archive_root / f"investing_calendar_structured_{start.isoformat()}_{end.isoformat()}"
    earnings_dir = archive_root / f"investing_earnings_{start.isoformat()}_{end.isoformat()}"
    calendar_dir.mkdir(parents=True, exist_ok=True)
    earnings_dir.mkdir(parents=True, exist_ok=True)

    calendar_page = page_fetcher(SOURCE_CALENDAR_URL) if page_fetcher else ""
    earnings_page = page_fetcher(SOURCE_EARNINGS_URL) if page_fetcher else ""
    countries = _country_map_from_page(calendar_page, key="eventAndHolidayCountries") if calendar_page else {}
    if (
        earnings_browser_capture
        and earnings_access_token is None
        and earnings_loaded_page_path is None
        and not _investing_earnings_access_token()
        and not earnings_page
    ):
        capture_path = earnings_dir / "browser_pages" / "investing_earnings_calendar_loaded_page.html"
        if earnings_page_capturer is not None:
            earnings_loaded_page_path = earnings_page_capturer(capture_path)
        else:
            earnings_loaded_page_path = capture_investing_earnings_loaded_page(
                output_path=capture_path,
                user_data_dir=earnings_browser_user_data_dir,
                executable_path=earnings_browser_executable,
                headless=earnings_browser_headless,
                timeout_ms=earnings_browser_timeout_ms,
            )
    loaded_earnings_page = _loaded_earnings_page_html(earnings_loaded_page_path)
    if loaded_earnings_page:
        earnings_page = loaded_earnings_page
    stock_countries = _country_map_from_page(earnings_page, key="stockCountries") if earnings_page else {}
    access_token = earnings_access_token or _investing_earnings_access_token()
    if not access_token and earnings_page:
        access_token = _access_token_from_page(earnings_page)
    if access_token:
        _validate_token_not_expired(access_token)

    event_rows, holiday_rows, calendar_reports = _fetch_calendar_archive(
        calendar_dir=calendar_dir,
        start=start,
        end=end,
        countries=countries,
        country_ids=calendar_country_ids,
        json_fetcher=json_fetcher,
    )
    if access_token:
        raw_earnings, earnings_reports = _fetch_earnings_archive(
            earnings_dir=earnings_dir,
            start=start,
            end=end,
            access_token=access_token,
            country_ids=earnings_country_ids,
            json_fetcher=json_fetcher,
        )
        instruments, instrument_reports = _fetch_instruments(
            earnings_dir=earnings_dir,
            ids=sorted({int(item["instrument_id"]) for item in raw_earnings if item.get("instrument_id")}),
            json_fetcher=json_fetcher,
        )
        key_metrics, metrics_reports = _fetch_key_metrics(
            earnings_dir=earnings_dir,
            ids=sorted({int(item["instrument_id"]) for item in raw_earnings if item.get("instrument_id")}),
            json_fetcher=json_fetcher,
        )
        earnings_rows = _normalize_earnings_rows(raw_earnings, instruments, key_metrics, stock_countries)
        earnings_reports.extend(instrument_reports)
        earnings_reports.extend(metrics_reports)
    else:
        earnings_rows = []
        earnings_reports = [{
            "date_from": start.isoformat(),
            "date_to": end.isoformat(),
            "status": "skipped",
            "reason": "missing_INVESTING_EARNINGS_ACCESS_TOKEN",
        }]

    _write_csv(calendar_dir / f"investing_economic_events_{start}_{end}.csv", event_rows, required_fields=["occurrence_time_utc"])
    _write_csv(calendar_dir / f"investing_holidays_{start}_{end}.csv", holiday_rows, required_fields=["holiday_start_utc"])
    _write_jsonl(calendar_dir / f"investing_calendar_combined_{start}_{end}.jsonl", event_rows + holiday_rows)
    (calendar_dir / "fetch_report.json").write_text(
        json.dumps(
            {
                "source_url": SOURCE_CALENDAR_URL,
                "date_from": start.isoformat(),
                "date_to": end.isoformat(),
                "event_rows": len(event_rows),
                "holiday_rows": len(holiday_rows),
                "combined_rows": len(event_rows) + len(holiday_rows),
                "chunk_reports": calendar_reports,
            },
            indent=2,
            sort_keys=True,
        )
    )
    _write_csv(earnings_dir / f"investing_earnings_{start}_{end}.csv", earnings_rows, required_fields=["date"])
    _write_jsonl(earnings_dir / f"investing_earnings_{start}_{end}.jsonl", earnings_rows)
    (earnings_dir / "fetch_report.json").write_text(
        json.dumps(
            {
                "source_url": SOURCE_EARNINGS_URL,
                "date_from": start.isoformat(),
                "date_to": end.isoformat(),
                "earning_rows": len(earnings_rows),
                "chunk_reports": earnings_reports,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _fetch_calendar_archive(
    *,
    calendar_dir: Path,
    start: dt.date,
    end: dt.date,
    countries: dict[str, dict[str, object]],
    country_ids: list[int],
    json_fetcher: JsonFetcher,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    event_rows: list[dict[str, object]] = []
    holiday_rows: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for month_start, month_end in _month_ranges(start, end):
        occurrences, events, page_reports = _fetch_calendar_pages(
            calendar_dir=calendar_dir,
            endpoint="/v1/calendars/economic/events/occurrences",
            kind="events",
            start=month_start,
            end=month_end,
            country_ids=country_ids,
            json_fetcher=json_fetcher,
        )
        event_rows.extend(_normalize_event_rows(occurrences, events, month_start, month_end, countries))
        reports.extend(page_reports)
        holidays, _, page_reports = _fetch_calendar_pages(
            calendar_dir=calendar_dir,
            endpoint="/v1/calendars/holidays",
            kind="holidays",
            start=month_start,
            end=month_end,
            country_ids=country_ids,
            json_fetcher=json_fetcher,
        )
        holiday_rows.extend(_normalize_holiday_rows(holidays, month_start, month_end))
        reports.extend(page_reports)
    return _dedupe(event_rows), _dedupe(holiday_rows), reports


def _fetch_calendar_pages(
    *,
    calendar_dir: Path,
    endpoint: str,
    kind: str,
    start: dt.date,
    end: dt.date,
    country_ids: list[int],
    json_fetcher: JsonFetcher,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cursor = ""
    page = 0
    items: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    while True:
        page += 1
        start_date = f"{start.isoformat()}T00:00:00.000{TIMEZONE_OFFSET}"
        end_date = f"{end.isoformat()}T23:59:59.999{TIMEZONE_OFFSET}"
        params = {
            "domain_id": DOMAIN_ID,
            "limit": "200",
            "start_date": start_date,
            "end_date": end_date,
            "country_ids": ",".join(str(item) for item in country_ids),
        }
        if cursor:
            params["cursor"] = cursor
        payload = json_fetcher(f"{CALENDAR_BASE}{endpoint}", params, _calendar_headers())
        if not isinstance(payload, dict):
            raise RuntimeError(f"unexpected Investing.com {kind} payload: {type(payload).__name__}")
        raw_path = calendar_dir / "raw_monthly" / kind / f"{kind}_{start}_{end}_page_{page:03d}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        if kind == "events":
            page_items = [item for item in payload.get("occurrences", []) if isinstance(item, dict)]
            page_events = [item for item in payload.get("events", []) if isinstance(item, dict)]
            events.extend(page_events)
        else:
            page_items = [item for item in payload.get("holidays", []) if isinstance(item, dict)]
            page_events = []
        items.extend(page_items)
        cursor = str(payload.get("next_page_cursor") or "")
        reports.append({"kind": kind, "date_from": start.isoformat(), "date_to": end.isoformat(), "page": page, "raw_path": str(raw_path), "items": len(page_items), "events": len(page_events), "next_page_cursor_present": bool(cursor)})
        if not cursor:
            break
        time.sleep(0.1)
    return items, events, reports


def _fetch_earnings_archive(
    *,
    earnings_dir: Path,
    start: dt.date,
    end: dt.date,
    access_token: str,
    country_ids: list[int],
    json_fetcher: JsonFetcher,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for month_start, month_end in _month_ranges(start, end):
        cursor = ""
        page = 0
        while True:
            page += 1
            params = {
                "start_date": f"{month_start.isoformat()}T00:00:00.000Z",
                "end_date": f"{month_end.isoformat()}T00:00:00.000Z",
                "country_ids": ",".join(str(item) for item in country_ids),
                "limit": "200",
                "deduplicate": "true",
            }
            if cursor:
                params["cursor"] = cursor
            headers = _earnings_headers(access_token)
            payload = json_fetcher(f"{EARNINGS_BASE}/v1/instruments/earnings", params, headers)
            if not isinstance(payload, dict):
                raise RuntimeError(f"unexpected Investing.com earnings payload: {type(payload).__name__}")
            raw_path = earnings_dir / "raw_monthly" / f"earnings_{month_start}_{month_end}_page_{page:03d}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            page_rows = [item for item in payload.get("earnings", []) if isinstance(item, dict)]
            rows.extend(page_rows)
            cursor = str(payload.get("cursor") or "")
            reports.append({"date_from": month_start.isoformat(), "date_to": month_end.isoformat(), "page": page, "raw_path": str(raw_path), "rows": len(page_rows), "cursor_present": bool(cursor)})
            if not cursor:
                break
            time.sleep(0.1)
    return _dedupe(rows), reports


def _chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _fetch_instruments(
    *,
    earnings_dir: Path,
    ids: list[int],
    json_fetcher: JsonFetcher,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    by_id: dict[str, dict[str, object]] = {}
    reports: list[dict[str, object]] = []
    for index, batch in enumerate(_chunks(ids, 50), start=1):
        payload = json_fetcher(
            f"{CALENDAR_BASE}/v1/instruments",
            {"instrument_ids": ",".join(str(item) for item in batch), "domain_id": DOMAIN_ID},
            _calendar_headers(),
        )
        if not isinstance(payload, list):
            raise RuntimeError(f"unexpected Investing.com instruments payload: {type(payload).__name__}")
        raw_path = earnings_dir / "raw_instruments" / f"instruments_batch_{index:04d}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        for item in payload:
            if isinstance(item, dict) and item.get("id") is not None:
                by_id[str(item.get("id"))] = item
        reports.append({"kind": "instruments", "batch": index, "requested": len(batch), "rows": len(payload), "raw_path": str(raw_path)})
        time.sleep(0.05)
    return by_id, reports


def _fetch_key_metrics(
    *,
    earnings_dir: Path,
    ids: list[int],
    json_fetcher: JsonFetcher,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    by_id: dict[str, dict[str, object]] = {}
    reports: list[dict[str, object]] = []
    for index, batch in enumerate(_chunks(ids, 50), start=1):
        payload = json_fetcher(
            f"{CALENDAR_BASE}/v1/instruments/key-metrics",
            {"instrument_ids": ",".join(str(item) for item in batch), "domain_id": DOMAIN_ID},
            _calendar_headers(),
        )
        if not isinstance(payload, list):
            raise RuntimeError(f"unexpected Investing.com key-metrics payload: {type(payload).__name__}")
        raw_path = earnings_dir / "raw_instruments" / f"key_metrics_batch_{index:04d}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        for item in payload:
            if isinstance(item, dict) and item.get("instrument_id") is not None:
                by_id[str(item.get("instrument_id"))] = item
        reports.append({"kind": "key_metrics", "batch": index, "requested": len(batch), "rows": len(payload), "raw_path": str(raw_path)})
        time.sleep(0.05)
    return by_id, reports


def _normalize_event_rows(occurrences: list[dict[str, object]], events: list[dict[str, object]], start: dt.date, end: dt.date, countries: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    mapped = {str(event.get("id") or event.get("event_id") or ""): event for event in events}
    rows = []
    for occurrence in occurrences:
        event_id = str(occurrence.get("event_id") or "")
        event = mapped.get(event_id, {})
        country_id = str(event.get("country_id") or "")
        country = countries.get(country_id, {})
        rows.append({
            "kind": "event", "requested_date_from": start.isoformat(), "requested_date_to": end.isoformat(),
            "occurrence_id": occurrence.get("occurrence_id", ""), "event_id": event_id,
            "occurrence_time_utc": occurrence.get("occurrence_time", ""), "actual_time_utc": occurrence.get("actual_time", ""),
            "country_id": country_id, "country_code": country.get("country_code", ""), "country": country.get("name", ""),
            "currency": event.get("currency", ""), "category": event.get("category", ""), "importance": event.get("importance", ""),
            "event_type": event.get("event_type", ""), "is_speech": event.get("event_type", "") == "speech", "is_report": event.get("event_type", "") == "report",
            "event": event.get("event_translated", ""), "event_short_name": event.get("short_name", ""), "event_long_name": event.get("long_name", ""),
            "event_description": event.get("description", ""), "period": occurrence.get("reference_period", ""), "unit": event.get("unit", ""),
            "precision": occurrence.get("precision", event.get("precision", "")), "actual": occurrence.get("actual", ""), "forecast": occurrence.get("forecast", ""),
            "previous": occurrence.get("previous", ""), "revised_from": occurrence.get("revised_from", ""), "preliminary": occurrence.get("preliminary", ""),
            "event_source": event.get("source", ""), "event_source_url": event.get("source_url", ""), "event_path": event.get("page_link", ""),
        })
    return rows


def _normalize_holiday_rows(holidays: list[dict[str, object]], start: dt.date, end: dt.date) -> list[dict[str, object]]:
    rows = []
    for holiday in holidays:
        exchange = holiday.get("exchange") if isinstance(holiday.get("exchange"), dict) else {}
        rows.append({"kind": "holiday", "requested_date_from": start.isoformat(), "requested_date_to": end.isoformat(), "holiday_id": holiday.get("holiday_id", ""), "holiday_start_utc": holiday.get("holiday_start", ""), "holiday_end_utc": holiday.get("holiday_end", ""), "country_id": exchange.get("country_id", ""), "country": exchange.get("country", ""), "exchange_id": holiday.get("exchange_id", ""), "exchange_short_name": exchange.get("short_name", ""), "exchange_long_name": exchange.get("long_name", ""), "exchange_time_zone": exchange.get("time_zone", ""), "name": holiday.get("holiday_name", ""), "exchange_closed": holiday.get("exchange_closed", "")})
    return rows


def _normalize_earnings_rows(
    earnings: list[dict[str, object]],
    instruments: dict[str, dict[str, object]],
    key_metrics: dict[str, dict[str, object]],
    countries: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for earning in earnings:
        instrument_id = str(earning.get("instrument_id") or "")
        instrument = instruments.get(instrument_id, {})
        metrics = key_metrics.get(instrument_id, {}).get("key_metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {}
        attributes = instrument.get("attributes") or {}
        if not isinstance(attributes, dict):
            attributes = {}
        price = instrument.get("price") or {}
        if not isinstance(price, dict):
            price = {}
        country_id = str(instrument.get("country_id") or earning.get("country_id") or "")
        country = countries.get(country_id, {})
        rows.append(
            {
                "kind": "earnings",
                "source_url": SOURCE_EARNINGS_URL,
                "date": earning.get("date", ""),
                "instrument_id": instrument_id,
                "company": instrument.get("long_name", earning.get("company", "")),
                "short_name": instrument.get("short_name", ""),
                "symbol": instrument.get("symbol", earning.get("symbol", "")),
                "display_symbol": instrument.get("display_symbol", ""),
                "country_id": country_id,
                "country": instrument.get("country", country.get("name", "")),
                "country_code": country.get("country_code", earning.get("country_code", "")),
                "exchange_id": instrument.get("exchange_id", ""),
                "exchange_short_name": instrument.get("exchange_short_name", ""),
                "currency_id": earning.get("currency_id", instrument.get("currency_id", "")),
                "currency_code": instrument.get("currency_code", ""),
                "sector_id": attributes.get("sector_id", ""),
                "importance": attributes.get("importance", ""),
                "instrument_type": instrument.get("type", metrics.get("instrument_type", "")),
                "market_phase": earning.get("market_phase", ""),
                "earning_date_type": earning.get("earning_date_type", ""),
                "report_month": earning.get("report_month", ""),
                "report_year": earning.get("report_year", ""),
                "eps_actual": earning.get("eps_actual", ""),
                "eps_forecast": earning.get("eps_forecast", ""),
                "revenue_actual": earning.get("revenue_actual", ""),
                "revenue_forecast": earning.get("revenue_forecast", ""),
                "market_cap": metrics.get("market_cap", ""),
                "price_last": price.get("last", ""),
                "price_change": price.get("change", ""),
                "price_change_percent": price.get("change_percent", ""),
                "last_price_timestamp_utc": price.get("last_price_timestamp", ""),
                "instrument_link": instrument.get("link", ""),
                "market_link": instrument.get("market_link", ""),
                "active": instrument.get("active", ""),
                "realtime": instrument.get("realtime", ""),
            }
        )
    return rows


def _month_ranges(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    ranges: list[tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor <= end:
        next_month = dt.date(cursor.year + 1, 1, 1) if cursor.month == 12 else dt.date(cursor.year, cursor.month + 1, 1)
        ranges.append((cursor, min(end, next_month - dt.timedelta(days=1))))
        cursor = next_month
    return ranges


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def capture_investing_earnings_loaded_page(
    *,
    output_path: Path,
    user_data_dir: Path | None = None,
    executable_path: Path | None = None,
    headless: bool | None = None,
    timeout_ms: int | None = None,
) -> Path:
    """Capture a browser-loaded Investing.com earnings page with a fresh token.

    Investing.com's earnings JSON endpoint requires a short-lived token embedded
    in the real browser page. Direct HTTP clients commonly receive a Cloudflare
    challenge, so this uses Playwright with a persistent Chrome profile. Set
    ``INVESTING_BROWSER_USER_DATA_DIR`` to reuse an existing solved session.
    """
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for automatic Investing.com earnings browser capture; "
            "install the browser extra or pass --investing-earnings-loaded-page"
        ) from exc

    resolved_user_data_dir = user_data_dir or Path(os.environ.get("INVESTING_BROWSER_USER_DATA_DIR", output_path.parent / "browser_profile"))
    env_executable = os.environ.get("INVESTING_BROWSER_EXECUTABLE", "").strip()
    resolved_executable_path = executable_path or (Path(env_executable) if env_executable else None)
    resolved_headless = (
        os.environ.get("INVESTING_BROWSER_HEADLESS", "0").strip().lower() in {"1", "true", "yes"}
        if headless is None
        else headless
    )
    resolved_timeout_ms = timeout_ms or int(os.environ.get("INVESTING_BROWSER_TIMEOUT_MS", "120000"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_user_data_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        launch_kwargs: dict[str, object] = {
            "headless": resolved_headless,
            "user_data_dir": str(resolved_user_data_dir),
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        if resolved_executable_path:
            launch_kwargs["executable_path"] = str(resolved_executable_path)
        context = playwright.chromium.launch_persistent_context(**launch_kwargs)
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(SOURCE_EARNINGS_URL, wait_until="domcontentloaded", timeout=resolved_timeout_ms)
            try:
                page.wait_for_function(
                    "() => document.documentElement.innerHTML.includes('accessToken')",
                    timeout=resolved_timeout_ms,
                )
            except PlaywrightTimeoutError as exc:
                output_path.write_text(page.content())
                raise RuntimeError(
                    "Investing.com earnings browser capture did not expose accessToken; "
                    f"saved current page to {output_path}. Complete the browser challenge and retry."
                ) from exc
            output_path.write_text(page.content())
        finally:
            context.close()
    token = _access_token_from_page(output_path.read_text(errors="replace"))
    _validate_token_not_expired(token)
    return output_path


def _request_json(url: str, params: dict[str, str], headers: dict[str, str]) -> object:
    request = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _calendar_headers() -> dict[str, str]:
    return {"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Origin": "https://in.investing.com", "Referer": SOURCE_CALENDAR_URL, "domain-id": DOMAIN_ID}


def _earnings_headers(access_token: str) -> dict[str, str]:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json, text/plain, */*", "Origin": "https://in.investing.com", "Referer": SOURCE_EARNINGS_URL, "domain-id": SUBDOMAIN}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers


def _investing_earnings_access_token() -> str:
    return os.environ.get("INVESTING_EARNINGS_ACCESS_TOKEN", "").strip()


def _loaded_earnings_page_html(path: Path | None) -> str:
    configured = path or _investing_earnings_loaded_page_path()
    if configured is None:
        return ""
    return configured.read_text(errors="replace")


def _investing_earnings_loaded_page_path() -> Path | None:
    configured = os.environ.get("INVESTING_EARNINGS_LOADED_PAGE", "").strip()
    return Path(configured) if configured else None


def _validate_token_not_expired(token: str) -> None:
    parts = token.split(".")
    if len(parts) != 3:
        return
    try:
        payload_bytes = base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4))
        payload = json.loads(payload_bytes)
    except Exception:
        return
    exp = payload.get("exp")
    if not isinstance(exp, int | float):
        return
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    if exp <= now:
        raise RuntimeError("Investing.com earnings accessToken is expired; reload the earnings calendar page and retry")


def _page_data(html: str) -> dict[str, object]:
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
    if not match:
        return {}
    return json.loads(match.group(1))


def _access_token_from_page(html: str) -> str:
    token = str(_page_data(html).get("props", {}).get("pageProps", {}).get("accessToken") or "")
    if not token:
        raise RuntimeError("Investing.com earnings page did not expose accessToken")
    return token


def _country_map_from_page(html: str, *, key: str) -> dict[str, dict[str, object]]:
    data = _page_data(html)
    groups = data.get("props", {}).get("pageProps", {}).get("state", {}).get("countryStore", {}).get(key, [])
    mapped: dict[str, dict[str, object]] = {}
    for group in groups:
        for country in group.get("countries", []):
            if isinstance(country, dict):
                mapped[str(country.get("id"))] = country
    return mapped


def _dedupe(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    out = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict[str, object]], *, required_fields: list[str] | None = None) -> None:
    fields = sorted({*(required_fields or []), *(key for row in rows for key in row)}) or ["empty"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
