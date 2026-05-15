from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
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
    stock_countries = _country_map_from_page(earnings_page, key="stockCountries") if earnings_page else {}
    access_token = earnings_access_token or _investing_earnings_access_token()
    if not access_token and earnings_page:
        access_token = _access_token_from_page(earnings_page)

    event_rows, holiday_rows, calendar_reports = _fetch_calendar_archive(
        calendar_dir=calendar_dir,
        start=start,
        end=end,
        countries=countries,
        country_ids=calendar_country_ids,
        json_fetcher=json_fetcher,
    )
    if access_token:
        earnings_rows, earnings_reports = _fetch_earnings_archive(
            earnings_dir=earnings_dir,
            start=start,
            end=end,
            access_token=access_token,
            countries=stock_countries,
            country_ids=earnings_country_ids,
            json_fetcher=json_fetcher,
        )
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
    countries: dict[str, dict[str, object]],
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
            rows.extend(_normalize_earnings_rows(page_rows, countries))
            cursor = str(payload.get("cursor") or "")
            reports.append({"date_from": month_start.isoformat(), "date_to": month_end.isoformat(), "page": page, "raw_path": str(raw_path), "rows": len(page_rows), "cursor_present": bool(cursor)})
            if not cursor:
                break
            time.sleep(0.1)
    return _dedupe(rows), reports


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


def _normalize_earnings_rows(earnings: list[dict[str, object]], countries: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for earning in earnings:
        country_id = str(earning.get("country_id") or "")
        country = countries.get(country_id, {})
        row = dict(earning)
        row.update({"kind": "earnings", "source_url": SOURCE_EARNINGS_URL, "country_code": country.get("country_code", row.get("country_code", ""))})
        rows.append(row)
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
