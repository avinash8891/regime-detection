"""Fetch Investing.com calendar and earnings artifacts into local archives."""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from regime_data_fetch._http import fetch_text
from regime_data_fetch.investing_earnings_browser import (
    access_token_from_page as _access_token_from_page,
    capture_investing_earnings_loaded_page as _capture_investing_earnings_loaded_page,
    capture_investing_earnings_page_with_token as _capture_investing_earnings_page_with_token,
    country_map_from_page as _country_map_from_page,
    investing_earnings_access_token as _investing_earnings_access_token,
    loaded_earnings_page_html as _loaded_earnings_page_html,
    redact_access_token as _redact_access_token,
    validate_token_not_expired as _validate_token_not_expired,
)
from regime_data_fetch.investing_archive import run_local_investing_archive_import
from regime_data_fetch.investing_live_constants import (
    CALENDAR_BASE,
    DOMAIN_ID,
    EARNINGS_BASE,
    SOURCE_CALENDAR_URL,
    SOURCE_EARNINGS_URL,
    SUBDOMAIN,
    TIMEZONE_OFFSET,
)
from regime_data_fetch.investing_live_normalizers import (
    normalize_earnings_rows as _normalize_earnings_rows,
    normalize_event_rows as _normalize_event_rows,
    normalize_holiday_rows as _normalize_holiday_rows,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CALENDAR_COUNTRY_IDS = [
    25,
    32,
    6,
    37,
    72,
    22,
    17,
    39,
    14,
    10,
    35,
    43,
    36,
    110,
    11,
    26,
    12,
    4,
    5,
    9,
    33,
    34,
    38,
    51,
    71,
    42,
    41,
    178,
    45,
    56,
]
DEFAULT_EARNINGS_COUNTRY_IDS = [5, 4, 14]
JsonFetcher = Callable[[str, dict[str, str], dict[str, str]], object]
PageFetcher = Callable[[str], str]
EarningsPageCapturer = Callable[[Path], Path]


@dataclass(frozen=True)
class EarningsBrowserConfig:
    user_data_dir: Path | None = None
    executable_path: Path | None = None
    headless: bool | None = None
    timeout_ms: int | None = None
    page_capturer: EarningsPageCapturer | None = None


def capture_investing_earnings_loaded_page(
    *,
    output_path: Path,
    user_data_dir: Path | None = None,
    executable_path: Path | None = None,
    headless: bool | None = None,
    timeout_ms: int | None = None,
) -> Path:
    return _capture_investing_earnings_loaded_page(
        output_path=output_path,
        user_data_dir=user_data_dir,
        executable_path=executable_path,
        headless=headless,
        timeout_ms=timeout_ms,
    )


# TODO(simplify, owner=regime-maintainers): the 7 passthrough `earnings_browser_*` kwargs on
# run_investing_live_fetch / capture_investing_live_archive should collapse
# into an `EarningsBrowserConfig` dataclass (user_data_dir, executable,
# headless, timeout_ms, page_capturer, loaded_page_path, access_token,
# capture). One param replaces the sprawl, and the 3-arg env-resolution
# ternary chain (~line 536) becomes a linear `_resolve_browser_config()`.
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
    earnings_browser_config = EarningsBrowserConfig(
        user_data_dir=earnings_browser_user_data_dir,
        executable_path=earnings_browser_executable,
        headless=earnings_browser_headless,
        timeout_ms=earnings_browser_timeout_ms,
        page_capturer=earnings_page_capturer,
    )
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
        earnings_browser_config=earnings_browser_config,
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
    earnings_browser_config: EarningsBrowserConfig | None = None,
) -> None:
    if end < start:
        raise ValueError("end must be >= start")
    json_fetcher = json_fetcher or _request_json
    calendar_dir = (
        archive_root
        / f"investing_calendar_structured_{start.isoformat()}_{end.isoformat()}"
    )
    earnings_dir = (
        archive_root / f"investing_earnings_{start.isoformat()}_{end.isoformat()}"
    )
    calendar_dir.mkdir(parents=True, exist_ok=True)
    earnings_dir.mkdir(parents=True, exist_ok=True)

    calendar_page = page_fetcher(SOURCE_CALENDAR_URL) if page_fetcher else ""
    earnings_page = page_fetcher(SOURCE_EARNINGS_URL) if page_fetcher else ""
    captured_access_token = None
    browser_config = earnings_browser_config or EarningsBrowserConfig()
    countries = (
        _country_map_from_page(calendar_page, key="eventAndHolidayCountries")
        if calendar_page
        else {}
    )
    if (
        earnings_browser_capture
        and earnings_access_token is None
        and earnings_loaded_page_path is None
        and not _investing_earnings_access_token()
        and not earnings_page
    ):
        capture_path = (
            earnings_dir
            / "browser_pages"
            / "investing_earnings_calendar_loaded_page.html"
        )
        if browser_config.page_capturer is not None:
            earnings_loaded_page_path = browser_config.page_capturer(capture_path)
            captured_html = earnings_loaded_page_path.read_text(errors="replace")
            captured_access_token = _access_token_from_page(captured_html)
            _validate_token_not_expired(captured_access_token)
            earnings_loaded_page_path.write_text(
                _redact_access_token(captured_html, captured_access_token)
            )
        else:
            captured_page = _capture_investing_earnings_page_with_token(
                output_path=capture_path,
                user_data_dir=browser_config.user_data_dir,
                executable_path=browser_config.executable_path,
                headless=browser_config.headless,
                timeout_ms=browser_config.timeout_ms,
            )
            earnings_loaded_page_path = captured_page.path
            captured_access_token = captured_page.access_token
    loaded_earnings_page = _loaded_earnings_page_html(earnings_loaded_page_path)
    if loaded_earnings_page:
        earnings_page = loaded_earnings_page
    stock_countries = (
        _country_map_from_page(earnings_page, key="stockCountries")
        if earnings_page
        else {}
    )
    access_token = _resolve_earnings_access_token(
        explicit_token=earnings_access_token or captured_access_token,
        earnings_page=earnings_page,
    )

    event_rows, holiday_rows, calendar_reports = _fetch_calendar_archive(
        calendar_dir=calendar_dir,
        start=start,
        end=end,
        countries=countries,
        country_ids=calendar_country_ids,
        json_fetcher=json_fetcher,
    )
    earnings_rows, earnings_reports = _fetch_and_normalize_earnings(
        earnings_dir=earnings_dir,
        start=start,
        end=end,
        access_token=access_token,
        country_ids=earnings_country_ids,
        json_fetcher=json_fetcher,
        stock_countries=stock_countries,
    )
    _write_live_archive_outputs(
        calendar_dir=calendar_dir,
        earnings_dir=earnings_dir,
        start=start,
        end=end,
        event_rows=event_rows,
        holiday_rows=holiday_rows,
        calendar_reports=calendar_reports,
        earnings_rows=earnings_rows,
        earnings_reports=earnings_reports,
    )


def _resolve_earnings_access_token(
    *, explicit_token: str | None, earnings_page: str
) -> str:
    access_token = explicit_token or _investing_earnings_access_token()
    if not access_token and earnings_page:
        access_token = _access_token_from_page(earnings_page)
    if not access_token:
        raise RuntimeError(
            "Investing.com earnings access token unavailable; enable browser capture, "
            "set INVESTING_EARNINGS_ACCESS_TOKEN, or pass --investing-earnings-loaded-page"
        )
    _validate_token_not_expired(access_token)
    return access_token


def _fetch_and_normalize_earnings(
    *,
    earnings_dir: Path,
    start: dt.date,
    end: dt.date,
    access_token: str,
    country_ids: list[int],
    json_fetcher: JsonFetcher,
    stock_countries: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_earnings, reports = _fetch_earnings_archive(
        earnings_dir=earnings_dir,
        start=start,
        end=end,
        access_token=access_token,
        country_ids=country_ids,
        json_fetcher=json_fetcher,
    )
    instrument_ids = sorted(
        {
            int(item["instrument_id"])
            for item in raw_earnings
            if item.get("instrument_id")
        }
    )
    instruments, instrument_reports = _fetch_instruments(
        earnings_dir=earnings_dir,
        ids=instrument_ids,
        json_fetcher=json_fetcher,
    )
    key_metrics, metrics_reports = _fetch_key_metrics(
        earnings_dir=earnings_dir,
        ids=instrument_ids,
        json_fetcher=json_fetcher,
    )
    rows = _normalize_earnings_rows(
        raw_earnings, instruments, key_metrics, stock_countries
    )
    return rows, [*reports, *instrument_reports, *metrics_reports]


def _write_live_archive_outputs(
    *,
    calendar_dir: Path,
    earnings_dir: Path,
    start: dt.date,
    end: dt.date,
    event_rows: list[dict[str, object]],
    holiday_rows: list[dict[str, object]],
    calendar_reports: list[dict[str, object]],
    earnings_rows: list[dict[str, object]],
    earnings_reports: list[dict[str, object]],
) -> None:
    _write_csv(
        calendar_dir / f"investing_economic_events_{start}_{end}.csv",
        event_rows,
        required_fields=["occurrence_time_utc"],
    )
    _write_csv(
        calendar_dir / f"investing_holidays_{start}_{end}.csv",
        holiday_rows,
        required_fields=["holiday_start_utc"],
    )
    _write_jsonl(
        calendar_dir / f"investing_calendar_combined_{start}_{end}.jsonl",
        event_rows + holiday_rows,
    )
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
    _write_csv(
        earnings_dir / f"investing_earnings_{start}_{end}.csv",
        earnings_rows,
        required_fields=["date"],
    )
    _write_jsonl(
        earnings_dir / f"investing_earnings_{start}_{end}.jsonl", earnings_rows
    )
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
        event_rows.extend(
            _normalize_event_rows(
                occurrences, events, month_start, month_end, countries
            )
        )
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
        payload = json_fetcher(
            f"{CALENDAR_BASE}{endpoint}", params, _calendar_headers()
        )
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"unexpected Investing.com {kind} payload: {type(payload).__name__}"
            )
        raw_path = (
            calendar_dir
            / "raw_monthly"
            / kind
            / f"{kind}_{start}_{end}_page_{page:03d}.json"
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        if kind == "events":
            page_items = [
                item
                for item in payload.get("occurrences", [])
                if isinstance(item, dict)
            ]
            page_events = [
                item for item in payload.get("events", []) if isinstance(item, dict)
            ]
            events.extend(page_events)
        else:
            page_items = [
                item for item in payload.get("holidays", []) if isinstance(item, dict)
            ]
            page_events = []
        items.extend(page_items)
        cursor = str(payload.get("next_page_cursor") or "")
        reports.append(
            {
                "kind": kind,
                "date_from": start.isoformat(),
                "date_to": end.isoformat(),
                "page": page,
                "raw_path": str(raw_path),
                "items": len(page_items),
                "events": len(page_events),
                "next_page_cursor_present": bool(cursor),
            }
        )
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
                "end_date": f"{month_end.isoformat()}T23:59:59.999Z",
                "country_ids": ",".join(str(item) for item in country_ids),
                "limit": "200",
                "deduplicate": "true",
            }
            if cursor:
                params["cursor"] = cursor
            headers = _earnings_headers(access_token)
            payload = json_fetcher(
                f"{EARNINGS_BASE}/v1/instruments/earnings", params, headers
            )
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"unexpected Investing.com earnings payload: {type(payload).__name__}"
                )
            raw_path = (
                earnings_dir
                / "raw_monthly"
                / f"earnings_{month_start}_{month_end}_page_{page:03d}.json"
            )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            page_rows = [
                item for item in payload.get("earnings", []) if isinstance(item, dict)
            ]
            rows.extend(page_rows)
            cursor = str(payload.get("cursor") or "")
            reports.append(
                {
                    "date_from": month_start.isoformat(),
                    "date_to": month_end.isoformat(),
                    "page": page,
                    "raw_path": str(raw_path),
                    "rows": len(page_rows),
                    "cursor_present": bool(cursor),
                }
            )
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
            {
                "instrument_ids": ",".join(str(item) for item in batch),
                "domain_id": DOMAIN_ID,
            },
            _calendar_headers(),
        )
        if not isinstance(payload, list):
            raise RuntimeError(
                f"unexpected Investing.com instruments payload: {type(payload).__name__}"
            )
        raw_path = (
            earnings_dir / "raw_instruments" / f"instruments_batch_{index:04d}.json"
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        for item in payload:
            if isinstance(item, dict) and item.get("id") is not None:
                by_id[str(item.get("id"))] = item
        reports.append(
            {
                "kind": "instruments",
                "batch": index,
                "requested": len(batch),
                "rows": len(payload),
                "raw_path": str(raw_path),
            }
        )
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
            {
                "instrument_ids": ",".join(str(item) for item in batch),
                "domain_id": DOMAIN_ID,
            },
            _calendar_headers(),
        )
        if not isinstance(payload, list):
            raise RuntimeError(
                f"unexpected Investing.com key-metrics payload: {type(payload).__name__}"
            )
        raw_path = (
            earnings_dir / "raw_instruments" / f"key_metrics_batch_{index:04d}.json"
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        for item in payload:
            if isinstance(item, dict) and item.get("instrument_id") is not None:
                by_id[str(item.get("instrument_id"))] = item
        reports.append(
            {
                "kind": "key_metrics",
                "batch": index,
                "requested": len(batch),
                "rows": len(payload),
                "raw_path": str(raw_path),
            }
        )
        time.sleep(0.05)
    return by_id, reports


def _month_ranges(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    ranges: list[tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor <= end:
        next_month = (
            dt.date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else dt.date(cursor.year, cursor.month + 1, 1)
        )
        ranges.append((cursor, min(end, next_month - dt.timedelta(days=1))))
        cursor = next_month
    return ranges


def _fetch_text(url: str) -> str:
    return fetch_text(url, timeout=60, urlopen=urllib.request.urlopen)


def _request_json(
    url: str,
    params: dict[str, str],
    headers: dict[str, str],
    *,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> object:
    request_url = f"{url}?{urllib.parse.urlencode(params)}"
    for attempt in range(1, max_retries + 1):
        try:
            return json.loads(
                fetch_text(
                    request_url,
                    headers=headers,
                    timeout=60,
                    urlopen=urllib.request.urlopen,
                )
            )
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt == max_retries:
                raise
            LOGGER.warning(
                "Investing.com JSON request failed for %s; retrying attempt %s/%s: %s",
                url,
                attempt + 1,
                max_retries,
                exc,
            )
            time.sleep(backoff_seconds * attempt)
    raise RuntimeError("unreachable Investing.com JSON retry state")


def _calendar_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Origin": "https://in.investing.com",
        "Referer": SOURCE_CALENDAR_URL,
        "domain-id": DOMAIN_ID,
    }


def _earnings_headers(access_token: str) -> dict[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://in.investing.com",
        "Referer": SOURCE_EARNINGS_URL,
        "domain-id": SUBDOMAIN,
    }
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers


def _dedupe(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    out = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    required_fields: list[str] | None = None,
) -> None:
    fields = sorted(
        {*(required_fields or []), *(key for row in rows for key in row)}
    ) or ["empty"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
