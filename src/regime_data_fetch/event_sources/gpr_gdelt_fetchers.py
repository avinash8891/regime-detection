from __future__ import annotations

import datetime as dt
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources._common import HTTP_USER_AGENT
from regime_data_fetch.event_sources.gpr_gdelt_conflict_parsers import _json_records

LOGGER = logging.getLogger(__name__)

ACLED_SOURCE_ID = "acled:events"
UCDP_SOURCE_ID = "ucdp:ged-candidate"
GPR_DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
GDELT_V2_MASTERFILELIST_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
GDELT_DAILY_EXPORT_URL_TEMPLATE = (
    "http://data.gdeltproject.org/gdeltv2/{date:%Y%m%d}*.export.CSV.zip"
)
ACLED_READ_URL = "https://acleddata.com/api/acled/read"
ACLED_TOKEN_URL = "https://acleddata.com/oauth/token"
UCDP_GED_CANDIDATE_URL = "https://ucdpapi.pcr.uu.se/api/gedevents/26.0.3"
GPR_MONTHLY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
AI_GPR_DAILY_URL = "https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv"
AI_GPR_EVENTTYPE_MONTHLY_URL = (
    "https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_eventtype_monthly.csv"
)
AI_GPR_COUNTRY_MONTHLY_URL = (
    "https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_country_monthly.csv"
)
CONFLICT_API_PAGE_SIZE = 1000
ConflictFetcher = Callable[[int, int], str | bytes | None]


@dataclass(frozen=True)
class SourceFetchStatus:
    source_id: str
    status: str
    rows: int = 0
    error: str | None = None
    attempted_fetches: int = 0
    failed_fetches: int = 0
    empty_payload: bool = False


@dataclass(frozen=True)
class FetchOutcome:
    rows: list[dict[str, object]]
    status: SourceFetchStatus


def fetch_gpr_daily() -> bytes:
    request = Request(GPR_DAILY_URL, headers={"User-Agent": HTTP_USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read()


def fetch_gpr_monthly() -> bytes:
    request = Request(GPR_MONTHLY_URL, headers={"User-Agent": HTTP_USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read()


def fetch_ai_gpr_daily() -> str:
    return _http_text(AI_GPR_DAILY_URL, headers={})


def fetch_ai_gpr_eventtype_monthly() -> str:
    return _http_text(AI_GPR_EVENTTYPE_MONTHLY_URL, headers={})


def fetch_ai_gpr_country_monthly() -> str:
    return _http_text(AI_GPR_COUNTRY_MONTHLY_URL, headers={})


def fetch_gdelt_daily_export(day: dt.date) -> bytes:
    master_list = _http_text(GDELT_V2_MASTERFILELIST_URL, headers={})
    export_urls = tuple(
        _gdelt_v2_export_urls_for_day(master_list=master_list, day=day)
    )
    return b"".join(_http_bytes(url) for url in export_urls)


def _gdelt_v2_export_urls_for_day(*, master_list: str, day: dt.date) -> list[str]:
    day_prefix = f"/gdeltv2/{day:%Y%m%d}"
    urls: list[str] = []
    for line in master_list.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        url = parts[2]
        if day_prefix in url and url.endswith(".export.CSV.zip"):
            urls.append(url)
    return urls


def fetch_acled_events(start_year: int, end_year: int) -> str | None:
    token = _acled_access_token()
    if token is None:
        LOGGER.error(
            "ACLED credentials unavailable; set ACLED_API_TOKEN or ACLED_USERNAME/ACLED_PASSWORD to fetch ACLED geopolitical events"
        )
        return None
    params = {
        "_format": "json",
        "event_date": f"{start_year}-01-01|{end_year}-12-31",
        "event_date_where": "BETWEEN",
        "fields": "event_date|event_type|sub_event_type|country|fatalities|source|notes",
        "limit": "5000",
    }
    records: list[dict[str, object]] = []
    page = 1
    while True:
        payload = _http_text(
            f"{ACLED_READ_URL}?{urlencode({**params, 'page': str(page)})}",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        page_records = _json_records(payload, container_keys=("data",))
        records.extend(page_records)
        if len(page_records) < 5000:
            break
        page += 1
    return json.dumps({"data": records}, sort_keys=True)


def fetch_ucdp_events(start_year: int, end_year: int) -> str | None:
    token = os.environ.get("UCDP_ACCESS_TOKEN", "").strip()
    if not token:
        LOGGER.error(
            "UCDP token unavailable; set UCDP_ACCESS_TOKEN to fetch UCDP GED Candidate geopolitical events"
        )
        return None
    return _fetch_paged_json(
        UCDP_GED_CANDIDATE_URL,
        headers={"x-ucdp-access-token": token},
        result_key="Result",
        extra_params={
            "StartDate": f"{start_year}-01-01",
            "EndDate": f"{end_year}-12-31",
        },
    )


def fetch_optional_conflict_rows(
    fetcher: ConflictFetcher,
    start_year: int,
    end_year: int,
    *,
    store: AcquisitionStore | None,
    run_id: int | None,
    source_id: str,
    source_identifier: str,
    source_url: str,
    parser: Callable[[str | bytes], list[dict[str, object]]],
) -> FetchOutcome:
    try:
        payload = fetcher(start_year, end_year)
    except OSError as exc:  # pragma: no cover - exercised via integration degradation
        LOGGER.error(
            "%s fetch failed; geopolitical candidates skipped: %s", source_id, exc
        )
        return FetchOutcome(
            rows=[],
            status=SourceFetchStatus(
                source_id=source_id,
                status="failed",
                error=str(exc),
                attempted_fetches=1,
                failed_fetches=1,
            ),
        )
    if payload is None:
        return FetchOutcome(
            rows=[],
            status=SourceFetchStatus(
                source_id=source_id, status="skipped", attempted_fetches=0
            ),
        )
    record_payload(
        store,
        run_id,
        source_id,
        source_identifier,
        payload,
        f"{source_id} geopolitical event data",
    )
    try:
        rows = [
            row
            for row in parser(payload, source_url=source_url)
            if start_year <= row["date"].year <= end_year
        ]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.error(
            "%s parse failed; geopolitical candidates skipped: %s", source_id, exc
        )
        return FetchOutcome(
            rows=[],
            status=SourceFetchStatus(
                source_id=source_id,
                status="failed",
                error=str(exc),
                attempted_fetches=1,
                failed_fetches=1,
                empty_payload=is_empty_payload(payload),
            ),
        )
    return FetchOutcome(
        rows=rows,
        status=SourceFetchStatus(
            source_id=source_id,
            status="ok" if rows else "empty",
            rows=len(rows),
            attempted_fetches=1,
            empty_payload=is_empty_payload(payload),
        ),
    )


def record_payload(
    store: AcquisitionStore | None,
    run_id: int | None,
    source_name: str,
    source_identifier: str,
    payload: str | bytes,
    notes: str,
) -> None:
    if store is None or run_id is None:
        return
    text = (
        payload.decode("utf-8", errors="replace")
        if isinstance(payload, bytes)
        else payload
    )
    store.record_text_artifact(
        run_id=run_id,
        source_name=source_name,
        artifact_kind="text",
        source_identifier=source_identifier,
        content_text=text,
        calendar_assumption="calendar day geopolitical signal",
        timezone="UTC",
        license_note="Public geopolitical-risk signal source",
        notes=notes,
    )


def is_empty_payload(payload: str | bytes) -> bool:
    return not payload.strip()


def _acled_access_token() -> str | None:
    token = os.environ.get("ACLED_API_TOKEN", "").strip()
    if token:
        return token
    username = os.environ.get("ACLED_USERNAME", "").strip()
    password = os.environ.get("ACLED_PASSWORD", "").strip()
    if not username or not password:
        return None
    body = urlencode(
        {
            "username": username,
            "password": password,
            "grant_type": "password",
            "client_id": "acled",
            "scope": "authenticated",
        }
    ).encode()
    request = Request(
        ACLED_TOKEN_URL,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": HTTP_USER_AGENT,
        },
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["access_token"])


def _fetch_paged_json(
    base_url: str,
    *,
    headers: dict[str, str],
    result_key: str,
    extra_params: dict[str, str],
) -> str:
    records: list[dict[str, object]] = []
    page = 1
    while True:
        params = {
            **extra_params,
            "pagesize": str(CONFLICT_API_PAGE_SIZE),
            "limit": str(CONFLICT_API_PAGE_SIZE),
            "page": str(page),
            "offset": str((page - 1) * CONFLICT_API_PAGE_SIZE),
        }
        payload = json.loads(
            _http_text(f"{base_url}?{urlencode(params)}", headers=headers)
        )
        page_records = payload.get(result_key, []) if isinstance(payload, dict) else []
        if not isinstance(page_records, list):
            break
        records.extend(record for record in page_records if isinstance(record, dict))
        total_count = _payload_total_count(payload)
        if len(page_records) < CONFLICT_API_PAGE_SIZE:
            if total_count is not None and len(records) < total_count:
                raise RuntimeError(
                    f"{base_url} returned short page before TotalCount was satisfied: "
                    f"records={len(records)} total_count={total_count} page={page}"
                )
            break
        if total_count is not None and len(records) >= total_count:
            break
        page += 1
    return json.dumps({result_key: records}, sort_keys=True)


def _payload_total_count(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in ("TotalCount", "total_count", "count"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _http_text(url: str, *, headers: dict[str, str]) -> str:
    request = Request(
        url, headers={"User-Agent": HTTP_USER_AGENT, **headers}
    )
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _http_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": HTTP_USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read()
