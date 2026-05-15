from __future__ import annotations

import datetime as dt
import html
import json
import logging
import re
import urllib.request
from collections.abc import Callable
from urllib.parse import urljoin
from typing import Any

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources.models import EventCandidate

LOGGER = logging.getLogger(__name__)
SOURCE_ID = "official-us-budget-discovery"
TREASURY_DEBT_LIMIT_URL = "https://home.treasury.gov/policy-issues/financial-markets-financial-institutions-and-fiscal-service/debt-limit"
DEFAULT_GOVINFO_PUBLIC_LAW_URLS = (
    "https://www.govinfo.gov/content/pkg/PLAW-118publ83/html/PLAW-118publ83.htm",
    "https://www.govinfo.gov/content/pkg/PLAW-118publ158/html/PLAW-118publ158.htm",
)
VALID_SUBTYPES = {"debt_ceiling", "shutdown", "cr_expiration"}
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
ANCHOR_RE = re.compile(r"<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
DATE_PARENS_RE = re.compile(r"\((\d{1,2})/(\d{1,2})/(\d{2,4})\)")
MONTH_DATE_RE = re.compile(
    r"\b("
    + "|".join(MONTHS)
    + r")\s+(\d{1,2}),\s+(\d{4})\b",
    re.IGNORECASE,
)
CR_TITLE_RE = re.compile(r"\b(CONTINUING APPROPRIATIONS[A-Z\s,-]*\d{4})\b", re.IGNORECASE)


class BudgetOfficialDiscoveryGenerator:
    source_id = SOURCE_ID

    def __init__(self, *, records_fetcher: Callable[[], str | list[dict[str, Any]]] | None = None, as_of_date: dt.date | None = None) -> None:
        self.records_fetcher = records_fetcher or fetch_official_budget_records
        self.as_of_date = as_of_date or dt.date.today()

    def generate(
        self,
        *,
        start_year: int,
        end_year: int,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[EventCandidate]:
        try:
            payload = self.records_fetcher()
        except Exception as exc:  # pragma: no cover - external degradation path
            LOGGER.error("official budget discovery failed; non-deterministic budget candidates skipped: %s", exc)
            return []
        _record_payload(store, run_id, payload)
        candidates = [
            _candidate(record, self.as_of_date)
            for record in parse_budget_official_records(payload)
            if start_year <= record["date"].year <= end_year
        ]
        return candidates


def parse_budget_official_records(payload: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_records = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(raw_records, list):
        raise ValueError("budget official discovery payload must be a list")
    records: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_records, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"budget official discovery record {idx} must be a mapping")
        event_subtype = str(raw.get("event_subtype", ""))
        if event_subtype not in VALID_SUBTYPES:
            raise ValueError(f"budget official discovery record {idx} has invalid event_subtype: {event_subtype}")
        source_id = str(raw.get("source_id", ""))
        if not source_id:
            raise ValueError(f"budget official discovery record {idx} missing source_id")
        records.append(
            {
                "date": dt.date.fromisoformat(str(raw["date"])),
                "event_subtype": event_subtype,
                "source_id": source_id,
                "source_url": str(raw.get("source_url", "")) or None,
                "raw_title": str(raw.get("raw_title", event_subtype)),
                "raw_snippet": str(raw.get("raw_snippet", "")) or None,
            }
        )
    return records


def _candidate(record: dict[str, Any], as_of_date: dt.date) -> EventCandidate:
    return EventCandidate(
        date=record["date"],
        event_type="budget",
        market="US",
        importance="high",
        source_id=record["source_id"],
        source_url=record["source_url"],
        raw_title=record["raw_title"],
        raw_snippet=record["raw_snippet"],
        is_future_scheduled=record["date"] > as_of_date,
        confidence="medium",
        requires_manual_review=True,
        event_subtype=record["event_subtype"],
    )


def _record_payload(store: AcquisitionStore | None, run_id: int | None, payload: object) -> None:
    if store is None or run_id is None:
        return
    content = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True)
    store.record_text_artifact(
        run_id=run_id,
        source_name=SOURCE_ID,
        artifact_kind="json",
        source_identifier="official_budget_records",
        content_text=content,
        calendar_assumption="US fiscal event dates",
        timezone="America/New_York",
        license_note="Official US government public-domain source records",
        notes="Official budget event discovery records",
    )


def fetch_official_budget_records(
    *,
    text_fetcher: Callable[[str], str] | None = None,
    govinfo_public_law_urls: list[str] | tuple[str, ...] = DEFAULT_GOVINFO_PUBLIC_LAW_URLS,
) -> list[dict[str, Any]]:
    fetcher = text_fetcher or _fetch_text
    records: list[dict[str, Any]] = []

    try:
        treasury_html = fetcher(TREASURY_DEBT_LIMIT_URL)
        records.extend(extract_treasury_debt_limit_records(treasury_html, source_url=TREASURY_DEBT_LIMIT_URL))
    except Exception as exc:  # pragma: no cover - external degradation path
        LOGGER.error("Treasury debt-limit discovery failed; debt-ceiling candidates skipped: %s", exc)

    for url in govinfo_public_law_urls:
        try:
            public_law_text = fetcher(url)
            records.extend(extract_govinfo_cr_records(public_law_text, source_url=url))
        except Exception as exc:  # pragma: no cover - external degradation path
            LOGGER.error("GovInfo CR discovery failed for %s; CR candidates skipped: %s", url, exc)

    return _dedupe_records(records)


def extract_treasury_debt_limit_records(index_html: str, *, source_url: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for match in ANCHOR_RE.finditer(index_html):
        href = html.unescape(match.group(1))
        title = _plain_text(match.group(2))
        normalized_title = title.lower()
        if "debt limit" not in normalized_title and "disp ending" not in normalized_title:
            continue

        event_date = _parse_disp_ending_date(title) or _parse_parenthesized_date(title)
        if event_date is None:
            continue

        if "disp ending" in normalized_title:
            snippet = f"Treasury debt-limit DISP period ending {event_date.isoformat()}."
        else:
            snippet = f"Treasury debt-limit notice dated {event_date.isoformat()}."
        records.append(
            {
                "date": event_date.isoformat(),
                "event_subtype": "debt_ceiling",
                "source_id": "treasury.gov:debt-limit",
                "source_url": urljoin(source_url, href),
                "raw_title": title,
                "raw_snippet": snippet,
            }
        )
    return _dedupe_records(records)


def extract_govinfo_cr_records(public_law_text: str, *, source_url: str) -> list[dict[str, Any]]:
    plain = _plain_text(public_law_text)
    if "CONTINUING APPROPRIATIONS" not in plain.upper():
        return []
    event_date = _parse_cr_expiration_date(plain)
    if event_date is None:
        return []
    return [
        {
            "date": event_date.isoformat(),
            "event_subtype": "cr_expiration",
            "source_id": "govinfo.gov:public-law",
            "source_url": source_url,
            "raw_title": _parse_cr_title(plain),
            "raw_snippet": f"GovInfo continuing appropriations expiration date {event_date.isoformat()}.",
        }
    ]


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "regime-detection-event-calendar/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", "replace")


def _plain_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(without_tags).split())


def _parse_parenthesized_date(value: str) -> dt.date | None:
    match = DATE_PARENS_RE.search(value)
    if not match:
        return None
    month, day, year_text = match.groups()
    year = int(year_text)
    if year < 100:
        year += 2000
    return dt.date(year, int(month), int(day))


def _parse_disp_ending_date(value: str) -> dt.date | None:
    match = re.search(r"DISP ending\s+([^()]+)", value, re.IGNORECASE)
    if not match:
        return None
    return _parse_month_date(match.group(1))


def _parse_cr_expiration_date(value: str) -> dt.date | None:
    explicit_note = re.search(r"Expiration date[^A-Za-z]+([A-Za-z]+\s+\d{1,2},\s+\d{4})", value, re.IGNORECASE)
    if explicit_note:
        return _parse_month_date(explicit_note.group(1))
    section_106_amendment = re.search(
        r"date specified in section\s+106\(3\).*?inserting\s+[`'\"]*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        value,
        re.IGNORECASE,
    )
    if section_106_amendment:
        return _parse_month_date(section_106_amendment.group(1))
    return None


def _parse_month_date(value: str) -> dt.date | None:
    match = MONTH_DATE_RE.search(value)
    if not match:
        return None
    month_name, day, year = match.groups()
    return dt.date(int(year), MONTHS[month_name.lower()], int(day))


def _parse_cr_title(value: str) -> str:
    match = CR_TITLE_RE.search(value)
    if not match:
        return "Continuing appropriations public law"
    return " ".join(match.group(1).split())


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str | None]] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        key = (str(record["date"]), str(record["event_subtype"]), str(record["source_id"]), record.get("source_url"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped
