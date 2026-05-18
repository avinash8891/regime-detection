from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources._common import (
    ECB_BASE_URL,
    FetchTextResult,
    absolute_url,
    fetch_text_result,
    strip_tags,
)
from regime_data_fetch.event_sources.models import EventCandidate

SOURCE_ID = "ecb.europa.eu:monetary-policy-decisions"
ARCHIVE_INDEX_URL = "https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html"
CURRENT_CALENDAR_URL = (
    "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html"
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarkupFetchStatus:
    url: str
    status: str
    error: str | None = None
    bytes_read: int = 0


class OfficialECBAdapter:
    source_id = SOURCE_ID

    def __init__(
        self,
        *,
        as_of_date: dt.date | None = None,
        text_fetcher: Callable[[str], str] | None = None,
        result_fetcher: Callable[[str], FetchTextResult] = fetch_text_result,
    ) -> None:
        self.as_of_date = as_of_date or dt.date.today()
        self.text_fetcher = text_fetcher
        self.result_fetcher = result_fetcher
        self.last_source_statuses: dict[str, MarkupFetchStatus] = {}
        self.last_run_status = "not_run"

    def fetch(
        self,
        *,
        start_year: int,
        end_year: int,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[EventCandidate]:
        self.last_source_statuses = {}
        self.last_run_status = "ok"
        html = self._fetch_text(ARCHIVE_INDEX_URL)
        _record_html(
            store,
            run_id,
            ARCHIVE_INDEX_URL,
            html,
            "ECB monetary-policy decisions archive index",
        )
        candidates: list[EventCandidate] = []
        snippet_urls = _archive_snippet_urls(
            html, start_year=start_year, end_year=end_year
        )
        if (
            html.strip()
            and not snippet_urls
            and self.last_source_statuses[ARCHIVE_INDEX_URL].status == "ok"
        ):
            self._record_markup_status(
                ARCHIVE_INDEX_URL,
                "parser_layout_drift",
                error="ECB archive index missing data-snippets attribute",
                bytes_read=len(html.encode("utf-8")),
            )
            LOGGER.error(
                "ECB archive index parser layout drift; data-snippets attribute missing from %s",
                ARCHIVE_INDEX_URL,
            )
        for snippet_url in snippet_urls:
            snippet = self._fetch_text(snippet_url)
            _record_html(
                store,
                run_id,
                snippet_url,
                snippet,
                "ECB monetary-policy decisions yearly archive snippet",
            )
            candidates.extend(
                parse_ecb_decision_archive(snippet, as_of_date=self.as_of_date)
            )

        calendar_html = self._fetch_text(CURRENT_CALENDAR_URL)
        _record_html(
            store,
            run_id,
            CURRENT_CALENDAR_URL,
            calendar_html,
            "ECB Governing Council current calendar",
        )
        candidates.extend(
            parse_ecb_current_calendar(calendar_html, as_of_date=self.as_of_date)
        )
        if any(
            status.status in {"failed", "partial", "parser_layout_drift"}
            for status in self.last_source_statuses.values()
        ):
            self.last_run_status = "partial"
        return _dedupe(candidates, start_year=start_year, end_year=end_year)

    def _fetch_text(self, url: str) -> str:
        if self.text_fetcher is not None:
            text = self.text_fetcher(url)
            self._record_markup_status(
                url, "ok" if text else "empty", bytes_read=len(text.encode("utf-8"))
            )
            return text
        result = self.result_fetcher(url)
        if not result.ok:
            self._record_markup_status(url, "failed", error=result.error)
            return ""
        text = result.text or ""
        self._record_markup_status(
            url, "ok" if text else "empty", bytes_read=len(text.encode("utf-8"))
        )
        return text

    def _record_markup_status(
        self,
        url: str,
        status: str,
        *,
        error: str | None = None,
        bytes_read: int = 0,
    ) -> None:
        self.last_source_statuses[url] = MarkupFetchStatus(
            url=url,
            status=status,
            error=error,
            bytes_read=bytes_read,
        )


def parse_ecb_decision_archive(
    html: str, *, as_of_date: dt.date
) -> list[EventCandidate]:
    candidates: list[EventCandidate] = []
    pattern = re.compile(
        r"<dt[^>]*isoDate=\"(?P<date>\d{4}-\d{2}-\d{2})\"[^>]*>.*?</dt>\s*"
        r"<dd[^>]*>.*?<a[^>]*href=\"(?P<href>[^\"]+)\"[^>]*>(?P<title>.*?)</a>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(html):
        title = strip_tags(match.group("title"))
        if "monetary policy decisions" not in title.lower():
            continue
        event_date = dt.date.fromisoformat(match.group("date"))
        candidates.append(
            _candidate(
                event_date=event_date,
                as_of_date=as_of_date,
                source_url=absolute_url(ECB_BASE_URL, match.group("href")),
                raw_title=title,
                raw_snippet=title,
                confidence="high",
            )
        )
    return candidates


def parse_ecb_current_calendar(
    html: str, *, as_of_date: dt.date
) -> list[EventCandidate]:
    candidates: list[EventCandidate] = []
    row_pattern = re.compile(
        r"<dt[^>]*>\s*(?P<date>\d{2}/\d{2}/\d{4})\s*</dt>\s*<dd[^>]*>(?P<description>.*?)</dd>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in row_pattern.finditer(html):
        description = strip_tags(match.group("description"))
        normalized = description.lower()
        if "non-monetary" in normalized or "monetary policy meeting" not in normalized:
            continue
        if "day 2" not in normalized and "press conference" not in normalized:
            continue
        day, month, year = (int(part) for part in match.group("date").split("/"))
        event_date = dt.date(year, month, day)
        candidates.append(
            _candidate(
                event_date=event_date,
                as_of_date=as_of_date,
                source_url=CURRENT_CALENDAR_URL,
                raw_title="ECB Governing Council monetary policy meeting",
                raw_snippet=description,
                confidence="medium" if event_date > as_of_date else "high",
            )
        )
    return candidates


def _candidate(
    *,
    event_date: dt.date,
    as_of_date: dt.date,
    source_url: str | None,
    raw_title: str,
    raw_snippet: str,
    confidence: str,
) -> EventCandidate:
    return EventCandidate(
        date=event_date,
        event_type="ECB_decision",
        market="GLOBAL",
        importance="high",
        source_id=SOURCE_ID,
        source_url=source_url,
        raw_title=raw_title,
        raw_snippet=raw_snippet,
        is_future_scheduled=event_date > as_of_date,
        confidence=confidence,  # type: ignore[arg-type]
        requires_manual_review=False,
    )


def _archive_snippet_urls(html: str, *, start_year: int, end_year: int) -> list[str]:
    match = re.search(r"data-snippets='(?P<snippets>[^']+)'", html)
    if match is None:
        return []
    urls: list[str] = []
    for raw in match.group("snippets").split(","):
        year_match = re.search(r"/(?P<year>20\d{2})/html/", raw)
        if year_match is None:
            continue
        year = int(year_match.group("year"))
        if start_year <= year <= end_year:
            urls.append(
                absolute_url(f"{ECB_BASE_URL}/press/govcdec/mopo/html/", raw) or raw
            )
    return urls


def _dedupe(
    candidates: list[EventCandidate], *, start_year: int, end_year: int
) -> list[EventCandidate]:
    deduped = {
        (candidate.event_type, candidate.date): candidate
        for candidate in candidates
        if start_year <= candidate.date.year <= end_year
    }
    return [
        deduped[key] for key in sorted(deduped, key=lambda item: (item[1], item[0]))
    ]


def _record_html(
    store: AcquisitionStore | None, run_id: int | None, url: str, html: str, notes: str
) -> None:
    if store is None or run_id is None:
        return
    store.record_text_artifact(
        run_id=run_id,
        source_name=SOURCE_ID,
        artifact_kind="html",
        source_identifier=url,
        content_text=html,
        calendar_assumption="NYSE trading calendar",
        timezone="America/New_York",
        license_note="ECB public webpage",
        notes=notes,
    )
