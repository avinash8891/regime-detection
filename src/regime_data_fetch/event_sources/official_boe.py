from __future__ import annotations

import datetime as dt
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlencode

from regime_data_fetch._http import fetch_text
from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources._common import (
    BOE_BASE_URL,
    MONTHS,
    FetchTextResult,
    absolute_url,
    fetch_text_result,
    strip_tags,
)
from regime_data_fetch.event_sources.models import EventCandidate

SOURCE_ID = "bankofengland.co.uk:mpc-decisions"
UPCOMING_MPC_URL = "https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates"
NEWS_SITEMAP_URL = "https://www.bankofengland.co.uk/sitemap/news"
NEWS_API_URL = "https://www.bankofengland.co.uk/_api/News/RefreshPagedNewsList"
NEWS_DATASOURCE_ID = "{CE377CC8-BFBC-418B-B4D9-DBC1C64774A8}"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarkupFetchStatus:
    url: str
    status: str
    error: str | None = None
    bytes_read: int = 0


class OfficialBOEAdapter:
    source_id = SOURCE_ID

    def __init__(
        self,
        *,
        as_of_date: dt.date | None = None,
        text_fetcher: Callable[[str], str] | None = None,
        result_fetcher: Callable[[str], FetchTextResult] = fetch_text_result,
        news_api_fetcher: Callable[[int], str] | None = None,
        stop_on_empty_news_page: bool | None = None,
    ) -> None:
        self.as_of_date = as_of_date or dt.date.today()
        self.text_fetcher = text_fetcher
        self.result_fetcher = result_fetcher
        self._stop_on_empty_news_page = (
            news_api_fetcher is not None
            if stop_on_empty_news_page is None
            else stop_on_empty_news_page
        )
        self.news_api_fetcher = news_api_fetcher or fetch_boe_news_api_page
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
        html = self._fetch_text(UPCOMING_MPC_URL)
        _record_html(store, run_id, UPCOMING_MPC_URL, html, "BoE upcoming MPC dates")
        candidates = parse_boe_upcoming_mpc_dates(html, as_of_date=self.as_of_date)
        sitemap_html = self._fetch_text(NEWS_SITEMAP_URL)
        _record_html(
            store,
            run_id,
            NEWS_SITEMAP_URL,
            sitemap_html,
            "BoE news sitemap for MPC dates pages",
        )
        for dates_url in _mpc_dates_page_urls(
            sitemap_html, start_year=start_year, end_year=end_year
        ):
            dates_html = self._fetch_text(dates_url)
            _record_html(
                store, run_id, dates_url, dates_html, "BoE annual MPC dates page"
            )
            candidates.extend(
                parse_boe_mpc_dates_page(
                    dates_html, source_url=dates_url, as_of_date=self.as_of_date
                )
            )
        for page in range(1, 30):
            page_url = f"{NEWS_API_URL}?page={page}"
            payload = self.news_api_fetcher(page)
            _record_html(
                store,
                run_id,
                page_url,
                payload,
                "BoE news API MPC archive page",
            )
            results_html = _extract_results_html(payload)
            if _payload_is_non_json(payload):
                message = "BoE search results not JSON; falling back to raw HTML parse"
                LOGGER.warning("%s for %s", message, page_url)
                self._record_markup_status(
                    page_url,
                    "partial",
                    error=message,
                    bytes_read=len(payload.encode("utf-8")),
                )
            page_candidates = parse_boe_news_api_results(
                results_html, as_of_date=self.as_of_date
            )
            candidates.extend(page_candidates)
            if (
                page_candidates
                and min(candidate.date.year for candidate in page_candidates)
                < start_year
            ):
                break
            if not page_candidates and self._stop_on_empty_news_page:
                break
        if any(
            status.status in {"failed", "partial"}
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


def fetch_boe_news_api_page(page: int) -> str:
    data = urlencode(
        {
            "SearchTerm": "Monetary Policy Summary and Minutes",
            "Id": NEWS_DATASOURCE_ID,
            "PageSize": "100",
            "Page": str(page),
            "Direction": "Latest",
        }
    ).encode("utf-8")
    return fetch_text(
        NEWS_API_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
        timeout=30,
    )


def parse_boe_upcoming_mpc_dates(
    html: str, *, as_of_date: dt.date
) -> list[EventCandidate]:
    candidates: list[EventCandidate] = []
    section_pattern = re.compile(
        r"<h2>\s*(?P<year>20\d{2})\s+(?:confirmed|provisional)\s+dates\s*</h2>(?P<section>.*?)(?=<h2>|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    row_pattern = re.compile(
        r"<tr[^>]*>(?P<row>.*?)</tr>", flags=re.IGNORECASE | re.DOTALL
    )
    cell_pattern = re.compile(
        r"<td[^>]*>(?P<cell>.*?)</td>", flags=re.IGNORECASE | re.DOTALL
    )
    date_pattern = re.compile(
        r"(?:Monday|Tuesday|Wednesday|Thursday|Friday)?\s*(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)",
        flags=re.IGNORECASE,
    )
    href_pattern = re.compile(r"href=\"(?P<href>[^\"]+)\"", flags=re.IGNORECASE)
    for section_match in section_pattern.finditer(html):
        year = int(section_match.group("year"))
        for row_match in row_pattern.finditer(section_match.group("section")):
            cells = cell_pattern.findall(row_match.group("row"))
            if len(cells) < 2:
                continue
            date_text = strip_tags(cells[0])
            date_match = date_pattern.search(date_text)
            if date_match is None:
                continue
            month = MONTHS[date_match.group("month").lower()]
            event_date = dt.date(year, month, int(date_match.group("day")))
            href_match = href_pattern.search(cells[1])
            title = strip_tags(cells[1])
            candidates.append(
                _candidate(
                    event_date,
                    as_of_date,
                    absolute_url(
                        BOE_BASE_URL, href_match.group("href") if href_match else None
                    ),
                    title,
                    date_text,
                )
            )
    if not candidates:
        candidates.extend(_parse_legacy_text_dates(html, as_of_date=as_of_date))
    return sorted(candidates, key=lambda candidate: candidate.date)


def parse_boe_mpc_dates_page(
    html: str, *, source_url: str, as_of_date: dt.date
) -> list[EventCandidate]:
    candidates: list[EventCandidate] = []
    row_pattern = re.compile(
        r"<tr[^>]*>(?P<row>.*?)</tr>", flags=re.IGNORECASE | re.DOTALL
    )
    cell_pattern = re.compile(
        r"<td[^>]*>(?P<cell>.*?)</td>", flags=re.IGNORECASE | re.DOTALL
    )
    full_date_pattern = re.compile(
        r"(?P<day>\d{1,2})\s+"
        r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+"
        r"(?P<year>20\d{2})",
        flags=re.IGNORECASE,
    )
    month_year_pattern = re.compile(
        r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+"
        r"(?P<year>20\d{2}).*?(?P<day>\d{1,2})",
        flags=re.IGNORECASE,
    )
    month_year_cell_pattern = re.compile(
        r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+"
        r"(?P<year>20\d{2})",
        flags=re.IGNORECASE,
    )
    day_cell_pattern = re.compile(
        r"(?:Monday|Tuesday|Wednesday|Thursday|Friday)?\s*(?P<day>\d{1,2})\b",
        flags=re.IGNORECASE,
    )
    section_pattern = re.compile(
        r"<h[23][^>]*>\s*(?P<year>20\d{2})\s+(?:confirmed|provisional|MPC)?\s*dates\s*</h[23]>(?P<section>.*?)(?=<h[23]|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    day_month_pattern = re.compile(
        r"(?:Monday|Tuesday|Wednesday|Thursday|Friday)?\s*(?P<day>\d{1,2})\s+"
        r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\b",
        flags=re.IGNORECASE,
    )
    for section_match in section_pattern.finditer(html):
        year = int(section_match.group("year"))
        for row_match in row_pattern.finditer(section_match.group("section")):
            cells = cell_pattern.findall(row_match.group("row"))
            if not cells:
                continue
            first_cell = strip_tags(cells[0])
            if (
                "no meeting" in first_cell.lower()
                or "publication" in first_cell.lower()
                or re.search(r"20\d{2}", first_cell)
            ):
                continue
            date_match = day_month_pattern.search(first_cell)
            if date_match is None:
                continue
            month = MONTHS[date_match.group("month").lower()]
            event_date = dt.date(year, month, int(date_match.group("day")))
            candidates.append(
                _candidate(
                    event_date,
                    as_of_date,
                    source_url,
                    "MPC Announcement and Minutes publication",
                    first_cell,
                )
            )
    for row_match in row_pattern.finditer(html):
        cells = cell_pattern.findall(row_match.group("row"))
        if not cells:
            continue
        first_cell = strip_tags(cells[0])
        if "no meeting" in first_cell.lower() or "publication" in first_cell.lower():
            continue
        date_match = full_date_pattern.search(first_cell)
        if date_match is None:
            date_match = month_year_pattern.search(first_cell)
        if date_match is not None:
            month = MONTHS[date_match.group("month").lower()]
            event_date = dt.date(
                int(date_match.group("year")), month, int(date_match.group("day"))
            )
        elif len(cells) >= 2:
            second_cell = strip_tags(cells[1])
            month_year_match = month_year_cell_pattern.search(first_cell)
            day_match = day_cell_pattern.search(second_cell)
            if (
                month_year_match is None
                or day_match is None
                or "no meeting" in second_cell.lower()
            ):
                continue
            month = MONTHS[month_year_match.group("month").lower()]
            event_date = dt.date(
                int(month_year_match.group("year")), month, int(day_match.group("day"))
            )
        else:
            continue
        candidates.append(
            _candidate(
                event_date,
                as_of_date,
                source_url,
                "MPC Announcement and Minutes publication",
                first_cell,
            )
        )
    deduped = {candidate.date: candidate for candidate in candidates}
    return [deduped[date] for date in sorted(deduped)]


def _parse_legacy_text_dates(html: str, *, as_of_date: dt.date) -> list[EventCandidate]:
    text = strip_tags(html)
    candidates: list[EventCandidate] = []
    current_year: int | None = None
    tokens = re.split(
        r"(?=(?:20\d{2})\s+(?:confirmed|provisional)\s+dates)|(?=(?:Monday|Tuesday|Wednesday|Thursday|Friday)\s+\d{1,2}\s+[A-Za-z]+)",
        text,
    )
    date_pattern = re.compile(
        r"(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday)\s+)?(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+).*?(?:MPC|Monetary Policy)",
        flags=re.IGNORECASE,
    )
    for token in tokens:
        year_match = re.search(
            r"\b(?P<year>20\d{2})\s+(?:confirmed|provisional)\s+dates\b",
            token,
            flags=re.IGNORECASE,
        )
        if year_match:
            current_year = int(year_match.group("year"))
        if current_year is None:
            continue
        date_match = date_pattern.search(token)
        if date_match is None:
            continue
        month = MONTHS.get(date_match.group("month").lower())
        if month is None:
            continue
        event_date = dt.date(current_year, month, int(date_match.group("day")))
        candidates.append(
            _candidate(
                event_date,
                as_of_date,
                UPCOMING_MPC_URL,
                "MPC Summary and minutes",
                token.strip(),
            )
        )
    return candidates


def parse_boe_news_api_results(
    results_html: str, *, as_of_date: dt.date
) -> list[EventCandidate]:
    candidates: list[EventCandidate] = []
    release_pattern = re.compile(
        r"<a[^>]+href=\"(?P<href>[^\"]*monetary-policy-summary-and-minutes[^\"]*)\"[^>]*>(?P<body>.*?)</a>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    date_pattern = re.compile(
        r"datetime=\"(?P<date>\d{4}-\d{2}-\d{2})\"", flags=re.IGNORECASE
    )
    title_pattern = re.compile(
        r"<h3[^>]*class=\"[^\"]*list[^\"]*\"[^>]*>(?P<title>.*?)</h3>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for release_match in release_pattern.finditer(results_html):
        body = release_match.group("body")
        date_match = date_pattern.search(body)
        title_match = title_pattern.search(body)
        if date_match is None or title_match is None:
            continue
        title = strip_tags(title_match.group("title"))
        if "monetary policy summary and minutes" not in title.lower():
            continue
        event_date = dt.date.fromisoformat(date_match.group("date"))
        candidates.append(
            _candidate(
                event_date,
                as_of_date,
                absolute_url(BOE_BASE_URL, release_match.group("href")),
                title,
                title,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.date)


def _candidate(
    event_date: dt.date,
    as_of_date: dt.date,
    source_url: str | None,
    raw_title: str,
    raw_snippet: str,
) -> EventCandidate:
    return EventCandidate(
        date=event_date,
        event_type="BOE_decision",
        market="GLOBAL",
        importance="high",
        source_id=SOURCE_ID,
        source_url=source_url,
        raw_title=raw_title,
        raw_snippet=raw_snippet,
        is_future_scheduled=event_date > as_of_date,
        confidence="medium" if event_date > as_of_date else "high",
        requires_manual_review=False,
    )


def _extract_results_html(payload: str) -> str:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return payload
    return str(parsed.get("Results", ""))


def _payload_is_non_json(payload: str) -> bool:
    try:
        json.loads(payload)
    except json.JSONDecodeError:
        return True
    return False


def _mpc_dates_page_urls(html: str, *, start_year: int, end_year: int) -> list[str]:
    urls: list[str] = []
    link_pattern = re.compile(
        r"<a[^>]+href=\"(?P<href>[^\"]+)\"[^>]*>(?P<title>.*?)</a>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in link_pattern.finditer(html):
        href = match.group("href")
        title = strip_tags(match.group("title"))
        combined = f"{href} {title}".lower()
        if (
            "monetary policy committee dates" not in combined
            and "mpc-dates" not in combined
            and "mpc-publication-dates" not in combined
            and "mpc-announcement-dates" not in combined
        ):
            continue
        years = [int(value) for value in re.findall(r"20\d{2}", combined)]
        if not any(start_year <= year <= end_year for year in years):
            continue
        if href.lower().endswith(".pdf"):
            continue
        url = absolute_url(BOE_BASE_URL, href)
        if url and url not in urls:
            urls.append(url)
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
        license_note="Bank of England public webpage/API",
        notes=notes,
    )
