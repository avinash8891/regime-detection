from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources._common import BOE_BASE_URL, MONTHS, absolute_url, fetch_text_url, strip_tags
from regime_data_fetch.event_sources.models import EventCandidate

SOURCE_ID = "bankofengland.co.uk:mpc-decisions"
UPCOMING_MPC_URL = "https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates"
NEWS_API_URL = "https://www.bankofengland.co.uk/_api/News/RefreshPagedNewsList"
NEWS_DATASOURCE_ID = "{CE377CC8-BFBC-418B-B4D9-DBC1C64774A8}"


class OfficialBOEAdapter:
    source_id = SOURCE_ID

    def __init__(
        self,
        *,
        as_of_date: dt.date | None = None,
        text_fetcher: Callable[[str], str] = fetch_text_url,
        news_api_fetcher: Callable[[int], str] | None = None,
    ) -> None:
        self.as_of_date = as_of_date or dt.date.today()
        self.text_fetcher = text_fetcher
        self.news_api_fetcher = news_api_fetcher or fetch_boe_news_api_page

    def fetch(
        self,
        *,
        start_year: int,
        end_year: int,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[EventCandidate]:
        html = self.text_fetcher(UPCOMING_MPC_URL)
        _record_html(store, run_id, UPCOMING_MPC_URL, html, "BoE upcoming MPC dates")
        candidates = parse_boe_upcoming_mpc_dates(html, as_of_date=self.as_of_date)
        for page in range(1, 30):
            payload = self.news_api_fetcher(page)
            _record_html(store, run_id, f"{NEWS_API_URL}?page={page}", payload, "BoE news API MPC archive page")
            page_candidates = parse_boe_news_api_results(_extract_results_html(payload), as_of_date=self.as_of_date)
            candidates.extend(page_candidates)
            if page_candidates and min(candidate.date.year for candidate in page_candidates) < start_year:
                break
            if not page_candidates and page > 3:
                break
        return _dedupe(candidates, start_year=start_year, end_year=end_year)


def fetch_boe_news_api_page(page: int) -> str:
    data = urlencode(
        {
            "SearchTerm": "",
            "Id": NEWS_DATASOURCE_ID,
            "PageSize": "100",
            "Page": str(page),
            "Direction": "Latest",
        }
    ).encode("utf-8")
    request = Request(
        NEWS_API_URL,
        data=data,
        headers={
            "User-Agent": "regime-detection-event-fetch/1.0",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_boe_upcoming_mpc_dates(html: str, *, as_of_date: dt.date) -> list[EventCandidate]:
    candidates: list[EventCandidate] = []
    section_pattern = re.compile(
        r"<h2>\s*(?P<year>20\d{2})\s+(?:confirmed|provisional)\s+dates\s*</h2>(?P<section>.*?)(?=<h2>|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    row_pattern = re.compile(r"<tr[^>]*>(?P<row>.*?)</tr>", flags=re.IGNORECASE | re.DOTALL)
    cell_pattern = re.compile(r"<td[^>]*>(?P<cell>.*?)</td>", flags=re.IGNORECASE | re.DOTALL)
    date_pattern = re.compile(r"(?:Monday|Tuesday|Wednesday|Thursday|Friday)?\s*(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)", flags=re.IGNORECASE)
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
            candidates.append(_candidate(event_date, as_of_date, absolute_url(BOE_BASE_URL, href_match.group("href") if href_match else None), title, date_text))
    return sorted(candidates, key=lambda candidate: candidate.date)


def parse_boe_news_api_results(results_html: str, *, as_of_date: dt.date) -> list[EventCandidate]:
    candidates: list[EventCandidate] = []
    release_pattern = re.compile(
        r"<a[^>]+href=\"(?P<href>[^\"]*monetary-policy-summary-and-minutes[^\"]*)\"[^>]*>(?P<body>.*?)</a>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    date_pattern = re.compile(r"datetime=\"(?P<date>\d{4}-\d{2}-\d{2})\"", flags=re.IGNORECASE)
    title_pattern = re.compile(r"<h3[^>]*class=\"[^\"]*list[^\"]*\"[^>]*>(?P<title>.*?)</h3>", flags=re.IGNORECASE | re.DOTALL)
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


def _candidate(event_date: dt.date, as_of_date: dt.date, source_url: str | None, raw_title: str, raw_snippet: str) -> EventCandidate:
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


def _dedupe(candidates: list[EventCandidate], *, start_year: int, end_year: int) -> list[EventCandidate]:
    deduped = {(candidate.event_type, candidate.date): candidate for candidate in candidates if start_year <= candidate.date.year <= end_year}
    return [deduped[key] for key in sorted(deduped, key=lambda item: (item[1], item[0]))]


def _record_html(store: AcquisitionStore | None, run_id: int | None, url: str, html: str, notes: str) -> None:
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
