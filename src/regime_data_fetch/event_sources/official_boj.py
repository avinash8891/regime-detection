from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources._common import BOJ_BASE_URL, MONTHS, FetchTextResult, absolute_url, fetch_text_result, strip_tags
from regime_data_fetch.event_sources.models import EventCandidate

SOURCE_ID = "boj.or.jp:monetary-policy-meetings"
CURRENT_URL = "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm"
PAST_URL = "https://www.boj.or.jp/en/mopo/mpmsche_minu/past.htm"


class OfficialBOJAdapter:
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

    def fetch(
        self,
        *,
        start_year: int,
        end_year: int,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[EventCandidate]:
        candidates: list[EventCandidate] = []
        for url, notes in [(CURRENT_URL, "BoJ current MPM schedule"), (PAST_URL, "BoJ past MPM schedule")]:
            html = self._fetch_text(url)
            _record_html(store, run_id, url, html, notes)
            candidates.extend(parse_boj_mpm_dates(html, as_of_date=self.as_of_date))
        return _dedupe(candidates, start_year=start_year, end_year=end_year)

    def _fetch_text(self, url: str) -> str:
        if self.text_fetcher is not None:
            return self.text_fetcher(url)
        result = self.result_fetcher(url)
        return result.text if result.ok and result.text is not None else ""


def parse_boj_mpm_dates(html: str, *, as_of_date: dt.date) -> list[EventCandidate]:
    candidates: list[EventCandidate] = []
    section_pattern = re.compile(
        r"<h2[^>]*(?:id=\"p(?P<idyear>20\d{2})\")?[^>]*>\s*(?P<year>20\d{2})\s*</h2>(?P<section>.*?)(?=<h2[^>]*>\s*20\d{2}\s*</h2>|<h2[^>]*id=\"p20\d{2}\"|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    row_pattern = re.compile(r"<tr[^>]*>(?P<row>.*?)</tr>", flags=re.IGNORECASE | re.DOTALL)
    cell_pattern = re.compile(r"<td[^>]*>(?P<cell>.*?)</td>", flags=re.IGNORECASE | re.DOTALL)
    href_pattern = re.compile(r"href=\"(?P<href>[^\"]+)\"", flags=re.IGNORECASE)
    for section_match in section_pattern.finditer(html):
        year = int(section_match.group("year"))
        for row_match in row_pattern.finditer(section_match.group("section")):
            cell_match = cell_pattern.search(row_match.group("row"))
            if cell_match is None:
                continue
            cell_html = cell_match.group("cell")
            cell_text = strip_tags(cell_html)
            event_date = _parse_mpm_date(cell_text, year)
            if event_date is None:
                continue
            href_match = href_pattern.search(cell_html)
            candidates.append(
                EventCandidate(
                    date=event_date,
                    event_type="BOJ_decision",
                    market="GLOBAL",
                    importance="high",
                    source_id=SOURCE_ID,
                    source_url=absolute_url(BOJ_BASE_URL, href_match.group("href") if href_match else CURRENT_URL),
                    raw_title="Bank of Japan Monetary Policy Meeting",
                    raw_snippet=cell_text,
                    is_future_scheduled=event_date > as_of_date,
                    confidence="medium" if event_date > as_of_date else "high",
                    requires_manual_review=False,
                )
            )
    if not candidates:
        candidates.extend(_parse_legacy_inline_dates(html, as_of_date=as_of_date))
    return sorted(candidates, key=lambda candidate: candidate.date)


def _parse_mpm_date(text: str, year: int) -> dt.date | None:
    pattern = re.compile(
        r"(?P<month1>Jan\.?|January|Feb\.?|February|Mar\.?|March|Apr\.?|April|May|June|July|Aug\.?|August|Sep\.?|Sept\.?|September|Oct\.?|October|Nov\.?|November|Dec\.?|December)\s+"
        r"(?P<day1>\d{1,2})(?:\s*\([^)]+\))?"
        r"(?:\s*,\s*(?:(?P<month2>Jan\.?|January|Feb\.?|February|Mar\.?|March|Apr\.?|April|May|June|July|Aug\.?|August|Sep\.?|Sept\.?|September|Oct\.?|October|Nov\.?|November|Dec\.?|December)\s+)?(?P<day2>\d{1,2})(?:\s*\([^)]+\))?)?",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        return None
    month_name = (match.group("month2") or match.group("month1")).lower()
    day = int(match.group("day2") or match.group("day1"))
    return dt.date(year, MONTHS[month_name], day)


def _parse_legacy_inline_dates(html: str, *, as_of_date: dt.date) -> list[EventCandidate]:
    text = strip_tags(html)
    pattern = re.compile(
        r"(?P<month>Jan\.?|January|Feb\.?|February|Mar\.?|March|Apr\.?|April|May|June|July|Aug\.?|August|Sep\.?|Sept\.?|September|Oct\.?|October|Nov\.?|November|Dec\.?|December)\s+"
        r"(?P<start>\d{1,2})(?:\s*(?:-|and)\s*(?P<end>\d{1,2}))?,\s*(?P<year>20\d{2})",
        flags=re.IGNORECASE,
    )
    candidates: list[EventCandidate] = []
    for match in pattern.finditer(text):
        event_date = dt.date(
            int(match.group("year")),
            MONTHS[match.group("month").lower()],
            int(match.group("end") or match.group("start")),
        )
        candidates.append(
            EventCandidate(
                date=event_date,
                event_type="BOJ_decision",
                market="GLOBAL",
                importance="high",
                source_id=SOURCE_ID,
                source_url=CURRENT_URL,
                raw_title="Bank of Japan Monetary Policy Meeting",
                raw_snippet=match.group(0),
                is_future_scheduled=event_date > as_of_date,
                confidence="medium" if event_date > as_of_date else "high",
                requires_manual_review=False,
            )
        )
    return candidates


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
        license_note="Bank of Japan public webpage",
        notes=notes,
    )
