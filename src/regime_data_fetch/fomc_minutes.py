from __future__ import annotations

import datetime as dt
import json
import re
import urllib.request
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from regime_data_fetch.acquisition_store import AcquisitionStore


LISTING_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
HISTORICAL_YEAR_INDEX_URL = "https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm"
SOURCE_NAME = "federalreserve.gov"
BASE_URL = "https://www.federalreserve.gov"

_LISTING_ENTRY_RE = re.compile(
    r'<div class="[^"]*fomc-meeting__month[^"]*"><strong>(?P<month>[A-Za-z/]+)</strong></div>\s*'
    r'<div class="[^"]*fomc-meeting__date[^"]*">(?P<meeting_days>[^<]+)</div>\s*'
    r'.*?<a href="(?P<pdf_path>/monetarypolicy/files/fomcminutes(?P<meeting_end>\d{8})\.pdf)">PDF</a>\s*\|\s*'
    r'<a href="(?P<html_path>/monetarypolicy/fomcminutes\d{8}\.htm)">HTML</a>\s*'
    r'<br>\s*\(Released (?P<release_date>[A-Za-z]+ \d{1,2}, \d{4})\)',
    flags=re.DOTALL,
)
_HISTORICAL_YEAR_RE = re.compile(r'href="(?P<path>/monetarypolicy/fomchistorical(?P<year>\d{4})\.htm)"', flags=re.IGNORECASE)
_HISTORICAL_ENTRY_RE = re.compile(
    r'<h5[^>]*>(?P<heading>[^<]+)</h5>.*?'
    r'Minutes \(Released (?P<release_date>[A-Za-z]+ \d{1,2}, \d{4})\):\s*<br\s*/?>\s*'
    r'<a href="(?P<html_path>/monetarypolicy/fomcminutes(?P<meeting_end>\d{8})\.htm)">HTML</a>\s*\|\s*'
    r'<a href="(?P<pdf_path>/monetarypolicy/files/fomcminutes\d{8}\.pdf)">[^<]*PDF</a>',
    flags=re.DOTALL | re.IGNORECASE,
)
_TITLE_RE = re.compile(
    r"<(?:h3|h1[^>]*)>\s*(?P<title>Minutes of the Federal Open Market Committee)\s*</(?:h3|h1)>",
    flags=re.IGNORECASE,
)
_MEETING_DATE_RE = re.compile(r"<p>\s*<strong>(?P<meeting_date_text>[^<]+)</strong>(?:<br\s*/?>)?", flags=re.IGNORECASE)
_ARTICLE_RE = re.compile(r'<div id="article"[^>]*>(?P<body>.*?)</div>', flags=re.DOTALL | re.IGNORECASE)
_LEGACY_ARTICLE_RE = re.compile(r'<div id="leftText">(?P<body>.*?)</div>\s*</div>', flags=re.DOTALL | re.IGNORECASE)


class FOMCMinutesFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class FOMCMinutesListingEntry:
    meeting_end_date: dt.date
    html_url: str
    pdf_url: str
    release_date: dt.date


@dataclass(frozen=True)
class FOMCMinutesArticle:
    meeting_end_date: dt.date
    release_timestamp: dt.datetime
    title: str
    meeting_date_text: str
    body_text: str
    source: str
    source_url: str
    pdf_url: str


def parse_fomc_minutes_listing(html: str) -> list[FOMCMinutesListingEntry]:
    entries: list[FOMCMinutesListingEntry] = []
    for match in _LISTING_ENTRY_RE.finditer(html):
        meeting_end_date = dt.datetime.strptime(match.group("meeting_end"), "%Y%m%d").date()
        release_date = _parse_release_date_text(match.group("release_date"))
        entries.append(
            FOMCMinutesListingEntry(
                meeting_end_date=meeting_end_date,
                html_url=BASE_URL + match.group("html_path"),
                pdf_url=BASE_URL + match.group("pdf_path"),
                release_date=release_date,
            )
        )

    if not entries:
        raise FOMCMinutesFetchError("FOMC calendars page did not contain parseable minutes entries")
    return entries


def parse_fomc_historical_year_index(html: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in _HISTORICAL_YEAR_RE.finditer(html):
        year = int(match.group("year"))
        if year >= 2021 or year < 1993:
            continue
        url = BASE_URL + match.group("path")
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    if not urls:
        raise FOMCMinutesFetchError("FOMC historical year index did not contain pre-2021 year pages")
    return urls


def parse_fomc_minutes_historical_listing(html: str) -> list[FOMCMinutesListingEntry]:
    entries: list[FOMCMinutesListingEntry] = []
    for match in _HISTORICAL_ENTRY_RE.finditer(html):
        meeting_end_date = dt.datetime.strptime(match.group("meeting_end"), "%Y%m%d").date()
        release_date = _parse_release_date_text(match.group("release_date"))
        entries.append(
            FOMCMinutesListingEntry(
                meeting_end_date=meeting_end_date,
                html_url=BASE_URL + match.group("html_path"),
                pdf_url=BASE_URL + match.group("pdf_path"),
                release_date=release_date,
            )
        )
    return entries


def fetch_release_timestamp(release_date: dt.date) -> dt.datetime:
    return dt.datetime(release_date.year, release_date.month, release_date.day, 14, 0, tzinfo=ZoneInfo("America/New_York"))


def parse_fomc_minutes_article(html: str, *, source_url: str, release_timestamp: dt.datetime) -> FOMCMinutesArticle:
    title_match = _TITLE_RE.search(html)
    if not title_match:
        raise FOMCMinutesFetchError("FOMC minutes page missing title")

    date_match = _MEETING_DATE_RE.search(html)
    if not date_match:
        raise FOMCMinutesFetchError("FOMC minutes page missing meeting date text")

    article_match = _ARTICLE_RE.search(html) or _LEGACY_ARTICLE_RE.search(html)
    if not article_match:
        raise FOMCMinutesFetchError("FOMC minutes page missing article body")

    body_html = article_match.group("body")
    body_text = _clean_html_text(body_html)
    if not body_text:
        raise FOMCMinutesFetchError("FOMC minutes page produced empty article body")

    meeting_end_date = _meeting_end_date_from_url(source_url)
    return FOMCMinutesArticle(
        meeting_end_date=meeting_end_date,
        release_timestamp=release_timestamp,
        title=unescape(title_match.group("title")).strip(),
        meeting_date_text=unescape(date_match.group("meeting_date_text")).strip(),
        body_text=body_text,
        source=SOURCE_NAME,
        source_url=source_url,
        pdf_url=f"{BASE_URL}/monetarypolicy/files/fomcminutes{meeting_end_date.strftime('%Y%m%d')}.pdf",
    )


def fetch_fomc_minutes_listing() -> str:
    return _http_get_text(LISTING_URL)


def fetch_fomc_historical_year_index() -> str:
    return _http_get_text(HISTORICAL_YEAR_INDEX_URL)


def fetch_fomc_historical_year_page(url: str) -> str:
    return _http_get_text(url)


def fetch_fomc_minutes_article(url: str) -> str:
    return _http_get_text(url)


def run_fomc_minutes_fetch(
    *,
    out_dir: Path,
    listing_fetcher=fetch_fomc_minutes_listing,
    historical_index_fetcher=fetch_fomc_historical_year_index,
    historical_page_fetcher=fetch_fomc_historical_year_page,
    article_fetcher=fetch_fomc_minutes_article,
    acquisition_db_path: Path | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    store = AcquisitionStore(acquisition_db_path) if acquisition_db_path else None
    fetch_run = (
        store.start_fetch_run(
            fetch_type="fomc_minutes",
            params={
                "listing_url": LISTING_URL,
                "historical_index_url": HISTORICAL_YEAR_INDEX_URL,
            },
        )
        if store
        else None
    )

    try:
        listing_html = listing_fetcher()
        if store and fetch_run:
            store.record_text_artifact(
                run_id=fetch_run.run_id,
                source_name="federalreserve:fomc_listing",
                artifact_kind="html",
                source_identifier=LISTING_URL,
                content_text=listing_html,
                timezone="America/New_York",
                license_note="Raw Federal Reserve FOMC listing page",
                notes="Current FOMC listing HTML persisted before parsing",
            )
        entries = parse_fomc_minutes_listing(listing_html)

        historical_index_html = historical_index_fetcher()
        if store and fetch_run:
            store.record_text_artifact(
                run_id=fetch_run.run_id,
                source_name="federalreserve:fomc_historical_index",
                artifact_kind="html",
                source_identifier=HISTORICAL_YEAR_INDEX_URL,
                content_text=historical_index_html,
                timezone="America/New_York",
                license_note="Raw Federal Reserve FOMC historical year index page",
                notes="FOMC historical year index HTML persisted before parsing",
            )
        historical_urls = parse_fomc_historical_year_index(historical_index_html)
        for historical_url in historical_urls:
            historical_html = historical_page_fetcher(historical_url)
            if store and fetch_run:
                store.record_text_artifact(
                    run_id=fetch_run.run_id,
                    source_name="federalreserve:fomc_historical_year",
                    artifact_kind="html",
                    source_identifier=historical_url,
                    content_text=historical_html,
                    timezone="America/New_York",
                    license_note="Raw Federal Reserve FOMC historical year page",
                    notes="FOMC historical year HTML persisted before parsing",
                )
            entries.extend(parse_fomc_minutes_historical_listing(historical_html))

        deduped_entries: dict[dt.date, FOMCMinutesListingEntry] = {}
        for entry in entries:
            deduped_entries[entry.meeting_end_date] = entry
        entries = sorted(deduped_entries.values(), key=lambda item: item.meeting_end_date, reverse=True)

        rows: list[FOMCMinutesArticle] = []
        for entry in entries:
            article_html = article_fetcher(entry.html_url)
            if store and fetch_run:
                store.record_text_artifact(
                    run_id=fetch_run.run_id,
                    source_name="federalreserve:fomc_minutes_article",
                    artifact_kind="html",
                    source_identifier=entry.html_url,
                    content_text=article_html,
                    effective_date=entry.meeting_end_date.isoformat(),
                    timezone="America/New_York",
                    license_note="Raw Federal Reserve FOMC minutes article page",
                    notes="FOMC minutes article HTML persisted before normalization",
                )
            rows.append(
                parse_fomc_minutes_article(
                    article_html,
                    source_url=entry.html_url,
                    release_timestamp=fetch_release_timestamp(entry.release_date),
                )
            )

        df = pd.DataFrame(
            [
                {
                    "meeting_end_date": row.meeting_end_date.isoformat(),
                    "release_timestamp": row.release_timestamp.isoformat(),
                    "title": row.title,
                    "meeting_date_text": row.meeting_date_text,
                    "body_text": row.body_text,
                    "source": row.source,
                    "source_url": row.source_url,
                    "pdf_url": row.pdf_url,
                }
                for row in rows
            ]
        )

        out_path_dir = out_dir / "fomc_minutes"
        out_path_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_path_dir / "fomc_minutes.parquet"
        df.to_parquet(parquet_path, index=False)

        report = {
            "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": SOURCE_NAME,
            "listing_url": LISTING_URL,
            "counts": {
                "rows": int(len(df)),
                "meeting_dates": int(df["meeting_end_date"].nunique()),
            },
            "date_range": {
                "min_meeting_end_date": str(df["meeting_end_date"].min()) if not df.empty else None,
                "max_meeting_end_date": str(df["meeting_end_date"].max()) if not df.empty else None,
                "min_release_timestamp": str(df["release_timestamp"].min()) if not df.empty else None,
                "max_release_timestamp": str(df["release_timestamp"].max()) if not df.empty else None,
            },
            "paths": {
                "fomc_minutes_parquet": str(parquet_path),
                "acquisition_db": str(acquisition_db_path) if acquisition_db_path else None,
            },
        }
        report_path = out_dir / "fomc_minutes_fetch_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if store and fetch_run:
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="fomc_minutes_parquet",
                path=parquet_path,
                row_count=len(df),
                min_date=str(df["meeting_end_date"].min()) if not df.empty else None,
                max_date=str(df["meeting_end_date"].max()) if not df.empty else None,
                notes="FOMC minutes parquet output",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="fomc_minutes_report",
                path=report_path,
                row_count=len(df),
                min_date=str(df["meeting_end_date"].min()) if not df.empty else None,
                max_date=str(df["meeting_end_date"].max()) if not df.empty else None,
                notes="FOMC minutes fetch report",
            )
            store.finish_fetch_run(run_id=fetch_run.run_id, status="ok")
        return report_path
    except Exception as exc:
        if store and fetch_run:
            store.finish_fetch_run(run_id=fetch_run.run_id, status="failed", notes=str(exc))
        raise


def _http_get_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _clean_html_text(html: str) -> str:
    text = re.sub(r"<sup[^>]*>.*?</sup>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<a[^>]*>.*?</a>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _meeting_end_date_from_url(url: str) -> dt.date:
    match = re.search(r"fomcminutes(?P<stamp>\d{8})\.htm", url)
    if not match:
        raise FOMCMinutesFetchError(f"Could not determine meeting date from URL: {url}")
    return dt.datetime.strptime(match.group("stamp"), "%Y%m%d").date()


def _parse_release_date_text(value: str) -> dt.date:
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise FOMCMinutesFetchError(f"Unparseable FOMC release date: {value!r}")
