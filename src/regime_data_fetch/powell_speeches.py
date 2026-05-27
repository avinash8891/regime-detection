from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportUnknownParameterType=false, reportMissingParameterType=false

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

INDEX_URL = (
    "https://www.federalreserve.gov/newsevents/speeches.htm?speaker=Jerome+H.+Powell"
)
BASE_URL = "https://www.federalreserve.gov"
SOURCE_NAME = "federalreserve.gov"

_YEAR_PAGE_RE = re.compile(
    r'href="(?P<path>/newsevents/\d{4}-speeches\.htm)"', flags=re.IGNORECASE
)
_YEAR_ROW_RE = re.compile(
    r"<time>(?P<date>\d{1,2}/\d{1,2}/\d{4})</time>\s*</div>\s*"
    r'<div class="col-xs-9 col-md-10 eventlist__event">\s*'
    r'<p><a href="(?P<path>/newsevents/speech/powell[^"#]+\.htm)"><em>(?P<title>.*?)</em></a></p>\s*'
    r'<p class="news__speaker">(?P<speaker>.*?)</p>\s*'
    r"<p>(?P<location>.*?)</p>",
    flags=re.DOTALL | re.IGNORECASE,
)
_ARTICLE_TIME_RE = re.compile(
    r"<p class=['\"]article__time['\"]>(?P<date>[^<]+)</p>", flags=re.IGNORECASE
)
_ARTICLE_TITLE_RE = re.compile(
    r"<h3 class=['\"]title['\"]>\s*<em>(?P<title>.*?)</em>\s*</h3>",
    flags=re.IGNORECASE | re.DOTALL,
)
_ARTICLE_SPEAKER_RE = re.compile(
    r'<p class="speaker">\s*(?P<speaker>.*?)</p>', flags=re.IGNORECASE | re.DOTALL
)
_ARTICLE_LOCATION_RE = re.compile(
    r"<p class=['\"]location['\"]>(?P<location>.*?)</p>",
    flags=re.IGNORECASE | re.DOTALL,
)
_ARTICLE_RE = re.compile(
    r'<div id="article">(?P<body>.*?)</div>\s*</div>', flags=re.IGNORECASE | re.DOTALL
)


class PowellSpeechFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class PowellSpeechListing:
    speech_date: dt.date
    speech_url: str


@dataclass(frozen=True)
class PowellSpeechArticle:
    speech_date: dt.date
    publication_timestamp: dt.datetime
    publication_timestamp_precision: str
    title: str
    speaker: str
    location: str
    body_text: str
    source: str
    source_url: str


def parse_powell_speeches_year_index(html: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in _YEAR_PAGE_RE.finditer(html):
        url = BASE_URL + match.group("path")
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    if not urls:
        raise PowellSpeechFetchError(
            "Powell speeches index did not contain yearly archive links"
        )
    return urls


def parse_powell_speech_year_page(html: str) -> list[PowellSpeechListing]:
    rows: list[PowellSpeechListing] = []
    for match in _YEAR_ROW_RE.finditer(html):
        rows.append(
            PowellSpeechListing(
                speech_date=dt.datetime.strptime(
                    match.group("date"), "%m/%d/%Y"
                ).date(),
                speech_url=BASE_URL + match.group("path"),
            )
        )
    return rows


def publication_timestamp_for_date(speech_date: dt.date) -> tuple[dt.datetime, str]:
    tz = ZoneInfo("America/New_York")
    return (
        dt.datetime(
            speech_date.year, speech_date.month, speech_date.day, 0, 0, tzinfo=tz
        ),
        "date_only",
    )


def parse_powell_speech_article(
    html: str,
    *,
    source_url: str,
    publication_timestamp: dt.datetime,
    publication_timestamp_precision: str,
) -> PowellSpeechArticle:
    date_match = _ARTICLE_TIME_RE.search(html)
    title_match = _ARTICLE_TITLE_RE.search(html)
    speaker_match = _ARTICLE_SPEAKER_RE.search(html)
    location_match = _ARTICLE_LOCATION_RE.search(html)
    article_match = _ARTICLE_RE.search(html)

    if not date_match:
        raise PowellSpeechFetchError("Powell speech page missing article date")
    if not title_match:
        raise PowellSpeechFetchError("Powell speech page missing title")
    if not speaker_match:
        raise PowellSpeechFetchError("Powell speech page missing speaker")
    if not location_match:
        raise PowellSpeechFetchError("Powell speech page missing location")
    if not article_match:
        raise PowellSpeechFetchError("Powell speech page missing article body")

    speech_date = dt.datetime.strptime(
        unescape(date_match.group("date")).strip(), "%B %d, %Y"
    ).date()
    body_text = _clean_html_text(article_match.group("body"))
    if not body_text:
        raise PowellSpeechFetchError("Powell speech page produced empty article body")

    return PowellSpeechArticle(
        speech_date=speech_date,
        publication_timestamp=publication_timestamp,
        publication_timestamp_precision=publication_timestamp_precision,
        title=_clean_html_text(title_match.group("title")),
        speaker=_clean_html_text(speaker_match.group("speaker")),
        location=_clean_html_text(location_match.group("location")),
        body_text=body_text,
        source=SOURCE_NAME,
        source_url=source_url,
    )


def fetch_powell_speeches_index() -> str:
    return _http_get_text(INDEX_URL)


def fetch_powell_speeches_year_page(url: str) -> str:
    return _http_get_text(url)


def fetch_powell_speech_article(url: str) -> str:
    return _http_get_text(url)


def run_powell_speeches_fetch(
    *,
    out_dir: Path,
    index_fetcher=fetch_powell_speeches_index,
    year_page_fetcher=fetch_powell_speeches_year_page,
    article_fetcher=fetch_powell_speech_article,
    acquisition_db_path: Path | None = None,
    artifact_store_root: str | Path | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    store = (
        AcquisitionStore(acquisition_db_path, artifact_store_root=artifact_store_root)
        if acquisition_db_path
        else None
    )
    fetch_run = (
        store.start_fetch_run(
            fetch_type="powell_speeches",
            params={
                "index_url": INDEX_URL,
            },
        )
        if store
        else None
    )

    try:
        index_html = index_fetcher()
        if store and fetch_run:
            store.record_text_artifact(
                run_id=fetch_run.run_id,
                source_name="federalreserve:powell_index",
                artifact_kind="html",
                source_identifier=INDEX_URL,
                content_text=index_html,
                timezone="America/New_York",
                license_note="Raw Federal Reserve Powell speeches index page",
                notes="Powell speeches index HTML persisted before parsing",
            )
        year_urls = parse_powell_speeches_year_index(index_html)
        listings: dict[dt.date, PowellSpeechListing] = {}
        for year_url in year_urls:
            year_html = year_page_fetcher(year_url)
            if store and fetch_run:
                store.record_text_artifact(
                    run_id=fetch_run.run_id,
                    source_name="federalreserve:powell_year_page",
                    artifact_kind="html",
                    source_identifier=year_url,
                    content_text=year_html,
                    timezone="America/New_York",
                    license_note="Raw Federal Reserve Powell yearly speeches page",
                    notes="Powell yearly archive HTML persisted before parsing",
                )
            for row in parse_powell_speech_year_page(year_html):
                listings[row.speech_date] = row

        if not listings:
            raise PowellSpeechFetchError(
                "No Powell speeches found in the yearly Federal Reserve archives"
            )

        articles: list[PowellSpeechArticle] = []
        for row in sorted(
            listings.values(), key=lambda item: item.speech_date, reverse=True
        ):
            ts, precision = publication_timestamp_for_date(row.speech_date)
            article_html = article_fetcher(row.speech_url)
            if store and fetch_run:
                store.record_text_artifact(
                    run_id=fetch_run.run_id,
                    source_name="federalreserve:powell_article",
                    artifact_kind="html",
                    source_identifier=row.speech_url,
                    content_text=article_html,
                    effective_date=row.speech_date.isoformat(),
                    timezone="America/New_York",
                    license_note="Raw Federal Reserve Powell speech article page",
                    notes="Powell speech article HTML persisted before normalization",
                )
            articles.append(
                parse_powell_speech_article(
                    article_html,
                    source_url=row.speech_url,
                    publication_timestamp=ts,
                    publication_timestamp_precision=precision,
                )
            )

        df = pd.DataFrame(
            [
                {
                    "speech_date": article.speech_date.isoformat(),
                    "publication_timestamp": article.publication_timestamp.isoformat(),
                    "publication_timestamp_precision": article.publication_timestamp_precision,
                    "title": article.title,
                    "speaker": article.speaker,
                    "location": article.location,
                    "body_text": article.body_text,
                    "source": article.source,
                    "source_url": article.source_url,
                }
                for article in articles
            ]
        )

        out_path_dir = out_dir / "powell_speeches"
        out_path_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_path_dir / "powell_speeches.parquet"
        df.to_parquet(parquet_path, index=False)

        report = {
            "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": SOURCE_NAME,
            "index_url": INDEX_URL,
            "counts": {
                "rows": int(len(df)),
                "speech_dates": int(df["speech_date"].nunique()),
            },
            "date_range": {
                "min_speech_date": (
                    str(df["speech_date"].min()) if not df.empty else None
                ),
                "max_speech_date": (
                    str(df["speech_date"].max()) if not df.empty else None
                ),
            },
            "paths": {
                "powell_speeches_parquet": str(parquet_path),
                "acquisition_db": (
                    str(acquisition_db_path) if acquisition_db_path else None
                ),
            },
        }
        report_path = out_dir / "powell_speeches_fetch_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if store and fetch_run:
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="powell_speeches_parquet",
                path=parquet_path,
                row_count=len(df),
                min_date=str(df["speech_date"].min()) if not df.empty else None,
                max_date=str(df["speech_date"].max()) if not df.empty else None,
                notes="Powell speeches parquet output",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="powell_speeches_report",
                path=report_path,
                row_count=len(df),
                min_date=str(df["speech_date"].min()) if not df.empty else None,
                max_date=str(df["speech_date"].max()) if not df.empty else None,
                notes="Powell speeches fetch report",
            )
            store.finish_fetch_run(run_id=fetch_run.run_id, status="ok")
        return report_path
    except Exception as exc:
        if store and fetch_run:
            store.finish_fetch_run(
                run_id=fetch_run.run_id, status="failed", notes=str(exc)
            )
        raise


def _http_get_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _clean_html_text(html: str) -> str:
    text = re.sub(r"<a[^>]*>", "", html, flags=re.IGNORECASE)
    text = re.sub(r"</a>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
