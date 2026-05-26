from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sqlite3
from contextlib import closing

import pandas as pd

from regime_data_fetch.powell_speeches import (
    PowellSpeechFetchError,
    publication_timestamp_for_date,
    parse_powell_speech_article,
    parse_powell_speech_year_page,
    parse_powell_speeches_year_index,
    run_powell_speeches_fetch,
)

FIXTURES = Path("tests/fixtures/raw/powell")


def test_parse_powell_speeches_year_index_extracts_year_pages() -> None:
    html = (FIXTURES / "powell_speeches_index_snippet.html").read_text()
    urls = parse_powell_speeches_year_index(html)

    assert urls == [
        "https://www.federalreserve.gov/newsevents/2026-speeches.htm",
        "https://www.federalreserve.gov/newsevents/2025-speeches.htm",
        "https://www.federalreserve.gov/newsevents/2024-speeches.htm",
    ]


def test_parse_powell_speech_year_page_filters_powell_only() -> None:
    html = (FIXTURES / "powell_2026_speeches_snippet.html").read_text()
    rows = parse_powell_speech_year_page(html)

    assert len(rows) == 2
    assert rows[0].speech_date == dt.date(2026, 3, 21)
    assert (
        rows[0].speech_url
        == "https://www.federalreserve.gov/newsevents/speech/powell20260321a.htm"
    )
    assert rows[1].speech_date == dt.date(2026, 1, 11)


def test_parse_powell_speech_year_page_allows_years_without_powell_rows() -> None:
    html = """
    <div class="row">
        <div class="col-xs-3 col-md-2 eventlist__time"><time>3/12/2026</time></div>
        <div class="col-xs-9 col-md-10 eventlist__event">
            <p><a href="/newsevents/speech/bowman20260312a.htm"><em>Capital Rules for the Real Economy</em></a></p>
            <p class="news__speaker">Vice Chair for Supervision Michelle W. Bowman</p>
            <p>At the Cato Institute, Washington, D.C.</p>
        </div>
    </div>
    """
    assert parse_powell_speech_year_page(html) == []


def test_publication_timestamp_for_date_marks_date_only_precision() -> None:
    ts, precision = publication_timestamp_for_date(dt.date(2026, 3, 21))
    assert ts.isoformat() == "2026-03-21T00:00:00-04:00"
    assert precision == "date_only"


def test_parse_powell_speech_article_extracts_body() -> None:
    html = (FIXTURES / "powell_speech_detail_snippet.html").read_text()
    ts, precision = publication_timestamp_for_date(dt.date(2026, 3, 21))

    article = parse_powell_speech_article(
        html,
        source_url="https://www.federalreserve.gov/newsevents/speech/powell20260321a.htm",
        publication_timestamp=ts,
        publication_timestamp_precision=precision,
    )

    assert article.title == "Acceptance Remarks"
    assert article.speaker == "Chair Jerome H. Powell"
    assert "Integrity in public institutions is essential" in article.body_text
    assert article.publication_timestamp_precision == "date_only"


def test_run_powell_speeches_fetch_writes_parquet_and_report(tmp_path: Path) -> None:
    index_html = (FIXTURES / "powell_speeches_index_snippet.html").read_text()
    year_html = (FIXTURES / "powell_2026_speeches_snippet.html").read_text()
    article_html = (FIXTURES / "powell_speech_detail_snippet.html").read_text()
    second_article_html = (
        article_html.replace("March 21, 2026", "January 11, 2026")
        .replace(
            "Acceptance Remarks",
            "Statement from Federal Reserve Chair Jerome H. Powell",
        )
        .replace(
            "At the American Society for Public Administration Annual Conference: Paul A. Volcker Public Integrity Award Ceremony (via pre-recorded video)",
            "Washington, D.C.",
        )
    )

    def fake_index_fetcher() -> str:
        return index_html

    def fake_year_fetcher(url: str) -> str:
        assert url.endswith("-speeches.htm")
        return year_html

    def fake_article_fetcher(url: str) -> str:
        assert url.startswith("https://www.federalreserve.gov/newsevents/speech/powell")
        if url.endswith("powell20260321a.htm"):
            return article_html
        if url.endswith("powell20260111a.htm"):
            return second_article_html
        raise AssertionError(f"Unexpected article URL: {url}")

    report_path = run_powell_speeches_fetch(
        out_dir=tmp_path,
        index_fetcher=fake_index_fetcher,
        year_page_fetcher=fake_year_fetcher,
        article_fetcher=fake_article_fetcher,
    )

    report = json.loads(report_path.read_text())
    assert report["counts"]["rows"] == 2
    assert report["source"] == "federalreserve.gov"
    assert report["paths"]["powell_speeches_parquet"] == str(
        tmp_path / "powell_speeches" / "powell_speeches.parquet"
    )

    df = pd.read_parquet(tmp_path / "powell_speeches" / "powell_speeches.parquet")
    assert list(df.columns) == [
        "speech_date",
        "publication_timestamp",
        "publication_timestamp_precision",
        "title",
        "speaker",
        "location",
        "body_text",
        "source",
        "source_url",
    ]
    assert set(df["speech_date"]) == {"2026-03-21", "2026-01-11"}


def test_run_powell_speeches_fetch_raises_on_missing_article(tmp_path: Path) -> None:
    def fake_index_fetcher() -> str:
        return (FIXTURES / "powell_speeches_index_snippet.html").read_text()

    def fake_year_fetcher(url: str) -> str:
        del url
        return (FIXTURES / "powell_2026_speeches_snippet.html").read_text()

    def bad_article_fetcher(url: str) -> str:
        del url
        return "<div id='article'><p class='article__time'>March 21, 2026</p></div>"

    try:
        run_powell_speeches_fetch(
            out_dir=tmp_path,
            index_fetcher=fake_index_fetcher,
            year_page_fetcher=fake_year_fetcher,
            article_fetcher=bad_article_fetcher,
        )
    except PowellSpeechFetchError as exc:
        assert "title" in str(exc).lower()
    else:
        raise AssertionError("Expected PowellSpeechFetchError")


def test_run_powell_speeches_fetch_raises_if_no_powell_rows_exist(
    tmp_path: Path,
) -> None:
    def fake_index_fetcher() -> str:
        return (FIXTURES / "powell_speeches_index_snippet.html").read_text()

    def empty_year_fetcher(url: str) -> str:
        del url
        return """
        <div class="row">
            <div class="col-xs-3 col-md-2 eventlist__time"><time>3/12/2026</time></div>
            <div class="col-xs-9 col-md-10 eventlist__event">
                <p><a href="/newsevents/speech/bowman20260312a.htm"><em>Capital Rules for the Real Economy</em></a></p>
                <p class="news__speaker">Vice Chair for Supervision Michelle W. Bowman</p>
                <p>At the Cato Institute, Washington, D.C.</p>
            </div>
        </div>
        """

    def fake_article_fetcher(url: str) -> str:
        raise AssertionError(f"Unexpected article fetch: {url}")

    try:
        run_powell_speeches_fetch(
            out_dir=tmp_path,
            index_fetcher=fake_index_fetcher,
            year_page_fetcher=empty_year_fetcher,
            article_fetcher=fake_article_fetcher,
        )
    except PowellSpeechFetchError as exc:
        assert "no powell speeches found" in str(exc).lower()
    else:
        raise AssertionError("Expected PowellSpeechFetchError")


def test_run_powell_speeches_fetch_records_raw_html_and_outputs_in_sqlite(
    tmp_path: Path,
) -> None:
    index_html = (FIXTURES / "powell_speeches_index_snippet.html").read_text()
    year_html = (FIXTURES / "powell_2026_speeches_snippet.html").read_text()
    article_html = (FIXTURES / "powell_speech_detail_snippet.html").read_text()
    second_article_html = (
        article_html.replace("March 21, 2026", "January 11, 2026")
        .replace(
            "Acceptance Remarks",
            "Statement from Federal Reserve Chair Jerome H. Powell",
        )
        .replace(
            "At the American Society for Public Administration Annual Conference: Paul A. Volcker Public Integrity Award Ceremony (via pre-recorded video)",
            "Washington, D.C.",
        )
    )
    acquisition_db = tmp_path / "acquisition.db"

    def fake_article_fetcher(url: str) -> str:
        if url.endswith("powell20260321a.htm"):
            return article_html
        if url.endswith("powell20260111a.htm"):
            return second_article_html
        raise AssertionError(f"Unexpected article URL: {url}")

    report_path = run_powell_speeches_fetch(
        out_dir=tmp_path,
        index_fetcher=lambda: index_html,
        year_page_fetcher=lambda url: year_html,
        article_fetcher=fake_article_fetcher,
        acquisition_db_path=acquisition_db,
    )

    report = json.loads(report_path.read_text())
    assert report["paths"]["acquisition_db"] == str(acquisition_db)

    with closing(sqlite3.connect(acquisition_db)) as conn:
        fetch_runs = conn.execute(
            "SELECT fetch_type, status FROM fetch_runs"
        ).fetchall()
        artifacts = conn.execute(
            "SELECT source_name, artifact_kind, count(*) FROM artifacts GROUP BY source_name, artifact_kind ORDER BY source_name, artifact_kind"
        ).fetchall()
        outputs = conn.execute(
            "SELECT output_kind FROM derived_outputs ORDER BY output_id"
        ).fetchall()

    assert fetch_runs == [("powell_speeches", "ok")]
    assert artifacts == [
        ("federalreserve:powell_article", "html", 2),
        ("federalreserve:powell_index", "html", 1),
        ("federalreserve:powell_year_page", "html", 3),
    ]
    assert outputs == [
        ("powell_speeches_parquet",),
        ("powell_speeches_report",),
    ]
