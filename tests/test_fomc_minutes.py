from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from regime_data_fetch.fomc_minutes import (
    FOMCMinutesFetchError,
    fetch_release_timestamp,
    parse_fomc_minutes_article,
    parse_fomc_historical_year_index,
    parse_fomc_minutes_historical_listing,
    parse_fomc_minutes_listing,
    run_fomc_minutes_fetch,
)


FIXTURES = Path("tests/fixtures/raw/fomc")


def test_parse_fomc_minutes_listing_extracts_entries() -> None:
    html = (FIXTURES / "fomc_calendars_snippet.html").read_text()

    entries = parse_fomc_minutes_listing(html)

    assert len(entries) == 2
    assert entries[0].meeting_end_date == dt.date(2026, 1, 28)
    assert entries[0].html_url == "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260128.htm"
    assert entries[0].release_date == dt.date(2026, 2, 18)
    assert entries[1].meeting_end_date == dt.date(2026, 3, 18)


def test_parse_fomc_historical_year_index_filters_pre_2021_years() -> None:
    html = (FIXTURES / "fomc_historical_year_snippet.html").read_text()

    urls = parse_fomc_historical_year_index(html)

    assert urls == [
        "https://www.federalreserve.gov/monetarypolicy/fomchistorical2020.htm",
        "https://www.federalreserve.gov/monetarypolicy/fomchistorical2019.htm",
        "https://www.federalreserve.gov/monetarypolicy/fomchistorical2018.htm",
        "https://www.federalreserve.gov/monetarypolicy/fomchistorical1993.htm",
    ]


def test_parse_fomc_minutes_historical_listing_extracts_entries() -> None:
    html = (FIXTURES / "fomchistorical2019_snippet.html").read_text()

    entries = parse_fomc_minutes_historical_listing(html)

    assert len(entries) == 2
    assert entries[0].meeting_end_date == dt.date(2019, 1, 30)
    assert entries[0].release_date == dt.date(2019, 2, 20)
    assert entries[1].meeting_end_date == dt.date(2019, 12, 11)
    assert entries[1].release_date == dt.date(2020, 1, 3)


def test_fetch_release_timestamp_uses_2pm_et() -> None:
    ts = fetch_release_timestamp(dt.date(2026, 2, 18))
    assert ts.isoformat() == "2026-02-18T14:00:00-05:00"


def test_parse_fomc_minutes_article_extracts_title_and_body() -> None:
    html = (FIXTURES / "fomcminutes20250319_snippet.html").read_text()

    article = parse_fomc_minutes_article(
        html,
        source_url="https://www.federalreserve.gov/monetarypolicy/fomcminutes20250319.htm",
        release_timestamp=fetch_release_timestamp(dt.date(2025, 4, 9)),
    )

    assert article.title == "Minutes of the Federal Open Market Committee"
    assert article.meeting_date_text == "March 18–19, 2025"
    assert "Participants reviewed recent developments in financial markets" in article.body_text
    assert article.source == "federalreserve.gov"


def test_parse_fomc_minutes_article_extracts_legacy_2011_shape() -> None:
    html = (FIXTURES / "fomcminutes20110126_snippet.html").read_text()

    article = parse_fomc_minutes_article(
        html,
        source_url="https://www.federalreserve.gov/monetarypolicy/fomcminutes20110126.htm",
        release_timestamp=fetch_release_timestamp(dt.date(2011, 2, 16)),
    )

    assert article.title == "Minutes of the Federal Open Market Committee"
    assert article.meeting_date_text == "January 25-26, 2011"
    assert "Participants discussed inflation, labor market conditions" in article.body_text


def test_run_fomc_minutes_fetch_writes_parquet_and_report(tmp_path: Path) -> None:
    listing_html = (FIXTURES / "fomc_calendars_snippet.html").read_text()
    historical_2019_html = (FIXTURES / "fomchistorical2019_snippet.html").read_text()
    article_html = (FIXTURES / "fomcminutes20250319_snippet.html").read_text()

    def fake_listing_fetcher() -> str:
        return listing_html

    def fake_historical_index_fetcher() -> str:
        return '<a href="/monetarypolicy/fomchistorical2019.htm">2019</a>'

    def fake_historical_page_fetcher(url: str) -> str:
        if url.endswith("fomchistorical2019.htm"):
            return historical_2019_html
        raise AssertionError(f"Unexpected historical URL: {url}")

    def fake_article_fetcher(url: str) -> str:
        assert url.startswith("https://www.federalreserve.gov/monetarypolicy/fomcminutes")
        return article_html

    report_path = run_fomc_minutes_fetch(
        out_dir=tmp_path,
        listing_fetcher=fake_listing_fetcher,
        historical_index_fetcher=fake_historical_index_fetcher,
        historical_page_fetcher=fake_historical_page_fetcher,
        article_fetcher=fake_article_fetcher,
    )

    report = json.loads(report_path.read_text())
    assert report["counts"]["rows"] == 4
    assert report["source"] == "federalreserve.gov"
    assert report["paths"]["fomc_minutes_parquet"] == str(tmp_path / "fomc_minutes" / "fomc_minutes.parquet")

    df = pd.read_parquet(tmp_path / "fomc_minutes" / "fomc_minutes.parquet")
    assert list(df.columns) == [
        "meeting_end_date",
        "release_timestamp",
        "title",
        "meeting_date_text",
        "body_text",
        "source",
        "source_url",
        "pdf_url",
    ]
    assert df.iloc[0]["title"] == "Minutes of the Federal Open Market Committee"
    assert "2019-01-30" in set(df["meeting_end_date"])


def test_run_fomc_minutes_fetch_raises_on_missing_article(tmp_path: Path) -> None:
    def fake_listing_fetcher() -> str:
        return (FIXTURES / "fomc_calendars_snippet.html").read_text()

    def fake_historical_index_fetcher() -> str:
        return '<a href="/monetarypolicy/fomchistorical2019.htm">2019</a>'

    def fake_historical_page_fetcher(url: str) -> str:
        if url.endswith("fomchistorical2019.htm"):
            return (FIXTURES / "fomchistorical2019_snippet.html").read_text()
        raise AssertionError(f"Unexpected historical URL: {url}")

    def bad_article_fetcher(url: str) -> str:
        del url
        return "<h3>Minutes of the Federal Open Market Committee</h3><p><strong>March 18–19, 2025</strong><br /></p>"

    try:
        run_fomc_minutes_fetch(
            out_dir=tmp_path,
            listing_fetcher=fake_listing_fetcher,
            historical_index_fetcher=fake_historical_index_fetcher,
            historical_page_fetcher=fake_historical_page_fetcher,
            article_fetcher=bad_article_fetcher,
        )
    except FOMCMinutesFetchError as exc:
        assert "article body" in str(exc).lower()
    else:
        raise AssertionError("Expected FOMCMinutesFetchError")
