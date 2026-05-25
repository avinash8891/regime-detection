from __future__ import annotations

import datetime as dt
import json
import urllib.error

from regime_data_fetch.bls_schedule import (
    BLSReleaseDate,
    build_bls_schedule_year_urls,
    fetch_bls_year_releases,
    parse_bls_schedule_page,
    fetch_bls_schedule_page_text,
)


def test_build_bls_schedule_year_urls_includes_modern_and_legacy_patterns() -> None:
    urls = build_bls_schedule_year_urls(start_year=2000, end_year=2001)

    assert urls[0] == "https://www.bls.gov/schedule/2000/"
    assert urls[1] == "https://www.bls.gov/schedule/2000/home.htm"
    assert urls[2] == "https://www.bls.gov/schedule/2001/"
    assert urls[3] == "https://www.bls.gov/schedule/2001/home.htm"


def test_parse_bls_schedule_page_extracts_modern_cpi_and_nfp_rows() -> None:
    html = """
    <h1>January 2024</h1>
    Friday, January 05, 2024
    08:30 AM
    Employment Situation for December 2023
    Thursday, January 11, 2024
    08:30 AM
    Consumer Price Index for December 2023
    Thursday, January 11, 2024
    08:30 AM
    Real Earnings for December 2023
    <h1>February 2024</h1>
    Friday, February 02, 2024
    08:30 AM
    Employment Situation for January 2024
    Tuesday, February 13, 2024
    08:30 AM
    Consumer Price Index for January 2024
    """

    releases = parse_bls_schedule_page(
        html,
        source_url="https://www.bls.gov/schedule/2024/",
        default_year=2024,
    )

    assert releases == [
        BLSReleaseDate(
            date=dt.date(2024, 1, 5),
            release_timestamp_et=dt.datetime(
                2024, 1, 5, 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))
            ),
            type="NFP",
            source_url="https://www.bls.gov/schedule/2024/",
            reference_period="December 2023",
        ),
        BLSReleaseDate(
            date=dt.date(2024, 1, 11),
            release_timestamp_et=dt.datetime(
                2024, 1, 11, 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))
            ),
            type="CPI",
            source_url="https://www.bls.gov/schedule/2024/",
            reference_period="December 2023",
        ),
        BLSReleaseDate(
            date=dt.date(2024, 2, 2),
            release_timestamp_et=dt.datetime(
                2024, 2, 2, 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))
            ),
            type="NFP",
            source_url="https://www.bls.gov/schedule/2024/",
            reference_period="January 2024",
        ),
        BLSReleaseDate(
            date=dt.date(2024, 2, 13),
            release_timestamp_et=dt.datetime(
                2024, 2, 13, 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))
            ),
            type="CPI",
            source_url="https://www.bls.gov/schedule/2024/",
            reference_period="January 2024",
        ),
    ]


def test_parse_bls_schedule_page_extracts_legacy_rows() -> None:
    html = """
    <h1>Schedule of Releases for 2001</h1>
    Consumer Price Index, December 2000 Jan. 17 8:30 am
    The Employment Situation, January 2001 Feb.  2, 2001 8:30 am
    Consumer Price Indexes, January 2001 Feb. 21 8:30 am
    The Employment Situation, February 2001 March  9 8:30 am
    """

    releases = parse_bls_schedule_page(
        html,
        source_url="https://www.bls.gov/schedule/2001/home.htm",
        default_year=2001,
    )

    assert releases == [
        BLSReleaseDate(
            date=dt.date(2001, 1, 17),
            release_timestamp_et=dt.datetime(
                2001, 1, 17, 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))
            ),
            type="CPI",
            source_url="https://www.bls.gov/schedule/2001/home.htm",
            reference_period="December 2000",
        ),
        BLSReleaseDate(
            date=dt.date(2001, 2, 2),
            release_timestamp_et=dt.datetime(
                2001, 2, 2, 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))
            ),
            type="NFP",
            source_url="https://www.bls.gov/schedule/2001/home.htm",
            reference_period="January 2001",
        ),
        BLSReleaseDate(
            date=dt.date(2001, 2, 21),
            release_timestamp_et=dt.datetime(
                2001, 2, 21, 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))
            ),
            type="CPI",
            source_url="https://www.bls.gov/schedule/2001/home.htm",
            reference_period="January 2001",
        ),
        BLSReleaseDate(
            date=dt.date(2001, 3, 9),
            release_timestamp_et=dt.datetime(
                2001, 3, 9, 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))
            ),
            type="NFP",
            source_url="https://www.bls.gov/schedule/2001/home.htm",
            reference_period="February 2001",
        ),
    ]


def test_parse_bls_schedule_page_supports_sept_abbreviation_in_legacy_rows() -> None:
    html = """
    <h1>Schedule of Releases for 2000</h1>
    Consumer Price Indexes, August 2000 Sept. 15 8:30 am
    The Employment Situation, August 2000 Sept.  1, 2000 8:30 am
    """

    releases = parse_bls_schedule_page(
        html,
        source_url="https://www.bls.gov/schedule/2000/home.htm",
        default_year=2000,
    )

    assert releases == [
        BLSReleaseDate(
            date=dt.date(2000, 9, 1),
            release_timestamp_et=dt.datetime(
                2000, 9, 1, 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-4))
            ),
            type="NFP",
            source_url="https://www.bls.gov/schedule/2000/home.htm",
            reference_period="August 2000",
        ),
        BLSReleaseDate(
            date=dt.date(2000, 9, 15),
            release_timestamp_et=dt.datetime(
                2000, 9, 15, 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-4))
            ),
            type="CPI",
            source_url="https://www.bls.gov/schedule/2000/home.htm",
            reference_period="August 2000",
        ),
    ]


def test_parse_bls_schedule_page_rolls_legacy_year_forward_for_january_release_of_december_period() -> (
    None
):
    html = """
    <h1>Schedule of Releases for 2000</h1>
    Consumer Price Index, December 2000 Jan. 17 8:30 am
    """

    releases = parse_bls_schedule_page(
        html,
        source_url="https://www.bls.gov/schedule/2000/home.htm",
        default_year=2000,
    )

    assert releases == [
        BLSReleaseDate(
            date=dt.date(2001, 1, 17),
            release_timestamp_et=dt.datetime(
                2001, 1, 17, 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-5))
            ),
            type="CPI",
            source_url="https://www.bls.gov/schedule/2000/home.htm",
            reference_period="December 2000",
        )
    ]


def test_fetch_bls_year_releases_dedupes_cross_year_rollover_releases() -> None:
    pages = {
        "https://www.bls.gov/schedule/2000/": """
        Consumer Price Index, December 2000 Jan. 17 8:30 am
        """,
        "https://www.bls.gov/schedule/2001/": """
        Consumer Price Index, December 2000 Jan. 17 8:30 am
        Consumer Price Indexes, January 2001 Feb. 21 8:30 am
        """,
    }

    def fake_page_fetcher(url: str) -> str:
        if url in pages:
            return pages[url]
        raise AssertionError(f"Unexpected URL: {url}")

    releases = fetch_bls_year_releases(
        start_year=2000,
        end_year=2001,
        page_fetcher=fake_page_fetcher,
    )

    assert [(release.date.isoformat(), release.type) for release in releases] == [
        ("2001-01-17", "CPI"),
        ("2001-02-21", "CPI"),
    ]


def test_fetch_bls_schedule_page_text_falls_back_to_tinyfish(monkeypatch) -> None:
    calls: list[str] = []

    class FakeResponse:
        def __init__(self, body: bytes):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return self.body

    def fake_urlopen(request, timeout):
        del timeout
        url = request.full_url
        calls.append(url)
        if url.startswith("https://www.bls.gov/"):
            raise urllib.error.HTTPError(url, 403, "Forbidden", hdrs=None, fp=None)
        assert url == "https://api.fetch.tinyfish.ai"
        payload = {
            "results": [
                {
                    "url": "https://www.bls.gov/schedule/2026/",
                    "text": "Friday, January 09, 2026 08:30 AM Employment Situation for December 2025",
                }
            ],
            "errors": [],
        }
        return FakeResponse(json.dumps(payload).encode())

    monkeypatch.setenv("TINYFISH_API_KEY", "test-key")
    monkeypatch.setattr(
        "regime_data_fetch.bls_schedule.urllib.request.urlopen", fake_urlopen
    )

    text = fetch_bls_schedule_page_text("https://www.bls.gov/schedule/2026/")

    assert (
        text
        == "Friday, January 09, 2026 08:30 AM Employment Situation for December 2025"
    )
    assert calls == [
        "https://www.bls.gov/schedule/2026/",
        "https://api.fetch.tinyfish.ai",
    ]
