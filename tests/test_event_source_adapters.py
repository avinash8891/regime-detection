from __future__ import annotations

import datetime as dt

from regime_data_fetch.event_sources.deterministic_election import ElectionAdapter
from regime_data_fetch.event_sources.official_boe import (
    NEWS_API_URL,
    OfficialBOEAdapter,
    parse_boe_mpc_dates_page,
    parse_boe_news_api_results,
    parse_boe_upcoming_mpc_dates,
)
from regime_data_fetch.event_sources.official_boj import (
    OfficialBOJAdapter,
    parse_boj_mpm_dates,
)
from regime_data_fetch.event_sources.official_ecb import (
    ARCHIVE_INDEX_URL,
    OfficialECBAdapter,
    parse_ecb_current_calendar,
    parse_ecb_decision_archive,
)
from regime_data_fetch.event_sources._common import FetchTextResult


def test_ecb_parses_archive_snippet_and_current_calendar_day_two_only() -> None:
    archive_html = """
    <dt isoDate="2026-04-30"><div class="date">30 April 2026</div></dt>
    <dd><div class="title"><a href="/press/pr/date/2026/html/ecb.mp260430~81b7179e6f.en.html">Monetary policy decisions</a></div></dd>
    """
    current_html = """
    <dt>10/06/2026</dt>
    <dd>Governing Council of the ECB: monetary policy meeting in Frankfurt (Day 1)</dd>
    <dt>11/06/2026</dt>
    <dd>Governing Council of the ECB: monetary policy meeting in Frankfurt (Day 2), followed by press conference</dd>
    <dt>25/06/2026</dt>
    <dd>General Council meeting of the ECB (virtual)</dd>
    """

    archive = parse_ecb_decision_archive(archive_html, as_of_date=dt.date(2026, 5, 14))
    current = parse_ecb_current_calendar(current_html, as_of_date=dt.date(2026, 5, 14))

    assert [
        (candidate.date, candidate.event_type, candidate.source_url)
        for candidate in archive
    ] == [
        (
            dt.date(2026, 4, 30),
            "ECB_decision",
            "https://www.ecb.europa.eu/press/pr/date/2026/html/ecb.mp260430~81b7179e6f.en.html",
        )
    ]
    assert [candidate.date for candidate in current] == [dt.date(2026, 6, 11)]
    assert current[0].is_future_scheduled is True


def test_boe_parses_current_page_and_news_api_results() -> None:
    upcoming_html = """
    <h2>2026 confirmed dates</h2>
    <table><tbody>
    <tr><td>Thursday 5 February</td><td><a href="/monetary-policy-summary-and-minutes/2026/february-2026">February MPC Summary and minutes</a></td></tr>
    <tr><td>Thursday 19 March</td><td><a href="/monetary-policy-summary-and-minutes/2026/march-2026">March MPC Summary and minutes</a></td></tr>
    </tbody></table>
    """
    api_results = """
    <a href="/monetary-policy-summary-and-minutes/2025/december-2025" class="release release-news">
      <time class="release-date" itemprop="datePublished" datetime="2025-12-18">18 December 2025</time>
      <h3 itemprop="name" class="list exclude-navigation">Bank Rate reduced to 3.75% - December 2025 Monetary Policy Summary and Minutes</h3>
    </a>
    """

    upcoming = parse_boe_upcoming_mpc_dates(
        upcoming_html, as_of_date=dt.date(2026, 1, 1)
    )
    historical = parse_boe_news_api_results(api_results, as_of_date=dt.date(2026, 1, 1))

    assert [candidate.date for candidate in upcoming] == [
        dt.date(2026, 2, 5),
        dt.date(2026, 3, 19),
    ]
    assert (
        upcoming[0].source_url
        == "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2026/february-2026"
    )
    assert [(candidate.date, candidate.raw_title) for candidate in historical] == [
        (
            dt.date(2025, 12, 18),
            "Bank Rate reduced to 3.75% - December 2025 Monetary Policy Summary and Minutes",
        )
    ]


def test_boe_parses_annual_mpc_dates_page() -> None:
    html = """
    <h1>Monetary Policy Committee dates for 2025</h1>
    <table><tbody>
    <tr><td><strong>MPC Announcement and Minutes publication</strong></td><td>Monetary Policy Report publication</td></tr>
    <tr><td>&nbsp;6 February 2025</td><td>&nbsp;6 February 2025</td></tr>
    <tr><td>&nbsp;20 March 2025</td><td>&nbsp;</td></tr>
    <tr><td>No meeting</td><td>&nbsp;</td></tr>
    </tbody></table>
    """

    candidates = parse_boe_mpc_dates_page(
        html,
        source_url="https://www.bankofengland.co.uk/news/2024/september/monetary-policy-committee-dates-for-2025",
        as_of_date=dt.date(2024, 9, 20),
    )

    assert [(candidate.date, candidate.source_url) for candidate in candidates] == [
        (
            dt.date(2025, 2, 6),
            "https://www.bankofengland.co.uk/news/2024/september/monetary-policy-committee-dates-for-2025",
        ),
        (
            dt.date(2025, 3, 20),
            "https://www.bankofengland.co.uk/news/2024/september/monetary-policy-committee-dates-for-2025",
        ),
    ]


def test_boe_parses_legacy_annual_mpc_month_day_table() -> None:
    html = """
    <h3>2017 MPC dates</h3>
    <table>
      <tr>
        <td><strong>Month</strong></td>
        <td><h3>MPC announcement and<br />meeting minutes publication</h3></td>
        <td><h3>Inflation Report publication</h3></td>
      </tr>
      <tr><td><strong>January 2017</strong></td><td>No meeting</td><td></td></tr>
      <tr><td><strong>February 2017</strong></td><td>Thursday 2</td><td>Thursday 2</td></tr>
      <tr><td><strong>March 2017</strong></td><td>Thursday 16</td><td></td></tr>
    </table>
    """

    candidates = parse_boe_mpc_dates_page(
        html,
        source_url="https://www.bankofengland.co.uk/news/2016/september/mpc-announcement-dates-for-2017-and-2018",
        as_of_date=dt.date(2016, 9, 29),
    )

    assert [candidate.date for candidate in candidates] == [
        dt.date(2017, 2, 2),
        dt.date(2017, 3, 16),
    ]


def test_boe_parses_annual_mpc_section_year_table() -> None:
    html = """
    <h3>2019 confirmed dates</h3>
    <table>
      <tr><td><p><strong>MPC Announcement and Minutes Publication</strong></p></td><td></td></tr>
      <tr><td><p>Thursday 7 February</p></td><td><p>Thursday 7 February</p></td></tr>
      <tr><td><p>Thursday 21 March</p></td><td></td></tr>
    </table>
    """

    candidates = parse_boe_mpc_dates_page(
        html,
        source_url="https://www.bankofengland.co.uk/news/2018/september/monetary-policy-committee-dates-for-2019",
        as_of_date=dt.date(2018, 9, 21),
    )

    assert [candidate.date for candidate in candidates] == [
        dt.date(2019, 2, 7),
        dt.date(2019, 3, 21),
    ]


def test_boe_annual_parser_ignores_malformed_table_rows() -> None:
    html = """
    <h3>2026 confirmed dates</h3>
    <table>
      <tr><td><p>MPC Announcement and Minutes Publication</p></td></tr>
      <tr><td><p>Publication date to be confirmed</p></td><td></td></tr>
      <tr><td><p>No meeting</p></td><td></td></tr>
    </table>
    """

    candidates = parse_boe_mpc_dates_page(
        html,
        source_url="https://www.bankofengland.co.uk/news/2025/september/monetary-policy-committee-dates-for-2026",
        as_of_date=dt.date(2025, 9, 20),
    )

    assert candidates == []


def test_boe_annual_parser_keeps_section_year_for_month_day_rollover() -> None:
    html = """
    <h3>2020 confirmed dates</h3>
    <table>
      <tr><td><p>MPC Announcement and Minutes Publication</p></td><td></td></tr>
      <tr><td><p>Thursday 19 December</p></td><td></td></tr>
    </table>
    <h3>2021 confirmed dates</h3>
    <table>
      <tr><td><p>MPC Announcement and Minutes Publication</p></td><td></td></tr>
      <tr><td><p>Thursday 4 February</p></td><td></td></tr>
    </table>
    """

    candidates = parse_boe_mpc_dates_page(
        html,
        source_url="https://www.bankofengland.co.uk/news/2019/september/monetary-policy-committee-dates",
        as_of_date=dt.date(2019, 9, 20),
    )

    assert [candidate.date for candidate in candidates] == [
        dt.date(2020, 12, 19),
        dt.date(2021, 2, 4),
    ]


def test_boe_adapter_continues_pagination_across_empty_news_pages() -> None:
    pages = {
        1: '{"Results": "<a href=\\"/monetary-policy-summary-and-minutes/2026/march-2026\\"><time datetime=\\"2026-03-19\\"></time><h3 class=\\"list\\">March 2026 Monetary Policy Summary and Minutes</h3></a>"}',
        2: '{"Results": ""}',
        3: '{"Results": "<a href=\\"/monetary-policy-summary-and-minutes/2025/december-2025\\"><time datetime=\\"2025-12-18\\"></time><h3 class=\\"list\\">December 2025 Monetary Policy Summary and Minutes</h3></a>"}',
        4: '{"Results": "<a href=\\"/monetary-policy-summary-and-minutes/2015/december-2015\\"><time datetime=\\"2015-12-17\\"></time><h3 class=\\"list\\">December 2015 Monetary Policy Summary and Minutes</h3></a>"}',
    }
    adapter = OfficialBOEAdapter(
        as_of_date=dt.date(2026, 5, 14),
        text_fetcher=lambda url: "",
        news_api_fetcher=lambda page: pages[page],
        stop_on_empty_news_page=False,
    )

    candidates = adapter.fetch(start_year=2025, end_year=2026, store=None, run_id=None)

    assert [candidate.date for candidate in candidates] == [
        dt.date(2025, 12, 18),
        dt.date(2026, 3, 19),
    ]


def test_official_rate_adapters_default_to_typed_text_results() -> None:
    fetched_urls: list[str] = []

    def fake_result_fetcher(url: str) -> FetchTextResult:
        fetched_urls.append(url)
        if "ecb.europa.eu/press/govcdec/mopo/html/index.en.html" in url:
            return FetchTextResult(
                text="<div data-snippets='/press/govcdec/mopo/2026/html/index_include.en.html'></div>"
            )
        if "ecb.europa.eu/press/govcdec/mopo/2026/html/index_include.en.html" in url:
            return FetchTextResult(
                text='<dt isoDate="2026-04-30"></dt><dd><a href="/press/pr/date/2026/html/ecb.en.html">Monetary policy decisions</a></dd>'
            )
        if "ecb.europa.eu/press/calendars/mgcgc/html/index.en.html" in url:
            return FetchTextResult(text=None, error="timeout")
        return FetchTextResult(text="")

    candidates = OfficialECBAdapter(
        as_of_date=dt.date(2026, 5, 14),
        result_fetcher=fake_result_fetcher,
    ).fetch(start_year=2026, end_year=2026, store=None, run_id=None)

    assert fetched_urls == [
        "https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html",
        "https://www.ecb.europa.eu/press/govcdec/mopo/2026/html/index_include.en.html",
        "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html",
    ]
    assert [candidate.date for candidate in candidates] == [dt.date(2026, 4, 30)]


def test_ecb_adapter_reports_failed_markup_fetch_in_status() -> None:
    def fake_result_fetcher(url: str) -> FetchTextResult:
        if url.endswith("/press/govcdec/mopo/html/index.en.html"):
            return FetchTextResult(text=None, error="archive timeout")
        return FetchTextResult(text="")

    adapter = OfficialECBAdapter(
        as_of_date=dt.date(2026, 5, 14),
        result_fetcher=fake_result_fetcher,
    )

    candidates = adapter.fetch(start_year=2026, end_year=2026, store=None, run_id=None)

    assert candidates == []
    assert (
        adapter.last_source_statuses[
            "https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html"
        ].status
        == "failed"
    )
    assert (
        adapter.last_source_statuses[
            "https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html"
        ].error
        == "archive timeout"
    )
    assert adapter.last_run_status == "partial"


def test_ecb_adapter_reports_archive_layout_drift_in_status() -> None:
    def fake_result_fetcher(url: str) -> FetchTextResult:
        if url == ARCHIVE_INDEX_URL:
            return FetchTextResult(text="<html>changed archive layout</html>")
        return FetchTextResult(text="")

    adapter = OfficialECBAdapter(
        as_of_date=dt.date(2026, 5, 14),
        result_fetcher=fake_result_fetcher,
    )

    candidates = adapter.fetch(start_year=2026, end_year=2026, store=None, run_id=None)

    assert candidates == []
    assert adapter.last_source_statuses[ARCHIVE_INDEX_URL].status == "parser_layout_drift"
    assert "data-snippets" in (adapter.last_source_statuses[ARCHIVE_INDEX_URL].error or "")
    assert adapter.last_run_status == "partial"


def test_boe_adapter_reports_failed_markup_fetch_in_status() -> None:
    def fake_result_fetcher(url: str) -> FetchTextResult:
        if url.endswith("/monetary-policy/upcoming-mpc-dates"):
            return FetchTextResult(text=None, error="upcoming timeout")
        return FetchTextResult(text="")

    adapter = OfficialBOEAdapter(
        as_of_date=dt.date(2026, 5, 14),
        result_fetcher=fake_result_fetcher,
        news_api_fetcher=lambda _page: '{"Results": ""}',
        stop_on_empty_news_page=True,
    )

    candidates = adapter.fetch(start_year=2026, end_year=2026, store=None, run_id=None)

    assert candidates == []
    assert (
        adapter.last_source_statuses[
            "https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates"
        ].status
        == "failed"
    )
    assert (
        adapter.last_source_statuses[
            "https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates"
        ].error
        == "upcoming timeout"
    )
    assert adapter.last_run_status == "partial"


def test_boe_adapter_reports_non_json_news_payload_in_status() -> None:
    adapter = OfficialBOEAdapter(
        as_of_date=dt.date(2026, 5, 14),
        text_fetcher=lambda _url: "",
        news_api_fetcher=lambda _page: "<html>cloudflare error page</html>",
        stop_on_empty_news_page=True,
    )

    candidates = adapter.fetch(start_year=2026, end_year=2026, store=None, run_id=None)

    assert candidates == []
    status = adapter.last_source_statuses[f"{NEWS_API_URL}?page=1"]
    assert status.status == "partial"
    assert "not JSON" in (status.error or "")
    assert adapter.last_run_status == "partial"


def test_official_rate_adapters_preserve_legacy_text_fetcher_override() -> None:
    def fail_result_fetcher(url: str) -> FetchTextResult:
        raise AssertionError(url)

    ecb_candidates = OfficialECBAdapter(
        as_of_date=dt.date(2026, 5, 14),
        text_fetcher=lambda _url: "",
        result_fetcher=fail_result_fetcher,
    ).fetch(start_year=2026, end_year=2026, store=None, run_id=None)
    boe_candidates = OfficialBOEAdapter(
        as_of_date=dt.date(2026, 5, 14),
        text_fetcher=lambda _url: "",
        result_fetcher=fail_result_fetcher,
        news_api_fetcher=lambda _page: '{"Results": ""}',
        stop_on_empty_news_page=True,
    ).fetch(start_year=2026, end_year=2026, store=None, run_id=None)
    boj_candidates = OfficialBOJAdapter(
        as_of_date=dt.date(2026, 5, 14),
        text_fetcher=lambda _url: "",
        result_fetcher=fail_result_fetcher,
    ).fetch(start_year=2026, end_year=2026, store=None, run_id=None)

    assert ecb_candidates == []
    assert boe_candidates == []
    assert boj_candidates == []


def test_boj_parses_current_and_past_mpm_tables_using_final_meeting_day() -> None:
    html = """
    <h2 id="p2026">2026</h2>
    <table><tbody>
    <tr><td><a href="/en/mopo/mpmdeci/mpr_2026/k260123a.pdf">Jan. 22 (Thurs.), 23 (Fri.) [PDF 171KB]</a></td></tr>
    <tr><td>June 15 (Mon.), 16 (Tues.)</td></tr>
    </tbody></table>
    <h2 id="p2025">2025</h2>
    <table><tbody>
    <tr><td><a href="/en/mopo/mpmdeci/mpr_2025/k250501a.pdf">Apr. 30 (Wed.), May 1 (Thurs.) [PDF 352KB]</a></td></tr>
    </tbody></table>
    """

    candidates = parse_boj_mpm_dates(html, as_of_date=dt.date(2026, 5, 14))

    assert [candidate.date for candidate in candidates] == [
        dt.date(2025, 5, 1),
        dt.date(2026, 1, 23),
        dt.date(2026, 6, 16),
    ]
    assert candidates[-1].is_future_scheduled is True


def test_election_adapter_computes_federal_general_elections() -> None:
    candidates = ElectionAdapter(as_of_date=dt.date(2026, 5, 14)).fetch(
        start_year=2016,
        end_year=2028,
        store=None,
        run_id=None,
    )

    assert [
        (candidate.date, candidate.raw_title, candidate.window_days)
        for candidate in candidates
    ] == [
        (dt.date(2016, 11, 8), "2016 presidential federal general election", (-5, 10)),
        (dt.date(2018, 11, 6), "2018 midterm federal general election", (-5, 10)),
        (dt.date(2020, 11, 3), "2020 presidential federal general election", (-5, 10)),
        (dt.date(2022, 11, 8), "2022 midterm federal general election", (-5, 10)),
        (dt.date(2024, 11, 5), "2024 presidential federal general election", (-5, 10)),
        (dt.date(2026, 11, 3), "2026 midterm federal general election", (-5, 10)),
        (dt.date(2028, 11, 7), "2028 presidential federal general election", (-5, 10)),
    ]
