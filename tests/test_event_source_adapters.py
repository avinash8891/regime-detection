from __future__ import annotations

import datetime as dt

from regime_data_fetch.event_sources.deterministic_election import ElectionAdapter
from regime_data_fetch.event_sources.official_boe import OfficialBOEAdapter, parse_boe_news_api_results, parse_boe_upcoming_mpc_dates
from regime_data_fetch.event_sources.official_boj import parse_boj_mpm_dates
from regime_data_fetch.event_sources.official_ecb import parse_ecb_current_calendar, parse_ecb_decision_archive


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

    assert [(candidate.date, candidate.event_type, candidate.source_url) for candidate in archive] == [
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

    upcoming = parse_boe_upcoming_mpc_dates(upcoming_html, as_of_date=dt.date(2026, 1, 1))
    historical = parse_boe_news_api_results(api_results, as_of_date=dt.date(2026, 1, 1))

    assert [candidate.date for candidate in upcoming] == [dt.date(2026, 2, 5), dt.date(2026, 3, 19)]
    assert upcoming[0].source_url == "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2026/february-2026"
    assert [(candidate.date, candidate.raw_title) for candidate in historical] == [
        (dt.date(2025, 12, 18), "Bank Rate reduced to 3.75% - December 2025 Monetary Policy Summary and Minutes")
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

    assert [candidate.date for candidate in candidates] == [dt.date(2025, 12, 18), dt.date(2026, 3, 19)]


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

    assert [(candidate.date, candidate.raw_title, candidate.window_days) for candidate in candidates] == [
        (dt.date(2016, 11, 8), "2016 presidential federal general election", (-5, 10)),
        (dt.date(2018, 11, 6), "2018 midterm federal general election", (-5, 10)),
        (dt.date(2020, 11, 3), "2020 presidential federal general election", (-5, 10)),
        (dt.date(2022, 11, 8), "2022 midterm federal general election", (-5, 10)),
        (dt.date(2024, 11, 5), "2024 presidential federal general election", (-5, 10)),
        (dt.date(2026, 11, 3), "2026 midterm federal general election", (-5, 10)),
        (dt.date(2028, 11, 7), "2028 presidential federal general election", (-5, 10)),
    ]
