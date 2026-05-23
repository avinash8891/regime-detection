from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from regime_data_fetch.event_calendar import GroupABuildResult
from regime_data_fetch.event_calendar_reporting import (
    build_group_b_report,
)
from regime_data_fetch.event_sources.approvals import (
    append_approval_record,
    load_approval_overlay,
)
from regime_data_fetch.event_sources.budget_official_discovery import (
    BudgetOfficialDiscoveryGenerator,
    extract_govinfo_cr_records,
    extract_treasury_debt_limit_records,
    fetch_official_budget_records,
    iter_govinfo_public_law_urls,
)
from regime_data_fetch.event_sources.deterministic_budget import (
    DeterministicBudgetAdapter,
)
from regime_data_fetch.event_sources.models import (
    ApprovalRecord,
    EventCandidate,
    PromotionDecision,
)
from regime_data_fetch.event_sources.orchestrator import EventSourceOrchestrator
from regime_data_fetch.event_sources.validators_tinyfish import TinyFishValidator
from regime_data_fetch.event_sources.gpr_gdelt_fetchers import (
    _fetch_paged_json,
    fetch_acled_events as _fetch_acled_events,
    fetch_hdx_hapi_conflict_events as _fetch_hdx_hapi_conflict_events,
)
from regime_data_fetch.event_sources.validators_gpr_gdelt import GPRGDELTSignalGenerator
from regime_data_fetch.event_sources.validators_gpr_gdelt import (
    parse_acled_events,
    parse_hdx_hapi_conflict_events,
    parse_ucdp_events,
)


def _build_group_b_report(result: GroupABuildResult) -> dict[str, object]:
    return build_group_b_report(
        candidates=result.candidates,
        decisions=result.decisions,
        approval_overlay=result.approval_overlay,
    )


def test_parse_acled_events_aggregates_conflict_rows_by_day() -> None:
    payload = """
{
  "status": 200,
  "data": [
    {
      "event_date": "2022-02-24",
      "event_type": "Battles",
      "sub_event_type": "Armed clash",
      "country": "Ukraine",
      "fatalities": "42",
      "source": "Reuters",
      "notes": "Russian forces entered Ukraine."
    },
    {
      "event_date": "2022-02-24",
      "event_type": "Protests",
      "sub_event_type": "Peaceful protest",
      "country": "Russia",
      "fatalities": "0",
      "source": "Local media",
      "notes": "Anti-war protest."
    }
  ]
}
"""

    rows = parse_acled_events(
        payload, source_url="https://acleddata.com/api/acled/read"
    )

    assert rows == [
        {
            "date": dt.date(2022, 2, 24),
            "event_count": 2,
            "fatalities": 42,
            "dominant_theme": "ACLED Battles / Protests: Ukraine, Russia",
            "source_url": "https://acleddata.com/api/acled/read",
        }
    ]


def test_acled_fetcher_paginates_until_short_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACLED_API_TOKEN", "fixture-token")
    calls: list[str] = []

    def fake_http_text(url: str, *, headers: dict[str, str]) -> str:
        calls.append(url)
        assert headers["Authorization"] == "Bearer fixture-token"
        page = "2" if "page=2" in url else "1"
        row_count = 5000 if page == "1" else 1
        rows = [
            {
                "event_date": "2022-02-24",
                "event_type": "Battles",
                "country": "Ukraine",
                "fatalities": "1",
            }
            for _ in range(row_count)
        ]
        return '{"data": ' + json.dumps(rows) + "}"

    monkeypatch.setattr(
        "regime_data_fetch.event_sources.gpr_gdelt_fetchers._http_text",
        fake_http_text,
    )

    payload = _fetch_acled_events(2022, 2022)
    rows = parse_acled_events(
        payload, source_url="https://acleddata.com/api/acled/read"
    )

    assert len(calls) == 2
    assert rows[0]["event_count"] == 5001


def test_parse_ucdp_events_aggregates_candidate_events_by_day() -> None:
    payload = """
{
  "Result": [
    {
      "date_start": "2022-02-24",
      "country": "Ukraine",
      "type_of_violence": 1,
      "best": 10,
      "source_article": "https://example.test/ucdp/1"
    },
    {
      "date_start": "2022-02-24",
      "country": "Ukraine",
      "type_of_violence": 3,
      "deaths_civilians": 5
    }
  ]
}
"""

    rows = parse_ucdp_events(
        payload, source_url="https://ucdpapi.pcr.uu.se/api/gedevents/26.0.3"
    )

    assert rows == [
        {
            "date": dt.date(2022, 2, 24),
            "event_count": 2,
            "fatalities": 15,
            "dominant_theme": "UCDP organized violence: Ukraine",
            "source_url": "https://example.test/ucdp/1",
        }
    ]


def test_parse_hdx_hapi_conflict_events_emits_monthly_admin_evidence() -> None:
    payload = """
{
  "data": [
    {
      "event_type": "political_violence",
      "events": 17,
      "fatalities": 91,
      "reference_period_start": "2022-02-01",
      "reference_period_end": "2022-02-28",
      "location_name": "Ukraine"
    }
  ]
}
"""

    rows = parse_hdx_hapi_conflict_events(
        payload,
        source_url="https://hapi.humdata.org/api/v2/coordination-context/conflict-events",
    )

    assert rows == [
        {
            "date": dt.date(2022, 2, 1),
            "event_count": 17,
            "fatalities": 91,
            "dominant_theme": "HDX HAPI monthly political_violence: Ukraine",
            "source_url": "https://hapi.humdata.org/api/v2/coordination-context/conflict-events",
        }
    ]


def test_hdx_hapi_fetcher_uses_generated_app_identifier_and_supported_date_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HDX_HAPI_APP_IDENTIFIER", raising=False)
    monkeypatch.setenv("HDX_HAPI_APP_NAME", "regime-detection-test")
    monkeypatch.setenv("HDX_HAPI_APP_EMAIL", "regime-detection@example.invalid")
    captured: dict[str, object] = {}

    def fake_fetch_paged_json(
        base_url: str,
        *,
        headers: dict[str, str],
        result_key: str,
        extra_params: dict[str, str],
    ) -> str:
        captured.update(
            base_url=base_url,
            headers=headers,
            result_key=result_key,
            extra_params=extra_params,
        )
        return '{"data": []}'

    monkeypatch.setattr(
        "regime_data_fetch.event_sources.gpr_gdelt_fetchers._fetch_paged_json",
        fake_fetch_paged_json,
    )

    payload = _fetch_hdx_hapi_conflict_events(2022, 2023)

    assert payload == '{"data": []}'
    assert captured["result_key"] == "data"
    extra_params = captured["extra_params"]
    assert isinstance(extra_params, dict)
    assert extra_params["start_date"] == "2022-01-01"
    assert extra_params["end_date"] == "2023-12-31"
    assert "reference_period_start" not in extra_params
    assert "reference_period_end" not in extra_params
    assert (
        extra_params["app_identifier"]
        == "cmVnaW1lLWRldGVjdGlvbi10ZXN0OnJlZ2ltZS1kZXRlY3Rpb25AZXhhbXBsZS5pbnZhbGlk"
    )


def test_hdx_hapi_fetcher_returns_none_and_logs_without_app_identifier(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "HDX_HAPI_APP_IDENTIFIER",
        "HDX_APP_IDENTIFIER",
        "HDX_HAPI_APP_NAME",
        "HDX_HAPI_APP_EMAIL",
    ):
        monkeypatch.delenv(key, raising=False)
    assert _fetch_hdx_hapi_conflict_events(2022, 2022) is None
    assert "HDX HAPI app identifier unavailable" in caplog.text


def test_conflict_pager_continues_without_total_count_until_short_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "regime_data_fetch.event_sources.gpr_gdelt_fetchers.CONFLICT_API_PAGE_SIZE", 2
    )
    calls: list[str] = []

    def fake_http_text(url: str, *, headers: dict[str, str]) -> str:
        del headers
        calls.append(url)
        if "offset=0" in url:
            return '{"data": [{"id": 1}, {"id": 2}]}'
        if "offset=2" in url:
            return '{"data": [{"id": 3}]}'
        raise AssertionError(f"unexpected page request: {url}")

    monkeypatch.setattr(
        "regime_data_fetch.event_sources.gpr_gdelt_fetchers._http_text",
        fake_http_text,
    )

    payload = _fetch_paged_json(
        "https://example.test/conflict-events",
        headers={},
        result_key="data",
        extra_params={"app_identifier": "fixture"},
    )

    assert json.loads(payload) == {"data": [{"id": 1}, {"id": 2}, {"id": 3}]}
    assert len(calls) == 2


def test_conflict_pager_fails_loudly_on_short_page_before_total_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "regime_data_fetch.event_sources.gpr_gdelt_fetchers.CONFLICT_API_PAGE_SIZE", 2
    )

    def fake_http_text(url: str, *, headers: dict[str, str]) -> str:
        del url, headers
        return '{"TotalCount": 5, "data": [{"id": 1}]}'

    monkeypatch.setattr(
        "regime_data_fetch.event_sources.gpr_gdelt_fetchers._http_text",
        fake_http_text,
    )

    with pytest.raises(RuntimeError, match="returned short page before TotalCount"):
        _fetch_paged_json(
            "https://example.test/conflict-events",
            headers={},
            result_key="data",
            extra_params={"app_identifier": "fixture"},
        )


def test_generator_includes_acled_ucdp_and_hdx_candidate_sources() -> None:
    gpr_csv = """date,gpr
2022-02-20,100
2022-02-21,101
2022-02-22,99
2022-02-23,101
2022-02-24,500
2022-02-25,120
"""
    gdelt_csv = """date,event_count,dominant_theme,source_url
2022-02-24,1200,Russia invasion of Ukraine,https://example.test/gdelt/20220224
"""
    acled_json = """{"status": 200, "data": [{"event_date": "2022-02-24", "event_type": "Battles", "country": "Ukraine", "fatalities": "42"}]}"""
    ucdp_json = """{"Result": [{"date_start": "2022-02-24", "country": "Ukraine", "best": 10}]}"""
    hdx_json = """{"data": [{"event_type": "political_violence", "events": 17, "fatalities": 91, "reference_period_start": "2022-02-01", "reference_period_end": "2022-02-28", "location_name": "Ukraine"}]}"""
    generator = GPRGDELTSignalGenerator(
        gpr_fetcher=lambda: gpr_csv,
        gdelt_fetcher=lambda: gdelt_csv,
        acled_fetcher=lambda start_year, end_year: acled_json,
        ucdp_fetcher=lambda start_year, end_year: ucdp_json,
        hdx_hapi_fetcher=lambda start_year, end_year: hdx_json,
        min_history_days=3,
        stddev_threshold=2.0,
    )

    candidates = generator.generate(
        start_year=2022, end_year=2022, store=None, run_id=None
    )

    assert [
        (
            candidate.date,
            candidate.source_id,
            candidate.event_subtype,
            candidate.requires_manual_review,
        )
        for candidate in candidates
    ] == [
        (
            dt.date(2022, 2, 1),
            "hdx-hapi:conflict-events",
            "hdx_hapi_monthly_conflict",
            True,
        ),
        (dt.date(2022, 2, 24), "acled:events", "acled_conflict_event", True),
        (dt.date(2022, 2, 24), "gdelt:events-v2", "gdelt_volume_spike", True),
        (
            dt.date(2022, 2, 24),
            "gpr:caldara-iacoviello",
            "gpr_headline_spike",
            True,
        ),
        (dt.date(2022, 2, 24), "ucdp:ged-candidate", "ucdp_organized_violence", True),
    ]


def test_budget_official_discovery_auto_promotes_two_independent_official_sources() -> (
    None
):
    records_json = """
[
  {
    "date": "2023-06-05",
    "event_subtype": "debt_ceiling",
    "source_id": "treasury.gov:debt-limit",
    "source_url": "https://home.treasury.gov/news/press-releases/jy1480",
    "raw_title": "Treasury debt limit X-date notice",
    "raw_snippet": "Treasury projected the X-date."
  },
  {
    "date": "2023-06-05",
    "event_subtype": "debt_ceiling",
    "source_id": "congress.gov:public-law",
    "source_url": "https://www.congress.gov/bill/118th-congress/house-bill/3746",
    "raw_title": "Fiscal Responsibility Act",
    "raw_snippet": "Congress suspended the debt limit."
  }
]
"""
    generator = BudgetOfficialDiscoveryGenerator(records_fetcher=lambda: records_json)
    candidates = generator.generate(
        start_year=2023, end_year=2023, store=None, run_id=None
    )
    orchestrator = EventSourceOrchestrator(
        primary_adapters=[], candidate_generators=[generator], validators=[]
    )

    _, _, decisions, rendered = orchestrator.run(
        start_year=2023, end_year=2023, store=None, run_id=None
    )

    assert [
        (candidate.date, candidate.event_subtype, candidate.source_id)
        for candidate in candidates
    ] == [
        (dt.date(2023, 6, 5), "debt_ceiling", "treasury.gov:debt-limit"),
        (dt.date(2023, 6, 5), "debt_ceiling", "congress.gov:public-law"),
    ]
    assert decisions[0].outcome == "promote"
    assert decisions[0].source_count == 2
    assert [(event.date, event.type) for event in rendered] == [
        (dt.date(2023, 6, 5), "budget")
    ]


def test_treasury_debt_limit_extractor_uses_official_index_links() -> None:
    index_html = """
<a href="/system/files/136/Debt-Limit-Letter-to-Congress-Members-20230526-McCarthy.pdf">
Secretary Yellen Sends Debt Limit Letter to Congress (5/26/23)</a>
<a href="/system/files/136/07282023_Letter_to_Speaker_McCarthy_2023_CSRDF_Report.pdf">
Assistant Secretary Davidson Sends Letter on Report on Fund Operations and Status of the CSRDF/PSRHBF under the DISP ending June 5, 2023 (7/28/23)</a>
"""

    records = extract_treasury_debt_limit_records(
        index_html,
        source_url="https://home.treasury.gov/policy-issues/financial-markets-financial-institutions-and-fiscal-service/debt-limit",
    )

    assert records == [
        {
            "date": "2023-05-26",
            "event_subtype": "debt_ceiling",
            "source_id": "treasury.gov:debt-limit",
            "source_url": "https://home.treasury.gov/system/files/136/Debt-Limit-Letter-to-Congress-Members-20230526-McCarthy.pdf",
            "raw_title": "Secretary Yellen Sends Debt Limit Letter to Congress (5/26/23)",
            "raw_snippet": "Treasury debt-limit notice dated 2023-05-26.",
        },
        {
            "date": "2023-06-05",
            "event_subtype": "debt_ceiling",
            "source_id": "treasury.gov:debt-limit",
            "source_url": "https://home.treasury.gov/system/files/136/07282023_Letter_to_Speaker_McCarthy_2023_CSRDF_Report.pdf",
            "raw_title": "Assistant Secretary Davidson Sends Letter on Report on Fund Operations and Status of the CSRDF/PSRHBF under the DISP ending June 5, 2023 (7/28/23)",
            "raw_snippet": "Treasury debt-limit DISP period ending 2023-06-05.",
        },
    ]


def test_govinfo_cr_extractor_reads_public_law_expiration_date() -> None:
    public_law_html = """
CONTINUING APPROPRIATIONS AND EXTENSIONS ACT, 2025
Sec. 106. Unless otherwise provided for in this Act or in the applicable
appropriations Act for fiscal year 2025, appropriations and funds made
available and authority granted pursuant to this Act shall be available
until whichever of the following first occurs:
(3) &lt;&lt;NOTE: Expiration date.&gt;&gt; December 20, 2024.
"""

    records = extract_govinfo_cr_records(
        public_law_html,
        source_url="https://www.govinfo.gov/content/pkg/PLAW-118publ83/html/PLAW-118publ83.htm",
    )

    assert records == [
        {
            "date": "2024-12-20",
            "event_subtype": "cr_expiration",
            "source_id": "govinfo.gov:public-law",
            "source_url": "https://www.govinfo.gov/content/pkg/PLAW-118publ83/html/PLAW-118publ83.htm",
            "raw_title": "CONTINUING APPROPRIATIONS AND EXTENSIONS ACT, 2025",
            "raw_snippet": "GovInfo continuing appropriations expiration date 2024-12-20.",
        }
    ]


def test_govinfo_cr_extractor_reads_amended_public_law_expiration_date() -> None:
    public_law_html = """
DIVISION A -- FURTHER CONTINUING APPROPRIATIONS ACT, 2025
Sec. 101. The Continuing Appropriations Act, 2025 (division A of
Public Law 118-83) is amended--
(1) by striking the date specified in section 106(3) and inserting
``March 14, 2025'';
Sec. 155. To remain available until September 30, 2029.
"""

    records = extract_govinfo_cr_records(
        public_law_html,
        source_url="https://www.govinfo.gov/content/pkg/PLAW-118publ158/html/PLAW-118publ158.htm",
    )

    assert [(record["date"], record["raw_snippet"]) for record in records] == [
        ("2025-03-14", "GovInfo continuing appropriations expiration date 2025-03-14.")
    ]


def test_default_official_budget_fetcher_combines_live_official_sources() -> None:
    def fake_fetch(url: str) -> str:
        if "home.treasury.gov" in url:
            return """
<a href="/system/files/136/Debt-Limit-Letter-to-Congress-20211116.pdf">
Secretary Yellen Sends Debt Limit Letter to Congress (11/16/2021)</a>
"""
        if "PLAW-118publ83" in url:
            return """
CONTINUING APPROPRIATIONS AND EXTENSIONS ACT, 2025
Sec. 106. (3) &lt;&lt;NOTE: Expiration date.&gt;&gt; December 20, 2024.
"""
        raise AssertionError(url)

    records = fetch_official_budget_records(
        text_fetcher=fake_fetch,
        govinfo_public_law_urls=[
            "https://www.govinfo.gov/content/pkg/PLAW-118publ83/html/PLAW-118publ83.htm"
        ],
    )

    assert [
        (record["date"], record["event_subtype"], record["source_id"])
        for record in records
    ] == [
        ("2021-11-16", "debt_ceiling", "treasury.gov:debt-limit"),
        ("2024-12-20", "cr_expiration", "govinfo.gov:public-law"),
    ]


def test_govinfo_public_law_url_enumerator_covers_2016_to_current_congresses() -> None:
    urls = list(
        iter_govinfo_public_law_urls(
            start_year=2016, end_year=2026, max_public_law_number=2
        )
    )

    assert (
        urls[0]
        == "https://www.govinfo.gov/content/pkg/PLAW-114publ1/html/PLAW-114publ1.htm"
    )
    assert (
        urls[-1]
        == "https://www.govinfo.gov/content/pkg/PLAW-119publ2/html/PLAW-119publ2.htm"
    )
    assert len(urls) == 12


def test_generator_default_fetcher_uses_requested_years_for_govinfo_discovery() -> None:
    fetched_urls: list[str] = []

    def fake_fetch(url: str) -> str:
        fetched_urls.append(url)
        if "home.treasury.gov" in url:
            return ""
        if "PLAW-117publ70" in url:
            return """
FURTHER CONTINUING APPROPRIATIONS ACT, 2022
The Continuing Appropriations Act, 2022 is amended by striking the date
specified in section 106(3) and inserting ``February 18, 2022'';
"""
        return ""

    generator = BudgetOfficialDiscoveryGenerator(
        text_fetcher=fake_fetch,
        max_govinfo_public_law_number=70,
        as_of_date=dt.date(2026, 5, 15),
    )

    candidates = generator.generate(
        start_year=2021, end_year=2022, store=None, run_id=None
    )

    assert any("PLAW-117publ70" in url for url in fetched_urls)
    assert [
        (candidate.date, candidate.event_subtype, candidate.source_id)
        for candidate in candidates
    ] == [(dt.date(2022, 2, 18), "cr_expiration", "govinfo.gov:public-law")]


def test_tinyfish_unavailable_returns_unknown_for_review_candidates() -> None:
    candidate = DeterministicBudgetAdapter(as_of_date=dt.date(2026, 5, 14)).fetch(
        start_year=2026,
        end_year=2026,
        store=None,
        run_id=None,
    )[0]
    validator = TinyFishValidator(
        search_fetcher=lambda candidate: (_ for _ in ()).throw(
            RuntimeError("not authenticated")
        )
    )

    validations = validator.validate([candidate], store=None, run_id=None)

    assert [
        (validation.candidate_key, validation.validator_id, validation.verdict)
        for validation in validations
    ] == [(("budget", dt.date(2026, 9, 30)), "tinyfish:search-extract", "unknown")]


@pytest.mark.parametrize("payload", ["{not json", ["not", "a", "mapping"]])
def test_tinyfish_invalid_payload_returns_unknown(payload: object) -> None:
    candidate = DeterministicBudgetAdapter(as_of_date=dt.date(2026, 5, 14)).fetch(
        start_year=2026,
        end_year=2026,
        store=None,
        run_id=None,
    )[0]
    validator = TinyFishValidator(search_fetcher=lambda candidate: payload)

    validations = validator.validate([candidate], store=None, run_id=None)

    assert [
        (validation.candidate_key, validation.validator_id, validation.verdict)
        for validation in validations
    ] == [(("budget", dt.date(2026, 9, 30)), "tinyfish:search-extract", "unknown")]


def test_append_approval_record_validates_and_round_trips(tmp_path: Path) -> None:
    overlay_path = tmp_path / "group_b_approvals.yaml"
    overlay_path.write_text("approvals: []\n")

    append_approval_record(
        overlay_path,
        event_type="geopolitical_event",
        event_date=dt.date(2022, 2, 24),
        candidate_id="abc123",
        source_count=2,
        approver="avinash",
        approved_at=dt.date(2026, 5, 14),
        notes="Russia invasion of Ukraine.",
    )

    approvals = load_approval_overlay(overlay_path)
    assert len(approvals) == 1
    assert approvals[0].event_type == "geopolitical_event"
    assert approvals[0].evidence_candidate_id == "abc123"
    assert approvals[0].evidence_source_count == 2
    assert approvals[0].notes == "Russia invasion of Ukraine."


def test_append_approval_record_rejects_duplicate_without_rewriting_overlay(
    tmp_path: Path,
) -> None:
    overlay_path = tmp_path / "group_b_approvals.yaml"
    overlay_path.write_text(
        """
approvals:
  - event_type: geopolitical_event
    date: "2022-02-24"
    approved_label: geopolitical_event
    approver: avinash
    approved_at: "2026-05-14"
    evidence_candidate_id: "abc123"
    evidence_source_count: 2
"""
    )
    original_text = overlay_path.read_text()

    with pytest.raises(ValueError, match="duplicate approval"):
        append_approval_record(
            overlay_path,
            event_type="geopolitical_event",
            event_date=dt.date(2022, 2, 24),
            candidate_id="def456",
            source_count=2,
            approver="avinash",
            approved_at=dt.date(2026, 5, 15),
            notes="duplicate",
        )

    assert overlay_path.read_text() == original_text


def test_group_b_report_surfaces_stale_approval_states() -> None:
    candidate = EventCandidate(
        date=dt.date(2022, 2, 24),
        event_type="geopolitical_event",
        market="GLOBAL",
        importance="high",
        source_id="gpr:caldara-iacoviello",
        source_url=None,
        raw_title="Russia invasion of Ukraine",
        raw_snippet="fixture",
        is_future_scheduled=False,
        confidence="medium",
        requires_manual_review=True,
        candidate_id="new-id",
    )
    contradicted = EventCandidate(
        **{
            **candidate.__dict__,
            "date": dt.date(2022, 2, 26),
            "candidate_id": "contradicted-id",
        }
    )
    approvals = [
        ApprovalRecord(
            "geopolitical_event",
            dt.date(2022, 2, 24),
            "geopolitical_event",
            "avinash",
            dt.date(2026, 5, 14),
            "old-id",
            2,
        ),
        ApprovalRecord(
            "geopolitical_event",
            dt.date(2022, 2, 25),
            "geopolitical_event",
            "avinash",
            dt.date(2026, 5, 14),
            "missing-id",
            1,
        ),
        ApprovalRecord(
            "geopolitical_event",
            dt.date(2022, 2, 26),
            "geopolitical_event",
            "avinash",
            dt.date(2026, 5, 14),
            "contradicted-id",
            1,
        ),
    ]
    result = GroupABuildResult(
        scheduled_events=[],
        candidates=[candidate, contradicted],
        validations=[],
        decisions=[
            PromotionDecision(
                ("geopolitical_event", dt.date(2022, 2, 24)),
                "promote",
                "medium",
                1,
                False,
                "overlay",
            ),
            PromotionDecision(
                ("geopolitical_event", dt.date(2022, 2, 26)),
                "quarantine",
                "low",
                1,
                True,
                "contradict",
            ),
        ],
        output_paths={},
        approval_overlay=approvals,
    )

    report = _build_group_b_report(result)

    assert report["stale_evidence"] == [
        {"event_type": "geopolitical_event", "date": "2022-02-24"}
    ]
    assert report["stale_approvals"] == [
        {"event_type": "geopolitical_event", "date": "2022-02-25"}
    ]
    assert report["contradicted_approvals"] == [
        {"event_type": "geopolitical_event", "date": "2022-02-26"}
    ]
