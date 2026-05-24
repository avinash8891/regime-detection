from __future__ import annotations

import datetime as dt
import io
import logging
from pathlib import Path
import zipfile

import pytest

from regime_data_fetch.event_calendar import GroupABuildResult
from regime_data_fetch.event_calendar_reporting import (
    build_group_b_report,
)
from regime_data_fetch.event_sources.approvals import (
    load_approval_overlay,
)
from regime_data_fetch.event_sources.deterministic_budget import (
    DeterministicBudgetAdapter,
)
from regime_data_fetch.event_sources.validators_gpr_gdelt import (
    ACLED_SOURCE_ID,
    ACLEDSignalGenerator,
    GDELTSignalGenerator,
    GPRSignalGenerator,
    detect_gpr_spikes,
    parse_ai_gpr_context,
    parse_gdelt_event_export,
    parse_gpr_monthly_country_context,
    parse_gpr_table,
)


def _build_group_b_report(result: GroupABuildResult) -> dict[str, object]:
    return build_group_b_report(
        candidates=result.candidates,
        decisions=result.decisions,
        approval_overlay=result.approval_overlay,
    )


def test_deterministic_budget_emits_fy_deadline_rows_on_nyse_sessions() -> None:
    candidates = DeterministicBudgetAdapter(as_of_date=dt.date(2026, 5, 14)).fetch(
        start_year=2016,
        end_year=2018,
        store=None,
        run_id=None,
    )

    assert [
        (
            candidate.date,
            candidate.event_type,
            candidate.event_subtype,
            candidate.source_id,
        )
        for candidate in candidates
    ] == [
        (
            dt.date(2016, 9, 30),
            "budget",
            "fy_deadline",
            "usa.gov:federal-budget-process",
        ),
        (
            dt.date(2017, 9, 29),
            "budget",
            "fy_deadline",
            "usa.gov:federal-budget-process",
        ),
        (
            dt.date(2018, 9, 28),
            "budget",
            "fy_deadline",
            "usa.gov:federal-budget-process",
        ),
    ]
    assert [candidate.requires_manual_review for candidate in candidates] == [
        False,
        False,
        False,
    ]
    assert [candidate.confidence for candidate in candidates] == [
        "high",
        "high",
        "high",
    ]


def test_deterministic_budget_rolls_weekend_deadlines_to_previous_nyse_session() -> (
    None
):
    candidates = DeterministicBudgetAdapter(as_of_date=dt.date(2026, 5, 14)).fetch(
        start_year=2017,
        end_year=2023,
        store=None,
        run_id=None,
    )

    by_year = {candidate.raw_title: candidate.date for candidate in candidates}

    assert by_year["US federal fiscal year 2017 deadline"] == dt.date(2017, 9, 29)
    assert by_year["US federal fiscal year 2018 deadline"] == dt.date(2018, 9, 28)
    assert by_year["US federal fiscal year 2023 deadline"] == dt.date(2023, 9, 29)


def test_load_approval_overlay_parses_valid_records(tmp_path: Path) -> None:
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
    evidence_source_count: 3
    importance: high
    window_days: [0, 0]
    notes: "Russia invasion of Ukraine."
"""
    )

    approvals = load_approval_overlay(overlay_path)

    assert len(approvals) == 1
    assert approvals[0].event_type == "geopolitical_event"
    assert approvals[0].date == dt.date(2022, 2, 24)
    assert approvals[0].evidence_candidate_id == "abc123"
    assert approvals[0].window_days == (0, 0)


def test_load_approval_overlay_rejects_duplicate_keys(tmp_path: Path) -> None:
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
    evidence_source_count: 3
  - event_type: geopolitical_event
    date: "2022-02-24"
    approved_label: geopolitical_event
    approver: avinash
    approved_at: "2026-05-14"
    evidence_candidate_id: "def456"
    evidence_source_count: 2
"""
    )

    with pytest.raises(ValueError, match="duplicate approval"):
        load_approval_overlay(overlay_path)


def test_gpr_gdelt_generator_flags_real_geopolitical_spike_date() -> None:
    gpr_csv = """date,gpr,gpr_act,gpr_threat,gpr_ma7,gpr_ma30,N10D
2022-02-20,100,100,100,100,100,100
2022-02-21,101,100,101,100,100,100
2022-02-22,99,99,100,100,100,100
2022-02-23,101,100,101,100,100,100
2022-02-24,500,700,125,220,150,900
2022-02-25,120,110,115,150,125,200
"""
    gdelt_csv = """date,event_count,dominant_theme,source_url
2022-02-24,1200,Russia invasion of Ukraine,https://example.test/gdelt/20220224
"""
    gpr = GPRSignalGenerator(
        gpr_fetcher=lambda: gpr_csv,
        min_history_days=3,
        stddev_threshold=2.0,
    )
    gdelt = GDELTSignalGenerator(
        gdelt_fetcher=lambda: gdelt_csv,
        merge_window_days=2,
    )

    candidates = sorted(
        [
            *gpr.generate(start_year=2022, end_year=2022, store=None, run_id=None),
            *gdelt.generate(start_year=2022, end_year=2022, store=None, run_id=None),
        ],
        key=lambda candidate: (candidate.date, candidate.source_id),
    )
    validations = [
        *gpr.validate(candidates, store=None, run_id=None),
        *gdelt.validate(candidates, store=None, run_id=None),
    ]

    assert [
        (
            candidate.date,
            candidate.event_type,
            candidate.source_id,
            candidate.event_subtype,
            candidate.importance,
            candidate.confidence,
            candidate.raw_title,
        )
        for candidate in candidates
    ] == [
        (
            dt.date(2022, 2, 24),
            "geopolitical_event",
            "gdelt:events-v2",
            "gdelt_volume_spike",
            "medium",
            "medium",
            "Russia invasion of Ukraine",
        ),
        (
            dt.date(2022, 2, 24),
            "geopolitical_event",
            "gpr:caldara-iacoviello",
            "gpr_acts_spike",
            "high",
            "high",
            "GPR acts-driven geopolitical risk spike",
        ),
    ]
    assert all(candidate.requires_manual_review for candidate in candidates)
    gpr_candidate = candidates[1]
    assert gpr_candidate.raw_snippet == (
        "GPR daily value 500.00 exceeded trailing threshold 102.22; "
        "components=headline,acts,threats,persistent_7d,persistent_30d; "
        "acts=700.00; threats=125.00; ma7=220.00; ma30=150.00; articles=900."
    )
    assert {
        (validation.candidate_key, validation.validator_id, validation.verdict)
        for validation in validations
    } == {
        (("geopolitical_event", dt.date(2022, 2, 24)), "gdelt:events-v2", "confirm"),
        (
            ("geopolitical_event", dt.date(2022, 2, 24)),
            "gpr:caldara-iacoviello",
            "confirm",
        ),
    }


def test_parse_gdelt_event_export_counts_material_conflict_rows() -> None:
    conflict = [""] * 61
    conflict[1] = "20220224"
    conflict[28] = "19"
    conflict[29] = "4"
    conflict[31] = "12"
    conflict[60] = "https://example.test/ukraine-invasion"
    cooperation = [""] * 61
    cooperation[1] = "20220224"
    cooperation[28] = "05"
    cooperation[29] = "1"
    cooperation[31] = "20"
    payload = ("\t".join(conflict) + "\n" + "\t".join(cooperation) + "\n").encode()

    rows = parse_gdelt_event_export(
        payload,
        source_url="http://data.gdeltproject.org/gdeltv2/20220224000000.export.CSV.zip",
    )

    assert rows == [
        {
            "date": dt.date(2022, 2, 24),
            "event_count": 12,
            "dominant_theme": "GDELT material conflict / protest volume",
            "source_url": "https://example.test/ukraine-invasion",
        }
    ]


def test_parse_gdelt_event_export_filters_to_expected_export_date() -> None:
    current = [""] * 61
    current[1] = "20220224"
    current[28] = "19"
    current[29] = "4"
    current[31] = "12"
    current[60] = "https://example.test/current"
    stale = [""] * 61
    stale[1] = "20210224"
    stale[28] = "19"
    stale[29] = "4"
    stale[31] = "99"
    stale[60] = "https://example.test/stale"
    payload = ("\t".join(stale) + "\n" + "\t".join(current) + "\n").encode()

    rows = parse_gdelt_event_export(
        payload,
        source_url="http://data.gdeltproject.org/gdeltv2/20220224000000.export.CSV.zip",
        expected_date=dt.date(2022, 2, 24),
    )

    assert [(row["date"], row["event_count"], row["source_url"]) for row in rows] == [
        (dt.date(2022, 2, 24), 12, "https://example.test/current")
    ]


def test_parse_gdelt_event_export_counts_concatenated_zip_archives() -> None:
    first = [""] * 61
    first[1] = "20220224"
    first[28] = "19"
    first[29] = "4"
    first[31] = "7"
    first[60] = "https://example.test/first"
    second = [""] * 61
    second[1] = "20220224"
    second[28] = "19"
    second[29] = "4"
    second[31] = "11"
    second[60] = "https://example.test/second"

    payload = b""
    for name, row in (
        ("20220224000000.export.CSV", first),
        ("20220224001500.export.CSV", second),
    ):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr(name, "\t".join(row) + "\n")
        payload += buffer.getvalue()

    rows = parse_gdelt_event_export(
        payload,
        source_url="http://data.gdeltproject.org/gdeltv2/20220224.export.CSV.zip",
        expected_date=dt.date(2022, 2, 24),
    )

    assert [(row["date"], row["event_count"]) for row in rows] == [
        (dt.date(2022, 2, 24), 18)
    ]


def test_gpr_generator_fetches_gdelt_daily_exports_for_spike_dates() -> None:
    gpr_csv = """date,gpr,gpr_act,gpr_threat,gpr_ma7,gpr_ma30,N10D
2022-02-20,100,100,100,100,100,100
2022-02-21,101,100,101,100,100,100
2022-02-22,99,99,100,100,100,100
2022-02-23,101,100,101,100,100,100
2022-02-24,500,700,125,220,150,900
2022-02-25,120,110,115,150,125,200
"""

    def gdelt_daily_fetcher(day: dt.date) -> bytes:
        row = [""] * 61
        row[1] = day.strftime("%Y%m%d")
        row[28] = "19"
        row[29] = "4"
        row[31] = "8"
        row[60] = f"https://example.test/gdelt/{day:%Y%m%d}"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr(f"{day:%Y%m%d}.export.CSV", "\t".join(row) + "\n")
        return buffer.getvalue()

    gpr = GPRSignalGenerator(
        gpr_fetcher=lambda: gpr_csv,
        min_history_days=3,
        stddev_threshold=2.0,
        merge_window_days=0,
    )
    gdelt = GDELTSignalGenerator(
        gdelt_daily_fetcher=gdelt_daily_fetcher,
        merge_window_days=0,
    )

    candidates = sorted(
        gpr.generate(start_year=2022, end_year=2022, store=None, run_id=None),
        key=lambda candidate: (candidate.date, candidate.source_id),
    )
    validations = [
        *gpr.validate(candidates, store=None, run_id=None),
        *gdelt.validate(candidates, store=None, run_id=None),
    ]
    candidates = sorted(
        [
            *candidates,
            *gdelt.generate(start_year=2022, end_year=2022, store=None, run_id=None),
        ],
        key=lambda candidate: (candidate.date, candidate.source_id),
    )

    assert [
        (candidate.date, candidate.source_id, candidate.raw_snippet)
        for candidate in candidates
    ] == [
        (
            dt.date(2022, 2, 24),
            "gpr:caldara-iacoviello",
            "GPR daily value 500.00 exceeded trailing threshold 102.22; "
            "components=headline,acts,threats,persistent_7d,persistent_30d; "
            "acts=700.00; threats=125.00; ma7=220.00; ma30=150.00; articles=900.",
        ),
    ]
    assert {
        (validation.candidate_key, validation.validator_id, validation.verdict)
        for validation in validations
    } == {
        (("geopolitical_event", dt.date(2022, 2, 24)), "gdelt:events-v2", "confirm"),
    }


def test_gpr_gdelt_generator_records_source_status_for_fetch_failures() -> None:
    def failing_gpr_fetcher() -> str:
        raise TimeoutError("gpr timed out")

    generator = GPRSignalGenerator(
        gpr_fetcher=failing_gpr_fetcher,
        min_history_days=3,
        stddev_threshold=2.0,
    )

    candidates = generator.generate(
        start_year=2022, end_year=2022, store=None, run_id=None
    )

    assert candidates == []
    assert generator.last_source_statuses["gpr:caldara-iacoviello"].status == "failed"
    assert (
        generator.last_source_statuses["gpr:caldara-iacoviello"].error
        == "gpr timed out"
    )
    assert generator.last_run_status == "partial"


def test_gdelt_generator_records_direct_gdelt_fetch_failure() -> None:
    def failing_gdelt_fetcher() -> str:
        raise TimeoutError("gdelt timed out")

    generator = GDELTSignalGenerator(gdelt_fetcher=failing_gdelt_fetcher)

    candidates = generator.generate(
        start_year=2022, end_year=2022, store=None, run_id=None
    )

    assert candidates == []
    assert generator.last_source_statuses["gdelt:events-v2"].status == "failed"
    assert generator.last_source_statuses["gdelt:events-v2"].error == "gdelt timed out"
    assert generator.last_run_status == "partial"


def test_acled_generator_records_optional_conflict_fetch_failure() -> None:
    def failing_acled_fetcher(_start_year: int, _end_year: int) -> str:
        raise TimeoutError("acled timed out")

    generator = ACLEDSignalGenerator(acled_fetcher=failing_acled_fetcher)

    candidates = generator.generate(
        start_year=2022, end_year=2022, store=None, run_id=None
    )

    assert candidates == []
    status = generator.last_source_statuses[ACLED_SOURCE_ID]
    assert status.status == "failed"
    assert status.error == "acled timed out"
    assert status.attempted_fetches == 1
    assert status.failed_fetches == 1
    assert generator.last_run_status == "partial"


def test_acled_generator_records_optional_conflict_parse_failure() -> None:
    generator = ACLEDSignalGenerator(
        acled_fetcher=lambda _start_year, _end_year: '{"data": [',
    )

    candidates = generator.generate(
        start_year=2022, end_year=2022, store=None, run_id=None
    )

    assert candidates == []
    status = generator.last_source_statuses[ACLED_SOURCE_ID]
    assert status.status == "failed"
    assert "Expecting value" in str(status.error)
    assert status.attempted_fetches == 1
    assert status.failed_fetches == 1
    assert generator.last_run_status == "partial"


def test_parse_gpr_table_logs_excel_parse_failure_before_csv_fallback(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_excel(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ValueError("not an excel workbook")

    monkeypatch.setattr(
        "regime_data_fetch.event_sources.validators_gpr_gdelt.pd.read_excel",
        fail_excel,
    )
    caplog.set_level(
        logging.DEBUG, logger="regime_data_fetch.event_sources.validators_gpr_gdelt"
    )

    frame = parse_gpr_table(b"date,gpr\n2026-05-01,123\n")

    assert frame.to_dict(orient="records") == [
        {
            "date": dt.date(2026, 5, 1),
            "gpr": 123,
            "gpr_act": None,
            "gpr_threat": None,
            "gpr_ma7": None,
            "gpr_ma30": None,
            "article_count": None,
            "event": "",
        }
    ]
    assert "GPR Excel parse failed; falling back to CSV parser" in caplog.text


def test_parse_gpr_table_keeps_daily_components() -> None:
    payload = """DAY,N10D,GPRD,GPRD_ACT,GPRD_THREAT,date,GPRD_MA30,GPRD_MA7,event
20220223,300,100,90,110,2022-02-23,95,98,
20220224,900,500,650,420,2022-02-24,130,180,Russia invasion of Ukraine
"""

    frame = parse_gpr_table(payload)

    assert frame.to_dict(orient="records") == [
        {
            "date": dt.date(2022, 2, 23),
            "gpr": 100,
            "gpr_act": 90,
            "gpr_threat": 110,
            "gpr_ma7": 98,
            "gpr_ma30": 95,
            "article_count": 300,
            "event": "",
        },
        {
            "date": dt.date(2022, 2, 24),
            "gpr": 500,
            "gpr_act": 650,
            "gpr_threat": 420,
            "gpr_ma7": 180,
            "gpr_ma30": 130,
            "article_count": 900,
            "event": "Russia invasion of Ukraine",
        },
    ]


def test_detect_gpr_spikes_classifies_acts_and_threats() -> None:
    frame = parse_gpr_table("""date,gpr,gpr_act,gpr_threat,gpr_ma7,gpr_ma30,N10D
2022-02-20,100,100,100,100,100,100
2022-02-21,101,100,101,100,100,100
2022-02-22,99,99,100,100,100,100
2022-02-23,101,100,101,100,100,100
2022-02-24,500,700,125,220,150,900
""")

    rows = detect_gpr_spikes(frame, min_history_days=3, stddev_threshold=2.0)

    assert rows == [
        {
            "date": dt.date(2022, 2, 24),
            "value": 500.0,
            "threshold": pytest.approx(102.21895141649746),
            "act_value": 700.0,
            "threat_value": 125.0,
            "ma7": 220.0,
            "ma30": 150.0,
            "article_count": 900.0,
            "event": "",
            "spike_components": (
                "headline",
                "acts",
                "threats",
                "persistent_7d",
                "persistent_30d",
            ),
            "dominant_component": "acts",
        }
    ]


def test_gpr_candidate_marks_threat_dominant_spikes() -> None:
    gpr_csv = """date,gpr,gpr_act,gpr_threat,gpr_ma7,gpr_ma30,N10D
2022-02-20,100,100,100,100,100,100
2022-02-21,101,100,101,100,100,100
2022-02-22,99,99,100,100,100,100
2022-02-23,101,100,101,100,100,100
2022-02-24,500,120,710,180,130,850
"""
    generator = GPRSignalGenerator(
        gpr_fetcher=lambda: gpr_csv,
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
            candidate.importance,
            candidate.confidence,
            candidate.raw_title,
        )
        for candidate in candidates
    ] == [
        (
            dt.date(2022, 2, 24),
            "gpr:caldara-iacoviello",
            "gpr_threats_spike",
            "high",
            "high",
            "GPR threats-driven geopolitical risk spike",
        )
    ]


def test_gpr_candidate_sets_persistence_window_days() -> None:
    gpr_csv = """date,gpr,gpr_act,gpr_threat,gpr_ma7,gpr_ma30,N10D
2022-02-20,100,100,100,100,100,100
2022-02-21,101,100,101,100,100,100
2022-02-22,99,99,100,100,100,100
2022-02-23,101,100,101,100,100,100
2022-02-24,500,700,125,220,150,900
"""
    generator = GPRSignalGenerator(
        gpr_fetcher=lambda: gpr_csv,
        min_history_days=3,
        stddev_threshold=2.0,
    )

    candidates = generator.generate(
        start_year=2022, end_year=2022, store=None, run_id=None
    )

    assert [(candidate.source_id, candidate.window_days) for candidate in candidates] == [
        ("gpr:caldara-iacoviello", (-2, 5))
    ]


def test_parse_gpr_monthly_country_context_returns_top_country_codes() -> None:
    payload = """month,GPRC_RUS,GPRC_UKR,GPRC_USA,GPRC_CHN
2022-02-01,320,280,20,110
2022-03-01,10,15,25,30
"""

    context = parse_gpr_monthly_country_context(
        payload, candidate_dates=[dt.date(2022, 2, 24)]
    )

    assert context == {
        dt.date(2022, 2, 24): "monthly_country_gpr=RUS:320.00,UKR:280.00,CHN:110.00"
    }


def test_parse_ai_gpr_context_combines_daily_event_type_and_country_role() -> None:
    daily = """Date,GPR_AI,GPR_AER,GPR_OIL
2022-02-24,475.5,350.0,20
"""
    event_type = """Date,GPR_AI,military_conflict,diplomatic_tension,sanctions
2022-02-01,220,175,40,90
"""
    country = """Date,GPR_AI,Russia_all,Russia_initiator,Ukraine_all,Ukraine_respondent,USA_all
2022-02-01,220,300,180,260,210,70
"""

    context = parse_ai_gpr_context(
        daily,
        event_type,
        country,
        candidate_dates=[dt.date(2022, 2, 24)],
    )

    assert context == {
        dt.date(2022, 2, 24): (
            "ai_gpr_daily=475.50; ai_gpr_event_type=military_conflict:175.00; "
            "ai_gpr_country=Russia_all:300.00,Ukraine_all:260.00,Ukraine_respondent:210.00"
        )
    }


def test_gpr_candidate_includes_monthly_and_ai_gpr_context() -> None:
    gpr_csv = """date,gpr,gpr_act,gpr_threat,gpr_ma7,gpr_ma30,N10D,event
2022-02-20,100,100,100,100,100,100,
2022-02-21,101,100,101,100,100,100,
2022-02-22,99,99,100,100,100,100,
2022-02-23,101,100,101,100,100,100,
2022-02-24,500,700,125,220,150,900,Russia invasion of Ukraine
"""
    monthly_country = """month,GPRC_RUS,GPRC_UKR,GPRC_USA
2022-02-01,320,280,20
"""
    ai_daily = """Date,GPR_AI,GPR_AER
2022-02-24,475.5,350.0
"""
    ai_event_type = """Date,GPR_AI,military_conflict,sanctions
2022-02-01,220,175,90
"""
    ai_country = """Date,GPR_AI,Russia_all,Ukraine_all,USA_all
2022-02-01,220,300,260,70
"""
    generator = GPRSignalGenerator(
        gpr_fetcher=lambda: gpr_csv,
        gpr_monthly_fetcher=lambda: monthly_country,
        ai_gpr_daily_fetcher=lambda: ai_daily,
        ai_gpr_eventtype_monthly_fetcher=lambda: ai_event_type,
        ai_gpr_country_monthly_fetcher=lambda: ai_country,
        min_history_days=3,
        stddev_threshold=2.0,
    )

    candidates = generator.generate(
        start_year=2022, end_year=2022, store=None, run_id=None
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.raw_title == "Russia invasion of Ukraine"
    assert "monthly_country_gpr=RUS:320.00,UKR:280.00,USA:20.00" in str(
        candidate.raw_snippet
    )
    assert "ai_gpr_daily=475.50" in str(candidate.raw_snippet)
    assert "ai_gpr_event_type=military_conflict:175.00" in str(candidate.raw_snippet)
    assert "ai_gpr_country=Russia_all:300.00,Ukraine_all:260.00,USA_all:70.00" in str(
        candidate.raw_snippet
    )


def test_gdelt_generator_records_partial_daily_gdelt_failure() -> None:
    gpr_csv = """date,gpr
2022-02-20,100
2022-02-21,101
2022-02-22,99
2022-02-23,101
2022-02-24,500
2022-02-25,120
"""

    def failing_gdelt_daily_fetcher(_day: dt.date) -> bytes:
        raise TimeoutError("gdelt timed out")

    gpr = GPRSignalGenerator(
        gpr_fetcher=lambda: gpr_csv,
        min_history_days=3,
        stddev_threshold=2.0,
        merge_window_days=0,
    )
    gdelt = GDELTSignalGenerator(
        gdelt_daily_fetcher=failing_gdelt_daily_fetcher,
        merge_window_days=0,
    )

    candidates = gpr.generate(
        start_year=2022, end_year=2022, store=None, run_id=None
    )
    validations = gdelt.validate(candidates, store=None, run_id=None)

    assert [(candidate.date, candidate.source_id) for candidate in candidates] == [
        (dt.date(2022, 2, 24), "gpr:caldara-iacoviello")
    ]
    assert validations == []
    assert gpr.last_source_statuses["gpr:caldara-iacoviello"].status == "ok"
    assert gdelt.last_source_statuses["gdelt:events-v2"].status == "partial"
    assert gdelt.last_source_statuses["gdelt:events-v2"].failed_fetches == 1
    assert gdelt.last_source_statuses["gdelt:events-v2"].empty_payload is False
