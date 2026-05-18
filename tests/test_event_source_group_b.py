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
from regime_data_fetch.event_sources.validators_gpr_gdelt import GPRGDELTSignalGenerator
from regime_data_fetch.event_sources.validators_gpr_gdelt import (
    ACLED_SOURCE_ID,
    parse_gdelt_event_export,
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
    generator = GPRGDELTSignalGenerator(
        gpr_fetcher=lambda: gpr_csv,
        gdelt_fetcher=lambda: gdelt_csv,
        min_history_days=3,
        stddev_threshold=2.0,
    )

    candidates = generator.generate(
        start_year=2022, end_year=2022, store=None, run_id=None
    )
    validations = generator.validate(candidates, store=None, run_id=None)

    assert [
        (candidate.date, candidate.event_type, candidate.source_id)
        for candidate in candidates
    ] == [
        (dt.date(2022, 2, 24), "geopolitical_event", "gdelt:events-v2"),
        (dt.date(2022, 2, 24), "geopolitical_event", "gpr:caldara-iacoviello"),
    ]
    assert all(candidate.requires_manual_review for candidate in candidates)
    assert all(candidate.confidence == "medium" for candidate in candidates)
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
    conflict = [""] * 58
    conflict[1] = "20220224"
    conflict[28] = "19"
    conflict[29] = "4"
    conflict[31] = "12"
    conflict[56] = "https://example.test/ukraine-invasion"
    cooperation = [""] * 58
    cooperation[1] = "20220224"
    cooperation[28] = "05"
    cooperation[29] = "1"
    cooperation[31] = "20"
    payload = ("\t".join(conflict) + "\n" + "\t".join(cooperation) + "\n").encode()

    rows = parse_gdelt_event_export(
        payload,
        source_url="https://data.gdeltproject.org/events/20220224.export.CSV.zip",
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
    current = [""] * 58
    current[1] = "20220224"
    current[28] = "19"
    current[29] = "4"
    current[31] = "12"
    current[56] = "https://example.test/current"
    stale = [""] * 58
    stale[1] = "20210224"
    stale[28] = "19"
    stale[29] = "4"
    stale[31] = "99"
    stale[56] = "https://example.test/stale"
    payload = ("\t".join(stale) + "\n" + "\t".join(current) + "\n").encode()

    rows = parse_gdelt_event_export(
        payload,
        source_url="https://data.gdeltproject.org/events/20220224.export.CSV.zip",
        expected_date=dt.date(2022, 2, 24),
    )

    assert [(row["date"], row["event_count"], row["source_url"]) for row in rows] == [
        (dt.date(2022, 2, 24), 12, "https://example.test/current")
    ]


def test_gpr_generator_fetches_gdelt_daily_exports_for_spike_dates() -> None:
    gpr_csv = """date,gpr
2022-02-20,100
2022-02-21,101
2022-02-22,99
2022-02-23,101
2022-02-24,500
2022-02-25,120
"""

    def gdelt_daily_fetcher(day: dt.date) -> bytes:
        row = [""] * 58
        row[1] = day.strftime("%Y%m%d")
        row[28] = "19"
        row[29] = "4"
        row[31] = "8"
        row[56] = f"https://example.test/gdelt/{day:%Y%m%d}"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr(f"{day:%Y%m%d}.export.CSV", "\t".join(row) + "\n")
        return buffer.getvalue()

    generator = GPRGDELTSignalGenerator(
        gpr_fetcher=lambda: gpr_csv,
        gdelt_daily_fetcher=gdelt_daily_fetcher,
        min_history_days=3,
        stddev_threshold=2.0,
        merge_window_days=0,
    )

    candidates = generator.generate(
        start_year=2022, end_year=2022, store=None, run_id=None
    )
    validations = generator.validate(candidates, store=None, run_id=None)

    assert [
        (candidate.date, candidate.source_id, candidate.raw_snippet)
        for candidate in candidates
    ] == [
        (
            dt.date(2022, 2, 24),
            "gdelt:events-v2",
            "GDELT geopolitical event volume: 8.",
        ),
        (
            dt.date(2022, 2, 24),
            "gpr:caldara-iacoviello",
            "GPR daily value 500.00 exceeded trailing threshold 102.22.",
        ),
    ]
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


def test_gpr_gdelt_generator_records_source_status_for_fetch_failures() -> None:
    def failing_gpr_fetcher() -> str:
        raise TimeoutError("gpr timed out")

    generator = GPRGDELTSignalGenerator(
        gpr_fetcher=failing_gpr_fetcher,
        gdelt_fetcher=lambda: "date,event_count,dominant_theme,source_url\n",
        acled_fetcher=lambda _start_year, _end_year: None,
        ucdp_fetcher=lambda _start_year, _end_year: None,
        hdx_hapi_fetcher=lambda _start_year, _end_year: None,
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
    assert generator.last_source_statuses["gdelt:events-v2"].status == "empty"
    assert generator.last_run_status == "partial"


def test_gpr_gdelt_generator_records_direct_gdelt_fetch_failure() -> None:
    gpr_csv = """date,gpr
2022-02-20,100
2022-02-21,101
2022-02-22,99
2022-02-23,101
2022-02-24,500
2022-02-25,120
"""

    def failing_gdelt_fetcher() -> str:
        raise TimeoutError("gdelt timed out")

    generator = GPRGDELTSignalGenerator(
        gpr_fetcher=lambda: gpr_csv,
        gdelt_fetcher=failing_gdelt_fetcher,
        acled_fetcher=lambda _start_year, _end_year: None,
        ucdp_fetcher=lambda _start_year, _end_year: None,
        hdx_hapi_fetcher=lambda _start_year, _end_year: None,
        min_history_days=3,
        stddev_threshold=2.0,
    )

    candidates = generator.generate(
        start_year=2022, end_year=2022, store=None, run_id=None
    )

    assert [(candidate.date, candidate.source_id) for candidate in candidates] == [
        (dt.date(2022, 2, 24), "gpr:caldara-iacoviello")
    ]
    assert generator.last_source_statuses["gpr:caldara-iacoviello"].status == "ok"
    assert generator.last_source_statuses["gdelt:events-v2"].status == "failed"
    assert generator.last_source_statuses["gdelt:events-v2"].error == "gdelt timed out"
    assert generator.last_run_status == "partial"


def test_gpr_gdelt_generator_records_optional_conflict_fetch_failure() -> None:
    def failing_acled_fetcher(_start_year: int, _end_year: int) -> str:
        raise TimeoutError("acled timed out")

    generator = GPRGDELTSignalGenerator(
        gpr_fetcher=lambda: "date,gpr\n2022-02-24,100\n",
        gdelt_fetcher=lambda: "date,event_count,dominant_theme,source_url\n",
        acled_fetcher=failing_acled_fetcher,
        ucdp_fetcher=lambda _start_year, _end_year: None,
        hdx_hapi_fetcher=lambda _start_year, _end_year: None,
        min_history_days=3,
        stddev_threshold=2.0,
    )

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


def test_gpr_gdelt_generator_records_optional_conflict_parse_failure() -> None:
    generator = GPRGDELTSignalGenerator(
        gpr_fetcher=lambda: "date,gpr\n2022-02-24,100\n",
        gdelt_fetcher=lambda: "date,event_count,dominant_theme,source_url\n",
        acled_fetcher=lambda _start_year, _end_year: '{"data": [',
        ucdp_fetcher=lambda _start_year, _end_year: None,
        hdx_hapi_fetcher=lambda _start_year, _end_year: None,
        min_history_days=3,
        stddev_threshold=2.0,
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
        {"date": dt.date(2026, 5, 1), "gpr": 123}
    ]
    assert "GPR Excel parse failed; falling back to CSV parser" in caplog.text


def test_gpr_gdelt_generator_records_partial_daily_gdelt_failure() -> None:
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

    generator = GPRGDELTSignalGenerator(
        gpr_fetcher=lambda: gpr_csv,
        gdelt_daily_fetcher=failing_gdelt_daily_fetcher,
        acled_fetcher=lambda _start_year, _end_year: None,
        ucdp_fetcher=lambda _start_year, _end_year: None,
        hdx_hapi_fetcher=lambda _start_year, _end_year: None,
        min_history_days=3,
        stddev_threshold=2.0,
        merge_window_days=0,
    )

    candidates = generator.generate(
        start_year=2022, end_year=2022, store=None, run_id=None
    )

    assert [(candidate.date, candidate.source_id) for candidate in candidates] == [
        (dt.date(2022, 2, 24), "gpr:caldara-iacoviello")
    ]
    assert generator.last_source_statuses["gpr:caldara-iacoviello"].status == "ok"
    assert generator.last_source_statuses["gdelt:events-v2"].status == "partial"
    assert generator.last_source_statuses["gdelt:events-v2"].failed_fetches == 1
    assert generator.last_source_statuses["gdelt:events-v2"].empty_payload is False
    assert generator.last_run_status == "partial"
