from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sqlite3

import pandas as pd
from openpyxl import Workbook

from regime_data_fetch.aggregate_eps import (
    AggregateEPSFetchError,
    EPSWaybackSnapshot,
    parse_wayback_cdx_json,
    parse_sp500_eps_workbook,
    run_wayback_aggregate_eps_fetch,
    run_aggregate_eps_fetch,
)


FIXTURES = Path("tests/fixtures/raw/eps")


def test_parse_sp500_eps_workbook_extracts_current_and_historical_snapshots() -> None:
    parsed = parse_sp500_eps_workbook(FIXTURES / "sp500_eps_est_fixture.xlsx")

    assert parsed.workbook_as_of_date == dt.date(2026, 1, 30)
    assert parsed.public_files_discontinued is True
    assert parsed.current_snapshot.observation_label == "current"
    assert parsed.current_snapshot.observation_date == dt.date(2026, 1, 30)
    assert parsed.current_snapshot.forward_estimate_label == "2026E"
    assert parsed.current_snapshot.forward_estimate_value == 310.24
    assert parsed.current_snapshot.estimate_2025e == 265.41
    assert parsed.current_snapshot.estimate_2026e == 310.24
    assert parsed.current_snapshot.change_vs_prior_observation_2026e == 0.02362412564339467
    assert len(parsed.historical_snapshots) == 7
    assert parsed.historical_snapshots[0].observation_date == dt.date(2024, 3, 31)
    assert parsed.historical_snapshots[-1].observation_date == dt.date(2025, 9, 30)
    assert parsed.historical_snapshots[-1].estimate_2026e == 303.08


def test_run_aggregate_eps_fetch_writes_parquet_and_report(tmp_path: Path) -> None:
    report_path = run_aggregate_eps_fetch(
        out_dir=tmp_path,
        workbook_path=FIXTURES / "sp500_eps_est_fixture.xlsx",
    )

    report = json.loads(report_path.read_text())
    assert report["counts"]["historical_snapshots"] == 7
    assert report["counts"]["current_snapshots"] == 1
    assert report["current_snapshot"]["forward_estimate_label"] == "2026E"
    assert report["current_snapshot"]["forward_estimate_value"] == 310.24
    assert report["current_snapshot"]["estimate_2026e"] == 310.24
    assert report["current_snapshot"]["change_vs_prior_observation_2026e"] == 0.02362412564339467
    assert report["limitations"]["aggregate_forward_eps_revision_direction_4w_available"] is False
    assert report["paths"]["aggregate_eps_parquet"] == str(
        tmp_path / "aggregate_forward_eps" / "sp500_eps_snapshots.parquet"
    )

    df = pd.read_parquet(tmp_path / "aggregate_forward_eps" / "sp500_eps_snapshots.parquet")
    assert list(df.columns) == [
        "workbook_as_of_date",
        "observation_date",
        "observation_label",
        "forward_estimate_label",
        "forward_estimate_value",
        "estimate_2025e",
        "estimate_q4_2025e",
        "estimate_2026e",
        "price",
        "pe_2025e",
        "pe_2026e",
        "change_vs_prior_observation_2025e",
        "change_vs_prior_observation_q4_2025e",
        "change_vs_prior_observation_2026e",
        "change_vs_prior_observation_price",
        "change_vs_prior_observation_pe_2025e",
        "change_vs_prior_observation_pe_2026e",
        "source",
        "source_path",
        "public_files_discontinued",
    ]
    current = df[df["observation_label"] == "current"].iloc[0]
    assert current["observation_date"] == dt.date(2026, 1, 30)
    assert current["forward_estimate_label"] == "2026E"
    assert current["forward_estimate_value"] == 310.24
    assert current["estimate_2026e"] == 310.24
    assert bool(current["public_files_discontinued"]) is True


def test_run_aggregate_eps_fetch_records_manual_workbook_in_sqlite(tmp_path: Path) -> None:
    acquisition_db = tmp_path / "acquisition.db"

    report_path = run_aggregate_eps_fetch(
        out_dir=tmp_path,
        workbook_path=FIXTURES / "sp500_eps_est_fixture.xlsx",
        acquisition_db_path=acquisition_db,
    )

    report = json.loads(report_path.read_text())
    assert report["paths"]["acquisition_db"] == str(acquisition_db)

    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute("SELECT fetch_type, status FROM fetch_runs").fetchall()
        artifacts = conn.execute(
            "SELECT source_name, artifact_kind, source_identifier, local_path, content_size_bytes, content_encoding FROM artifacts"
        ).fetchall()
        blob_sizes = conn.execute("SELECT length(content_bytes) FROM artifact_blobs").fetchall()
        outputs = conn.execute("SELECT output_kind FROM derived_outputs ORDER BY output_id").fetchall()

    assert fetch_runs == [("aggregate_eps", "ok")]
    assert artifacts == [
        (
            "S&P Global aggregate forward EPS workbook",
            "xlsx_manual",
            str(FIXTURES / "sp500_eps_est_fixture.xlsx"),
            str(FIXTURES / "sp500_eps_est_fixture.xlsx"),
            (FIXTURES / "sp500_eps_est_fixture.xlsx").stat().st_size,
            "binary",
        )
    ]
    assert blob_sizes == [((FIXTURES / "sp500_eps_est_fixture.xlsx").stat().st_size,)]
    assert outputs == [("aggregate_eps_parquet",), ("aggregate_eps_report",)]


def test_parse_sp500_eps_workbook_raises_when_expected_sheet_is_missing(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.xlsx"
    wb = Workbook()
    wb.active.title = "Sheet1"
    wb.save(bad_path)

    try:
        parse_sp500_eps_workbook(bad_path)
    except AggregateEPSFetchError as exc:
        assert "ESTIMATES&PEs" in str(exc)
    else:
        raise AssertionError("Expected AggregateEPSFetchError")


def test_parse_wayback_cdx_json_extracts_successful_workbook_snapshots() -> None:
    cdx_json = """
    [
      ["timestamp","original","statuscode","mimetype"],
      ["20200110123456","https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx","200","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
      ["20200214101010","https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx","200","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
    ]
    """

    rows = parse_wayback_cdx_json(cdx_json, target_url="https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx")

    assert rows == [
        EPSWaybackSnapshot(
            timestamp="20200110123456",
            archive_url="https://web.archive.org/web/20200110123456if_/https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx",
            snapshot_date=dt.date(2020, 1, 10),
        ),
        EPSWaybackSnapshot(
            timestamp="20200214101010",
            archive_url="https://web.archive.org/web/20200214101010if_/https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx",
            snapshot_date=dt.date(2020, 2, 14),
        ),
    ]


def test_run_wayback_aggregate_eps_fetch_builds_timeline_from_snapshots(tmp_path: Path) -> None:
    workbook_bytes = (FIXTURES / "sp500_eps_est_fixture.xlsx").read_bytes()

    def fake_cdx_fetcher() -> str:
        return """
        [
          ["timestamp","original","statuscode","mimetype"],
          ["20200110123456","https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx","200","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
          ["20200214101010","https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx","200","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
        ]
        """

    def fake_snapshot_fetcher(snapshot: EPSWaybackSnapshot) -> bytes:
        return workbook_bytes

    report_path = run_wayback_aggregate_eps_fetch(
        out_dir=tmp_path,
        cdx_fetcher=fake_cdx_fetcher,
        snapshot_fetcher=fake_snapshot_fetcher,
    )

    report = json.loads(report_path.read_text())
    assert report["counts"]["snapshots_listed"] == 2
    assert report["counts"]["snapshots_downloaded"] == 2
    assert report["counts"]["timeline_rows"] == 2
    assert report["timeline_preview"][0]["snapshot_date"] == "2020-01-10"
    assert report["timeline_preview"][0]["forward_estimate_label"] == "2026E"
    assert report["timeline_preview"][0]["forward_estimate_value"] == 310.24
    assert report["timeline_preview"][0]["estimate_2026e"] == 310.24
    assert report["paths"]["timeline_parquet"] == str(
        tmp_path / "aggregate_forward_eps_wayback" / "sp500_eps_wayback_timeline.parquet"
    )
    assert (tmp_path / "aggregate_forward_eps_wayback" / "snapshots" / "20200110123456.xlsx").exists()
    assert (tmp_path / "aggregate_forward_eps_wayback" / "snapshots" / "20200214101010.xlsx").exists()

    df = pd.read_parquet(tmp_path / "aggregate_forward_eps_wayback" / "sp500_eps_wayback_timeline.parquet")
    assert list(df.columns) == [
        "snapshot_date",
        "timestamp",
        "archive_url",
        "workbook_as_of_date",
        "forward_estimate_label",
        "forward_estimate_value",
        "estimate_2025e",
        "estimate_q4_2025e",
        "estimate_2026e",
        "price",
        "pe_2025e",
        "pe_2026e",
        "change_vs_prior_observation_2025e",
        "change_vs_prior_observation_q4_2025e",
        "change_vs_prior_observation_2026e",
        "change_vs_prior_observation_price",
        "change_vs_prior_observation_pe_2025e",
        "change_vs_prior_observation_pe_2026e",
        "public_files_discontinued",
        "source",
    ]
    assert list(df["snapshot_date"]) == [dt.date(2020, 1, 10), dt.date(2020, 2, 14)]
    assert list(df["estimate_2026e"]) == [310.24, 310.24]


def test_run_wayback_aggregate_eps_fetch_respects_bounds_and_writes_status_artifacts(tmp_path: Path) -> None:
    workbook_bytes = (FIXTURES / "sp500_eps_est_fixture.xlsx").read_bytes()

    def fake_cdx_fetcher() -> str:
        return """
        [
          ["timestamp","original","statuscode","mimetype"],
          ["20200110123456","https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx","200","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
          ["20200214101010","https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx","200","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
          ["20200320101010","https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx","200","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
        ]
        """

    def fake_snapshot_fetcher(snapshot: EPSWaybackSnapshot) -> bytes:
        return workbook_bytes

    report_path = run_wayback_aggregate_eps_fetch(
        out_dir=tmp_path,
        max_snapshots=2,
        from_date=dt.date(2020, 2, 1),
        to_date=dt.date(2020, 3, 31),
        stop_after_first_success=True,
        cdx_fetcher=fake_cdx_fetcher,
        snapshot_fetcher=fake_snapshot_fetcher,
    )

    report = json.loads(report_path.read_text())
    assert report["requested"] == {
        "max_snapshots": 2,
        "from_date": "2020-02-01",
        "to_date": "2020-03-31",
        "stop_after_first_success": True,
    }
    assert report["counts"]["snapshots_listed"] == 2
    assert report["counts"]["snapshots_downloaded"] == 1
    assert report["counts"]["snapshots_parsed_ok"] == 1
    assert report["counts"]["snapshots_failed"] == 0
    assert report["counts"]["timeline_rows"] == 1

    index_path = tmp_path / "aggregate_forward_eps_wayback" / "wayback_snapshot_index.json"
    status_path = tmp_path / "aggregate_forward_eps_wayback" / "snapshot_status.jsonl"
    assert index_path.exists()
    assert status_path.exists()

    index = json.loads(index_path.read_text())
    assert [row["snapshot_date"] for row in index] == ["2020-02-14", "2020-03-20"]
    statuses = [json.loads(line) for line in status_path.read_text().splitlines()]
    assert len(statuses) == 1
    assert statuses[0]["status"] == "parsed_ok"
    assert statuses[0]["snapshot_date"] == "2020-02-14"


def test_run_wayback_aggregate_eps_fetch_records_sqlite_artifacts_and_outputs(tmp_path: Path) -> None:
    workbook_bytes = (FIXTURES / "sp500_eps_est_fixture.xlsx").read_bytes()
    acquisition_db = tmp_path / "acquisition.db"

    def fake_cdx_fetcher() -> str:
        return """
        [
          ["timestamp","original","statuscode","mimetype"],
          ["20200110123456","https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx","200","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
          ["20200214101010","https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx","200","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
        ]
        """

    def fake_snapshot_fetcher(snapshot: EPSWaybackSnapshot) -> bytes:
        return workbook_bytes

    report_path = run_wayback_aggregate_eps_fetch(
        out_dir=tmp_path,
        acquisition_db_path=acquisition_db,
        cdx_fetcher=fake_cdx_fetcher,
        snapshot_fetcher=fake_snapshot_fetcher,
    )

    report = json.loads(report_path.read_text())
    assert report["paths"]["acquisition_db"] == str(acquisition_db)

    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute("SELECT fetch_type, status FROM fetch_runs").fetchall()
        artifacts = conn.execute(
            "SELECT source_name, artifact_kind, count(*) FROM artifacts GROUP BY source_name, artifact_kind ORDER BY source_name, artifact_kind"
        ).fetchall()
        outputs = conn.execute(
            "SELECT output_kind FROM derived_outputs ORDER BY output_id"
        ).fetchall()

    assert fetch_runs == [("aggregate_eps_wayback", "ok")]
    assert artifacts == [
        ("wayback:cdx", "json", 1),
        ("wayback:eps_workbook", "xlsx_wayback", 2),
    ]
    assert outputs == [
        ("aggregate_eps_wayback_snapshot_index",),
        ("aggregate_eps_wayback_status",),
        ("aggregate_eps_wayback_timeline",),
        ("aggregate_eps_wayback_report",),
    ]
