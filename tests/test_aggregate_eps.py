from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest
from openpyxl import Workbook

from regime_data_fetch.aggregate_eps import (
    EPS_DIR_NAME,
    EPS_REVISION_LOOKBACK_WEEKS,
    SOURCE_NAME,
    WAYBACK_DIR_NAME,
    WAYBACK_TIMELINE_FILENAME,
    WEEKLY_HISTORY_FILENAME,
    AggregateEPSFetchError,
    AggregateEPSSnapshot,
    append_weekly_eps_snapshot,
    compute_eps_revision_direction_4w,
    parse_sp500_eps_workbook,
    run_aggregate_eps_fetch,
    seed_weekly_history_from_wayback_timeline,
)
from regime_data_fetch.aggregate_eps_wayback import (
    EPSWaybackSnapshot,
    parse_wayback_cdx_json,
    run_wayback_aggregate_eps_fetch,
)


FIXTURES = Path("tests/fixtures/raw/eps")


def _eps_snapshot(observation_date: dt.date, forward_eps: float) -> AggregateEPSSnapshot:
    """Build a realistic AggregateEPSSnapshot with the two fields the
    weekly accumulator consumes populated; the rest left at None (the
    accumulator only reads observation_date / observation_label /
    forward_estimate_value)."""
    return AggregateEPSSnapshot(
        observation_date=observation_date,
        observation_label="current",
        forward_estimate_label="2026E",
        forward_estimate_value=forward_eps,
        estimate_2025e=None,
        estimate_q4_2025e=None,
        estimate_2026e=forward_eps,
        price=None,
        pe_2025e=None,
        pe_2026e=None,
        change_vs_prior_observation_2025e=None,
        change_vs_prior_observation_q4_2025e=None,
        change_vs_prior_observation_2026e=None,
        change_vs_prior_observation_price=None,
        change_vs_prior_observation_pe_2025e=None,
        change_vs_prior_observation_pe_2026e=None,
    )


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


def test_parse_sp500_eps_workbook_supports_legacy_xls_layout(monkeypatch, tmp_path: Path) -> None:
    legacy_path = tmp_path / "SP500eps.xls"
    legacy_path.write_text("placeholder")

    frame = pd.DataFrame(
        [
            ["S&P 500 EARNINGS AND ESTIMATE REPORT"],
            [None],
            [dt.datetime(2013, 12, 12)],
            [None],
            [None],
            [None],
            [None],
            [None],
            ["OBSERVATION", "Q4,'13 EST", "2013 EST", "2014 EST", "IDX PRICE"],
            [dt.datetime(2013, 3, 28), 29.76, 111.14, 124.73, 1569.18],
            [dt.datetime(2013, 6, 28), 29.32, 109.06, 123.01, 1606.27],
            [dt.datetime(2013, 9, 30), 28.89, 107.83, 121.83, 1681.54],
            ["current", 28.41, 107.46, 122.29, 1798.00],
        ]
    )

    monkeypatch.setattr("regime_data_fetch.aggregate_eps.pd.read_excel", lambda *args, **kwargs: frame)

    parsed = parse_sp500_eps_workbook(legacy_path)

    assert parsed.workbook_as_of_date == dt.date(2013, 12, 12)
    assert parsed.public_files_discontinued is False
    assert len(parsed.historical_snapshots) == 3
    assert parsed.current_snapshot.forward_estimate_label == "2014 EST"
    assert parsed.current_snapshot.forward_estimate_value == 122.29
    assert parsed.current_snapshot.price == 1798.0
    assert parsed.current_snapshot.estimate_q4_2025e == 28.41


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


# ---------------------------------------------------------------------------
# Weekly-snapshot accumulator + 4-week revision direction (Log #48 closure).
# ---------------------------------------------------------------------------


def test_append_weekly_eps_snapshot_creates_and_appends(tmp_path: Path) -> None:
    """Each call appends one weekly current-snapshot row; the parquet
    accumulates across calls, sorted ascending by observation_date."""
    eps_dir = tmp_path / "aggregate_forward_eps"
    eps_dir.mkdir(parents=True)

    first = append_weekly_eps_snapshot(
        eps_dir=eps_dir,
        current_snapshot=_eps_snapshot(dt.date(2026, 1, 9), 305.10),
    )
    assert list(first["observation_date"]) == [dt.date(2026, 1, 9)]

    second = append_weekly_eps_snapshot(
        eps_dir=eps_dir,
        current_snapshot=_eps_snapshot(dt.date(2026, 1, 16), 306.40),
    )
    assert list(second["observation_date"]) == [
        dt.date(2026, 1, 9),
        dt.date(2026, 1, 16),
    ]
    assert list(second["forward_estimate_value"]) == [305.10, 306.40]

    # The parquet on disk matches the returned frame.
    on_disk = pd.read_parquet(eps_dir / WEEKLY_HISTORY_FILENAME)
    assert len(on_disk) == 2


def test_append_weekly_eps_snapshot_is_idempotent_by_observation_date(
    tmp_path: Path,
) -> None:
    """Re-running a fetch for the same week overwrites that date's row
    rather than double-counting it."""
    eps_dir = tmp_path / "aggregate_forward_eps"
    eps_dir.mkdir(parents=True)

    append_weekly_eps_snapshot(
        eps_dir=eps_dir,
        current_snapshot=_eps_snapshot(dt.date(2026, 1, 9), 305.10),
    )
    # Re-run the SAME observation_date with a corrected estimate value.
    deduped = append_weekly_eps_snapshot(
        eps_dir=eps_dir,
        current_snapshot=_eps_snapshot(dt.date(2026, 1, 9), 305.55),
    )
    assert list(deduped["observation_date"]) == [dt.date(2026, 1, 9)]
    # The row carries the most recent value, not a duplicate.
    assert list(deduped["forward_estimate_value"]) == [305.55]


def test_compute_eps_revision_direction_4w_all_nan_below_lookback() -> None:
    """With <= EPS_REVISION_LOOKBACK_WEEKS weekly rows, the revision series
    is entirely NaN — the §2B earnings labels stay silent (cold-start)."""
    history = pd.DataFrame(
        {
            "observation_date": [
                dt.date(2026, 1, 2),
                dt.date(2026, 1, 9),
                dt.date(2026, 1, 16),
                dt.date(2026, 1, 23),
            ],
            "forward_estimate_value": [300.0, 301.0, 302.0, 303.0],
        }
    )
    revision = compute_eps_revision_direction_4w(history)
    assert len(revision) == 4
    assert revision.isna().all()


def test_compute_eps_revision_direction_4w_hand_computed() -> None:
    """5 weekly rows → the 5th row has a non-NaN 4-week revision equal to
    the hand-computed (fwd[t] - fwd[t-4]) / fwd[t-4]."""
    history = pd.DataFrame(
        {
            "observation_date": [
                dt.date(2026, 1, 2),
                dt.date(2026, 1, 9),
                dt.date(2026, 1, 16),
                dt.date(2026, 1, 23),
                dt.date(2026, 1, 30),
            ],
            "forward_estimate_value": [300.0, 301.0, 302.0, 303.0, 309.0],
        }
    )
    revision = compute_eps_revision_direction_4w(history)
    # Rows 0-3: NaN (cold-start, no 4-weeks-prior row).
    assert revision.iloc[:EPS_REVISION_LOOKBACK_WEEKS].isna().all()
    # Row 4: (309.0 - 300.0) / 300.0 == 0.03
    assert revision.iloc[4] == pytest.approx(0.03)
    # Series is indexed by observation_date.
    assert revision.index[4] == pd.Timestamp("2026-01-30")


def test_compute_eps_revision_direction_4w_handles_zero_prior() -> None:
    """A zero 4-weeks-prior estimate yields NaN, not a divide-by-zero."""
    history = pd.DataFrame(
        {
            "observation_date": [
                dt.date(2026, 1, 2),
                dt.date(2026, 1, 9),
                dt.date(2026, 1, 16),
                dt.date(2026, 1, 23),
                dt.date(2026, 1, 30),
            ],
            "forward_estimate_value": [0.0, 301.0, 302.0, 303.0, 309.0],
        }
    )
    revision = compute_eps_revision_direction_4w(history)
    assert pd.isna(revision.iloc[4])


def test_run_aggregate_eps_fetch_accumulates_weekly_history(tmp_path: Path) -> None:
    """Integration: run_aggregate_eps_fetch appends to the weekly-history
    parquet on each run. Re-running with the same fixture workbook is
    idempotent (same observation_date) — the report's weekly_history_rows
    count stays 1 and the 4-week revision stays unavailable."""
    report_path = run_aggregate_eps_fetch(
        out_dir=tmp_path,
        workbook_path=FIXTURES / "sp500_eps_est_fixture.xlsx",
    )
    report = json.loads(report_path.read_text())
    assert report["counts"]["weekly_history_rows"] == 1
    assert (
        report["limitations"][
            "aggregate_forward_eps_revision_direction_4w_available"
        ]
        is False
    )
    weekly_path = tmp_path / "aggregate_forward_eps" / WEEKLY_HISTORY_FILENAME
    assert weekly_path.exists()
    assert report["paths"]["aggregate_eps_weekly_history_parquet"] == str(weekly_path)

    # Re-run with the same workbook: dedup keeps the accumulator at 1 row.
    run_aggregate_eps_fetch(
        out_dir=tmp_path,
        workbook_path=FIXTURES / "sp500_eps_est_fixture.xlsx",
    )
    assert len(pd.read_parquet(weekly_path)) == 1


# --- Wayback-timeline accumulator seeding -----------------------------------


def _wayback_timeline_df(rows: list[tuple[dt.date, float | None]]) -> pd.DataFrame:
    """Build a synthetic Wayback EPS timeline frame shaped like the parquet
    run_wayback_aggregate_eps_fetch materialises. The seeding bridge reads
    only workbook_as_of_date + forward_estimate_value; the rest is realistic
    filler so the fixture matches the real timeline schema."""
    return pd.DataFrame(
        [
            {
                "snapshot_date": obs_date,
                "timestamp": obs_date.strftime("%Y%m%d000000"),
                "archive_url": (
                    f"https://web.archive.org/web/{obs_date:%Y%m%d}000000/"
                    "https://www.spglobal.com/spdji/en/documents/"
                    "additional-material/sp-500-eps-est.xlsx"
                ),
                "workbook_as_of_date": obs_date,
                "forward_estimate_label": "2026E",
                "forward_estimate_value": fwd,
                "source": "wayback_machine",
            }
            for obs_date, fwd in rows
        ]
    )


def _write_wayback_timeline(tmp_path: Path, df: pd.DataFrame) -> Path:
    wayback_dir = tmp_path / WAYBACK_DIR_NAME
    wayback_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = wayback_dir / WAYBACK_TIMELINE_FILENAME
    df.to_parquet(timeline_path, index=False)
    return timeline_path


def test_seed_weekly_history_creates_accumulator_from_timeline(
    tmp_path: Path,
) -> None:
    """With no existing accumulator, the seed bridges the Wayback timeline
    straight into sp500_eps_weekly_history.parquet — one accumulator row per
    timeline row, keyed by workbook_as_of_date, sorted ascending."""
    _write_wayback_timeline(
        tmp_path,
        _wayback_timeline_df(
            [
                (dt.date(2026, 1, 7), 271.00),
                (dt.date(2026, 1, 14), 272.50),
                (dt.date(2026, 1, 21), 273.10),
            ]
        ),
    )

    combined = seed_weekly_history_from_wayback_timeline(out_dir=tmp_path)

    assert list(combined["observation_date"]) == [
        dt.date(2026, 1, 7),
        dt.date(2026, 1, 14),
        dt.date(2026, 1, 21),
    ]
    assert list(combined["forward_estimate_value"]) == [271.00, 272.50, 273.10]
    assert set(combined["observation_label"]) == {"wayback_backfill"}
    assert set(combined["source"]) == {"wayback_machine"}

    on_disk = pd.read_parquet(
        tmp_path / EPS_DIR_NAME / WEEKLY_HISTORY_FILENAME
    )
    assert len(on_disk) == 3


def test_seed_weekly_history_existing_live_rows_win_on_collision(
    tmp_path: Path,
) -> None:
    """A live run_aggregate_eps_fetch row is authoritative — on an
    observation_date collision the existing accumulator row is kept, not the
    Wayback-archived snapshot for the same date."""
    eps_dir = tmp_path / EPS_DIR_NAME
    eps_dir.mkdir(parents=True)
    # A live fetch already recorded 2026-01-14 with its authoritative value.
    append_weekly_eps_snapshot(
        eps_dir=eps_dir,
        current_snapshot=_eps_snapshot(dt.date(2026, 1, 14), 272.50),
    )
    # The Wayback timeline carries a stale/different value for that date
    # plus two dates the accumulator doesn't have yet.
    _write_wayback_timeline(
        tmp_path,
        _wayback_timeline_df(
            [
                (dt.date(2026, 1, 7), 271.00),
                (dt.date(2026, 1, 14), 999.99),  # collides with the live row
                (dt.date(2026, 1, 21), 273.10),
            ]
        ),
    )

    combined = seed_weekly_history_from_wayback_timeline(out_dir=tmp_path)

    assert list(combined["observation_date"]) == [
        dt.date(2026, 1, 7),
        dt.date(2026, 1, 14),
        dt.date(2026, 1, 21),
    ]
    by_date = combined.set_index("observation_date")
    # The live row's value survived; the colliding Wayback value was dropped.
    assert by_date.loc[dt.date(2026, 1, 14), "forward_estimate_value"] == 272.50
    assert by_date.loc[dt.date(2026, 1, 14), "source"] == SOURCE_NAME
    # The two non-colliding Wayback dates were seeded in.
    assert by_date.loc[dt.date(2026, 1, 7), "source"] == "wayback_machine"


def test_seed_weekly_history_dedupes_timeline_keeping_last(
    tmp_path: Path,
) -> None:
    """Multiple Wayback snapshots can share one workbook_as_of_date — the
    last (freshest capture, timeline is snapshot-date sorted) is kept."""
    _write_wayback_timeline(
        tmp_path,
        _wayback_timeline_df(
            [
                (dt.date(2026, 1, 7), 271.00),
                (dt.date(2026, 1, 7), 271.85),  # later capture, same workbook date
            ]
        ),
    )

    combined = seed_weekly_history_from_wayback_timeline(out_dir=tmp_path)

    assert list(combined["observation_date"]) == [dt.date(2026, 1, 7)]
    assert list(combined["forward_estimate_value"]) == [271.85]


def test_seed_weekly_history_raises_when_timeline_missing(
    tmp_path: Path,
) -> None:
    """No Wayback timeline parquet → loud failure routing the operator to
    run the backfill first, not a silent empty seed."""
    with pytest.raises(AggregateEPSFetchError, match="No Wayback EPS timeline"):
        seed_weekly_history_from_wayback_timeline(out_dir=tmp_path)


def test_seed_weekly_history_collapses_earnings_cold_start(
    tmp_path: Path,
) -> None:
    """The point of the bridge: a one-time Wayback backfill + seed pre-fills
    the accumulator past EPS_REVISION_LOOKBACK_WEEKS so
    compute_eps_revision_direction_4w is non-NaN immediately — no waiting for
    >4 live weekly fetches."""
    _write_wayback_timeline(
        tmp_path,
        _wayback_timeline_df(
            [
                (dt.date(2026, 1, 7), 270.00),
                (dt.date(2026, 1, 14), 271.00),
                (dt.date(2026, 1, 21), 272.00),
                (dt.date(2026, 1, 28), 273.00),
                (dt.date(2026, 2, 4), 277.20),
            ]
        ),
    )

    combined = seed_weekly_history_from_wayback_timeline(out_dir=tmp_path)
    revision = compute_eps_revision_direction_4w(combined)

    # Cold-start rows stay NaN; the 5th row unlocks immediately post-seed.
    assert revision.iloc[:EPS_REVISION_LOOKBACK_WEEKS].isna().all()
    # (277.20 - 270.00) / 270.00 == 0.0266...
    assert revision.iloc[4] == pytest.approx((277.20 - 270.00) / 270.00)
