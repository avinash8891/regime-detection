from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sqlite3
from typing import get_args, get_type_hints
import urllib.error

import pandas as pd
import pytest
from openpyxl import Workbook

import regime_data_fetch.aggregate_eps_models as aggregate_eps_models
from regime_data_fetch.aggregate_eps import (
    EPS_REVISION_LOOKBACK_WEEKS,
    WEEKLY_HISTORY_FILENAME,
    AggregateEPSFetchError,
    AggregateEPSSnapshot,
    EPSWaybackSnapshot,
    append_weekly_eps_snapshot,
    compute_eps_revision_direction_4w,
    download_spglobal_eps_workbook,
    fetch_wayback_cdx,
    fetch_wayback_snapshot_bytes,
    parse_wayback_cdx_json,
    parse_sp500_eps_workbook,
    run_aggregate_eps_auto_fetch,
    run_wayback_aggregate_eps_fetch,
    run_aggregate_eps_fetch,
)
from regime_data_fetch.aggregate_eps_models import (
    AggregateEPSSnapshot as AggregateEPSSnapshotModel,
    EPSHorizonLabel,
)

FIXTURES = Path("tests/fixtures/raw/eps")


class _BytesResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "_BytesResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _eps_snapshot(
    observation_date: dt.date, forward_eps: float
) -> AggregateEPSSnapshot:
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


def test_aggregate_eps_snapshot_label_mappings_are_immutable() -> None:
    estimates = {"2026E": 310.24}
    pes = {"2026E": 20.1}
    changes = {"2026E": 0.02}
    pe_changes = {"2026E": 0.01}

    snapshot = AggregateEPSSnapshot(
        observation_date=dt.date(2026, 1, 30),
        observation_label="current",
        forward_estimate_label="2026E",
        forward_estimate_value=310.24,
        estimates_by_label=estimates,
        pe_by_label=pes,
        change_vs_prior_observation_by_label=changes,
        change_vs_prior_observation_pe_by_label=pe_changes,
    )
    estimates["2026E"] = 999.0

    assert snapshot.estimate_2026e == 310.24
    with pytest.raises(TypeError):
        snapshot.estimates_by_label["2026E"] = 999.0  # type: ignore[index]


def test_aggregate_eps_horizon_labels_are_closed_type() -> None:
    assert set(get_args(EPSHorizonLabel)) == {"2025E", "Q4 2025E", "2026E"}
    assert (
        get_type_hints(
            AggregateEPSSnapshotModel,
            globalns=vars(aggregate_eps_models),
        )["estimates_by_label"]
        == aggregate_eps_models.Mapping[EPSHorizonLabel, float | None]
    )
    with pytest.raises(AggregateEPSFetchError, match="unknown EPS horizon label"):
        AggregateEPSSnapshot(
            observation_date=dt.date(2026, 5, 15),
            observation_label="current",
            forward_estimate_label="2026E",
            forward_estimate_value=310.24,
            estimates_by_label={"2027E": 1.0},  # type: ignore[dict-item]
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
    assert (
        parsed.current_snapshot.change_vs_prior_observation_2026e == 0.02362412564339467
    )
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
    assert (
        report["current_snapshot"]["change_vs_prior_observation_2026e"]
        == 0.02362412564339467
    )
    assert (
        report["limitations"]["aggregate_forward_eps_revision_direction_4w_available"]
        is False
    )
    assert report["paths"]["aggregate_eps_parquet"] == str(
        tmp_path / "aggregate_forward_eps" / "sp500_eps_snapshots.parquet"
    )

    df = pd.read_parquet(
        tmp_path / "aggregate_forward_eps" / "sp500_eps_snapshots.parquet"
    )
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


def test_aggregate_eps_report_helper_matches_fetch_report(tmp_path: Path) -> None:
    report_path = run_aggregate_eps_fetch(
        out_dir=tmp_path,
        workbook_path=FIXTURES / "sp500_eps_est_fixture.xlsx",
    )

    from regime_data_fetch.aggregate_eps_reports import build_aggregate_eps_report

    report = json.loads(report_path.read_text())
    weekly_history = pd.read_parquet(
        tmp_path / "aggregate_forward_eps" / WEEKLY_HISTORY_FILENAME
    )
    parsed = parse_sp500_eps_workbook(FIXTURES / "sp500_eps_est_fixture.xlsx")
    expected = build_aggregate_eps_report(
        as_of_utc=report["as_of_utc"],
        workbook_path=FIXTURES / "sp500_eps_est_fixture.xlsx",
        parsed=parsed,
        weekly_history=weekly_history,
        revision_available=bool(
            compute_eps_revision_direction_4w(weekly_history).notna().any()
        ),
        parquet_path=tmp_path / "aggregate_forward_eps" / "sp500_eps_snapshots.parquet",
        weekly_history_path=tmp_path
        / "aggregate_forward_eps"
        / WEEKLY_HISTORY_FILENAME,
        acquisition_db_path=None,
    )

    assert report == expected


def test_run_aggregate_eps_fetch_records_manual_workbook_in_sqlite(
    tmp_path: Path,
) -> None:
    acquisition_db = tmp_path / "acquisition.db"

    report_path = run_aggregate_eps_fetch(
        out_dir=tmp_path,
        workbook_path=FIXTURES / "sp500_eps_est_fixture.xlsx",
        acquisition_db_path=acquisition_db,
    )

    report = json.loads(report_path.read_text())
    assert report["paths"]["acquisition_db"] == str(acquisition_db)

    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute(
            "SELECT fetch_type, status FROM fetch_runs"
        ).fetchall()
        artifacts = conn.execute(
            "SELECT source_name, artifact_kind, source_identifier, local_path, content_size_bytes, content_encoding FROM artifacts"
        ).fetchall()
        blob_sizes = conn.execute(
            "SELECT length(content_bytes) FROM artifact_blobs"
        ).fetchall()
        outputs = conn.execute(
            "SELECT output_kind FROM derived_outputs ORDER BY output_id"
        ).fetchall()

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


def test_run_aggregate_eps_auto_fetch_uses_browser_fallback_after_direct_download_failure(
    tmp_path: Path,
) -> None:
    workbook_bytes = (FIXTURES / "sp500_eps_est_fixture.xlsx").read_bytes()

    def failing_downloader(**kwargs) -> Path:
        raise AggregateEPSFetchError("direct download blocked")

    def browser_downloader(**kwargs) -> Path:
        out_path = kwargs["out_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(workbook_bytes)
        return out_path

    report_path = run_aggregate_eps_auto_fetch(
        out_dir=tmp_path,
        workbook_downloader=failing_downloader,
        browser_downloader=browser_downloader,
    )

    report = json.loads(report_path.read_text())
    assert report["source_path"] == str(
        tmp_path / "spglobal_eps" / "sp-500-eps-est.xlsx"
    )
    assert report["counts"]["current_snapshots"] == 1


def test_run_aggregate_eps_auto_fetch_has_no_disable_fallback_mode(
    tmp_path: Path,
) -> None:
    workbook_bytes = (FIXTURES / "sp500_eps_est_fixture.xlsx").read_bytes()
    calls: list[str] = []

    def failing_downloader(**kwargs) -> Path:
        calls.append("direct")
        raise AggregateEPSFetchError("direct download blocked")

    def browser_downloader(**kwargs) -> Path:
        calls.append("browser")
        out_path = kwargs["out_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(workbook_bytes)
        return out_path

    run_aggregate_eps_auto_fetch(
        out_dir=tmp_path,
        workbook_downloader=failing_downloader,
        browser_downloader=browser_downloader,
    )

    assert calls == ["direct", "browser"]


def test_download_spglobal_eps_workbook_writes_payload_and_sends_browser_headers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = (FIXTURES / "sp500_eps_est_fixture.xlsx").read_bytes()
    captured = {}

    def fake_urlopen(req, *, timeout: int):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["user_agent"] = req.headers["User-agent"]
        captured["accept"] = req.headers["Accept"]
        return _BytesResponse(payload)

    monkeypatch.setattr(
        "regime_data_fetch.aggregate_eps.urllib.request.urlopen",
        fake_urlopen,
    )
    out_path = tmp_path / "spglobal_eps" / "sp-500-eps-est.xlsx"

    returned = download_spglobal_eps_workbook(
        out_path=out_path,
        source_url="https://example.test/sp-500-eps-est.xlsx",
        timeout_seconds=13,
    )

    assert returned == out_path
    assert out_path.read_bytes() == payload
    assert captured["url"] == "https://example.test/sp-500-eps-est.xlsx"
    assert captured["timeout"] == 13
    assert "Chrome/126.0.0.0" in captured["user_agent"]
    assert (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        in captured["accept"]
    )


def test_download_spglobal_eps_workbook_403_raises_operator_instructions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_urlopen(req, *, timeout: int):
        raise urllib.error.HTTPError(
            url=req.full_url,
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(
        "regime_data_fetch.aggregate_eps.urllib.request.urlopen",
        fake_urlopen,
    )

    with pytest.raises(
        AggregateEPSFetchError, match="Akamai bot mitigation"
    ) as excinfo:
        download_spglobal_eps_workbook(
            out_path=tmp_path / "spglobal_eps" / "sp-500-eps-est.xlsx",
            source_url="https://example.test/sp-500-eps-est.xlsx",
        )

    message = str(excinfo.value)
    assert "copy it to data/raw/spglobal_eps/sp-500-eps-est.xlsx" in message
    assert "re-run --fetch eps-spglobal-auto" in message
    assert isinstance(excinfo.value.__cause__, urllib.error.HTTPError)


@pytest.mark.parametrize(
    ("exc", "match"),
    [
        (
            urllib.error.HTTPError(
                url="https://example.test/sp-500-eps-est.xlsx",
                code=404,
                msg="Not Found",
                hdrs=None,
                fp=None,
            ),
            "Failed to download S&P EPS workbook",
        ),
        (
            urllib.error.URLError("connection reset"),
            "Failed to download S&P EPS workbook",
        ),
    ],
)
def test_download_spglobal_eps_workbook_wraps_non_403_network_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exc: Exception,
    match: str,
) -> None:
    def fake_urlopen(_req, *, timeout: int):
        raise exc

    monkeypatch.setattr(
        "regime_data_fetch.aggregate_eps.urllib.request.urlopen",
        fake_urlopen,
    )

    with pytest.raises(AggregateEPSFetchError, match=match) as excinfo:
        download_spglobal_eps_workbook(
            out_path=tmp_path / "spglobal_eps" / "sp-500-eps-est.xlsx",
            source_url="https://example.test/sp-500-eps-est.xlsx",
        )

    assert excinfo.value.__cause__ is exc


def test_download_spglobal_eps_workbook_rejects_empty_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "regime_data_fetch.aggregate_eps.urllib.request.urlopen",
        lambda _req, *, timeout: _BytesResponse(b""),
    )
    out_path = tmp_path / "spglobal_eps" / "sp-500-eps-est.xlsx"

    with pytest.raises(AggregateEPSFetchError, match="returned empty payload"):
        download_spglobal_eps_workbook(
            out_path=out_path,
            source_url="https://example.test/sp-500-eps-est.xlsx",
        )

    assert not out_path.exists()


def test_parse_sp500_eps_workbook_raises_when_expected_sheet_is_missing(
    tmp_path: Path,
) -> None:
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


def test_parse_sp500_eps_workbook_supports_legacy_xls_layout(
    monkeypatch, tmp_path: Path
) -> None:
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

    monkeypatch.setattr(
        "regime_data_fetch.aggregate_eps.pd.read_excel", lambda *args, **kwargs: frame
    )

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

    rows = parse_wayback_cdx_json(
        cdx_json,
        target_url="https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx",
    )

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


def test_fetch_wayback_cdx_retries_transient_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'[["timestamp","original","statuscode","mimetype"]]'

    def fake_urlopen(*_args: object, **_kwargs: object) -> _Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.HTTPError(
                url="https://web.archive.org/cdx",
                code=503,
                msg="Service Unavailable",
                hdrs=None,
                fp=None,
            )
        return _Response()

    monkeypatch.setattr(
        "regime_data_fetch.aggregate_eps.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        "regime_data_fetch.aggregate_eps.time.sleep", lambda *_args: None
    )

    payload = fetch_wayback_cdx(max_attempts=2, backoff_seconds=0)

    assert calls["count"] == 2
    assert payload == '[["timestamp","original","statuscode","mimetype"]]'


def test_fetch_wayback_cdx_raises_after_repeated_url_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    def fake_urlopen(*_args: object, **_kwargs: object):
        raise urllib.error.URLError("Wayback unavailable")

    monkeypatch.setattr(
        "regime_data_fetch.aggregate_eps.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr("regime_data_fetch.aggregate_eps.time.sleep", sleeps.append)

    with pytest.raises(
        AggregateEPSFetchError, match="Wayback CDX fetch failed after 3 attempts"
    ) as excinfo:
        fetch_wayback_cdx(max_attempts=3, backoff_seconds=2.0)

    assert sleeps == [2.0, 4.0]
    assert isinstance(excinfo.value.__cause__, urllib.error.URLError)


def test_fetch_wayback_cdx_does_not_retry_non_transient_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def fake_urlopen(req, *, timeout: int):
        calls["count"] += 1
        raise urllib.error.HTTPError(
            url=req.full_url,
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(
        "regime_data_fetch.aggregate_eps.urllib.request.urlopen",
        fake_urlopen,
    )

    with pytest.raises(urllib.error.HTTPError):
        fetch_wayback_cdx(max_attempts=3, backoff_seconds=0)

    assert calls["count"] == 1


def test_fetch_wayback_snapshot_bytes_reads_archive_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    snapshot = EPSWaybackSnapshot(
        timestamp="20200110123456",
        archive_url="https://web.archive.org/web/20200110123456if_/https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx",
        snapshot_date=dt.date(2020, 1, 10),
    )

    def fake_urlopen(req, *, timeout: int):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["user_agent"] = req.headers["User-agent"]
        return _BytesResponse(b"xlsx-bytes")

    monkeypatch.setattr(
        "regime_data_fetch.aggregate_eps.urllib.request.urlopen",
        fake_urlopen,
    )

    assert fetch_wayback_snapshot_bytes(snapshot) == b"xlsx-bytes"
    assert captured == {
        "url": snapshot.archive_url,
        "timeout": 60,
        "user_agent": "Mozilla/5.0",
    }


def test_run_wayback_aggregate_eps_fetch_builds_timeline_from_snapshots(
    tmp_path: Path,
) -> None:
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
        tmp_path
        / "aggregate_forward_eps_wayback"
        / "sp500_eps_wayback_timeline.parquet"
    )
    assert (
        tmp_path / "aggregate_forward_eps_wayback" / "snapshots" / "20200110123456.xlsx"
    ).exists()
    assert (
        tmp_path / "aggregate_forward_eps_wayback" / "snapshots" / "20200214101010.xlsx"
    ).exists()
    assert report["counts"]["weekly_history_rows"] == 1
    assert report["paths"]["aggregate_eps_weekly_history_parquet"] == str(
        tmp_path / "aggregate_forward_eps" / WEEKLY_HISTORY_FILENAME
    )
    assert (tmp_path / "aggregate_forward_eps" / WEEKLY_HISTORY_FILENAME).exists()

    df = pd.read_parquet(
        tmp_path
        / "aggregate_forward_eps_wayback"
        / "sp500_eps_wayback_timeline.parquet"
    )
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


def test_run_wayback_aggregate_eps_fetch_respects_bounds_and_writes_status_artifacts(
    tmp_path: Path,
) -> None:
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

    index_path = (
        tmp_path / "aggregate_forward_eps_wayback" / "wayback_snapshot_index.json"
    )
    status_path = tmp_path / "aggregate_forward_eps_wayback" / "snapshot_status.jsonl"
    assert index_path.exists()
    assert status_path.exists()

    index = json.loads(index_path.read_text())
    assert [row["snapshot_date"] for row in index] == ["2020-02-14", "2020-03-20"]
    statuses = [json.loads(line) for line in status_path.read_text().splitlines()]
    assert len(statuses) == 1
    assert statuses[0]["status"] == "parsed_ok"
    assert statuses[0]["snapshot_date"] == "2020-02-14"


def test_run_wayback_aggregate_eps_fetch_records_sqlite_artifacts_and_outputs(
    tmp_path: Path,
) -> None:
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
        fetch_runs = conn.execute(
            "SELECT fetch_type, status FROM fetch_runs"
        ).fetchall()
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
        ("aggregate_eps_weekly_history",),
        ("aggregate_eps_wayback_report",),
    ]


def test_run_wayback_aggregate_eps_fetch_records_failed_status_when_all_snapshots_fail(
    tmp_path: Path,
) -> None:
    acquisition_db = tmp_path / "acquisition.db"

    def fake_cdx_fetcher() -> str:
        return """
        [
          ["timestamp","original","statuscode","mimetype"],
          ["20200110123456","https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx","200","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
        ]
        """

    def failing_snapshot_fetcher(snapshot: EPSWaybackSnapshot) -> bytes:
        raise urllib.error.URLError(f"archive missing for {snapshot.timestamp}")

    with pytest.raises(
        AggregateEPSFetchError,
        match="Wayback EPS backfill produced no parsed timeline rows",
    ):
        run_wayback_aggregate_eps_fetch(
            out_dir=tmp_path,
            acquisition_db_path=acquisition_db,
            cdx_fetcher=fake_cdx_fetcher,
            snapshot_fetcher=failing_snapshot_fetcher,
        )

    status_path = tmp_path / "aggregate_forward_eps_wayback" / "snapshot_status.jsonl"
    statuses = [json.loads(line) for line in status_path.read_text().splitlines()]
    assert statuses == [
        {
            "snapshot_date": "2020-01-10",
            "timestamp": "20200110123456",
            "archive_url": "https://web.archive.org/web/20200110123456if_/https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx",
            "status": "failed",
            "detail": "URLError: <urlopen error archive missing for 20200110123456>",
        }
    ]
    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute(
            "SELECT fetch_type, status, notes FROM fetch_runs"
        ).fetchall()
        outputs = conn.execute("SELECT output_kind FROM derived_outputs").fetchall()

    assert len(fetch_runs) == 1
    assert fetch_runs[0][0] == "aggregate_eps_wayback"
    assert fetch_runs[0][1] == "failed"
    assert "Wayback EPS backfill produced no parsed timeline rows" in fetch_runs[0][2]
    assert outputs == []


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
        report["limitations"]["aggregate_forward_eps_revision_direction_4w_available"]
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
