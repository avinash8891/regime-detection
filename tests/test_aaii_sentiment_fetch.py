from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
import urllib.error

import pandas as pd
import pytest

from regime_data_fetch import aaii_sentiment

_AAII_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "raw" / "aaii"


class _BytesResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "_BytesResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


class _FixedDate(dt.date):
    @classmethod
    def today(cls) -> "_FixedDate":
        return cls(2026, 1, 3)


def test_run_sentiment_fetch_records_canonical_artifact_ledger(monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "data" / "raw"
    sentiment_dir = out_dir / "sentiment"
    sentiment_dir.mkdir(parents=True)
    existing = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-01"),
                "bullish": 0.40,
                "neutral": 0.30,
                "bearish": 0.30,
                "bull_bear_spread": 0.10,
                "bull_bear_spread_8w_ma": 0.10,
            }
        ]
    )
    existing.to_parquet(sentiment_dir / "aaii_sentiment.parquet", index=False)
    acquisition_db = tmp_path / "acquisition.db"

    def fake_fetch_latest_rows(url: str, after_date, *, timeout: int = 30) -> pd.DataFrame:
        assert url == "https://example.test/aaii"
        assert after_date.isoformat() == "2026-05-01"
        return pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-05-08"),
                    "bullish": 0.45,
                    "neutral": 0.25,
                    "bearish": 0.30,
                }
            ]
        )

    monkeypatch.setattr(aaii_sentiment, "fetch_latest_rows", fake_fetch_latest_rows)

    report_path = aaii_sentiment.run_sentiment_fetch(
        out_dir=out_dir,
        url="https://example.test/aaii",
        acquisition_db_path=acquisition_db,
    )

    report = json.loads(report_path.read_text())
    assert report["rows"] == 2
    assert report["max_date"] == "2026-05-08"
    assert report["paths"]["acquisition_db"] == str(acquisition_db)

    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute("SELECT fetch_type, status FROM fetch_runs").fetchall()
        artifact_records = conn.execute(
            "SELECT name, stage, source_name, artifact_kind, row_count, min_date, max_date FROM artifact_records"
        ).fetchall()
        checkpoints = conn.execute(
            "SELECT source_name, cursor_key, cursor_value FROM source_checkpoints"
        ).fetchall()
        outputs = conn.execute("SELECT output_kind FROM derived_outputs ORDER BY output_id").fetchall()

    assert fetch_runs == [("sentiment", "ok")]
    assert artifact_records == [
        ("aaii_sentiment", "canonical", "aaii", "parquet", 2, "2026-05-01", "2026-05-08")
    ]
    assert checkpoints == [("aaii", "survey_week", "2026-05-08")]
    assert outputs == [
        ("aaii_sentiment_parquet",),
        ("aaii_sentiment_fetch_report",),
    ]


def test_run_sentiment_fetch_can_skip_missing_seed_for_unattended_all(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "data" / "raw"
    acquisition_db = tmp_path / "acquisition.db"

    report_path = aaii_sentiment.run_sentiment_fetch(
        out_dir=out_dir,
        acquisition_db_path=acquisition_db,
        required=False,
    )

    report = json.loads(report_path.read_text())
    assert report["status"] == "skipped"
    assert report["materializable"] is False
    assert "aaii_sentiment_historical.cfb" in report["reason"]

    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute(
            "SELECT fetch_type, status, notes FROM fetch_runs"
        ).fetchall()

    assert fetch_runs == [("sentiment", "skipped", report["reason"])]


def test_run_sentiment_fetch_required_missing_seed_fails_loudly(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "data" / "raw"

    with pytest.raises(FileNotFoundError, match="No parquet at"):
        aaii_sentiment.run_sentiment_fetch(out_dir=out_dir, required=True)


def test_fetch_latest_rows_parses_aaii_html_and_infers_year_rollover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (_AAII_FIXTURE_DIR / "sent_results_snippet.html").read_bytes()
    captured = {}

    def fake_urlopen(req, *, timeout: int):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["user_agent"] = req.headers["User-agent"]
        return _BytesResponse(payload)

    monkeypatch.setattr(aaii_sentiment.dt, "date", _FixedDate)
    monkeypatch.setattr(aaii_sentiment.urllib.request, "urlopen", fake_urlopen)

    frame = aaii_sentiment.fetch_latest_rows(
        "https://example.test/aaii",
        after_date=dt.date(2025, 12, 30),
        timeout=17,
    )

    assert captured == {
        "url": "https://example.test/aaii",
        "timeout": 17,
        "user_agent": "Mozilla/5.0 (compatible; regime-engine-fetcher/2.0)",
    }
    assert frame["date"].tolist() == [
        pd.Timestamp("2025-12-31"),
        pd.Timestamp("2026-01-02"),
    ]
    assert frame["bullish"].tolist() == pytest.approx([0.383, 0.41])
    assert frame["neutral"].tolist() == pytest.approx([0.312, 0.29])
    assert frame["bearish"].tolist() == pytest.approx([0.305, 0.3])


def test_fetch_latest_rows_filters_rows_not_newer_than_after_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (_AAII_FIXTURE_DIR / "sent_results_snippet.html").read_bytes()

    def fake_urlopen(_req, *, timeout: int):
        assert timeout == 30
        return _BytesResponse(payload)

    monkeypatch.setattr(aaii_sentiment.dt, "date", _FixedDate)
    monkeypatch.setattr(aaii_sentiment.urllib.request, "urlopen", fake_urlopen)

    frame = aaii_sentiment.fetch_latest_rows(
        "https://example.test/aaii",
        after_date=dt.date(2026, 1, 2),
    )

    assert list(frame.columns) == ["date", "bullish", "neutral", "bearish"]
    assert frame.empty


def test_fetch_latest_rows_wraps_urlerror_with_operator_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(_req, *, timeout: int):
        assert timeout == 30
        raise urllib.error.URLError("TLS handshake failed")

    monkeypatch.setattr(aaii_sentiment.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Failed to fetch AAII sentiment page") as excinfo:
        aaii_sentiment.fetch_latest_rows(
            "https://example.test/aaii",
            after_date=dt.date(2026, 1, 1),
        )

    assert isinstance(excinfo.value.__cause__, urllib.error.URLError)
    assert "TLS handshake failed" in str(excinfo.value)


def test_update_aaii_parquet_appends_scraped_rows_and_recomputes_rolling_spread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "sentiment" / "aaii_sentiment.parquet"
    out_path.parent.mkdir()
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-12-24"),
                "bullish": 0.30,
                "neutral": 0.30,
                "bearish": 0.40,
                "bull_bear_spread": -0.10,
                "bull_bear_spread_8w_ma": -0.10,
            },
            {
                "date": pd.Timestamp("2025-12-31"),
                "bullish": 0.383,
                "neutral": 0.312,
                "bearish": 0.305,
                "bull_bear_spread": 0.078,
                "bull_bear_spread_8w_ma": -0.011,
            },
        ]
    ).to_parquet(out_path, index=False)
    payload = (_AAII_FIXTURE_DIR / "sent_results_snippet.html").read_bytes()

    def fake_urlopen(_req, *, timeout: int):
        assert timeout == 30
        return _BytesResponse(payload)

    monkeypatch.setattr(aaii_sentiment.dt, "date", _FixedDate)
    monkeypatch.setattr(aaii_sentiment.urllib.request, "urlopen", fake_urlopen)

    combined = aaii_sentiment.update_aaii_parquet(
        raw_dir=tmp_path,
        out_path=out_path,
        url="https://example.test/aaii",
    )

    persisted = pd.read_parquet(out_path)
    pd.testing.assert_frame_equal(combined, persisted)
    assert combined["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-12-24",
        "2025-12-31",
        "2026-01-02",
    ]
    assert combined["bull_bear_spread"].round(3).tolist() == [-0.1, 0.078, 0.11]
    assert combined["bull_bear_spread_8w_ma"].round(6).tolist() == [
        -0.1,
        -0.011,
        0.029333,
    ]


def test_update_aaii_parquet_preserves_existing_file_when_no_new_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "sentiment" / "aaii_sentiment.parquet"
    out_path.parent.mkdir()
    existing = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "bullish": 0.41,
                "neutral": 0.29,
                "bearish": 0.30,
                "bull_bear_spread": 0.11,
                "bull_bear_spread_8w_ma": 0.11,
            }
        ]
    )
    existing.to_parquet(out_path, index=False)
    payload = (_AAII_FIXTURE_DIR / "sent_results_snippet.html").read_bytes()

    def fake_urlopen(_req, *, timeout: int):
        return _BytesResponse(payload)

    monkeypatch.setattr(aaii_sentiment.dt, "date", _FixedDate)
    monkeypatch.setattr(aaii_sentiment.urllib.request, "urlopen", fake_urlopen)

    combined = aaii_sentiment.update_aaii_parquet(
        raw_dir=tmp_path,
        out_path=out_path,
        url="https://example.test/aaii",
    )

    pd.testing.assert_frame_equal(combined, existing)
    pd.testing.assert_frame_equal(pd.read_parquet(out_path), existing)


def test_run_sentiment_fetch_marks_fetch_run_failed_when_live_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "data" / "raw"
    sentiment_dir = out_dir / "sentiment"
    sentiment_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "bullish": 0.41,
                "neutral": 0.29,
                "bearish": 0.30,
                "bull_bear_spread": 0.11,
                "bull_bear_spread_8w_ma": 0.11,
            }
        ]
    ).to_parquet(sentiment_dir / "aaii_sentiment.parquet", index=False)
    acquisition_db = tmp_path / "acquisition.db"

    def fake_urlopen(_req, *, timeout: int):
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(aaii_sentiment.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Failed to fetch AAII sentiment page"):
        aaii_sentiment.run_sentiment_fetch(
            out_dir=out_dir,
            url="https://example.test/aaii",
            acquisition_db_path=acquisition_db,
        )

    with sqlite3.connect(acquisition_db) as conn:
        fetch_runs = conn.execute(
            "SELECT fetch_type, status, notes FROM fetch_runs"
        ).fetchall()

    assert len(fetch_runs) == 1
    assert fetch_runs[0][0] == "sentiment"
    assert fetch_runs[0][1] == "failed"
    assert "Failed to fetch AAII sentiment page" in fetch_runs[0][2]
