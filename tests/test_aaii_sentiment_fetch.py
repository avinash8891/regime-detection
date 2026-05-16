from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from regime_data_fetch import aaii_sentiment


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
