from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from regime_data_fetch.event_sources.models import EventCandidate
from regime_data_fetch.event_sources.validators_hf_central_bank import HFCentralBankValidator


def _candidate(event_date: dt.date, event_type: str) -> EventCandidate:
    return EventCandidate(
        date=event_date,
        event_type=event_type,
        market="GLOBAL",
        importance="high",
        source_id="official",
        source_url=None,
        raw_title="official decision",
        raw_snippet=None,
        is_future_scheduled=False,
        confidence="medium",
        requires_manual_review=False,
    )


def test_hf_validator_confirms_contradicts_and_returns_unknown(tmp_path: Path) -> None:
    parquet_path = tmp_path / "hf_sample.parquet"
    pd.DataFrame(
        [
            {
                "central_bank": "European Central Bank",
                "doc_type": "Monetary policy decision",
                "title": "Monetary policy decisions",
                "url": "https://hf.test/ecb",
                "meeting_date": "2026-06-11",
            },
            {
                "central_bank": "Bank of England",
                "doc_type": "Monetary Policy Summary",
                "title": "Bank Rate decision",
                "url": "https://hf.test/boe",
                "meeting_date": "2026-03-20",
            },
            {
                "central_bank": "Bank of Japan",
                "doc_type": "Speech",
                "title": "Governor speech",
                "url": "https://hf.test/boj",
                "meeting_date": "2026-06-16",
            },
        ]
    ).to_parquet(parquet_path, index=False)
    validator = HFCentralBankValidator(parquet_fetcher=lambda: parquet_path.read_bytes())

    results = validator.validate(
        [
            _candidate(dt.date(2026, 6, 11), "ECB_decision"),
            _candidate(dt.date(2026, 3, 19), "BOE_decision"),
            _candidate(dt.date(2026, 6, 16), "BOJ_decision"),
        ],
        store=None,
        run_id=None,
    )

    assert [(result.candidate_key, result.verdict, result.evidence_url) for result in results] == [
        (("ECB_decision", dt.date(2026, 6, 11)), "confirm", "https://hf.test/ecb"),
        (("BOE_decision", dt.date(2026, 3, 19)), "contradict", "https://hf.test/boe"),
        (("BOJ_decision", dt.date(2026, 6, 16)), "unknown", None),
    ]
