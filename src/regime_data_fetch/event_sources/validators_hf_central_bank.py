from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import pandas as pd

from regime_data_fetch._http import fetch_bytes
from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources.models import EventCandidate, ValidationResult

LOGGER = logging.getLogger(__name__)
VALIDATOR_ID = "hf:aufklarer-central-bank-communications"
PARQUET_URL = (
    "https://huggingface.co/datasets/aufklarer/central-bank-communications/"
    "resolve/refs%2Fconvert%2Fparquet/documents_other/train/0000.parquet"
)

_EVENT_BANKS = {
    "ECB_decision": ("european central bank", "ecb"),
    "BOE_decision": ("bank of england", "boe"),
    "BOJ_decision": ("bank of japan", "boj"),
}
_DECISION_TERMS = ("decision", "monetary policy summary", "statement", "minutes")
_NON_DECISION_TERMS = ("speech", "interview", "remarks", "testimony")


class HFCentralBankValidator:
    validator_id = VALIDATOR_ID

    def __init__(
        self,
        *,
        parquet_fetcher: Callable[[], bytes] | None = None,
        confirmation_window_days: int = 1,
    ) -> None:
        self.parquet_fetcher = parquet_fetcher or fetch_hf_parquet
        self.confirmation_window_days = confirmation_window_days

    def validate(
        self,
        candidates: list[EventCandidate],
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[ValidationResult]:
        central_bank_candidates = [
            candidate
            for candidate in candidates
            if candidate.event_type in _EVENT_BANKS
        ]
        if not central_bank_candidates:
            return []
        try:
            parquet_bytes = self.parquet_fetcher()
        except (TimeoutError, OSError) as exc:
            LOGGER.warning(
                "hf_central_bank parquet fetch failed; returning unknown fallback "
                "validator_id=%s candidate_count=%s",
                VALIDATOR_ID,
                len(central_bank_candidates),
                exc_info=True,
            )
            return [
                _unknown(
                    candidate,
                    evidence_snippet=f"validator_source_unavailable: {type(exc).__name__}",
                )
                for candidate in central_bank_candidates
            ]
        frame = pd.read_parquet(BytesIO(parquet_bytes))

        if store is not None and run_id is not None:
            artifact_path = Path(
                "data/raw/event_calendar/hf_central_bank_documents_other.parquet"
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(parquet_bytes)
            store.record_file_artifact(
                run_id=run_id,
                source_name=VALIDATOR_ID,
                artifact_kind="parquet",
                source_identifier=PARQUET_URL,
                file_path=artifact_path,
                calendar_assumption="validator only",
                timezone="source dataset",
                license_note="Hugging Face public dataset snapshot",
                notes="Central-bank communications validator parquet",
                store_bytes=False,
            )

        return [
            self._validate_candidate(candidate, frame)
            for candidate in central_bank_candidates
        ]

    def _validate_candidate(
        self, candidate: EventCandidate, frame: pd.DataFrame
    ) -> ValidationResult:
        bank_terms = _EVENT_BANKS[candidate.event_type]
        bank_rows = frame[
            frame["central_bank"]
            .fillna("")
            .astype(str)
            .str.lower()
            .map(lambda value: any(term in value for term in bank_terms))
        ]
        if bank_rows.empty:
            return _unknown(candidate)

        decision_rows = bank_rows[
            bank_rows.apply(
                lambda row: _is_decision_doc(
                    str(row.get("doc_type", "")), str(row.get("title", ""))
                ),
                axis=1,
            )
        ].copy()
        if decision_rows.empty:
            return _unknown(candidate)

        decision_rows = decision_rows.assign(
            parsed_date=pd.to_datetime(
                decision_rows["meeting_date"], errors="coerce"
            ).dt.date
        )
        dated_rows = decision_rows.dropna(subset=["parsed_date"])
        if dated_rows.empty:
            return _unknown(candidate)

        window_start = candidate.date - dt.timedelta(days=self.confirmation_window_days)
        window_end = candidate.date + dt.timedelta(days=self.confirmation_window_days)
        window_rows = dated_rows[
            (dated_rows["parsed_date"] >= window_start)
            & (dated_rows["parsed_date"] <= window_end)
        ]
        same_day = window_rows[window_rows["parsed_date"] == candidate.date]
        if not same_day.empty:
            row = same_day.iloc[0]
            return ValidationResult(
                candidate_key=(candidate.event_type, candidate.date),
                validator_id=VALIDATOR_ID,
                verdict="confirm",
                evidence_url=_optional_str(row.get("url")),
                evidence_snippet=_optional_str(row.get("title")),
            )
        if not window_rows.empty:
            row = window_rows.iloc[0]
            return ValidationResult(
                candidate_key=(candidate.event_type, candidate.date),
                validator_id=VALIDATOR_ID,
                verdict="contradict",
                evidence_url=_optional_str(row.get("url")),
                evidence_snippet=_optional_str(row.get("title")),
            )
        return _unknown(candidate)


def fetch_hf_parquet() -> bytes:
    return fetch_bytes(PARQUET_URL, timeout=60)


def _is_decision_doc(doc_type: str, title: str) -> bool:
    combined = f"{doc_type} {title}".lower()
    if any(term in combined for term in _NON_DECISION_TERMS):
        return False
    return any(term in combined for term in _DECISION_TERMS)


def _unknown(
    candidate: EventCandidate, *, evidence_snippet: str | None = None
) -> ValidationResult:
    return ValidationResult(
        candidate_key=(candidate.event_type, candidate.date),
        validator_id=VALIDATOR_ID,
        verdict="unknown",
        evidence_url=None,
        evidence_snippet=evidence_snippet,
    )


def _optional_str(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)
