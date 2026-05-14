from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Callable
from typing import Any

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources.models import EventCandidate

LOGGER = logging.getLogger(__name__)
SOURCE_ID = "official-us-budget-discovery"
VALID_SUBTYPES = {"debt_ceiling", "shutdown", "cr_expiration"}


class BudgetOfficialDiscoveryGenerator:
    source_id = SOURCE_ID

    def __init__(self, *, records_fetcher: Callable[[], str | list[dict[str, Any]]] | None = None, as_of_date: dt.date | None = None) -> None:
        self.records_fetcher = records_fetcher or (lambda: [])
        self.as_of_date = as_of_date or dt.date.today()

    def generate(
        self,
        *,
        start_year: int,
        end_year: int,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[EventCandidate]:
        try:
            payload = self.records_fetcher()
        except Exception as exc:  # pragma: no cover - external degradation path
            LOGGER.error("official budget discovery failed; non-deterministic budget candidates skipped: %s", exc)
            return []
        _record_payload(store, run_id, payload)
        candidates = [
            _candidate(record, self.as_of_date)
            for record in parse_budget_official_records(payload)
            if start_year <= record["date"].year <= end_year
        ]
        return candidates


def parse_budget_official_records(payload: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_records = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(raw_records, list):
        raise ValueError("budget official discovery payload must be a list")
    records: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_records, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"budget official discovery record {idx} must be a mapping")
        event_subtype = str(raw.get("event_subtype", ""))
        if event_subtype not in VALID_SUBTYPES:
            raise ValueError(f"budget official discovery record {idx} has invalid event_subtype: {event_subtype}")
        source_id = str(raw.get("source_id", ""))
        if not source_id:
            raise ValueError(f"budget official discovery record {idx} missing source_id")
        records.append(
            {
                "date": dt.date.fromisoformat(str(raw["date"])),
                "event_subtype": event_subtype,
                "source_id": source_id,
                "source_url": str(raw.get("source_url", "")) or None,
                "raw_title": str(raw.get("raw_title", event_subtype)),
                "raw_snippet": str(raw.get("raw_snippet", "")) or None,
            }
        )
    return records


def _candidate(record: dict[str, Any], as_of_date: dt.date) -> EventCandidate:
    return EventCandidate(
        date=record["date"],
        event_type="budget",
        market="US",
        importance="high",
        source_id=record["source_id"],
        source_url=record["source_url"],
        raw_title=record["raw_title"],
        raw_snippet=record["raw_snippet"],
        is_future_scheduled=record["date"] > as_of_date,
        confidence="medium",
        requires_manual_review=True,
        event_subtype=record["event_subtype"],
    )


def _record_payload(store: AcquisitionStore | None, run_id: int | None, payload: object) -> None:
    if store is None or run_id is None:
        return
    content = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True)
    store.record_text_artifact(
        run_id=run_id,
        source_name=SOURCE_ID,
        artifact_kind="json",
        source_identifier="official_budget_records",
        content_text=content,
        calendar_assumption="US fiscal event dates",
        timezone="America/New_York",
        license_note="Official US government public-domain source records",
        notes="Official budget event discovery records",
    )
