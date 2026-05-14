from __future__ import annotations

import json
import logging
from collections.abc import Callable

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources.models import EventCandidate, ValidationResult

LOGGER = logging.getLogger(__name__)
VALIDATOR_ID = "tinyfish:search-extract"


class TinyFishValidator:
    validator_id = VALIDATOR_ID

    def __init__(self, *, search_fetcher: Callable[[EventCandidate], str | dict[str, object]] | None = None) -> None:
        self.search_fetcher = search_fetcher

    def validate(
        self,
        candidates: list[EventCandidate],
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[ValidationResult]:
        del store, run_id
        return [self._validate_candidate(candidate) for candidate in candidates if candidate.event_type in {"geopolitical_event", "budget"}]

    def _validate_candidate(self, candidate: EventCandidate) -> ValidationResult:
        key = (candidate.event_type, candidate.date)
        if self.search_fetcher is None:
            return ValidationResult(key, self.validator_id, "unknown", None, "TinyFish fetcher not configured")
        try:
            payload = self.search_fetcher(candidate)
        except Exception as exc:
            LOGGER.error("TinyFish unavailable for %s %s; verdict unknown: %s", candidate.event_type, candidate.date, exc)
            return ValidationResult(key, self.validator_id, "unknown", None, "TinyFish unavailable or unauthenticated")
        parsed = json.loads(payload) if isinstance(payload, str) else payload
        if bool(parsed.get("confirm")):
            return ValidationResult(
                key,
                self.validator_id,
                "confirm",
                str(parsed.get("url", "")) or None,
                str(parsed.get("snippet", "TinyFish corroborated candidate")),
            )
        return ValidationResult(key, self.validator_id, "unknown", str(parsed.get("url", "")) or None, str(parsed.get("snippet", "TinyFish found no confirmation")))
