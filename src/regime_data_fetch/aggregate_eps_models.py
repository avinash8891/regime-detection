from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


class AggregateEPSFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class AggregateEPSSnapshot:
    observation_date: dt.date
    observation_label: str
    forward_estimate_label: str | None
    forward_estimate_value: float | None
    estimate_2025e: float | None
    estimate_q4_2025e: float | None
    estimate_2026e: float | None
    price: float | None
    pe_2025e: float | None
    pe_2026e: float | None
    change_vs_prior_observation_2025e: float | None
    change_vs_prior_observation_q4_2025e: float | None
    change_vs_prior_observation_2026e: float | None
    change_vs_prior_observation_price: float | None
    change_vs_prior_observation_pe_2025e: float | None
    change_vs_prior_observation_pe_2026e: float | None


@dataclass(frozen=True)
class ParsedAggregateEPSWorkbook:
    workbook_as_of_date: dt.date
    public_files_discontinued: bool
    historical_snapshots: list[AggregateEPSSnapshot]
    current_snapshot: AggregateEPSSnapshot


@dataclass(frozen=True)
class EPSWaybackSnapshot:
    timestamp: str
    archive_url: str
    snapshot_date: dt.date
