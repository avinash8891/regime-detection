from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

from regime_data_fetch.event_sources.models import (
    ApprovalRecord,
    EventCandidate,
    PromotionDecision,
    ValidationResult,
)

US_EASTERN = dt.timezone(dt.timedelta(hours=-5))
EventMarket = Literal["US", "GLOBAL"]
EventType = Literal[
    "FOMC",
    "CPI",
    "NFP",
    "budget",
    "election",
    "geopolitical_event",
    "global_rate_decision",
    "ECB_decision",
    "BOE_decision",
    "BOJ_decision",
]
EventImportance = Literal["high", "medium"]
EventSource = Literal[
    "federalreserve.gov:fomccalendars",
    "bls.gov:schedule:consumer-price-index",
    "bls.gov:schedule:employment-situation",
    "operator:event_calendar_v2",
    "ecb.europa.eu:governing-council-calendar",
    "ecb.europa.eu:monetary-policy-decisions",
    "bankofengland.co.uk:mpc-dates",
    "bankofengland.co.uk:mpc-decisions",
    "boj.or.jp:monetary-policy-meeting-schedule",
    "boj.or.jp:monetary-policy-meetings",
    "fec.gov:election-dates",
    "usa.gov:federal-budget-process",
    "official-us-budget-discovery",
    "congress.gov:public-law",
    "gdelt:events-v2",
    "gpr:caldara-iacoviello",
    "acled:events",
    "ucdp:ged-candidate",
]
EVENT_MARKETS = frozenset(cast(tuple[EventMarket, ...], ("US", "GLOBAL")))
EVENT_TYPES = frozenset(cast(tuple[EventType, ...], get_args(EventType)))
EVENT_IMPORTANCES = frozenset(cast(tuple[EventImportance, ...], ("high", "medium")))
EVENT_SOURCES = frozenset(cast(tuple[EventSource, ...], get_args(EventSource)))


class EventCalendarFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScheduledEvent:
    date: dt.date
    release_timestamp_et: dt.datetime
    market: EventMarket
    type: EventType
    importance: EventImportance
    source: EventSource
    window_days: tuple[int, int] | None = None
    approved_label: str | None = None

    def __post_init__(self) -> None:
        if self.market not in EVENT_MARKETS:
            raise ValueError(f"unknown scheduled event market: {self.market}")
        if self.type not in EVENT_TYPES:
            raise ValueError(f"unknown scheduled event type: {self.type}")
        if self.importance not in EVENT_IMPORTANCES:
            raise ValueError(
                f"unknown scheduled event importance: {self.importance}"
            )
        if self.source not in EVENT_SOURCES:
            raise ValueError(f"unknown scheduled event source: {self.source}")
        if self.window_days is not None:
            lower, upper = self.window_days
            if lower > upper:
                raise ValueError(
                    f"window_days lower bound must be <= upper bound: {self.window_days}"
                )


@dataclass(frozen=True)
class EventLabelResolution:
    matching_labels: tuple[str, ...]
    primary_label: str


@dataclass(frozen=True)
class GroupABuildResult:
    scheduled_events: list[ScheduledEvent]
    candidates: list[EventCandidate]
    validations: list[ValidationResult]
    decisions: list[PromotionDecision]
    output_paths: dict[str, Path]
    approval_overlay: list[ApprovalRecord] | None = None
