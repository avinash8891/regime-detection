from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

US_EASTERN = dt.timezone(dt.timedelta(hours=-5))


class EventCalendarFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScheduledEvent:
    date: dt.date
    release_timestamp_et: dt.datetime
    market: str
    type: str
    importance: str
    source: str
    window_days: tuple[int, int] | None = None
    approved_label: str | None = None


@dataclass(frozen=True)
class EventLabelResolution:
    all_matching_events: list[str]
    selected_via_precedence: str


@dataclass(frozen=True)
class GroupABuildResult:
    scheduled_events: list[ScheduledEvent]
    candidates: list[object]
    validations: list[object]
    decisions: list[object]
    output_paths: dict[str, Path]
    approval_overlay: list[object] | None = None
