from __future__ import annotations

import datetime as dt

import pandas_market_calendars as mcal

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources.models import EventCandidate

SOURCE_ID = "usa.gov:federal-budget-process"
_NYSE = mcal.get_calendar("NYSE")


class DeterministicBudgetAdapter:
    source_id = SOURCE_ID

    def __init__(self, *, as_of_date: dt.date | None = None) -> None:
        self.as_of_date = as_of_date or dt.date.today()

    def fetch(
        self,
        *,
        start_year: int,
        end_year: int,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[EventCandidate]:
        del store, run_id
        candidates: list[EventCandidate] = []
        for year in range(start_year, end_year + 1):
            statutory_deadline = dt.date(year, 9, 30)
            emitted_date = _previous_nyse_session(statutory_deadline)
            candidates.append(
                EventCandidate(
                    date=emitted_date,
                    event_type="budget",
                    market="US",
                    importance="medium",
                    source_id=SOURCE_ID,
                    source_url="https://www.usa.gov/federal-budget-process",
                    raw_title=f"US federal fiscal year {year} deadline",
                    raw_snippet="The US federal fiscal year ends on September 30.",
                    is_future_scheduled=emitted_date > self.as_of_date,
                    confidence="high",
                    requires_manual_review=False,
                    event_subtype="fy_deadline",
                )
            )
        return candidates


def _previous_nyse_session(value: dt.date) -> dt.date:
    schedule = _NYSE.schedule(
        start_date=value - dt.timedelta(days=10),
        end_date=value,
    )
    if schedule.empty:
        raise RuntimeError(
            f"NYSE calendar returned no sessions before {value.isoformat()}"
        )
    return schedule.index[-1].date()
