from __future__ import annotations

import datetime as dt

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources.models import EventCandidate

SOURCE_ID = "usa.gov:federal-budget-process"


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
        return [
            EventCandidate(
                date=dt.date(year, 9, 30),
                event_type="budget",
                market="US",
                importance="medium",
                source_id=SOURCE_ID,
                source_url="https://www.usa.gov/federal-budget-process",
                raw_title=f"US federal fiscal year {year} deadline",
                raw_snippet="The US federal fiscal year ends on September 30.",
                is_future_scheduled=dt.date(year, 9, 30) > self.as_of_date,
                confidence="high",
                requires_manual_review=False,
                event_subtype="fy_deadline",
            )
            for year in range(start_year, end_year + 1)
        ]
