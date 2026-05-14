from __future__ import annotations

import datetime as dt

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources.models import EventCandidate

SOURCE_ID = "fec.gov:election-dates"
SOURCE_URL = "https://www.fec.gov/introduction-campaign-finance/election-and-voting-information/federal-elections/"


class ElectionAdapter:
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
            if year % 2 != 0:
                continue
            candidates.append(
                EventCandidate(
                    date=us_general_election_date(year),
                    event_type="election",
                    market="US",
                    importance="high",
                    source_id=SOURCE_ID,
                    source_url=SOURCE_URL,
                    raw_title=f"{year} {election_kind(year)} federal general election",
                    raw_snippet="Computed under 2 U.S.C. §7: Tuesday after the first Monday in November.",
                    is_future_scheduled=us_general_election_date(year) > self.as_of_date,
                    confidence="high",
                    requires_manual_review=False,
                    window_days=(-5, 10),
                )
            )
        return candidates


def election_kind(year: int) -> str:
    return "presidential" if year % 4 == 0 else "midterm"


def us_general_election_date(year: int) -> dt.date:
    first_november = dt.date(year, 11, 1)
    first_monday = first_november + dt.timedelta(days=(0 - first_november.weekday()) % 7)
    return first_monday + dt.timedelta(days=1)
