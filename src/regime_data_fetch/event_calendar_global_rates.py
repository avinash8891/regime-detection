"""Parse central-bank rate-decision calendars into normalized event dates."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
import re

from regime_data_fetch.event_sources._common import MONTHS

SOURCE_ECB = "ecb.europa.eu:governing-council-calendar"
SOURCE_BOE = "bankofengland.co.uk:mpc-dates"
SOURCE_BOJ = "boj.or.jp:monetary-policy-meeting-schedule"

GLOBAL_RATE_URLS = {
    "ecb": "https://www.ecb.europa.eu/events/calendar/mgcgc/html/index.en.html",
    "boe": "https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates",
    "boj": "https://www.boj.or.jp/en/mopo/mpmsche_minu/",
}

class UnsupportedGlobalRateSource(ValueError):
    pass


@dataclass(frozen=True)
class GlobalRateDecisionEvent:
    date: dt.date
    event_type: str
    source: str


def parse_global_rate_decision_events(*, source_key: str, text: str) -> list[GlobalRateDecisionEvent]:
    if source_key == "ecb":
        return parse_ecb_decision_events(text)
    if source_key == "boj":
        return parse_boj_decision_events(text)
    normalized = re.sub(r"<[^>]+>", " ", text)
    normalized = re.sub(r"\s+", " ", normalized)
    if source_key == "boe":
        return parse_boe_decision_events(normalized)
    raise UnsupportedGlobalRateSource(f"Unsupported global rate calendar source: {source_key}")


def parse_ecb_decision_events(text: str) -> list[GlobalRateDecisionEvent]:
    events: list[GlobalRateDecisionEvent] = []
    row_pattern = re.compile(
        r"<dt[^>]*>\s*(?P<date>\d{2}/\d{2}/\d{4})\s*</dt>\s*<dd[^>]*>(?P<description>.*?)</dd>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in row_pattern.finditer(text):
        description = re.sub(r"<[^>]+>", " ", match.group("description"))
        if "non-monetary" in description.lower():
            continue
        if "monetary policy meeting" not in description.lower():
            continue
        if "day 2" not in description.lower() and "press conference" not in description.lower():
            continue
        day, month, year = (int(part) for part in match.group("date").split("/"))
        events.append(GlobalRateDecisionEvent(dt.date(year, month, day), "ECB_decision", SOURCE_ECB))
    return _dedupe_events(events)


def parse_boe_decision_events(text: str) -> list[GlobalRateDecisionEvent]:
    events: list[GlobalRateDecisionEvent] = []
    current_year: int | None = None
    tokens = re.split(
        r"(?=(?:20\d{2})\s+(?:confirmed|provisional)\s+dates)"
        r"|(?=(?:Monday|Tuesday|Wednesday|Thursday|Friday)\s+\d{1,2}\s+[A-Za-z]+)",
        text,
    )
    date_pattern = re.compile(
        r"(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday)\s+)?(?P<day>\d{1,2})\s+"
        r"(?P<month>[A-Za-z]+).*?(?:MPC|Monetary Policy)",
        flags=re.IGNORECASE,
    )
    for token in tokens:
        year_match = re.search(r"\b(?P<year>20\d{2})\s+(?:confirmed|provisional)\s+dates\b", token, flags=re.IGNORECASE)
        if year_match:
            current_year = int(year_match.group("year"))
        if current_year is None:
            continue
        date_match = date_pattern.search(token)
        if not date_match:
            continue
        month = MONTHS.get(date_match.group("month").lower())
        if month is None:
            continue
        event_date = dt.date(current_year, month, int(date_match.group("day")))
        events.append(GlobalRateDecisionEvent(event_date, "BOE_decision", SOURCE_BOE))
    return _dedupe_events(events)


def parse_boj_decision_events(text: str) -> list[GlobalRateDecisionEvent]:
    events: list[GlobalRateDecisionEvent] = []
    section_pattern = re.compile(
        r"<h2[^>]*>\s*(?P<year>20\d{2})\s*</h2>(?P<section>.*?)(?=<h2[^>]*>\s*20\d{2}\s*</h2>|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    row_pattern = re.compile(r"<tr[^>]*>(?P<row>.*?)</tr>", flags=re.IGNORECASE | re.DOTALL)
    cell_pattern = re.compile(r"<td[^>]*>(?P<cell>.*?)</td>", flags=re.IGNORECASE | re.DOTALL)
    table_date_pattern = re.compile(
        r"(?P<month>Jan\.?|January|Feb\.?|February|Mar\.?|March|Apr\.?|April|May|June|July|Aug\.?|August|Sep\.?|Sept\.?|September|Oct\.?|October|Nov\.?|November|Dec\.?|December)\s+"
        r"(?P<start>\d{1,2})(?:\s*\([^)]+\))?(?:\s*,\s*(?P<end>\d{1,2})(?:\s*\([^)]+\))?)?",
        flags=re.IGNORECASE,
    )
    for section_match in section_pattern.finditer(text):
        year = int(section_match.group("year"))
        for row_match in row_pattern.finditer(section_match.group("section")):
            cell_match = cell_pattern.search(row_match.group("row"))
            if cell_match is None:
                continue
            cell_text = re.sub(r"<[^>]+>", " ", cell_match.group("cell"))
            cell_text = re.sub(r"\s+", " ", cell_text)
            date_match = table_date_pattern.search(cell_text)
            if date_match is None:
                continue
            month = MONTHS[date_match.group("month").lower()]
            day = int(date_match.group("end") or date_match.group("start"))
            events.append(GlobalRateDecisionEvent(dt.date(year, month, day), "BOJ_decision", SOURCE_BOJ))

    normalized = re.sub(r"<[^>]+>", " ", text)
    normalized = re.sub(r"\s+", " ", normalized)
    pattern = re.compile(
        r"(?P<month>Jan\.?|January|Feb\.?|February|Mar\.?|March|Apr\.?|April|May|June|July|Aug\.?|August|Sep\.?|Sept\.?|September|Oct\.?|October|Nov\.?|November|Dec\.?|December)\s+"
        r"(?P<start>\d{1,2})(?:\s*(?:-|and)\s*(?P<end>\d{1,2}))?,\s*(?P<year>20\d{2})",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(normalized):
        month = MONTHS[match.group("month").lower()]
        day = int(match.group("end") or match.group("start"))
        event_date = dt.date(int(match.group("year")), month, day)
        events.append(GlobalRateDecisionEvent(event_date, "BOJ_decision", SOURCE_BOJ))
    return _dedupe_events(events)


def global_rate_source_name(source_key: str) -> str:
    return {"ecb": SOURCE_ECB, "boe": SOURCE_BOE, "boj": SOURCE_BOJ}.get(source_key, source_key)


def _dedupe_events(events: list[GlobalRateDecisionEvent]) -> list[GlobalRateDecisionEvent]:
    return list({(event.date, event.event_type, event.source): event for event in events}.values())
