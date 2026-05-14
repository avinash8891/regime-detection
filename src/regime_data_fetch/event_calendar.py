from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from urllib.request import Request, urlopen

import pandas_market_calendars as mcal
import yaml

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.bls_schedule import BLSScheduleFetchError, fetch_bls_schedule_page_text, fetch_bls_year_releases
from regime_data_fetch.earnings_season_calendar import is_in_earnings_season
from regime_data_fetch.expiry_calendar import expand_trading_day_window, compute_monthly_options_expiry_anchor
from regime_data_fetch.fomc_minutes import (
    fetch_fomc_historical_year_index,
    fetch_fomc_historical_year_page,
    fetch_fomc_minutes_listing,
    fetch_release_timestamp,
    parse_fomc_historical_year_index,
    parse_fomc_minutes_historical_listing,
    parse_fomc_minutes_listing,
)

US_EASTERN = dt.timezone(dt.timedelta(hours=-5))
SOURCE_FOMC = "federalreserve.gov:fomccalendars"
SOURCE_CPI = "bls.gov:schedule:consumer-price-index"
SOURCE_NFP = "bls.gov:schedule:employment-situation"
SOURCE_ECB = "ecb.europa.eu:governing-council-calendar"
SOURCE_BOE = "bankofengland.co.uk:mpc-dates"
SOURCE_BOJ = "boj.or.jp:monetary-policy-meeting-schedule"
SOURCE_FEC = "fec.gov:election-dates"
SOURCE_US_BUDGET = "usa.gov:federal-budget-process"
_FOMC_MINUTES_LINK_RE = re.compile(r"/monetarypolicy/fomcminutes(?P<meeting_end>\d{8})\.htm", flags=re.IGNORECASE)
_NYSE = mcal.get_calendar("NYSE")
_GLOBAL_RATE_URLS = {
    "ecb": "https://www.ecb.europa.eu/events/calendar/mgcgc/html/index.en.html",
    "boe": "https://www.bankofengland.co.uk/monetary-policy/upcoming-mpc-dates",
    "boj": "https://www.boj.or.jp/en/mopo/mpmsche_minu/",
}
_BLS_OFFICIAL_CANCELED_RELEASE_COUNTS = {
    # BLS 2025 lapse page: October 2025 CPI and Employment Situation
    # news releases were canceled, so release-date year 2025 has 11 rows.
    ("CPI", 2025): 1,
    ("NFP", 2025): 1,
}
_MONTHS = {
    "january": 1,
    "jan": 1,
    "jan.": 1,
    "february": 2,
    "feb": 2,
    "feb.": 2,
    "march": 3,
    "mar": 3,
    "mar.": 3,
    "april": 4,
    "apr": 4,
    "apr.": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "jun.": 6,
    "july": 7,
    "jul": 7,
    "jul.": 7,
    "august": 8,
    "aug": 8,
    "aug.": 8,
    "september": 9,
    "sept": 9,
    "sept.": 9,
    "sep": 9,
    "sep.": 9,
    "october": 10,
    "oct": 10,
    "oct.": 10,
    "november": 11,
    "nov": 11,
    "nov.": 11,
    "december": 12,
    "dec": 12,
    "dec.": 12,
}
EVENT_PRECEDENCE = (
    "geopolitical_event",
    "election_window",
    "fed_week",
    "global_rate_decision",
    "budget_week",
    "cpi_week",
    "nfp_week",
    "expiry_week",
    "earnings_season",
    "normal_calendar",
    "unknown",
)
SCHEDULED_EVENT_WINDOWS: dict[str, tuple[str, int, int]] = {
    "FOMC": ("fed_week", 2, 2),
    "CPI": ("cpi_week", 1, 1),
    "NFP": ("nfp_week", 1, 1),
    "budget": ("budget_week", 0, 0),
    "election": ("election_window", 5, 10),
    "geopolitical_event": ("geopolitical_event", 0, 0),
    "global_rate_decision": ("global_rate_decision", 0, 0),
    "ECB_decision": ("global_rate_decision", 0, 0),
    "BOE_decision": ("global_rate_decision", 0, 0),
    "BOJ_decision": ("global_rate_decision", 0, 0),
}


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


@dataclass(frozen=True)
class EventLabelResolution:
    all_matching_events: list[str]
    selected_via_precedence: str


def fetch_bls_release_timestamp(release_date: dt.date) -> dt.datetime:
    return dt.datetime(
        release_date.year,
        release_date.month,
        release_date.day,
        8,
        30,
        tzinfo=US_EASTERN,
    )


def load_scheduled_events_yaml(path: Path) -> list[ScheduledEvent]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise EventCalendarFetchError(f"Event calendar YAML at {path} did not contain an events list")

    events: list[ScheduledEvent] = []
    for idx, entry in enumerate(payload["events"], start=1):
        if not isinstance(entry, dict):
            raise EventCalendarFetchError(f"Event calendar YAML entry {idx} was not a mapping")
        try:
            event_date = dt.date.fromisoformat(str(entry["date"]))
            release_timestamp_raw = entry.get("release_timestamp_et")
            release_timestamp = (
                dt.datetime.fromisoformat(str(release_timestamp_raw))
                if release_timestamp_raw is not None
                else dt.datetime(
                    event_date.year,
                    event_date.month,
                    event_date.day,
                    0,
                    0,
                    tzinfo=US_EASTERN,
                )
            )
            market = str(entry["market"])
            event_type = str(entry["type"])
            importance = str(entry["importance"])
            source = str(entry["source"])
            window_days = _parse_window_days(entry.get("window_days"))
        except KeyError as exc:
            raise EventCalendarFetchError(f"Event calendar YAML entry {idx} missing field {exc.args[0]!r}") from exc
        except ValueError as exc:
            raise EventCalendarFetchError(f"Event calendar YAML entry {idx} contained an invalid date/timestamp") from exc

        events.append(
            ScheduledEvent(
                date=event_date,
                release_timestamp_et=release_timestamp,
                market=market,
                type=event_type,
                importance=importance,
                source=source,
                window_days=window_days,
            )
        )
    return events


def resolve_event_label(
    *,
    as_of_date: dt.date,
    scheduled_events: list[ScheduledEvent],
) -> EventLabelResolution:
    matching_events: set[str] = set()
    matching_events.update(_matching_scheduled_event_labels(as_of_date=as_of_date, scheduled_events=scheduled_events))

    expiry_anchor = compute_monthly_options_expiry_anchor(year=as_of_date.year, month=as_of_date.month)
    expiry_window = set(expand_trading_day_window(anchor_date=expiry_anchor, lookback_trading_days=2, lookahead_trading_days=0))
    if as_of_date in expiry_window:
        matching_events.add("expiry_week")

    if is_in_earnings_season(as_of_date=as_of_date):
        matching_events.add("earnings_season")

    ordered_matches = [label for label in EVENT_PRECEDENCE if label in matching_events]
    if not ordered_matches:
        return EventLabelResolution(all_matching_events=[], selected_via_precedence="normal_calendar")
    return EventLabelResolution(
        all_matching_events=ordered_matches,
        selected_via_precedence=ordered_matches[0],
    )


def validate_fomc_listing_integrity(
    *,
    html: str,
    parsed_entries: list[ScheduledEvent],
    min_year: int | None = None,
    max_year: int | None = None,
) -> None:
    raw_dates = {
        dt.datetime.strptime(match.group("meeting_end"), "%Y%m%d").date()
        for match in _FOMC_MINUTES_LINK_RE.finditer(html)
    }
    if min_year is not None:
        raw_dates = {value for value in raw_dates if value.year >= min_year}
    if max_year is not None:
        raw_dates = {value for value in raw_dates if value.year <= max_year}

    parsed_dates = {event.date for event in parsed_entries}
    if min_year is not None:
        parsed_dates = {value for value in parsed_dates if value.year >= min_year}
    if max_year is not None:
        parsed_dates = {value for value in parsed_dates if value.year <= max_year}

    if parsed_dates != raw_dates:
        missing = [value.isoformat() for value in sorted(raw_dates - parsed_dates)]
        extra = [value.isoformat() for value in sorted(parsed_dates - raw_dates)]
        raise EventCalendarFetchError(
            "FOMC listing parse mismatch against raw minute-link scan: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    counts_by_year = Counter(value.year for value in parsed_dates)
    years = sorted(counts_by_year)
    if years:
        last_year = max(years)
        for year in years:
            if year == last_year:
                continue
            count = counts_by_year[year]
            if count != 8:
                raise EventCalendarFetchError(
                    f"FOMC year {year} had {count} scheduled meetings after parse; expected 8 for a complete scheduled year"
                )


def run_us_event_calendar_fetch(
    *,
    repo_root: Path,
    fred_api_key: str | None,
    fomc_listing_fetcher=fetch_fomc_minutes_listing,
    fomc_historical_index_fetcher=fetch_fomc_historical_year_index,
    fomc_historical_page_fetcher=fetch_fomc_historical_year_page,
    bls_page_fetcher=None,
    acquisition_db_path: Path | None = None,
    bls_start_year: int = 2000,
    bls_end_year: int | None = None,
    include_v2_curated_candidates: bool = False,
    global_rate_calendar_text_fetchers: Mapping[str, Callable[[], str]] | None = None,
) -> Path:
    del fred_api_key

    store = AcquisitionStore(acquisition_db_path) if acquisition_db_path else None
    fetch_run = (
        store.start_fetch_run(
            fetch_type="events",
            params={
                "bls_start_year": bls_start_year,
                "bls_end_year": bls_end_year or dt.date.today().year,
            },
        )
        if store
        else None
    )

    try:
        events = _fetch_fomc_events(
            listing_fetcher=fomc_listing_fetcher,
            historical_index_fetcher=fomc_historical_index_fetcher,
            historical_page_fetcher=fomc_historical_page_fetcher,
            store=store,
            run_id=fetch_run.run_id if fetch_run else None,
        )
        events.extend(
            _fetch_bls_events(
                page_fetcher=bls_page_fetcher,
                start_year=bls_start_year,
                end_year=bls_end_year or dt.date.today().year,
                store=store,
                run_id=fetch_run.run_id if fetch_run else None,
            )
        )
        if include_v2_curated_candidates:
            events.extend(
                _build_v2_curated_candidate_events(
                    start_year=bls_start_year,
                    end_year=bls_end_year or dt.date.today().year,
                    global_rate_calendar_text_fetchers=global_rate_calendar_text_fetchers,
                    store=store,
                    run_id=fetch_run.run_id if fetch_run else None,
                )
            )

        events.sort(key=lambda event: (event.release_timestamp_et, event.type))

        output_dir = repo_root / "configs" / "events"
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = output_dir / "us_events.yaml"
        yaml_path.write_text(_render_events_yaml(events))

        counts = Counter(event.type for event in events)
        report = {
            "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": {
                "fomc": SOURCE_FOMC,
                "cpi": SOURCE_CPI,
                "nfp": SOURCE_NFP,
                "election": SOURCE_FEC if include_v2_curated_candidates else None,
                "budget": SOURCE_US_BUDGET if include_v2_curated_candidates else None,
                "ecb": SOURCE_ECB if include_v2_curated_candidates else None,
                "boe": SOURCE_BOE if include_v2_curated_candidates else None,
                "boj": SOURCE_BOJ if include_v2_curated_candidates else None,
            },
            "counts": {
                "total_events": len(events),
                "by_type": {key: counts[key] for key in sorted(counts)},
            },
            "paths": {
                "event_calendar_yaml": str(yaml_path),
                "acquisition_db": str(acquisition_db_path) if acquisition_db_path else None,
            },
        }
        report_path = repo_root / "event_calendar_fetch_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if store and fetch_run:
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="event_calendar_yaml",
                path=yaml_path,
                row_count=len(events),
                min_date=min(event.date for event in events).isoformat() if events else None,
                max_date=max(event.date for event in events).isoformat() if events else None,
                notes="Generated scheduled US event calendar",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="event_calendar_report",
                path=report_path,
                row_count=len(events),
                min_date=min(event.date for event in events).isoformat() if events else None,
                max_date=max(event.date for event in events).isoformat() if events else None,
                notes="Event calendar fetch report",
            )
            store.finish_fetch_run(run_id=fetch_run.run_id, status="ok")
        return report_path
    except Exception as exc:
        if store and fetch_run:
            store.finish_fetch_run(run_id=fetch_run.run_id, status="failed", notes=str(exc))
        raise


def _matching_scheduled_event_labels(*, as_of_date: dt.date, scheduled_events: list[ScheduledEvent]) -> set[str]:
    labels: set[str] = set()
    for event in scheduled_events:
        if event.market not in {"US", "GLOBAL"}:
            continue
        window_spec = SCHEDULED_EVENT_WINDOWS.get(event.type)
        if window_spec is None:
            continue
        label, lookback_days, lookahead_days = window_spec
        if event.window_days is not None:
            start_offset, end_offset = event.window_days
            lookback_days = abs(start_offset)
            lookahead_days = end_offset
        window = set(
            _expand_nyse_window_for_scheduled_event(
                anchor_date=event.date,
                lookback_trading_days=lookback_days,
                lookahead_trading_days=lookahead_days,
            )
        )
        if as_of_date in window:
            labels.add(label)
    return labels


def _parse_window_days(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = yaml.safe_load(value)
    else:
        parsed = value
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        raise EventCalendarFetchError(
            f"window_days must be a two-item list, got {value!r}"
        )
    try:
        return (int(parsed[0]), int(parsed[1]))
    except (TypeError, ValueError) as exc:
        raise EventCalendarFetchError(
            f"window_days entries must be integers, got {value!r}"
        ) from exc


def _fetch_fomc_events(
    *,
    listing_fetcher,
    historical_index_fetcher,
    historical_page_fetcher,
    store: AcquisitionStore | None,
    run_id: int | None,
) -> list[ScheduledEvent]:
    listing_html = listing_fetcher()
    if store and run_id is not None:
        store.record_text_artifact(
            run_id=run_id,
            source_name=SOURCE_FOMC,
            artifact_kind="html",
            source_identifier="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            content_text=listing_html,
            calendar_assumption="NYSE trading calendar",
            timezone="America/New_York",
            license_note="Federal Reserve public webpage",
            notes="Current FOMC calendar page",
        )
    entries = parse_fomc_minutes_listing(listing_html)
    validate_fomc_listing_integrity(
        html=listing_html,
        parsed_entries=[
            ScheduledEvent(
                date=entry.meeting_end_date,
                release_timestamp_et=fetch_release_timestamp(entry.release_date),
                market="US",
                type="FOMC",
                importance="high",
                source=SOURCE_FOMC,
            )
            for entry in entries
        ],
        min_year=2021,
    )
    historical_index_html = historical_index_fetcher()
    if store and run_id is not None and historical_index_html.strip():
        store.record_text_artifact(
            run_id=run_id,
            source_name=SOURCE_FOMC,
            artifact_kind="html",
            source_identifier="https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm",
            content_text=historical_index_html,
            calendar_assumption="NYSE trading calendar",
            timezone="America/New_York",
            license_note="Federal Reserve public webpage",
            notes="FOMC historical year index page",
        )
    if historical_index_html.strip():
        for url in parse_fomc_historical_year_index(historical_index_html):
            historical_html = historical_page_fetcher(url)
            if store and run_id is not None:
                store.record_text_artifact(
                    run_id=run_id,
                    source_name=SOURCE_FOMC,
                    artifact_kind="html",
                    source_identifier=url,
                    content_text=historical_html,
                    calendar_assumption="NYSE trading calendar",
                    timezone="America/New_York",
                    license_note="Federal Reserve public webpage",
                    notes="FOMC historical yearly page",
                )
            historical_entries = parse_fomc_minutes_historical_listing(historical_html)
            year = int(url.rsplit("fomchistorical", 1)[1].split(".htm", 1)[0])
            validate_fomc_listing_integrity(
                html=historical_html,
                parsed_entries=[
                    ScheduledEvent(
                        date=entry.meeting_end_date,
                        release_timestamp_et=fetch_release_timestamp(entry.release_date),
                        market="US",
                        type="FOMC",
                        importance="high",
                        source=SOURCE_FOMC,
                    )
                    for entry in historical_entries
                ],
                min_year=year,
                max_year=year,
            )
            entries.extend(historical_entries)

    deduped: dict[dt.date, ScheduledEvent] = {}
    for entry in entries:
        deduped[entry.meeting_end_date] = ScheduledEvent(
            date=entry.meeting_end_date,
            release_timestamp_et=fetch_release_timestamp(entry.release_date),
            market="US",
            type="FOMC",
            importance="high",
            source=SOURCE_FOMC,
        )
    return list(deduped.values())


def _fetch_bls_events(
    *,
    page_fetcher,
    start_year: int,
    end_year: int,
    store: AcquisitionStore | None,
    run_id: int | None,
) -> list[ScheduledEvent]:
    recording_page_fetcher = page_fetcher
    if store and run_id is not None:
        base_fetcher = page_fetcher

        def recording_page_fetcher(url: str) -> str:
            html = base_fetcher(url) if base_fetcher else fetch_bls_schedule_page_text(url)
            store.record_text_artifact(
                run_id=run_id,
                source_name="bls.gov:schedule",
                artifact_kind="html",
                source_identifier=url,
                content_text=html,
                calendar_assumption="NYSE trading calendar",
                timezone="America/New_York",
                license_note="BLS public release-schedule page",
                notes="BLS yearly release schedule page",
            )
            return html

    try:
        releases = fetch_bls_year_releases(
            start_year=start_year,
            end_year=end_year,
            page_fetcher=recording_page_fetcher,
        )
    except BLSScheduleFetchError as exc:
        raise EventCalendarFetchError(str(exc)) from exc

    if not releases:
        raise EventCalendarFetchError("BLS schedule fetch returned no CPI/NFP releases")

    events: list[ScheduledEvent] = []
    for release in releases:
        events.append(
            ScheduledEvent(
                date=release.date,
                release_timestamp_et=release.release_timestamp_et,
                market="US",
                type=release.type,
                importance="high",
                source=SOURCE_CPI if release.type == "CPI" else SOURCE_NFP,
            )
        )

    _validate_bls_events(events=events, start_year=start_year, end_year=end_year)
    return events


def _validate_bls_events(*, events: list[ScheduledEvent], start_year: int, end_year: int) -> None:
    grouped: dict[str, list[ScheduledEvent]] = {"CPI": [], "NFP": []}
    for event in events:
        if event.type in grouped:
            grouped[event.type].append(event)

    for event_type, typed_events in grouped.items():
        if not typed_events:
            raise EventCalendarFetchError(f"BLS schedule returned no {event_type} events")

        dates = [event.date for event in typed_events]
        if dates != sorted(dates):
            raise EventCalendarFetchError(f"BLS {event_type} events are not sorted ascending")
        if len(dates) != len(set(dates)):
            raise EventCalendarFetchError(f"BLS {event_type} events contained duplicate release dates")

        counts_by_year = Counter(value.year for value in dates)
        for year in range(start_year, end_year + 1):
            if year == end_year:
                continue
            count = counts_by_year.get(year, 0)
            expected_count = 12 - _BLS_OFFICIAL_CANCELED_RELEASE_COUNTS.get((event_type, year), 0)
            if count != expected_count:
                raise EventCalendarFetchError(
                    f"BLS {event_type} year {year} had {count} release dates; expected {expected_count} monthly releases"
                )


def _build_v2_curated_candidate_events(
    *,
    start_year: int,
    end_year: int,
    global_rate_calendar_text_fetchers: Mapping[str, Callable[[], str]] | None,
    store: AcquisitionStore | None,
    run_id: int | None,
) -> list[ScheduledEvent]:
    events: list[ScheduledEvent] = []
    for year in range(start_year, end_year + 1):
        if year % 2 == 0:
            events.append(
                ScheduledEvent(
                    date=_us_general_election_date(year),
                    release_timestamp_et=_midnight_et(_us_general_election_date(year)),
                    market="US",
                    type="election",
                    importance="high",
                    source=SOURCE_FEC,
                    window_days=(-5, 10),
                )
            )
        budget_date = dt.date(year, 9, 30)
        events.append(
            ScheduledEvent(
                date=budget_date,
                release_timestamp_et=_midnight_et(budget_date),
                market="US",
                type="budget",
                importance="medium",
                source=SOURCE_US_BUDGET,
            )
        )

    text_fetchers = global_rate_calendar_text_fetchers or {
        key: _build_url_text_fetcher(url) for key, url in _GLOBAL_RATE_URLS.items()
    }
    for source_key, fetcher in text_fetchers.items():
        text = fetcher()
        source_name = _global_rate_source_name(source_key)
        if store and run_id is not None:
            store.record_text_artifact(
                run_id=run_id,
                source_name=source_name,
                artifact_kind="html",
                source_identifier=_GLOBAL_RATE_URLS.get(source_key, source_key),
                content_text=text,
                calendar_assumption="NYSE trading calendar",
                timezone="America/New_York",
                license_note="Official central-bank public calendar page",
                notes=f"{source_key.upper()} monetary-policy calendar candidate source",
            )
        events.extend(
            event
            for event in _parse_global_rate_decision_events(source_key=source_key, text=text)
            if start_year <= event.date.year <= end_year
        )

    deduped = {(event.date, event.type, event.source): event for event in events}
    return list(deduped.values())


def _build_url_text_fetcher(url: str) -> Callable[[], str]:
    def fetch() -> str:
        request = Request(url, headers={"User-Agent": "regime-detection-event-fetch/1.0"})
        with urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")

    return fetch


def _parse_global_rate_decision_events(*, source_key: str, text: str) -> list[ScheduledEvent]:
    if source_key == "ecb":
        return _parse_ecb_decision_events(text)
    if source_key == "boj":
        return _parse_boj_decision_events(text)
    normalized = re.sub(r"<[^>]+>", " ", text)
    normalized = re.sub(r"\s+", " ", normalized)
    if source_key == "boe":
        return _parse_boe_decision_events(normalized)
    raise EventCalendarFetchError(f"Unsupported global rate calendar source: {source_key}")


def _parse_ecb_decision_events(text: str) -> list[ScheduledEvent]:
    events: list[ScheduledEvent] = []
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
        event_date = dt.date(year, month, day)
        events.append(_global_rate_event(event_date, "ECB_decision", SOURCE_ECB))
    return _dedupe_events(events)


def _parse_boe_decision_events(text: str) -> list[ScheduledEvent]:
    events: list[ScheduledEvent] = []
    current_year: int | None = None
    tokens = re.split(r"(?=(?:20\d{2})\s+(?:confirmed|provisional)\s+dates)|(?=(?:Monday|Tuesday|Wednesday|Thursday|Friday)\s+\d{1,2}\s+[A-Za-z]+)", text)
    date_pattern = re.compile(
        r"(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday)\s+)?(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+).*?(?:MPC|Monetary Policy)",
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
        month = _MONTHS.get(date_match.group("month").lower())
        if month is None:
            continue
        event_date = dt.date(current_year, month, int(date_match.group("day")))
        events.append(_global_rate_event(event_date, "BOE_decision", SOURCE_BOE))
    return _dedupe_events(events)


def _parse_boj_decision_events(text: str) -> list[ScheduledEvent]:
    events: list[ScheduledEvent] = []
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
            month = _MONTHS[date_match.group("month").lower()]
            day = int(date_match.group("end") or date_match.group("start"))
            events.append(_global_rate_event(dt.date(year, month, day), "BOJ_decision", SOURCE_BOJ))

    normalized = re.sub(r"<[^>]+>", " ", text)
    normalized = re.sub(r"\s+", " ", normalized)
    pattern = re.compile(
        r"(?P<month>Jan\.?|January|Feb\.?|February|Mar\.?|March|Apr\.?|April|May|June|July|Aug\.?|August|Sep\.?|Sept\.?|September|Oct\.?|October|Nov\.?|November|Dec\.?|December)\s+"
        r"(?P<start>\d{1,2})(?:\s*(?:-|and)\s*(?P<end>\d{1,2}))?,\s*(?P<year>20\d{2})",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(normalized):
        month = _MONTHS[match.group("month").lower()]
        day = int(match.group("end") or match.group("start"))
        event_date = dt.date(int(match.group("year")), month, day)
        events.append(_global_rate_event(event_date, "BOJ_decision", SOURCE_BOJ))
    return _dedupe_events(events)


def _global_rate_event(event_date: dt.date, event_type: str, source: str) -> ScheduledEvent:
    return ScheduledEvent(
        date=event_date,
        release_timestamp_et=_midnight_et(event_date),
        market="GLOBAL",
        type=event_type,
        importance="high",
        source=source,
    )


def _global_rate_source_name(source_key: str) -> str:
    return {"ecb": SOURCE_ECB, "boe": SOURCE_BOE, "boj": SOURCE_BOJ}.get(source_key, source_key)


def _us_general_election_date(year: int) -> dt.date:
    first_november = dt.date(year, 11, 1)
    days_until_monday = (0 - first_november.weekday()) % 7
    first_monday = first_november + dt.timedelta(days=days_until_monday)
    return first_monday + dt.timedelta(days=1)


def _midnight_et(value: dt.date) -> dt.datetime:
    return dt.datetime(value.year, value.month, value.day, 0, 0, tzinfo=US_EASTERN)


def _dedupe_events(events: list[ScheduledEvent]) -> list[ScheduledEvent]:
    return list({(event.date, event.type, event.source): event for event in events}.values())


def _render_events_yaml(events: list[ScheduledEvent]) -> str:
    lines = ["events:"]
    for event in events:
        lines.extend(
            [
                f'  - date: "{event.date.isoformat()}"',
                f'    release_timestamp_et: "{event.release_timestamp_et.isoformat()}"',
                f'    market: "{event.market}"',
                f'    type: "{event.type}"',
                f'    importance: "{event.importance}"',
                f'    source: "{event.source}"',
            ]
        )
        if event.window_days is not None:
            lines.append(f"    window_days: [{event.window_days[0]}, {event.window_days[1]}]")
    return "\n".join(lines) + "\n"


def _expand_nyse_window_for_scheduled_event(
    *,
    anchor_date: dt.date,
    lookback_trading_days: int,
    lookahead_trading_days: int,
) -> list[dt.date]:
    start_date = anchor_date - dt.timedelta(days=14)
    end_date = anchor_date + dt.timedelta(days=14)
    schedule = _NYSE.schedule(start_date.isoformat(), end_date.isoformat())
    trading_days = [index.date() for index in schedule.index]
    try:
        anchor_idx = trading_days.index(anchor_date)
    except ValueError as exc:
        raise EventCalendarFetchError(f"Scheduled event date {anchor_date.isoformat()} is not an NYSE trading day") from exc

    window_start = anchor_idx - lookback_trading_days
    window_end = anchor_idx + lookahead_trading_days
    if window_start < 0 or window_end >= len(trading_days):
        raise EventCalendarFetchError(
            f"NYSE window [{lookback_trading_days}, {lookahead_trading_days}] around {anchor_date.isoformat()} exceeded available trading-day slice"
        )
    return trading_days[window_start : window_end + 1]
