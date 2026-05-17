from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
import yaml

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.bls_schedule import (
    BLSScheduleFetchError,
    fetch_bls_schedule_page_text,
    fetch_bls_year_releases,
)
from regime_data_fetch.earnings_season_calendar import is_in_earnings_season
from regime_data_fetch import event_calendar_global_rates as _global_rates
from regime_data_fetch.event_calendar_reporting import (
    build_candidate_artifact_records,
    build_group_a_report,
    build_group_b_report,
    report_path as format_report_path,
)
from regime_data_fetch.event_calendar_global_rate_parsers import (
    global_rate_event as _global_rate_event,
    global_rate_source_name as _global_rate_source_name,
    midnight_et as _midnight_et,
    parse_boe_decision_events as _parse_boe_decision_events,
    parse_boj_decision_events as _parse_boj_decision_events,
    parse_ecb_decision_events as _parse_ecb_decision_events,
    parse_global_rate_decision_events as _parse_global_rate_decision_events,
)
from regime_data_fetch.event_calendar_models import (
    EventCalendarFetchError,
    EventImportance,
    EventLabelResolution,
    EventMarket,
    EventSource,
    EventType,
    GroupABuildResult,
    ScheduledEvent,
    US_EASTERN,
)
from regime_data_fetch.event_calendar_rendering import (
    render_events_yaml as _render_events_yaml,
)
from regime_data_fetch.event_calendar_windows import (
    expand_nyse_window_for_scheduled_event as _expand_nyse_window_for_scheduled_event,
    us_general_election_date as _us_general_election_date,
)
from regime_data_fetch.expiry_calendar import (
    expand_trading_day_window,
    compute_monthly_options_expiry_anchor,
)
from regime_data_fetch.fomc_minutes import (
    FOMCMinutesFetchError,
    fetch_fomc_historical_year_index,
    fetch_fomc_historical_year_page,
    fetch_fomc_minutes_listing,
    fetch_release_timestamp,
    parse_fomc_historical_year_index,
    parse_fomc_minutes_historical_listing,
    parse_fomc_minutes_listing,
)

SOURCE_FOMC = "federalreserve.gov:fomccalendars"
SOURCE_CPI = "bls.gov:schedule:consumer-price-index"
SOURCE_NFP = "bls.gov:schedule:employment-situation"
SOURCE_ECB = _global_rates.SOURCE_ECB
SOURCE_BOE = _global_rates.SOURCE_BOE
SOURCE_BOJ = _global_rates.SOURCE_BOJ
SOURCE_FEC = "fec.gov:election-dates"
SOURCE_US_BUDGET = "usa.gov:federal-budget-process"
_FOMC_MINUTES_LINK_RE = re.compile(
    r"/monetarypolicy/fomcminutes(?P<meeting_end>\d{8})\.htm", flags=re.IGNORECASE
)
_GLOBAL_RATE_URLS = _global_rates.GLOBAL_RATE_URLS
_BLS_OFFICIAL_CANCELED_RELEASE_COUNTS = {
    # BLS 2025 lapse page: October 2025 CPI and Employment Situation
    # news releases were canceled, so release-date year 2025 has 11 rows.
    ("CPI", 2025): 1,
    ("NFP", 2025): 1,
}
ROUTINE_LAYER_EVENT_TYPES = (
    "BOE_decision",
    "BOJ_decision",
    "CPI",
    "ECB_decision",
    "FOMC",
    "NFP",
    "budget",
    "election",
)
APPROVAL_GATED_EVENT_TYPES = ("geopolitical_event",)
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

__all__ = [
    "EventImportance",
    "EventMarket",
    "EventSource",
    "EventType",
    "_expand_nyse_window_for_scheduled_event",
    "_global_rate_event",
    "_global_rate_source_name",
    "_midnight_et",
    "_parse_boe_decision_events",
    "_parse_boj_decision_events",
    "_parse_ecb_decision_events",
    "_parse_global_rate_decision_events",
    "_render_events_yaml",
    "_us_general_election_date",
]


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
        raise EventCalendarFetchError(
            f"Event calendar YAML at {path} did not contain an events list"
        )

    events: list[ScheduledEvent] = []
    for idx, entry in enumerate(payload["events"], start=1):
        if not isinstance(entry, dict):
            raise EventCalendarFetchError(
                f"Event calendar YAML entry {idx} was not a mapping"
            )
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
            approved_label = (
                str(entry["approved_label"])
                if entry.get("approved_label") is not None
                else None
            )
        except KeyError as exc:
            raise EventCalendarFetchError(
                f"Event calendar YAML entry {idx} missing field {exc.args[0]!r}"
            ) from exc
        except ValueError as exc:
            raise EventCalendarFetchError(
                f"Event calendar YAML entry {idx} contained an invalid date/timestamp"
            ) from exc

        events.append(
            ScheduledEvent(
                date=event_date,
                release_timestamp_et=release_timestamp,
                market=market,
                type=event_type,
                importance=importance,
                source=source,
                window_days=window_days,
                approved_label=approved_label,
            )
        )
    return events


def resolve_event_label(
    *,
    as_of_date: dt.date,
    scheduled_events: list[ScheduledEvent],
) -> EventLabelResolution:
    matching_events: set[str] = set()
    matching_events.update(
        _matching_scheduled_event_labels(
            as_of_date=as_of_date, scheduled_events=scheduled_events
        )
    )

    expiry_anchor = compute_monthly_options_expiry_anchor(
        year=as_of_date.year, month=as_of_date.month
    )
    expiry_window = set(
        expand_trading_day_window(
            anchor_date=expiry_anchor, lookback_trading_days=2, lookahead_trading_days=0
        )
    )
    if as_of_date in expiry_window:
        matching_events.add("expiry_week")

    if is_in_earnings_season(as_of_date=as_of_date):
        matching_events.add("earnings_season")

    ordered_matches = [label for label in EVENT_PRECEDENCE if label in matching_events]
    if not ordered_matches:
        return EventLabelResolution(
            all_matching_events=[], selected_via_precedence="normal_calendar"
        )
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
    artifact_store_root: str | Path | None = None,
    bls_start_year: int = 2000,
    bls_end_year: int | None = None,
    include_v2_curated_candidates: bool = False,
    global_rate_calendar_text_fetchers: Mapping[str, Callable[[], str]] | None = None,
    group_a_text_fetcher: Callable[[str], str] | None = None,
    group_a_boe_news_fetcher: Callable[[int], str] | None = None,
    group_a_hf_parquet_fetcher: Callable[[], bytes] | None = None,
    as_of_date: dt.date | None = None,
) -> Path:
    del fred_api_key
    if as_of_date is None and (bls_end_year is None or include_v2_curated_candidates):
        raise ValueError(
            "run_us_event_calendar_fetch requires as_of_date for deterministic replay"
        )
    effective_end_year = bls_end_year if bls_end_year is not None else as_of_date.year

    store = (
        AcquisitionStore(acquisition_db_path, artifact_store_root=artifact_store_root)
        if acquisition_db_path
        else None
    )
    fetch_run = (
        store.start_fetch_run(
            fetch_type="events",
            params={
                "bls_start_year": bls_start_year,
                "bls_end_year": effective_end_year,
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
                end_year=effective_end_year,
                store=store,
                run_id=fetch_run.run_id if fetch_run else None,
            )
        )
        group_a_result: GroupABuildResult | None = None
        if include_v2_curated_candidates:
            group_a_result = _build_v2_curated_candidate_events(
                repo_root=repo_root,
                start_year=bls_start_year,
                end_year=effective_end_year,
                as_of_date=as_of_date,
                global_rate_calendar_text_fetchers=global_rate_calendar_text_fetchers,
                group_a_text_fetcher=group_a_text_fetcher,
                group_a_boe_news_fetcher=group_a_boe_news_fetcher,
                group_a_hf_parquet_fetcher=group_a_hf_parquet_fetcher,
                store=store,
                run_id=fetch_run.run_id if fetch_run else None,
            )
            events.extend(group_a_result.scheduled_events)

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
            "coverage": _build_event_type_coverage(counts),
            "paths": {
                "event_calendar_yaml": format_report_path(
                    yaml_path, repo_root=repo_root
                ),
                "acquisition_db": format_report_path(
                    acquisition_db_path, repo_root=repo_root
                )
                if acquisition_db_path
                else None,
            },
        }
        if group_a_result is not None:
            report["group_a"] = build_group_a_report(
                candidates=group_a_result.candidates,
                decisions=group_a_result.decisions,
                output_paths=group_a_result.output_paths,
                repo_root=repo_root,
            )
            report["group_b"] = build_group_b_report(
                candidates=group_a_result.candidates,
                decisions=group_a_result.decisions,
                approval_overlay=group_a_result.approval_overlay,
            )
        report_path = repo_root / "event_calendar_fetch_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if store and fetch_run:
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="event_calendar_yaml",
                path=yaml_path,
                row_count=len(events),
                min_date=min(event.date for event in events).isoformat()
                if events
                else None,
                max_date=max(event.date for event in events).isoformat()
                if events
                else None,
                notes="Generated scheduled US event calendar",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="event_calendar_report",
                path=report_path,
                row_count=len(events),
                min_date=min(event.date for event in events).isoformat()
                if events
                else None,
                max_date=max(event.date for event in events).isoformat()
                if events
                else None,
                notes="Event calendar fetch report",
            )
            store.finish_fetch_run(run_id=fetch_run.run_id, status="ok")
        return report_path
    except (
        BLSScheduleFetchError,
        EventCalendarFetchError,
        FOMCMinutesFetchError,
        OSError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        if store and fetch_run:
            store.finish_fetch_run(
                run_id=fetch_run.run_id, status="failed", notes=str(exc)
            )
        raise


def _build_event_type_coverage(counts: Counter[str]) -> dict[str, object]:
    return {
        "routine_expected_types": list(ROUTINE_LAYER_EVENT_TYPES),
        "routine_missing_types": [
            event_type
            for event_type in ROUTINE_LAYER_EVENT_TYPES
            if counts[event_type] == 0
        ],
        "approval_gated_types": list(APPROVAL_GATED_EVENT_TYPES),
        "approval_gated_by_type": {
            event_type: counts[event_type] for event_type in APPROVAL_GATED_EVENT_TYPES
        },
    }


def _matching_scheduled_event_labels(
    *, as_of_date: dt.date, scheduled_events: list[ScheduledEvent]
) -> set[str]:
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
                        release_timestamp_et=fetch_release_timestamp(
                            entry.release_date
                        ),
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
            html = (
                base_fetcher(url) if base_fetcher else fetch_bls_schedule_page_text(url)
            )
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


def _validate_bls_events(
    *, events: list[ScheduledEvent], start_year: int, end_year: int
) -> None:
    grouped: dict[str, list[ScheduledEvent]] = {"CPI": [], "NFP": []}
    for event in events:
        if event.type in grouped:
            grouped[event.type].append(event)

    for event_type, typed_events in grouped.items():
        if not typed_events:
            raise EventCalendarFetchError(
                f"BLS schedule returned no {event_type} events"
            )

        dates = [event.date for event in typed_events]
        if dates != sorted(dates):
            raise EventCalendarFetchError(
                f"BLS {event_type} events are not sorted ascending"
            )
        if len(dates) != len(set(dates)):
            raise EventCalendarFetchError(
                f"BLS {event_type} events contained duplicate release dates"
            )

        counts_by_year = Counter(value.year for value in dates)
        for year in range(start_year, end_year + 1):
            if year == end_year:
                continue
            count = counts_by_year.get(year, 0)
            expected_count = 12 - _BLS_OFFICIAL_CANCELED_RELEASE_COUNTS.get(
                (event_type, year), 0
            )
            if count != expected_count:
                raise EventCalendarFetchError(
                    f"BLS {event_type} year {year} had {count} release dates; expected {expected_count} monthly releases"
                )


def _build_v2_curated_candidate_events(
    *,
    repo_root: Path,
    start_year: int,
    end_year: int,
    as_of_date: dt.date,
    global_rate_calendar_text_fetchers: Mapping[str, Callable[[], str]] | None,
    group_a_text_fetcher: Callable[[str], str] | None,
    group_a_boe_news_fetcher: Callable[[int], str] | None,
    group_a_hf_parquet_fetcher: Callable[[], bytes] | None,
    store: AcquisitionStore | None,
    run_id: int | None,
) -> GroupABuildResult:
    from regime_data_fetch.event_sources.deterministic_election import ElectionAdapter
    from regime_data_fetch.event_sources.approvals import load_approval_overlay
    from regime_data_fetch.event_sources.budget_official_discovery import (
        BudgetOfficialDiscoveryGenerator,
    )
    from regime_data_fetch.event_sources.deterministic_budget import (
        DeterministicBudgetAdapter,
    )
    from regime_data_fetch.event_sources.official_boe import OfficialBOEAdapter
    from regime_data_fetch.event_sources.official_boj import OfficialBOJAdapter
    from regime_data_fetch.event_sources.official_ecb import OfficialECBAdapter
    from regime_data_fetch.event_sources.orchestrator import EventSourceOrchestrator
    from regime_data_fetch.event_sources.validators_hf_central_bank import (
        HFCentralBankValidator,
    )
    from regime_data_fetch.event_sources.validators_gpr_gdelt import (
        GPRGDELTSignalGenerator,
    )
    from regime_data_fetch.event_sources.validators_tinyfish import TinyFishValidator

    events: list[ScheduledEvent] = []
    text_fetcher = group_a_text_fetcher or _group_a_text_fetcher_from_legacy_map(
        global_rate_calendar_text_fetchers
    )
    hf_parquet_fetcher = group_a_hf_parquet_fetcher
    if hf_parquet_fetcher is None and global_rate_calendar_text_fetchers is not None:

        def hf_parquet_fetcher() -> bytes:
            raise TimeoutError(
                "hf central-bank parquet fetcher not provided for legacy replay"
            )

    boe_news_fetcher = group_a_boe_news_fetcher
    if boe_news_fetcher is None and global_rate_calendar_text_fetchers is not None:

        def boe_news_fetcher(page: int) -> str:
            return '{"Results": ""}'

    live_group_b_sources = (
        global_rate_calendar_text_fetchers is None and group_a_text_fetcher is None
    )
    group_b_generators = []
    group_b_validators = []
    if live_group_b_sources:
        gpr_gdelt = GPRGDELTSignalGenerator()
        group_b_generators.extend([BudgetOfficialDiscoveryGenerator(), gpr_gdelt])
        group_b_validators.extend([gpr_gdelt, TinyFishValidator()])

    approval_overlay = load_approval_overlay(
        repo_root / "configs" / "events" / "group_b_approvals.yaml"
    )
    orchestrator = EventSourceOrchestrator(
        primary_adapters=[
            OfficialECBAdapter(
                as_of_date=as_of_date,
                text_fetcher=text_fetcher or _build_url_text_fetcher_with_arg(),
            ),
            OfficialBOEAdapter(
                as_of_date=as_of_date,
                text_fetcher=text_fetcher or _build_url_text_fetcher_with_arg(),
                news_api_fetcher=boe_news_fetcher,
            ),
            OfficialBOJAdapter(
                as_of_date=as_of_date,
                text_fetcher=text_fetcher or _build_url_text_fetcher_with_arg(),
            ),
            ElectionAdapter(as_of_date=as_of_date),
            DeterministicBudgetAdapter(as_of_date=as_of_date),
        ],
        candidate_generators=group_b_generators,
        validators=[
            HFCentralBankValidator(parquet_fetcher=hf_parquet_fetcher),
            *group_b_validators,
        ],
        approval_overlay=approval_overlay,
    )
    candidates, validations, decisions, promoted_events = orchestrator.run(
        start_year=start_year,
        end_year=end_year,
        store=store,
        run_id=run_id,
    )
    events.extend(promoted_events)

    deduped = {(event.date, event.type, event.source): event for event in events}
    output_paths = _write_group_a_artifacts(
        repo_root=repo_root,
        candidates=candidates,
        validations=validations,
        decisions=decisions,
        store=store,
        run_id=run_id,
    )
    return GroupABuildResult(
        scheduled_events=list(deduped.values()),
        candidates=candidates,
        validations=validations,
        decisions=decisions,
        output_paths=output_paths,
        approval_overlay=approval_overlay,
    )


def _build_url_text_fetcher_with_arg() -> Callable[[str], str]:
    def fetch(url: str) -> str:
        return _build_url_text_fetcher(url)()

    return fetch


def _group_a_text_fetcher_from_legacy_map(
    global_rate_calendar_text_fetchers: Mapping[str, Callable[[], str]] | None,
) -> Callable[[str], str] | None:
    if global_rate_calendar_text_fetchers is None:
        return None

    def fetch(url: str) -> str:
        if "ecb.europa.eu/press/govcdec/mopo/html/index.en.html" in url:
            return ""
        if "ecb.europa.eu/press/govcdec/mopo/" in url:
            return ""
        if (
            "ecb.europa.eu/press/calendars/mgcgc/html/index.en.html" in url
            and "ecb" in global_rate_calendar_text_fetchers
        ):
            return global_rate_calendar_text_fetchers["ecb"]()
        if (
            "bankofengland.co.uk/monetary-policy/upcoming-mpc-dates" in url
            and "boe" in global_rate_calendar_text_fetchers
        ):
            return global_rate_calendar_text_fetchers["boe"]()
        if "bankofengland.co.uk/sitemap/news" in url:
            return ""
        if (
            "boj.or.jp/en/mopo/mpmsche_minu/index.htm" in url
            and "boj" in global_rate_calendar_text_fetchers
        ):
            return global_rate_calendar_text_fetchers["boj"]()
        if "boj.or.jp/en/mopo/mpmsche_minu/past.htm" in url:
            return ""
        return _build_url_text_fetcher(url)()

    return fetch


def _write_group_a_artifacts(
    *,
    repo_root: Path,
    candidates: list[object],
    validations: list[object],
    decisions: list[object],
    store: AcquisitionStore | None,
    run_id: int | None,
) -> dict[str, Path]:
    output_dir = repo_root / "data" / "raw" / "event_calendar" / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "event_candidates.parquet"
    validation_path = output_dir / "event_validations.parquet"
    quarantine_path = output_dir / "quarantine.parquet"

    records = build_candidate_artifact_records(
        candidates=candidates, validations=validations, decisions=decisions
    )

    candidate_df = pd.DataFrame(records.candidates)
    pd.DataFrame(records.validations).to_parquet(validation_path, index=False)
    candidate_df.to_parquet(candidate_path, index=False)
    pd.DataFrame(records.quarantine, columns=candidate_df.columns).to_parquet(
        quarantine_path, index=False
    )

    if store and run_id is not None:
        _record_group_a_output(
            store,
            run_id,
            "event_group_a_candidates",
            candidate_path,
            records.candidates,
        )
        _record_group_a_output(
            store,
            run_id,
            "event_group_a_validations",
            validation_path,
            records.validations,
        )
        _record_group_a_output(
            store,
            run_id,
            "event_group_a_quarantine",
            quarantine_path,
            records.quarantine,
        )

    return {
        "candidates": candidate_path,
        "validations": validation_path,
        "quarantine": quarantine_path,
    }


def _record_group_a_output(
    store: AcquisitionStore,
    run_id: int,
    output_kind: str,
    path: Path,
    records: list[dict[str, object | None]],
) -> None:
    dates = [str(record["date"]) for record in records if record.get("date")]
    store.record_output(
        run_id=run_id,
        output_kind=output_kind,
        path=path,
        row_count=len(records),
        min_date=min(dates) if dates else None,
        max_date=max(dates) if dates else None,
        notes="Group A event-source derived artifact",
    )


def _build_url_text_fetcher(url: str) -> Callable[[], str]:
    def fetch() -> str:
        request = Request(
            url, headers={"User-Agent": "regime-detection-event-fetch/1.0"}
        )
        try:
            with urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
        except URLError:
            return ""

    return fetch
