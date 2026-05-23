from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources._common import HTTP_USER_AGENT
from regime_data_fetch.event_calendar_models import (
    EventCalendarFetchError,
    GroupABuildResult,
    ScheduledEvent,
)
from regime_data_fetch.event_calendar_reporting import build_candidate_artifact_records
from regime_data_fetch.event_sources.models import (
    EventCandidate,
    PromotionDecision,
    ValidationResult,
)

LOGGER = logging.getLogger(__name__)


def build_v2_curated_candidate_events(
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
    from regime_data_fetch.event_sources.approvals import load_approval_overlay
    from regime_data_fetch.event_sources.budget_official_discovery import (
        BudgetOfficialDiscoveryGenerator,
    )
    from regime_data_fetch.event_sources.deterministic_budget import (
        DeterministicBudgetAdapter,
    )
    from regime_data_fetch.event_sources.deterministic_election import ElectionAdapter
    from regime_data_fetch.event_sources.official_boe import OfficialBOEAdapter
    from regime_data_fetch.event_sources.official_boj import OfficialBOJAdapter
    from regime_data_fetch.event_sources.official_ecb import OfficialECBAdapter
    from regime_data_fetch.event_sources.orchestrator import EventSourceOrchestrator
    from regime_data_fetch.event_sources.validators_gpr_gdelt import (
        ACLEDSignalGenerator,
        GDELTSignalGenerator,
        GPRSignalGenerator,
        HDXHAPISignalGenerator,
        UCDPSignalGenerator,
    )
    from regime_data_fetch.event_sources.validators_hf_central_bank import (
        HFCentralBankValidator,
    )
    from regime_data_fetch.event_sources.validators_tinyfish import TinyFishValidator

    events: list[ScheduledEvent] = []
    text_fetcher = group_a_text_fetcher or group_a_text_fetcher_from_legacy_map(
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
        gpr = GPRSignalGenerator()
        gdelt = GDELTSignalGenerator()
        acled = ACLEDSignalGenerator()
        ucdp = UCDPSignalGenerator()
        hdx = HDXHAPISignalGenerator()
        group_b_generators.extend(
            [
                BudgetOfficialDiscoveryGenerator(),
                gpr,
                gdelt,
                acled,
                ucdp,
                hdx,
            ]
        )
        group_b_validators.extend([gpr, gdelt, acled, ucdp, hdx, TinyFishValidator()])

    approval_overlay = load_approval_overlay(
        repo_root / "configs" / "events" / "group_b_approvals.yaml"
    )
    orchestrator = EventSourceOrchestrator(
        primary_adapters=[
            OfficialECBAdapter(
                as_of_date=as_of_date,
                text_fetcher=text_fetcher or build_url_text_fetcher_with_arg(),
            ),
            OfficialBOEAdapter(
                as_of_date=as_of_date,
                text_fetcher=text_fetcher or build_url_text_fetcher_with_arg(),
                news_api_fetcher=boe_news_fetcher,
            ),
            OfficialBOJAdapter(
                as_of_date=as_of_date,
                text_fetcher=text_fetcher or build_url_text_fetcher_with_arg(),
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
    output_paths = write_group_a_artifacts(
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


def build_url_text_fetcher_with_arg() -> Callable[[str], str]:
    def fetch(url: str) -> str:
        return build_url_text_fetcher(url)()

    return fetch


def group_a_text_fetcher_from_legacy_map(
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
        return build_url_text_fetcher(url)()

    return fetch


def write_group_a_artifacts(
    *,
    repo_root: Path,
    candidates: list[EventCandidate],
    validations: list[ValidationResult],
    decisions: list[PromotionDecision],
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
        record_group_a_output(
            store,
            run_id,
            "event_group_a_candidates",
            candidate_path,
            records.candidates,
        )
        record_group_a_output(
            store,
            run_id,
            "event_group_a_validations",
            validation_path,
            records.validations,
        )
        record_group_a_output(
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


def record_group_a_output(
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


def build_url_text_fetcher(url: str) -> Callable[[], str]:
    def fetch() -> str:
        request = Request(url, headers={"User-Agent": HTTP_USER_AGENT})
        try:
            with urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            LOGGER.error(
                "event calendar source fetch failed for %s; aborting event fetch: %s",
                url,
                reason,
            )
            raise EventCalendarFetchError(
                f"event calendar source fetch failed for {url}: {reason}"
            ) from exc

    return fetch
