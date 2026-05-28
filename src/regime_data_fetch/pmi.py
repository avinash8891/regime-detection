from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from regime_data_fetch._http import fetch_text
from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources._common import MONTHS
from regime_data_fetch.ism import release_timestamp_for

DBNOMICS_URLS = {
    "manufacturing": "https://db.nomics.world/ISM/pmi/pm?tab=table",
    "services": "https://db.nomics.world/ISM/nm-pmi/pm?tab=table",
}
TRADINGECONOMICS_URLS = {
    "manufacturing": "https://tradingeconomics.com/united-states/business-confidence",
    "services": "https://tradingeconomics.com/united-states/non-manufacturing-pmi",
}
MANUAL_PMI_SOURCE_URLS = {
    "manufacturing": "https://in.investing.com/economic-calendar/ism-manufacturing-pmi-173",
    "services": "https://www.investing.com/economic-calendar/united-states-ism-non-manufacturing-pmi-176",
}
DEFAULT_MANUAL_PMI_HISTORY_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "manual_inputs" / "pmi"
)
OPERATOR_PASTED_SOURCE_NOTE = "Operator copied Investing.com historical PMI release-history tables into repo-local TSV files."
LOGGER = logging.getLogger(__name__)

_DBNOMICS_ROW_RE = re.compile(
    r"(?P<period>\d{4}-\d{2})\s+(?P<value>-?\d+(?:\.\d+)?)", re.IGNORECASE
)
_TE_TITLE_RE = re.compile(
    r"<title>\s*United States ISM (?P<series>Manufacturing|Services) PMI\s*</title>",
    re.IGNORECASE,
)
_TE_MFG_DESC_RE = re.compile(
    r"remained [^ ]+ at (?P<value>\d+(?:\.\d+)?) points in (?P<month>[A-Za-z]+)",
    re.IGNORECASE,
)
_TE_SVC_DESC_RE = re.compile(
    r"Non Manufacturing PMI in the United States [^ ]+ to (?P<value>\d+(?:\.\d+)?) points in (?P<month>[A-Za-z]+)",
    re.IGNORECASE,
)


class PMIFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class PMIObservation:
    series_name: str
    period: str
    value: float
    release_timestamp: dt.datetime
    source: str
    source_url: str


@dataclass(frozen=True)
class PMIFetchBundle:
    source_name: str
    raw_pages: dict[str, str]
    observations: list[PMIObservation]


@dataclass(frozen=True)
class ManualPMIHistoryRow:
    series_name: str
    period: str
    release_date_local: str
    time_local: str
    actual: float
    forecast: float | None
    previous: float | None
    source: str
    source_url: str


def release_timestamp_for_period(*, series_name: str, period: str) -> dt.datetime:
    year, month = map(int, period.split("-"))
    if month == 12:
        release_year = year + 1
        release_month = 1
    else:
        release_year = year
        release_month = month + 1

    business_day_index = 1 if series_name == "manufacturing" else 3
    return release_timestamp_for(
        year=release_year, month=release_month, business_day_index=business_day_index
    )


def parse_dbnomics_html(
    html: str, *, series_name: str, source_url: str
) -> list[PMIObservation]:
    cleaned = re.sub(r"<![^>]*>", " ", html)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    observations: list[PMIObservation] = []
    for match in _DBNOMICS_ROW_RE.finditer(cleaned):
        period = match.group("period")
        value = float(match.group("value"))
        observations.append(
            PMIObservation(
                series_name=series_name,
                period=period,
                value=value,
                release_timestamp=release_timestamp_for_period(
                    series_name=series_name, period=period
                ),
                source="dbnomics",
                source_url=source_url,
            )
        )
    if not observations:
        raise PMIFetchError(
            f"DBnomics page did not contain parseable PMI rows for {series_name}"
        )
    return observations


def parse_tradingeconomics_html(
    html: str, *, series_name: str, source_url: str
) -> PMIObservation:
    _ensure_series_title(html, expected=series_name)
    pattern = _TE_MFG_DESC_RE if series_name == "manufacturing" else _TE_SVC_DESC_RE
    match = pattern.search(html)
    if not match:
        raise PMIFetchError(
            f"TradingEconomics page did not contain parseable PMI description for {series_name}"
        )

    month_name = match.group("month")
    month = MONTHS[month_name[:3].lower()]
    year = _extract_reference_year(html)
    period = f"{year:04d}-{month:02d}"
    return PMIObservation(
        series_name=series_name,
        period=period,
        value=float(match.group("value")),
        release_timestamp=release_timestamp_for_period(
            series_name=series_name, period=period
        ),
        source="tradingeconomics",
        source_url=source_url,
    )


def choose_latest_available(
    *, observations: list[PMIObservation], as_of_timestamp: dt.datetime
) -> PMIObservation:
    eligible = [obs for obs in observations if obs.release_timestamp <= as_of_timestamp]
    if not eligible:
        raise PMIFetchError(
            "No PMI observation available as of the requested timestamp"
        )
    return max(eligible, key=lambda obs: (obs.release_timestamp, obs.period))


def expected_latest_period(*, series_name: str, as_of_timestamp: dt.datetime) -> str:
    candidate = dt.date(as_of_timestamp.year, as_of_timestamp.month, 1)
    while True:
        period = candidate.strftime("%Y-%m")
        if (
            release_timestamp_for_period(series_name=series_name, period=period)
            <= as_of_timestamp
        ):
            return period

        if candidate.month == 1:
            candidate = dt.date(candidate.year - 1, 12, 1)
        else:
            candidate = dt.date(candidate.year, candidate.month - 1, 1)


def validate_latest_observations(
    *, latest_rows: list[PMIObservation], as_of_timestamp: dt.datetime, source_name: str
) -> None:
    stale: list[str] = []
    for row in latest_rows:
        expected = expected_latest_period(
            series_name=row.series_name, as_of_timestamp=as_of_timestamp
        )
        if row.period != expected:
            stale.append(f"{row.series_name} expected {expected} got {row.period}")

    if stale:
        details = "; ".join(stale)
        raise PMIFetchError(f"{source_name} returned stale PMI data: {details}")


def fetch_pmi_dbnomics(*, as_of_date: dt.date) -> PMIFetchBundle:
    del as_of_date
    rows: list[PMIObservation] = []
    raw_pages: dict[str, str] = {}
    for series_name, url in DBNOMICS_URLS.items():
        html = _http_get_text(url)
        raw_pages[series_name] = html
        rows.extend(parse_dbnomics_html(html, series_name=series_name, source_url=url))
    return PMIFetchBundle(
        source_name="dbnomics", raw_pages=raw_pages, observations=rows
    )


def fetch_pmi_tradingeconomics(*, as_of_date: dt.date) -> PMIFetchBundle:
    del as_of_date
    rows: list[PMIObservation] = []
    raw_pages: dict[str, str] = {}
    for series_name, url in TRADINGECONOMICS_URLS.items():
        html = _http_get_text(url)
        raw_pages[series_name] = html
        rows.append(
            parse_tradingeconomics_html(html, series_name=series_name, source_url=url)
        )
    return PMIFetchBundle(
        source_name="tradingeconomics", raw_pages=raw_pages, observations=rows
    )


def run_pmi_fetch(
    *,
    out_dir: Path,
    as_of_date: dt.date,
    primary_fetcher=fetch_pmi_dbnomics,
    backup_fetcher=fetch_pmi_tradingeconomics,
    acquisition_db_path: Path | None = None,
    artifact_store_root: str | Path | None = None,
    manual_history_dir: Path | None = None,
) -> Path:
    if manual_history_dir is not None:
        return run_manual_pmi_history_import(
            out_dir=out_dir,
            as_of_date=as_of_date,
            history_dir=manual_history_dir,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=artifact_store_root,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    store = (
        AcquisitionStore(acquisition_db_path, artifact_store_root=artifact_store_root)
        if acquisition_db_path
        else None
    )
    fetch_run = (
        store.start_fetch_run(
            fetch_type="pmi",
            params={
                "as_of_date": as_of_date.isoformat(),
            },
        )
        if store
        else None
    )
    as_of_timestamp = dt.datetime.combine(
        as_of_date, dt.time(23, 59, 59), tzinfo=dt.timezone.utc
    ).astimezone(release_timestamp_for(year=2026, month=4, business_day_index=1).tzinfo)
    attempts: list[dict[str, str]] = []
    bundles_by_source: dict[str, PMIFetchBundle] = {}

    chosen_rows: list[PMIObservation] | None = None
    chosen_bundle: PMIFetchBundle | None = None
    selected_source: str | None = None
    try:
        for source_name, fetcher in [
            ("dbnomics", primary_fetcher),
            ("tradingeconomics", backup_fetcher),
        ]:
            try:
                fetch_result = _normalize_fetch_result(
                    fetcher(as_of_date=as_of_date), source_name=source_name
                )
                bundles_by_source[source_name] = fetch_result
                if store and fetch_run and fetch_result.raw_pages:
                    for series_name, html in fetch_result.raw_pages.items():
                        store.record_text_artifact(
                            run_id=fetch_run.run_id,
                            source_name=f"{source_name}:pmi",
                            artifact_kind="html",
                            source_identifier=f"{source_name}:{series_name}:{as_of_date.isoformat()}",
                            content_text=html,
                            effective_date=as_of_date.isoformat(),
                            timezone="America/New_York",
                            license_note=f"Raw {source_name} PMI page fetched before normalization",
                            notes=f"Raw {source_name} PMI page for {series_name}",
                        )

                latest = [
                    choose_latest_available(
                        observations=[
                            obs
                            for obs in fetch_result.observations
                            if obs.series_name == series_name
                        ],
                        as_of_timestamp=as_of_timestamp,
                    )
                    for series_name in ("manufacturing", "services")
                ]
                validate_latest_observations(
                    latest_rows=latest,
                    as_of_timestamp=as_of_timestamp,
                    source_name=source_name,
                )
                chosen_rows = latest
                chosen_bundle = fetch_result
                selected_source = source_name
                attempts.append({"source": source_name, "status": "success"})
                break
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "PMI source %s failed: %s",
                    source_name,
                    exc,
                    exc_info=True,
                )
                attempts.append(
                    {"source": source_name, "status": "failure", "error": str(exc)}
                )

        if chosen_rows is None or selected_source is None or chosen_bundle is None:
            raise PMIFetchError(f"All PMI sources failed: {attempts}")

        latest_df = pd.DataFrame(
            [
                {
                    "series_name": row.series_name,
                    "period": row.period,
                    "value": row.value,
                    "release_timestamp": row.release_timestamp.isoformat(),
                    "source": row.source,
                    "source_url": row.source_url,
                }
                for row in chosen_rows
            ]
        )
        history_path = out_dir / "pmi" / "us_ism_pmi_history.parquet"
        history_rows = _select_history_rows(
            bundles_by_source=bundles_by_source,
            chosen_bundle=chosen_bundle,
            as_of_timestamp=as_of_timestamp,
        )
        history_rows = _merge_existing_history_rows(
            history_path=history_path,
            new_rows=[*history_rows, *chosen_rows],
            as_of_timestamp=as_of_timestamp,
        )
        history_df = pd.DataFrame(
            [
                {
                    "series_name": row.series_name,
                    "period": row.period,
                    "value": row.value,
                    "release_timestamp": row.release_timestamp.isoformat(),
                    "source": row.source,
                    "source_url": row.source_url,
                }
                for row in history_rows
            ]
        )
        pmi_dir = out_dir / "pmi"
        pmi_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = pmi_dir / "us_ism_pmi.parquet"
        latest_df.to_parquet(parquet_path, index=False)
        history_df.to_parquet(history_path, index=False)

        report = {
            "as_of_date": as_of_date.isoformat(),
            "selected_source": selected_source,
            "history_source": history_rows[0].source if history_rows else None,
            "attempts": attempts,
            "counts": {
                "rows": int(len(latest_df)),
                "history_rows": int(len(history_df)),
            },
            "paths": {
                "pmi_parquet": str(parquet_path),
                "pmi_history_parquet": str(history_path),
                "acquisition_db": (
                    str(acquisition_db_path) if acquisition_db_path else None
                ),
            },
        }
        report_path = out_dir / "pmi_fetch_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if store and fetch_run:
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="pmi_parquet",
                path=parquet_path,
                row_count=len(latest_df),
                min_date=min(latest_df["period"]) if not latest_df.empty else None,
                max_date=max(latest_df["period"]) if not latest_df.empty else None,
                notes="Normalized PMI parquet output",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="pmi_history_parquet",
                path=history_path,
                row_count=len(history_df),
                min_date=min(history_df["period"]) if not history_df.empty else None,
                max_date=max(history_df["period"]) if not history_df.empty else None,
                notes="Normalized PMI history parquet output",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="pmi_report",
                path=report_path,
                row_count=len(latest_df),
                min_date=min(latest_df["period"]) if not latest_df.empty else None,
                max_date=max(latest_df["period"]) if not latest_df.empty else None,
                notes="PMI fetch report",
            )
            store.finish_fetch_run(run_id=fetch_run.run_id, status="ok")
        return report_path
    except Exception as exc:
        if store and fetch_run:
            store.finish_fetch_run(
                run_id=fetch_run.run_id, status="failed", notes=str(exc)
            )
        raise


def run_manual_pmi_history_import(
    *,
    out_dir: Path,
    as_of_date: dt.date,
    history_dir: Path,
    acquisition_db_path: Path | None = None,
    artifact_store_root: str | Path | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    store = (
        AcquisitionStore(acquisition_db_path, artifact_store_root=artifact_store_root)
        if acquisition_db_path
        else None
    )
    fetch_run = (
        store.start_fetch_run(
            fetch_type="pmi",
            params={
                "as_of_date": as_of_date.isoformat(),
                "history_dir": str(history_dir),
                "source_mode": "manual_investing_history",
                "source_note": OPERATOR_PASTED_SOURCE_NOTE,
                "source_urls": MANUAL_PMI_SOURCE_URLS,
            },
        )
        if store
        else None
    )
    try:
        rows = load_manual_pmi_history(history_dir=history_dir)
        cutoff = dt.datetime.combine(
            as_of_date, dt.time(23, 59, 59), tzinfo=dt.timezone.utc
        ).astimezone(ZoneInfo("America/New_York"))
        history_rows = [row for row in rows if row.release_timestamp <= cutoff]
        latest_rows = [
            choose_latest_available(
                observations=[
                    row for row in history_rows if row.series_name == series_name
                ],
                as_of_timestamp=cutoff,
            )
            for series_name in ("manufacturing", "services")
        ]

        if store and fetch_run:
            for file_path, series_name in _manual_pmi_history_files(
                history_dir
            ).items():
                store.record_text_artifact(
                    run_id=fetch_run.run_id,
                    source_name="investing:pmi",
                    artifact_kind="tsv",
                    source_identifier=f"investing:{series_name}:{as_of_date.isoformat()}",
                    content_text=file_path.read_text(),
                    effective_date=as_of_date.isoformat(),
                    start_date=min(
                        row.period
                        for row in history_rows
                        if row.series_name == series_name
                    ),
                    end_date=max(
                        row.period
                        for row in history_rows
                        if row.series_name == series_name
                    ),
                    timezone="America/New_York",
                    license_note=(
                        "Operator-pasted Investing.com PMI release-history table "
                        "supplied for backtest-aligned periods"
                    ),
                    notes=f"{OPERATOR_PASTED_SOURCE_NOTE} Series: {series_name}",
                )

        latest_df = pd.DataFrame(
            [
                {
                    "series_name": row.series_name,
                    "period": row.period,
                    "value": row.value,
                    "release_timestamp": row.release_timestamp.isoformat(),
                    "source": row.source,
                    "source_url": row.source_url,
                }
                for row in latest_rows
            ]
        )
        history_df = pd.DataFrame(
            [
                {
                    "series_name": row.series_name,
                    "period": row.period,
                    "value": row.value,
                    "release_timestamp": row.release_timestamp.isoformat(),
                    "source": row.source,
                    "source_url": row.source_url,
                }
                for row in history_rows
            ]
        )

        pmi_dir = out_dir / "pmi"
        pmi_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = pmi_dir / "us_ism_pmi.parquet"
        latest_df.to_parquet(parquet_path, index=False)
        history_path = pmi_dir / "us_ism_pmi_history.parquet"
        history_df.to_parquet(history_path, index=False)

        report = {
            "as_of_date": as_of_date.isoformat(),
            "selected_source": "manual_investing_history",
            "history_source": "manual_investing_history",
            "source_note": OPERATOR_PASTED_SOURCE_NOTE,
            "source_urls": MANUAL_PMI_SOURCE_URLS,
            "attempts": [{"source": "manual_investing_history", "status": "success"}],
            "counts": {
                "rows": int(len(latest_df)),
                "history_rows": int(len(history_df)),
            },
            "paths": {
                "pmi_parquet": str(parquet_path),
                "pmi_history_parquet": str(history_path),
                "manual_pmi_manufacturing_tsv": {
                    "path": str(history_dir / "ism_manufacturing_pmi.tsv"),
                    "local_path": "data/manual_inputs/pmi/ism_manufacturing_pmi.tsv",
                },
                "manual_pmi_services_tsv": {
                    "path": str(history_dir / "ism_services_pmi.tsv"),
                    "local_path": "data/manual_inputs/pmi/ism_services_pmi.tsv",
                },
                "acquisition_db": (
                    str(acquisition_db_path) if acquisition_db_path else None
                ),
            },
        }
        report_path = out_dir / "pmi_fetch_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if store and fetch_run:
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="pmi_parquet",
                path=parquet_path,
                row_count=len(latest_df),
                min_date=min(latest_df["period"]) if not latest_df.empty else None,
                max_date=max(latest_df["period"]) if not latest_df.empty else None,
                notes="Normalized PMI parquet output",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="pmi_history_parquet",
                path=history_path,
                row_count=len(history_df),
                min_date=min(history_df["period"]) if not history_df.empty else None,
                max_date=max(history_df["period"]) if not history_df.empty else None,
                notes="Normalized PMI history parquet output",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="pmi_report",
                path=report_path,
                row_count=len(latest_df),
                min_date=min(latest_df["period"]) if not latest_df.empty else None,
                max_date=max(latest_df["period"]) if not latest_df.empty else None,
                notes="PMI fetch report",
            )
            store.finish_fetch_run(run_id=fetch_run.run_id, status="ok")
        return report_path
    except Exception as exc:
        if store and fetch_run:
            store.finish_fetch_run(
                run_id=fetch_run.run_id, status="failed", notes=str(exc)
            )
        raise


def _normalize_fetch_result(
    fetch_result: object, *, source_name: str
) -> PMIFetchBundle:
    if isinstance(fetch_result, PMIFetchBundle):
        return fetch_result
    if isinstance(fetch_result, list):
        return PMIFetchBundle(
            source_name=source_name, raw_pages={}, observations=fetch_result
        )
    raise TypeError(
        f"Unexpected PMI fetch result for {source_name}: {type(fetch_result).__name__}"
    )


def _select_history_rows(
    *,
    bundles_by_source: dict[str, PMIFetchBundle],
    chosen_bundle: PMIFetchBundle,
    as_of_timestamp: dt.datetime,
) -> list[PMIObservation]:
    candidate_bundles: list[PMIFetchBundle] = []
    if "dbnomics" in bundles_by_source:
        candidate_bundles.append(bundles_by_source["dbnomics"])
    if chosen_bundle not in candidate_bundles:
        candidate_bundles.append(chosen_bundle)

    best_rows: list[PMIObservation] = []
    best_key: tuple[int, int] = (-1, -1)
    for bundle in candidate_bundles:
        rows = [
            row
            for row in bundle.observations
            if row.release_timestamp <= as_of_timestamp
        ]
        rows = _dedupe_history_rows(rows)
        key = (len(rows), len({row.period for row in rows}))
        if key > best_key:
            best_rows = rows
            best_key = key
    return best_rows


def _merge_existing_history_rows(
    *,
    history_path: Path,
    new_rows: list[PMIObservation],
    as_of_timestamp: dt.datetime,
) -> list[PMIObservation]:
    if not history_path.exists():
        return _dedupe_history_rows(
            [row for row in new_rows if row.release_timestamp <= as_of_timestamp]
        )

    existing_df = pd.read_parquet(history_path)
    required = {
        "series_name",
        "period",
        "value",
        "release_timestamp",
        "source",
        "source_url",
    }
    if not required.issubset(existing_df.columns):
        missing = sorted(required - set(existing_df.columns))
        raise PMIFetchError(f"existing PMI history missing columns: {missing}")

    existing_rows = []
    for row in existing_df.itertuples(index=False):
        release_timestamp = pd.Timestamp(row.release_timestamp)
        if release_timestamp.tzinfo is None:
            release_timestamp = release_timestamp.tz_localize(dt.UTC)
        else:
            release_timestamp = release_timestamp.tz_convert(dt.UTC)
        existing_rows.append(
            PMIObservation(
                series_name=str(row.series_name),
                period=str(row.period),
                value=float(row.value),
                release_timestamp=release_timestamp.to_pydatetime(),
                source=str(row.source),
                source_url=str(row.source_url),
            )
        )

    current_rows = [row for row in new_rows if row.release_timestamp <= as_of_timestamp]
    return _dedupe_history_rows([*existing_rows, *current_rows])


def _dedupe_history_rows(rows: list[PMIObservation]) -> list[PMIObservation]:
    by_key: dict[tuple[str, str], PMIObservation] = {}
    for row in sorted(
        rows, key=lambda item: (item.series_name, item.period, item.release_timestamp)
    ):
        by_key[(row.series_name, row.period)] = row
    return sorted(by_key.values(), key=lambda item: (item.period, item.series_name))


def _ensure_series_title(html: str, *, expected: str) -> None:
    match = _TE_TITLE_RE.search(html)
    if not match:
        raise PMIFetchError("TradingEconomics page missing expected title")
    got = match.group("series").lower()
    if expected not in got:
        raise PMIFetchError(
            f"TradingEconomics title mismatch: expected {expected}, got {got}"
        )


def _extract_reference_year(html: str) -> int:
    match = re.search(
        r"Reference\s+[A-Za-z]{3,9}\s+(?P<year>\d{4})", html, flags=re.IGNORECASE
    )
    if match:
        return int(match.group("year"))

    match = re.search(
        r'content="[^"]*?\bof\s+(?P<year>\d{4})\b[^"]*"', html, flags=re.IGNORECASE
    )
    if match:
        return int(match.group("year"))

    temporal = re.search(
        r'temporalCoverage"\s*:\s*"(?P<start>\d{4}-\d{2}-\d{2})/(?P<end>\d{4})-(?P<month>\d{2})-\d{2}"',
        html,
        flags=re.IGNORECASE,
    )
    if temporal:
        return int(temporal.group("end"))

    raise PMIFetchError("TradingEconomics page missing reference year")


def _http_get_text(url: str) -> str:
    return fetch_text(url, timeout=30)


def load_manual_pmi_history(*, history_dir: Path) -> list[PMIObservation]:
    rows: list[PMIObservation] = []
    for file_path, series_name in _manual_pmi_history_files(history_dir).items():
        with file_path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for record in reader:
                manual_row = ManualPMIHistoryRow(
                    series_name=series_name,
                    period=record["period"].strip(),
                    release_date_local=record["release_date_local"].strip(),
                    time_local=record["time_local"].strip(),
                    actual=float(record["actual"]),
                    forecast=_parse_optional_float(record.get("forecast")),
                    previous=_parse_optional_float(record.get("previous")),
                    source="investing_manual",
                    source_url=MANUAL_PMI_SOURCE_URLS[series_name],
                )
                rows.append(_manual_history_row_to_observation(manual_row))
    return sorted(rows, key=lambda item: (item.period, item.series_name))


def _manual_pmi_history_files(history_dir: Path) -> dict[Path, str]:
    mapping = {
        history_dir / "ism_manufacturing_pmi.tsv": "manufacturing",
        history_dir / "ism_services_pmi.tsv": "services",
    }
    missing = [str(path) for path in mapping if not path.exists()]
    if missing:
        raise PMIFetchError(f"Missing manual PMI history files: {missing}")
    return mapping


def _manual_history_row_to_observation(row: ManualPMIHistoryRow) -> PMIObservation:
    release_date = dt.datetime.strptime(row.release_date_local, "%d-%m-%Y").date()
    release_timestamp = dt.datetime.combine(
        release_date, dt.time(10, 0), tzinfo=ZoneInfo("America/New_York")
    )
    return PMIObservation(
        series_name=row.series_name,
        period=row.period,
        value=row.actual,
        release_timestamp=release_timestamp,
        source=row.source,
        source_url=row.source_url,
    )


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return float(text)
