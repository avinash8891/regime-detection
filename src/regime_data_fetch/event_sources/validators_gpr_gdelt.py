from __future__ import annotations

import datetime as dt
import io
import logging
import zipfile
from collections.abc import Callable

import pandas as pd

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources.gpr_gdelt_conflict_parsers import _parse_positive_int
from regime_data_fetch.event_sources.gpr_gdelt_fetchers import (
    ACLED_READ_URL,
    ACLED_SOURCE_ID,
    ConflictFetcher,
    GDELT_DAILY_EXPORT_URL_TEMPLATE,
    GPR_DAILY_URL,
    HDX_HAPI_CONFLICT_EVENTS_URL,
    HDX_HAPI_SOURCE_ID,
    SourceFetchStatus,
    UCDP_GED_CANDIDATE_URL,
    UCDP_SOURCE_ID,
    fetch_acled_events as _fetch_acled_events,
    fetch_gdelt_daily_export as _fetch_gdelt_daily_export,
    fetch_gpr_daily as _fetch_gpr_daily,
    fetch_hdx_hapi_conflict_events as _fetch_hdx_hapi_conflict_events,
    fetch_optional_conflict_rows as _fetch_optional_conflict_rows,
    fetch_ucdp_events as _fetch_ucdp_events,
    is_empty_payload as _is_empty_payload,
    record_payload as _record_payload,
)
from regime_data_fetch.event_sources.gpr_gdelt_conflict_parsers import (
    parse_acled_events,
    parse_hdx_hapi_conflict_events,
    parse_ucdp_events,
)
from regime_data_fetch.event_sources.models import EventCandidate, ValidationResult

LOGGER = logging.getLogger(__name__)
GPR_SOURCE_ID = "gpr:caldara-iacoviello"
GDELT_SOURCE_ID = "gdelt:events-v2"
GDELT_EVENT_ROOT_CODES = {"14", "18", "19", "20"}
GDELT_SQLDATE_IDX = 1
GDELT_EVENT_ROOT_CODE_IDX = 28
GDELT_QUAD_CLASS_IDX = 29
GDELT_NUM_MENTIONS_IDX = 31
GDELT_SOURCE_URL_IDX = 56


class GPRGDELTSignalGenerator:
    source_id = "gpr-gdelt:geopolitical-signals"
    validator_id = "gpr-gdelt:cross-source"

    def __init__(
        self,
        *,
        gpr_fetcher: Callable[[], str | bytes] | None = None,
        gdelt_fetcher: Callable[[], str | bytes] | None = None,
        gdelt_daily_fetcher: Callable[[dt.date], bytes] | None = None,
        acled_fetcher: ConflictFetcher | None = None,
        ucdp_fetcher: ConflictFetcher | None = None,
        hdx_hapi_fetcher: ConflictFetcher | None = None,
        min_history_days: int = 252,
        stddev_threshold: float = 3.0,
        merge_window_days: int = 2,
    ) -> None:
        self.gpr_fetcher = gpr_fetcher or _fetch_gpr_daily
        self.gdelt_fetcher = gdelt_fetcher
        self.gdelt_daily_fetcher = gdelt_daily_fetcher or _fetch_gdelt_daily_export
        self.acled_fetcher = acled_fetcher or _fetch_acled_events
        self.ucdp_fetcher = ucdp_fetcher or _fetch_ucdp_events
        self.hdx_hapi_fetcher = hdx_hapi_fetcher or _fetch_hdx_hapi_conflict_events
        self.min_history_days = min_history_days
        self.stddev_threshold = stddev_threshold
        self.merge_window_days = merge_window_days
        self._last_source_dates: dict[str, set[dt.date]] = {}
        self.last_source_statuses: dict[str, SourceFetchStatus] = {}
        self.last_run_status = "not_run"

    def generate(
        self,
        *,
        start_year: int,
        end_year: int,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[EventCandidate]:
        self.last_source_statuses = {}
        self.last_run_status = "ok"
        candidates: list[EventCandidate] = []
        gpr_spikes = [
            row
            for row in self._fetch_gpr_spikes(store=store, run_id=run_id)
            if start_year <= row["date"].year <= end_year
        ]
        gdelt_spikes = self._fetch_gdelt_spikes(gpr_spikes, store=store, run_id=run_id)
        acled_events = self._fetch_acled_event_rows(
            start_year, end_year, store=store, run_id=run_id
        )
        ucdp_events = self._fetch_ucdp_event_rows(
            start_year, end_year, store=store, run_id=run_id
        )
        hdx_events = self._fetch_hdx_hapi_event_rows(
            start_year, end_year, store=store, run_id=run_id
        )
        self._last_source_dates = {
            GPR_SOURCE_ID: {row["date"] for row in gpr_spikes},
            GDELT_SOURCE_ID: {row["date"] for row in gdelt_spikes},
            ACLED_SOURCE_ID: {row["date"] for row in acled_events},
            UCDP_SOURCE_ID: {row["date"] for row in ucdp_events},
            HDX_HAPI_SOURCE_ID: {row["date"] for row in hdx_events},
        }

        for row in hdx_events:
            if start_year <= row["date"].year <= end_year:
                candidates.append(
                    _signal_candidate(
                        row,
                        source_id=HDX_HAPI_SOURCE_ID,
                        event_subtype="hdx_hapi_monthly_conflict",
                    )
                )
        for row in acled_events:
            if start_year <= row["date"].year <= end_year:
                candidates.append(
                    _signal_candidate(
                        row,
                        source_id=ACLED_SOURCE_ID,
                        event_subtype="acled_conflict_event",
                    )
                )
        for row in gdelt_spikes:
            if start_year <= row["date"].year <= end_year:
                candidates.append(_gdelt_candidate(row))
        for row in gpr_spikes:
            if start_year <= row["date"].year <= end_year:
                anchor = _anchor_date(
                    row["date"],
                    self._last_source_dates[GDELT_SOURCE_ID],
                    self.merge_window_days,
                )
                candidates.append(_gpr_candidate(row, anchor))
        for row in ucdp_events:
            if start_year <= row["date"].year <= end_year:
                candidates.append(
                    _signal_candidate(
                        row,
                        source_id=UCDP_SOURCE_ID,
                        event_subtype="ucdp_organized_violence",
                    )
                )
        self.last_run_status = (
            "partial"
            if any(
                status.status in {"failed", "partial"}
                for status in self.last_source_statuses.values()
            )
            else "ok"
        )
        return sorted(
            candidates, key=lambda candidate: (candidate.date, candidate.source_id)
        )

    def validate(
        self,
        candidates: list[EventCandidate],
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[ValidationResult]:
        del store, run_id
        validations: list[ValidationResult] = []
        for candidate in candidates:
            if candidate.event_type != "geopolitical_event":
                continue
            key = (candidate.event_type, candidate.date)
            for source_id, dates in sorted(self._last_source_dates.items()):
                if source_id == candidate.source_id:
                    continue
                if _has_nearby(candidate.date, dates, self.merge_window_days):
                    validations.append(
                        ValidationResult(
                            key,
                            source_id,
                            "confirm",
                            candidate.source_url,
                            f"{source_id} corroborated nearby geopolitical signal",
                        )
                    )
        return validations

    def _fetch_gpr_spikes(
        self, *, store: AcquisitionStore | None, run_id: int | None
    ) -> list[dict[str, object]]:
        try:
            payload = self.gpr_fetcher()
            _record_payload(
                store, run_id, GPR_SOURCE_ID, "gpr_daily", payload, "GPR daily index"
            )
            rows = detect_gpr_spikes(
                parse_gpr_table(payload),
                min_history_days=self.min_history_days,
                stddev_threshold=self.stddev_threshold,
            )
        except (
            OSError,
            ValueError,
        ) as exc:  # pragma: no cover - exercised via integration degradation
            LOGGER.error(
                "GPR fetch/parse failed; geopolitical GPR candidates skipped: %s", exc
            )
            self._record_source_status(
                GPR_SOURCE_ID,
                "failed",
                error=str(exc),
                attempted_fetches=1,
                failed_fetches=1,
            )
            return []
        self._record_source_status(
            GPR_SOURCE_ID,
            "ok" if rows else "empty",
            rows=len(rows),
            attempted_fetches=1,
            empty_payload=_is_empty_payload(payload),
        )
        return rows

    def _fetch_gdelt_spikes(
        self,
        gpr_spikes: list[dict[str, object]],
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[dict[str, object]]:
        if self.gdelt_fetcher is None:
            return self._fetch_gdelt_daily_spike_windows(
                gpr_spikes, store=store, run_id=run_id
            )
        try:
            payload = self.gdelt_fetcher()
            _record_payload(
                store,
                run_id,
                GDELT_SOURCE_ID,
                "gdelt_geopolitical_volume",
                payload,
                "GDELT geopolitical volume sample",
            )
            rows = parse_gdelt_volume_table(payload)
        except (
            OSError,
            ValueError,
        ) as exc:  # pragma: no cover - exercised via integration degradation
            LOGGER.error(
                "GDELT fetch/parse failed; geopolitical GDELT candidates skipped: %s",
                exc,
            )
            self._record_source_status(
                GDELT_SOURCE_ID,
                "failed",
                error=str(exc),
                attempted_fetches=1,
                failed_fetches=1,
            )
            return []
        self._record_source_status(
            GDELT_SOURCE_ID,
            "ok" if rows else "empty",
            rows=len(rows),
            attempted_fetches=1,
            empty_payload=_is_empty_payload(payload),
        )
        return rows

    def _fetch_gdelt_daily_spike_windows(
        self,
        gpr_spikes: list[dict[str, object]],
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[dict[str, object]]:
        dates = sorted(
            {
                day
                for row in gpr_spikes
                for day in _window_dates(row["date"], self.merge_window_days)
            }
        )
        rows: list[dict[str, object]] = []
        failed_fetches = 0
        last_error: str | None = None
        for day in dates:
            try:
                payload = self.gdelt_daily_fetcher(day)
            except (
                OSError
            ) as exc:  # pragma: no cover - exercised via integration degradation
                LOGGER.error(
                    "GDELT daily export fetch failed for %s; date skipped: %s",
                    day.isoformat(),
                    exc,
                )
                failed_fetches += 1
                last_error = str(exc)
                continue
            source_identifier = f"gdelt_daily_export_{day:%Y%m%d}"
            _record_payload(
                store,
                run_id,
                GDELT_SOURCE_ID,
                source_identifier,
                payload,
                "GDELT daily event export",
            )
            rows.extend(
                parse_gdelt_event_export(
                    payload,
                    source_url=GDELT_DAILY_EXPORT_URL_TEMPLATE.format(date=day),
                    expected_date=day,
                )
            )
        rows = sorted(rows, key=lambda row: (row["date"], row["event_count"]))
        status = "partial" if failed_fetches else "ok" if rows else "empty"
        self._record_source_status(
            GDELT_SOURCE_ID,
            status,
            rows=len(rows),
            error=last_error,
            attempted_fetches=len(dates),
            failed_fetches=failed_fetches,
        )
        return rows

    def _record_source_status(
        self,
        source_id: str,
        status: str,
        *,
        rows: int = 0,
        error: str | None = None,
        attempted_fetches: int = 0,
        failed_fetches: int = 0,
        empty_payload: bool = False,
    ) -> None:
        self.last_source_statuses[source_id] = SourceFetchStatus(
            source_id=source_id,
            status=status,
            rows=rows,
            error=error,
            attempted_fetches=attempted_fetches,
            failed_fetches=failed_fetches,
            empty_payload=empty_payload,
        )

    def _fetch_acled_event_rows(
        self,
        start_year: int,
        end_year: int,
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[dict[str, object]]:
        outcome = _fetch_optional_conflict_rows(
            self.acled_fetcher,
            start_year,
            end_year,
            store=store,
            run_id=run_id,
            source_id=ACLED_SOURCE_ID,
            source_identifier="acled_events",
            source_url=ACLED_READ_URL,
            parser=parse_acled_events,
        )
        self.last_source_statuses[ACLED_SOURCE_ID] = outcome.status
        return outcome.rows

    def _fetch_ucdp_event_rows(
        self,
        start_year: int,
        end_year: int,
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[dict[str, object]]:
        outcome = _fetch_optional_conflict_rows(
            self.ucdp_fetcher,
            start_year,
            end_year,
            store=store,
            run_id=run_id,
            source_id=UCDP_SOURCE_ID,
            source_identifier="ucdp_ged_candidate",
            source_url=UCDP_GED_CANDIDATE_URL,
            parser=parse_ucdp_events,
        )
        self.last_source_statuses[UCDP_SOURCE_ID] = outcome.status
        return outcome.rows

    def _fetch_hdx_hapi_event_rows(
        self,
        start_year: int,
        end_year: int,
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[dict[str, object]]:
        outcome = _fetch_optional_conflict_rows(
            self.hdx_hapi_fetcher,
            start_year,
            end_year,
            store=store,
            run_id=run_id,
            source_id=HDX_HAPI_SOURCE_ID,
            source_identifier="hdx_hapi_conflict_events",
            source_url=HDX_HAPI_CONFLICT_EVENTS_URL,
            parser=parse_hdx_hapi_conflict_events,
        )
        self.last_source_statuses[HDX_HAPI_SOURCE_ID] = outcome.status
        return outcome.rows


def parse_gpr_table(payload: str | bytes) -> pd.DataFrame:
    if isinstance(payload, bytes):
        try:
            df = pd.read_excel(io.BytesIO(payload))
        except ValueError:
            LOGGER.debug(
                "GPR Excel parse failed; falling back to CSV parser source_id=%s",
                GPR_SOURCE_ID,
                exc_info=True,
            )
            df = pd.read_csv(io.BytesIO(payload))
    else:
        df = pd.read_csv(io.StringIO(payload))
    lower_columns = {str(column).strip().lower(): column for column in df.columns}
    date_column = lower_columns.get("date") or lower_columns.get("day")
    value_column = (
        lower_columns.get("gpr")
        or lower_columns.get("gprd")
        or lower_columns.get("gpr_daily")
    )
    if date_column is None or value_column is None:
        raise ValueError("GPR table must contain date and gpr/GPRD columns")
    out = df[[date_column, value_column]].rename(
        columns={date_column: "date", value_column: "gpr"}
    )
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["gpr"] = pd.to_numeric(out["gpr"], errors="coerce")
    return out.dropna(subset=["gpr"]).sort_values("date").reset_index(drop=True)


def detect_gpr_spikes(
    df: pd.DataFrame, *, min_history_days: int, stddev_threshold: float
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    values = df["gpr"].astype(float)
    rolling_mean = (
        values.shift(1).rolling(min_history_days, min_periods=min_history_days).mean()
    )
    rolling_std = (
        values.shift(1)
        .rolling(min_history_days, min_periods=min_history_days)
        .std(ddof=0)
    )
    for idx, row in df.iterrows():
        mean = rolling_mean.iloc[idx]
        std = rolling_std.iloc[idx]
        if pd.isna(mean) or pd.isna(std) or std == 0:
            continue
        threshold = float(mean) + stddev_threshold * float(std)
        value = float(row["gpr"])
        if value > threshold:
            rows.append({"date": row["date"], "value": value, "threshold": threshold})
    return rows


def parse_gdelt_volume_table(payload: str | bytes) -> list[dict[str, object]]:
    text = (
        payload.decode("utf-8", errors="replace")
        if isinstance(payload, bytes)
        else payload
    )
    df = pd.read_csv(io.StringIO(text))
    lower_columns = {str(column).strip().lower(): column for column in df.columns}
    date_column = lower_columns.get("date")
    count_column = lower_columns.get("event_count") or lower_columns.get("count")
    if date_column is None or count_column is None:
        raise ValueError(
            "GDELT volume table must contain date and event_count/count columns"
        )
    theme_column = lower_columns.get("dominant_theme")
    url_column = lower_columns.get("source_url")
    rows: list[dict[str, object]] = []
    for record in df.to_dict("records"):
        event_count = int(record[count_column])
        if event_count <= 0:
            continue
        rows.append(
            {
                "date": dt.date.fromisoformat(str(record[date_column])),
                "event_count": event_count,
                "dominant_theme": str(
                    record.get(theme_column, "geopolitical volume spike")
                )
                if theme_column
                else "geopolitical volume spike",
                "source_url": str(record.get(url_column, "")) if url_column else None,
            }
        )
    return rows


def parse_gdelt_event_export(
    payload: str | bytes,
    *,
    source_url: str,
    expected_date: dt.date | None = None,
) -> list[dict[str, object]]:
    text = _decode_gdelt_export(payload)
    totals: dict[dt.date, dict[str, object]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) <= GDELT_SOURCE_URL_IDX:
            continue
        root_code = columns[GDELT_EVENT_ROOT_CODE_IDX].strip()
        quad_class = columns[GDELT_QUAD_CLASS_IDX].strip()
        if root_code not in GDELT_EVENT_ROOT_CODES and quad_class != "4":
            continue
        try:
            event_date = dt.datetime.strptime(
                columns[GDELT_SQLDATE_IDX], "%Y%m%d"
            ).date()
        except ValueError:
            continue
        if expected_date is not None and event_date != expected_date:
            continue
        mentions = _parse_positive_int(columns[GDELT_NUM_MENTIONS_IDX], default=1)
        current = totals.setdefault(
            event_date,
            {
                "date": event_date,
                "event_count": 0,
                "dominant_theme": "GDELT material conflict / protest volume",
                "source_url": _source_url_or_export_url(
                    columns[GDELT_SOURCE_URL_IDX], source_url
                ),
            },
        )
        current["event_count"] = int(current["event_count"]) + mentions
    return [row for _, row in sorted(totals.items()) if int(row["event_count"]) > 0]


def _decode_gdelt_export(payload: str | bytes) -> str:
    if isinstance(payload, str):
        return payload
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            first_name = archive.namelist()[0]
            return archive.read(first_name).decode("utf-8", errors="replace")
    return payload.decode("utf-8", errors="replace")


def _source_url_or_export_url(value: str, export_url: str) -> str:
    candidate = value.strip()
    return candidate if candidate.startswith(("http://", "https://")) else export_url


def _gpr_candidate(row: dict[str, object], anchor_date: dt.date) -> EventCandidate:
    return EventCandidate(
        date=anchor_date,
        event_type="geopolitical_event",
        market="GLOBAL",
        importance="medium",
        source_id=GPR_SOURCE_ID,
        source_url=GPR_DAILY_URL,
        raw_title="GPR geopolitical risk spike",
        raw_snippet=f"GPR daily value {row['value']:.2f} exceeded trailing threshold {row['threshold']:.2f}.",
        is_future_scheduled=False,
        confidence="medium",
        requires_manual_review=True,
        event_subtype="gpr_spike",
    )


def _gdelt_candidate(row: dict[str, object]) -> EventCandidate:
    return EventCandidate(
        date=row["date"],
        event_type="geopolitical_event",
        market="GLOBAL",
        importance="medium",
        source_id=GDELT_SOURCE_ID,
        source_url=row["source_url"] or None,
        raw_title=str(row["dominant_theme"]),
        raw_snippet=f"GDELT geopolitical event volume: {row['event_count']}.",
        is_future_scheduled=False,
        confidence="medium",
        requires_manual_review=True,
        event_subtype="gdelt_volume_spike",
    )


def _signal_candidate(
    row: dict[str, object], *, source_id: str, event_subtype: str
) -> EventCandidate:
    return EventCandidate(
        date=row["date"],
        event_type="geopolitical_event",
        market="GLOBAL",
        importance="medium",
        source_id=source_id,
        source_url=row["source_url"] or None,
        raw_title=str(row["dominant_theme"]),
        raw_snippet=f"{source_id} geopolitical event count: {row['event_count']}; fatalities: {row.get('fatalities', 0)}.",
        is_future_scheduled=False,
        confidence="medium",
        requires_manual_review=True,
        event_subtype=event_subtype,
    )


def _anchor_date(
    gpr_date: dt.date, gdelt_dates: set[dt.date], window_days: int
) -> dt.date:
    nearby = sorted(
        date for date in gdelt_dates if abs((date - gpr_date).days) <= window_days
    )
    return gpr_date if gpr_date in nearby or not nearby else gpr_date


def _has_nearby(event_date: dt.date, dates: set[dt.date], window_days: int) -> bool:
    return any(abs((date - event_date).days) <= window_days for date in dates)


def _window_dates(anchor: dt.date, window_days: int) -> list[dt.date]:
    return [
        anchor + dt.timedelta(days=offset)
        for offset in range(-window_days, window_days + 1)
    ]
