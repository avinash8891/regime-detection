from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportUnusedFunction=false

import datetime as dt
import io
import logging
import zipfile
from collections.abc import Callable

import pandas as pd

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources.gpr_gdelt_conflict_parsers import (
    _parse_positive_int,
)
from regime_data_fetch.event_sources.gpr_gdelt_fetchers import (
    ACLED_READ_URL,
    ACLED_SOURCE_ID,
    ConflictFetcher,
    GDELT_DAILY_EXPORT_URL_TEMPLATE,
    GPR_DAILY_URL,
    SourceFetchStatus,
    UCDP_GED_CANDIDATE_URL,
    UCDP_SOURCE_ID,
    fetch_acled_events as _fetch_acled_events,
    fetch_ai_gpr_country_monthly as _fetch_ai_gpr_country_monthly,
    fetch_ai_gpr_daily as _fetch_ai_gpr_daily,
    fetch_ai_gpr_eventtype_monthly as _fetch_ai_gpr_eventtype_monthly,
    fetch_gdelt_daily_export as _fetch_gdelt_daily_export,
    fetch_gpr_daily as _fetch_gpr_daily,
    fetch_gpr_monthly as _fetch_gpr_monthly,
    fetch_optional_conflict_rows as _fetch_optional_conflict_rows,
    fetch_ucdp_events as _fetch_ucdp_events,
    is_empty_payload as _is_empty_payload,
    record_payload as _record_payload,
)
from regime_data_fetch.event_sources.gpr_gdelt_conflict_parsers import (
    parse_acled_events,
    parse_ucdp_events,
)
from regime_data_fetch.event_sources.models import EventCandidate, ValidationResult

LOGGER = logging.getLogger(__name__)
GPR_SOURCE_ID = "gpr:caldara-iacoviello"
GPR_MONTHLY_SOURCE_ID = "gpr:caldara-iacoviello-monthly"
AI_GPR_SOURCE_ID = "ai-gpr:iacoviello-tong"
GDELT_SOURCE_ID = "gdelt:events-v2"
GDELT_EVENT_ROOT_CODES = {"14", "18", "19", "20"}
GDELT_SQLDATE_IDX = 1
GDELT_EVENT_ROOT_CODE_IDX = 28
GDELT_QUAD_CLASS_IDX = 29
GDELT_NUM_MENTIONS_IDX = 31
GDELT_SOURCE_URL_IDX = 60


def _nearby_validations(
    *,
    candidates: list[EventCandidate],
    source_id: str,
    dates: set[dt.date],
    window_days: int,
) -> list[ValidationResult]:
    validations: list[ValidationResult] = []
    for candidate in candidates:
        if (
            candidate.event_type != "geopolitical_event"
            or candidate.source_id == source_id
        ):
            continue
        if _has_nearby(candidate.date, dates, window_days):
            validations.append(
                ValidationResult(
                    (candidate.event_type, candidate.date),
                    source_id,
                    "confirm",
                    candidate.source_url,
                    f"{source_id} corroborated nearby geopolitical signal",
                )
            )
    return validations


class GPRSignalGenerator:
    source_id = GPR_SOURCE_ID
    validator_id = GPR_SOURCE_ID

    def __init__(
        self,
        *,
        gpr_fetcher: Callable[[], str | bytes] | None = None,
        gpr_monthly_fetcher: Callable[[], str | bytes] | None = None,
        ai_gpr_daily_fetcher: Callable[[], str | bytes] | None = None,
        ai_gpr_eventtype_monthly_fetcher: Callable[[], str | bytes] | None = None,
        ai_gpr_country_monthly_fetcher: Callable[[], str | bytes] | None = None,
        min_history_days: int = 252,
        stddev_threshold: float = 3.0,
        merge_window_days: int = 2,
    ) -> None:
        self.gpr_fetcher = gpr_fetcher or _fetch_gpr_daily
        self.gpr_monthly_fetcher = (
            gpr_monthly_fetcher
            if gpr_monthly_fetcher is not None
            else _fetch_gpr_monthly if gpr_fetcher is None else None
        )
        self.ai_gpr_daily_fetcher = (
            ai_gpr_daily_fetcher
            if ai_gpr_daily_fetcher is not None
            else _fetch_ai_gpr_daily if gpr_fetcher is None else None
        )
        self.ai_gpr_eventtype_monthly_fetcher = (
            ai_gpr_eventtype_monthly_fetcher
            if ai_gpr_eventtype_monthly_fetcher is not None
            else _fetch_ai_gpr_eventtype_monthly if gpr_fetcher is None else None
        )
        self.ai_gpr_country_monthly_fetcher = (
            ai_gpr_country_monthly_fetcher
            if ai_gpr_country_monthly_fetcher is not None
            else _fetch_ai_gpr_country_monthly if gpr_fetcher is None else None
        )
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
        gpr_spikes = [
            row
            for row in self._fetch_gpr_spikes(store=store, run_id=run_id)
            if start_year <= row["date"].year <= end_year
        ]
        gpr_context = self._fetch_gpr_context(gpr_spikes, store=store, run_id=run_id)
        self._last_source_dates = {
            GPR_SOURCE_ID: {row["date"] for row in gpr_spikes},
            AI_GPR_SOURCE_ID: set(gpr_context.get("ai_gpr_dates", set())),
        }
        self.last_run_status = (
            "partial"
            if any(
                status.status in {"failed", "partial"}
                for status in self.last_source_statuses.values()
            )
            else "ok"
        )
        return sorted(
            (_gpr_candidate(row, row["date"], gpr_context) for row in gpr_spikes),
            key=lambda candidate: (candidate.date, candidate.source_id),
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
        for source_id, dates in sorted(self._last_source_dates.items()):
            validations.extend(
                _nearby_validations(
                    candidates=candidates,
                    source_id=source_id,
                    dates=dates,
                    window_days=self.merge_window_days,
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
        except (OSError, ValueError) as exc:
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

    def _fetch_gpr_context(
        self,
        gpr_spikes: list[dict[str, object]],
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> dict[str, object]:
        candidate_dates = [row["date"] for row in gpr_spikes]
        context: dict[str, object] = {
            "monthly_country": {},
            "ai_gpr": {},
            "ai_gpr_dates": set(),
        }
        if not candidate_dates:
            self._record_source_status(
                GPR_MONTHLY_SOURCE_ID, "empty", rows=0, attempted_fetches=0
            )
            self._record_source_status(
                AI_GPR_SOURCE_ID, "empty", rows=0, attempted_fetches=0
            )
            return context
        if self.gpr_monthly_fetcher is not None:
            try:
                payload = self.gpr_monthly_fetcher()
                _record_payload(
                    store,
                    run_id,
                    GPR_MONTHLY_SOURCE_ID,
                    "gpr_monthly_country",
                    payload,
                    "GPR monthly country context",
                )
                monthly_country = parse_gpr_monthly_country_context(
                    payload, candidate_dates=candidate_dates
                )
                context["monthly_country"] = monthly_country
                self._record_source_status(
                    GPR_MONTHLY_SOURCE_ID,
                    "ok" if monthly_country else "empty",
                    rows=len(monthly_country),
                    attempted_fetches=1,
                    empty_payload=_is_empty_payload(payload),
                )
            except (OSError, ValueError) as exc:
                LOGGER.error(
                    "GPR monthly context fetch/parse failed; context skipped: %s", exc
                )
                self._record_source_status(
                    GPR_MONTHLY_SOURCE_ID,
                    "failed",
                    error=str(exc),
                    attempted_fetches=1,
                    failed_fetches=1,
                )
        else:
            self._record_source_status(
                GPR_MONTHLY_SOURCE_ID, "skipped", attempted_fetches=0
            )

        if (
            self.ai_gpr_daily_fetcher is None
            or self.ai_gpr_eventtype_monthly_fetcher is None
            or self.ai_gpr_country_monthly_fetcher is None
        ):
            self._record_source_status(AI_GPR_SOURCE_ID, "skipped", attempted_fetches=0)
            return context
        try:
            daily_payload = self.ai_gpr_daily_fetcher()
            eventtype_payload = self.ai_gpr_eventtype_monthly_fetcher()
            country_payload = self.ai_gpr_country_monthly_fetcher()
            _record_payload(
                store,
                run_id,
                AI_GPR_SOURCE_ID,
                "ai_gpr_daily",
                daily_payload,
                "AI-GPR daily context",
            )
            _record_payload(
                store,
                run_id,
                AI_GPR_SOURCE_ID,
                "ai_gpr_eventtype_monthly",
                eventtype_payload,
                "AI-GPR monthly event-type context",
            )
            _record_payload(
                store,
                run_id,
                AI_GPR_SOURCE_ID,
                "ai_gpr_country_monthly",
                country_payload,
                "AI-GPR monthly country context",
            )
            ai_context = parse_ai_gpr_context(
                daily_payload,
                eventtype_payload,
                country_payload,
                candidate_dates=candidate_dates,
            )
            context["ai_gpr"] = ai_context
            context["ai_gpr_dates"] = set(ai_context)
            self._record_source_status(
                AI_GPR_SOURCE_ID,
                "ok" if ai_context else "empty",
                rows=len(ai_context),
                attempted_fetches=3,
                empty_payload=all(
                    _is_empty_payload(payload)
                    for payload in (daily_payload, eventtype_payload, country_payload)
                ),
            )
        except (OSError, ValueError) as exc:
            LOGGER.error("AI-GPR context fetch/parse failed; context skipped: %s", exc)
            self._record_source_status(
                AI_GPR_SOURCE_ID,
                "failed",
                error=str(exc),
                attempted_fetches=3,
                failed_fetches=1,
            )
        return context

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


class GDELTSignalGenerator:
    source_id = GDELT_SOURCE_ID
    validator_id = GDELT_SOURCE_ID

    def __init__(
        self,
        *,
        gdelt_fetcher: Callable[[], str | bytes] | None = None,
        gdelt_daily_fetcher: Callable[[dt.date], bytes] | None = None,
        merge_window_days: int = 2,
    ) -> None:
        self.gdelt_fetcher = gdelt_fetcher
        self.gdelt_daily_fetcher = gdelt_daily_fetcher or _fetch_gdelt_daily_export
        self.merge_window_days = merge_window_days
        self._last_source_dates: set[dt.date] = set()
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
        rows = self._fetch_gdelt_spikes(store=store, run_id=run_id)
        rows = [row for row in rows if start_year <= row["date"].year <= end_year]
        self._last_source_dates = {row["date"] for row in rows}
        self.last_run_status = (
            "partial"
            if any(
                status.status in {"failed", "partial"}
                for status in self.last_source_statuses.values()
            )
            else "ok"
        )
        return sorted(
            (_gdelt_candidate(row) for row in rows), key=lambda item: item.date
        )

    def validate(
        self,
        candidates: list[EventCandidate],
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[ValidationResult]:
        if not self._last_source_dates:
            rows = self._fetch_gdelt_daily_candidate_windows(
                candidates, store=store, run_id=run_id
            )
            self._last_source_dates = {row["date"] for row in rows}
        return _nearby_validations(
            candidates=candidates,
            source_id=GDELT_SOURCE_ID,
            dates=self._last_source_dates,
            window_days=self.merge_window_days,
        )

    def _fetch_gdelt_spikes(
        self, *, store: AcquisitionStore | None, run_id: int | None
    ) -> list[dict[str, object]]:
        if self.gdelt_fetcher is None:
            self._record_source_status(GDELT_SOURCE_ID, "skipped", attempted_fetches=0)
            return []
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
        except (OSError, ValueError) as exc:
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

    def _fetch_gdelt_daily_candidate_windows(
        self,
        candidates: list[EventCandidate],
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[dict[str, object]]:
        # TODO(gpr-gdelt-backlog): keep the approved event calendar simple.
        # Add a separate geopolitical_risk_signals artifact/table keyed by a
        # signal id, with one row per GPR spike/signal date: headline GPR,
        # ACT/THREAT components, MA7/MA30 persistence, article count,
        # dominant_component, confidence, suggested_window_days, monthly
        # country context, and AI-GPR context. Approval overlays should
        # reference that signal id as evidence_candidate_id instead of
        # duplicating rich GPR fields in us_events.yaml. Final approved
        # geopolitical_event rows should remain ordinary event rows with date,
        # type, window_days, approved_label, and notes.
        dates = sorted(
            {
                day
                for candidate in candidates
                if candidate.event_type == "geopolitical_event"
                for day in _window_dates(candidate.date, self.merge_window_days)
            }
        )
        rows: list[dict[str, object]] = []
        failed_fetches = 0
        last_error: str | None = None
        for day in dates:
            try:
                payload = self.gdelt_daily_fetcher(day)
            except OSError as exc:
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


class _ConflictSignalGenerator:
    source_id = ""
    event_subtype = ""
    source_identifier = ""
    source_url = ""

    def __init__(
        self,
        *,
        fetcher: ConflictFetcher,
        parser: Callable[..., list[dict[str, object]]],
        merge_window_days: int = 2,
    ) -> None:
        self.fetcher = fetcher
        self.parser = parser
        self.merge_window_days = merge_window_days
        self._last_source_dates: set[dt.date] = set()
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
        outcome = _fetch_optional_conflict_rows(
            self.fetcher,
            start_year,
            end_year,
            store=store,
            run_id=run_id,
            source_id=self.source_id,
            source_identifier=self.source_identifier,
            source_url=self.source_url,
            parser=self.parser,
        )
        self.last_source_statuses[self.source_id] = outcome.status
        self._last_source_dates = {row["date"] for row in outcome.rows}
        self.last_run_status = (
            "partial" if outcome.status.status in {"failed", "partial"} else "ok"
        )
        return sorted(
            (
                _signal_candidate(
                    row,
                    source_id=self.source_id,
                    event_subtype=self.event_subtype,
                )
                for row in outcome.rows
            ),
            key=lambda item: item.date,
        )

    def validate(
        self,
        candidates: list[EventCandidate],
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[ValidationResult]:
        del store, run_id
        return _nearby_validations(
            candidates=candidates,
            source_id=self.source_id,
            dates=self._last_source_dates,
            window_days=self.merge_window_days,
        )


class ACLEDSignalGenerator(_ConflictSignalGenerator):
    source_id = ACLED_SOURCE_ID
    event_subtype = "acled_conflict_event"
    source_identifier = "acled_events"
    source_url = ACLED_READ_URL

    def __init__(self, *, acled_fetcher: ConflictFetcher | None = None) -> None:
        super().__init__(
            fetcher=acled_fetcher or _fetch_acled_events,
            parser=parse_acled_events,
        )


class UCDPSignalGenerator(_ConflictSignalGenerator):
    source_id = UCDP_SOURCE_ID
    event_subtype = "ucdp_organized_violence"
    source_identifier = "ucdp_ged_candidate"
    source_url = UCDP_GED_CANDIDATE_URL

    def __init__(self, *, ucdp_fetcher: ConflictFetcher | None = None) -> None:
        super().__init__(
            fetcher=ucdp_fetcher or _fetch_ucdp_events,
            parser=parse_ucdp_events,
        )


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
    optional_columns = {
        "gpr_act": lower_columns.get("gpr_act") or lower_columns.get("gprd_act"),
        "gpr_threat": lower_columns.get("gpr_threat")
        or lower_columns.get("gprd_threat"),
        "gpr_ma7": lower_columns.get("gpr_ma7") or lower_columns.get("gprd_ma7"),
        "gpr_ma30": lower_columns.get("gpr_ma30") or lower_columns.get("gprd_ma30"),
        "article_count": lower_columns.get("n10d")
        or lower_columns.get("article_count")
        or lower_columns.get("articles"),
        "event": lower_columns.get("event"),
    }
    out = pd.DataFrame(
        {
            "date": _parse_gpr_dates(df[date_column]),
            "gpr": pd.to_numeric(df[value_column], errors="coerce"),
        }
    )
    for output_column, source_column in optional_columns.items():
        if output_column == "event":
            if source_column is None:
                out[output_column] = ""
            else:
                out[output_column] = df[source_column].fillna("").astype(str)
            continue
        out[output_column] = (
            pd.to_numeric(df[source_column], errors="coerce")
            if source_column is not None
            else pd.NA
        )
    return out.dropna(subset=["gpr"]).sort_values("date").reset_index(drop=True)


def parse_gpr_monthly_country_context(
    payload: str | bytes,
    *,
    candidate_dates: list[dt.date],
    top_n: int = 3,
) -> dict[dt.date, str]:
    df = _read_csv_or_excel(payload)
    lower_columns = {str(column).strip().lower(): column for column in df.columns}
    month_column = lower_columns.get("month") or lower_columns.get("date")
    if month_column is None:
        raise ValueError("GPR monthly table must contain month/date column")
    df = df.copy()
    df["_month"] = pd.to_datetime(df[month_column], errors="coerce").dt.to_period("M")
    country_columns = [
        column
        for column in df.columns
        if str(column).upper().startswith("GPRC_")
        and not str(column).upper().startswith("GPRHC_")
    ]
    if not country_columns:
        raise ValueError("GPR monthly table must contain GPRC_* country columns")
    context: dict[dt.date, str] = {}
    for candidate_date in candidate_dates:
        month = pd.Period(candidate_date, freq="M")
        rows = df[df["_month"] == month]
        if rows.empty:
            continue
        record = rows.iloc[-1]
        country_values: list[tuple[str, float]] = []
        for column in country_columns:
            value = _optional_float(record.get(column))
            if value is None or value <= 0:
                continue
            country_values.append((str(column).upper().removeprefix("GPRC_"), value))
        country_values.sort(key=lambda item: item[1], reverse=True)
        if country_values:
            context[candidate_date] = "monthly_country_gpr=" + ",".join(
                f"{code}:{value:.2f}" for code, value in country_values[:top_n]
            )
    return context


def parse_ai_gpr_context(
    daily_payload: str | bytes,
    eventtype_monthly_payload: str | bytes,
    country_monthly_payload: str | bytes,
    *,
    candidate_dates: list[dt.date],
    top_n: int = 3,
) -> dict[dt.date, str]:
    daily = _read_csv_or_excel(daily_payload)
    eventtype = _read_csv_or_excel(eventtype_monthly_payload)
    country = _read_csv_or_excel(country_monthly_payload)
    daily_date_column = _column_by_lower(daily, "date")
    eventtype_date_column = _column_by_lower(eventtype, "date")
    country_date_column = _column_by_lower(country, "date")
    daily_ai_column = _column_by_lower(daily, "gpr_ai")
    if (
        daily_date_column is None
        or eventtype_date_column is None
        or country_date_column is None
    ):
        raise ValueError("AI-GPR tables must contain Date columns")
    daily = daily.copy()
    eventtype = eventtype.copy()
    country = country.copy()
    daily["_date"] = pd.to_datetime(daily[daily_date_column], errors="coerce").dt.date
    eventtype["_month"] = pd.to_datetime(
        eventtype[eventtype_date_column], errors="coerce"
    ).dt.to_period("M")
    country["_month"] = pd.to_datetime(
        country[country_date_column], errors="coerce"
    ).dt.to_period("M")

    context: dict[dt.date, str] = {}
    for candidate_date in candidate_dates:
        parts: list[str] = []
        daily_row = daily[daily["_date"] == candidate_date]
        if not daily_row.empty:
            ai_value = _optional_float(daily_row.iloc[-1].get(daily_ai_column))
            if ai_value is not None:
                parts.append(f"ai_gpr_daily={ai_value:.2f}")
        month = pd.Period(candidate_date, freq="M")
        eventtype_row = eventtype[eventtype["_month"] == month]
        if not eventtype_row.empty:
            event_values = _top_numeric_columns(
                eventtype_row.iloc[-1],
                exclude={"Date", "GPR_AI", "_month"},
                top_n=1,
            )
            if event_values:
                name, value = event_values[0]
                parts.append(f"ai_gpr_event_type={name}:{value:.2f}")
        country_row = country[country["_month"] == month]
        if not country_row.empty:
            country_values = _top_numeric_columns(
                country_row.iloc[-1],
                exclude={"Date", "GPR_AI", "_month"},
                top_n=top_n,
            )
            if country_values:
                parts.append(
                    "ai_gpr_country="
                    + ",".join(f"{name}:{value:.2f}" for name, value in country_values)
                )
        if parts:
            context[candidate_date] = "; ".join(parts)
    return context


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
            act_value = _optional_float(row.get("gpr_act"))
            threat_value = _optional_float(row.get("gpr_threat"))
            ma7 = _optional_float(row.get("gpr_ma7"))
            ma30 = _optional_float(row.get("gpr_ma30"))
            article_count = _optional_float(row.get("article_count"))
            components = ["headline"]
            if act_value is not None and act_value > threshold:
                components.append("acts")
            if threat_value is not None and threat_value > threshold:
                components.append("threats")
            if ma7 is not None and ma7 > threshold:
                components.append("persistent_7d")
            if ma30 is not None and ma30 > threshold:
                components.append("persistent_30d")
            dominant_component = "headline"
            component_values = {
                "acts": act_value,
                "threats": threat_value,
            }
            valid_components = {
                name: component_value
                for name, component_value in component_values.items()
                if component_value is not None and component_value > threshold
            }
            if valid_components:
                dominant_component = max(
                    valid_components, key=valid_components.__getitem__
                )
            rows.append(
                {
                    "date": row["date"],
                    "value": value,
                    "threshold": threshold,
                    "act_value": act_value,
                    "threat_value": threat_value,
                    "ma7": ma7,
                    "ma30": ma30,
                    "article_count": article_count,
                    "event": str(row.get("event", "")),
                    "spike_components": tuple(components),
                    "dominant_component": dominant_component,
                }
            )
    return rows


def _parse_gpr_dates(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    compact = text.str.fullmatch(r"\d{8}")
    parsed = pd.to_datetime(series, errors="coerce")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d")
    return parsed.dt.date


def _optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_csv_or_excel(payload: str | bytes) -> pd.DataFrame:
    if isinstance(payload, bytes):
        try:
            return pd.read_excel(io.BytesIO(payload))
        except ValueError:
            return pd.read_csv(io.BytesIO(payload))
    return pd.read_csv(io.StringIO(payload))


def _column_by_lower(df: pd.DataFrame, name: str) -> object | None:
    lower_columns = {str(column).strip().lower(): column for column in df.columns}
    return lower_columns.get(name)


def _top_numeric_columns(
    record: pd.Series,
    *,
    exclude: set[str],
    top_n: int,
) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    exclude_lower = {item.lower() for item in exclude}
    for column, raw_value in record.items():
        column_name = str(column)
        if column_name.lower() in exclude_lower or column_name.startswith("_"):
            continue
        value = _optional_float(raw_value)
        if value is None or value <= 0:
            continue
        values.append((column_name, value))
    values.sort(key=lambda item: item[1], reverse=True)
    return values[:top_n]


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
                "dominant_theme": (
                    str(record.get(theme_column, "geopolitical volume spike"))
                    if theme_column
                    else "geopolitical volume spike"
                ),
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
    # TODO(gdelt-relevance): GDELT 2.0 exports are broad, noisy volume
    # corroboration. Use them after a GPR signal anchor exists: filter or score
    # rows by same-day/near-day window, CAMEO severity, actor/country relevance
    # from monthly GPR or AI-GPR context, mention/article volume, and top source
    # URLs. GDELT-only volume must not create or promote a geopolitical_event.
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
        return "\n".join(
            _decode_gdelt_zip_archive(archive_payload)
            for archive_payload in _split_concatenated_zip_archives(payload)
        )
    return payload.decode("utf-8", errors="replace")


def _decode_gdelt_zip_archive(payload: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        first_name = archive.namelist()[0]
        return archive.read(first_name).decode("utf-8", errors="replace")


def _split_concatenated_zip_archives(payload: bytes) -> list[bytes]:
    chunks: list[bytes] = []
    start = 0
    while start < len(payload):
        end_record = payload.find(b"PK\x05\x06", start)
        if end_record == -1 or end_record + 22 > len(payload):
            return [payload] if not chunks else chunks
        comment_length = int.from_bytes(
            payload[end_record + 20 : end_record + 22], "little"
        )
        end = end_record + 22 + comment_length
        chunks.append(payload[start:end])
        start = end
    return chunks


def _source_url_or_export_url(value: str, export_url: str) -> str:
    candidate = value.strip()
    return candidate if candidate.startswith(("http://", "https://")) else export_url


def _gpr_candidate(
    row: dict[str, object],
    anchor_date: dt.date,
    context: dict[str, object] | None = None,
) -> EventCandidate:
    dominant_component = str(row.get("dominant_component", "headline"))
    components = tuple(row.get("spike_components", ("headline",)))
    article_count = _optional_float(row.get("article_count"))
    has_directional_component = bool({"acts", "threats"}.intersection(components))
    importance = (
        "high"
        if len(components) >= 3 or (article_count is not None and article_count >= 500)
        else "medium"
    )
    confidence = (
        "high"
        if has_directional_component
        and article_count is not None
        and article_count >= 500
        else "medium"
    )
    event_text = str(row.get("event", "")).strip()
    raw_title = (
        event_text
        if event_text
        else f"GPR {_component_title(dominant_component)} geopolitical risk spike"
    )
    return EventCandidate(
        date=anchor_date,
        event_type="geopolitical_event",
        market="GLOBAL",
        importance=importance,
        source_id=GPR_SOURCE_ID,
        source_url=GPR_DAILY_URL,
        raw_title=raw_title,
        raw_snippet=_gpr_snippet(row, components, article_count, context),
        is_future_scheduled=False,
        confidence=confidence,
        requires_manual_review=True,
        window_days=_gpr_window_days(components),
        event_subtype=f"gpr_{dominant_component}_spike",
    )


def _component_title(component: str) -> str:
    return {
        "acts": "acts-driven",
        "threats": "threats-driven",
        "headline": "headline",
    }.get(component, "headline")


def _gpr_snippet(
    row: dict[str, object],
    components: tuple[object, ...],
    article_count: float | None,
    context: dict[str, object] | None,
) -> str:
    parts = [
        f"GPR daily value {row['value']:.2f} exceeded trailing threshold {row['threshold']:.2f}",
        f"components={','.join(str(component) for component in components)}",
    ]
    optional_parts = (
        ("acts", row.get("act_value")),
        ("threats", row.get("threat_value")),
        ("ma7", row.get("ma7")),
        ("ma30", row.get("ma30")),
    )
    for label, value in optional_parts:
        number = _optional_float(value)
        if number is not None:
            parts.append(f"{label}={number:.2f}")
    if article_count is not None:
        parts.append(f"articles={article_count:.0f}")
    candidate_date = row.get("date")
    if isinstance(candidate_date, dt.date) and context is not None:
        monthly_country = context.get("monthly_country", {})
        ai_gpr = context.get("ai_gpr", {})
        if isinstance(monthly_country, dict) and candidate_date in monthly_country:
            parts.append(str(monthly_country[candidate_date]))
        if isinstance(ai_gpr, dict) and candidate_date in ai_gpr:
            parts.append(str(ai_gpr[candidate_date]))
    return "; ".join(parts) + "."


def _gpr_window_days(components: tuple[object, ...]) -> tuple[int, int]:
    component_names = {str(component) for component in components}
    if "persistent_30d" in component_names:
        return (-2, 5)
    if "persistent_7d" in component_names:
        return (-1, 3)
    return (0, 0)


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
