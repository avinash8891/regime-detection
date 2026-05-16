from __future__ import annotations

import base64
import datetime as dt
import io
import json
import logging
import os
import zipfile
from collections.abc import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources.models import EventCandidate, ValidationResult

LOGGER = logging.getLogger(__name__)
GPR_SOURCE_ID = "gpr:caldara-iacoviello"
GDELT_SOURCE_ID = "gdelt:events-v2"
ACLED_SOURCE_ID = "acled:events"
UCDP_SOURCE_ID = "ucdp:ged-candidate"
HDX_HAPI_SOURCE_ID = "hdx-hapi:conflict-events"
GPR_DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
GDELT_DAILY_EXPORT_URL_TEMPLATE = "http://data.gdeltproject.org/events/{date:%Y%m%d}.export.CSV.zip"
ACLED_READ_URL = "https://acleddata.com/api/acled/read"
ACLED_TOKEN_URL = "https://acleddata.com/oauth/token"
UCDP_GED_CANDIDATE_URL = "https://ucdpapi.pcr.uu.se/api/gedevents/26.0.3"
HDX_HAPI_CONFLICT_EVENTS_URL = "https://hapi.humdata.org/api/v2/coordination-context/conflict-events"
GDELT_EVENT_ROOT_CODES = {"14", "18", "19", "20"}
GDELT_SQLDATE_IDX = 1
GDELT_EVENT_ROOT_CODE_IDX = 28
GDELT_QUAD_CLASS_IDX = 29
GDELT_NUM_MENTIONS_IDX = 31
GDELT_SOURCE_URL_IDX = 56
CONFLICT_API_PAGE_SIZE = 1000
ConflictFetcher = Callable[[int, int], str | bytes | None]


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

    def generate(
        self,
        *,
        start_year: int,
        end_year: int,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[EventCandidate]:
        candidates: list[EventCandidate] = []
        gpr_spikes = [
            row
            for row in self._fetch_gpr_spikes(store=store, run_id=run_id)
            if start_year <= row["date"].year <= end_year
        ]
        gdelt_spikes = self._fetch_gdelt_spikes(gpr_spikes, store=store, run_id=run_id)
        acled_events = self._fetch_acled_event_rows(start_year, end_year, store=store, run_id=run_id)
        ucdp_events = self._fetch_ucdp_event_rows(start_year, end_year, store=store, run_id=run_id)
        hdx_events = self._fetch_hdx_hapi_event_rows(start_year, end_year, store=store, run_id=run_id)
        self._last_source_dates = {
            GPR_SOURCE_ID: {row["date"] for row in gpr_spikes},
            GDELT_SOURCE_ID: {row["date"] for row in gdelt_spikes},
            ACLED_SOURCE_ID: {row["date"] for row in acled_events},
            UCDP_SOURCE_ID: {row["date"] for row in ucdp_events},
            HDX_HAPI_SOURCE_ID: {row["date"] for row in hdx_events},
        }

        for row in hdx_events:
            if start_year <= row["date"].year <= end_year:
                candidates.append(_signal_candidate(row, source_id=HDX_HAPI_SOURCE_ID, event_subtype="hdx_hapi_monthly_conflict"))
        for row in acled_events:
            if start_year <= row["date"].year <= end_year:
                candidates.append(_signal_candidate(row, source_id=ACLED_SOURCE_ID, event_subtype="acled_conflict_event"))
        for row in gdelt_spikes:
            if start_year <= row["date"].year <= end_year:
                candidates.append(_gdelt_candidate(row))
        for row in gpr_spikes:
            if start_year <= row["date"].year <= end_year:
                anchor = _anchor_date(row["date"], self._last_source_dates[GDELT_SOURCE_ID], self.merge_window_days)
                candidates.append(_gpr_candidate(row, anchor))
        for row in ucdp_events:
            if start_year <= row["date"].year <= end_year:
                candidates.append(_signal_candidate(row, source_id=UCDP_SOURCE_ID, event_subtype="ucdp_organized_violence"))
        return sorted(candidates, key=lambda candidate: (candidate.date, candidate.source_id))

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

    def _fetch_gpr_spikes(self, *, store: AcquisitionStore | None, run_id: int | None) -> list[dict[str, object]]:
        try:
            payload = self.gpr_fetcher()
        except Exception as exc:  # pragma: no cover - exercised via integration degradation
            LOGGER.error("GPR fetch failed; geopolitical GPR candidates skipped: %s", exc)
            return []
        _record_payload(store, run_id, GPR_SOURCE_ID, "gpr_daily", payload, "GPR daily index")
        return detect_gpr_spikes(parse_gpr_table(payload), min_history_days=self.min_history_days, stddev_threshold=self.stddev_threshold)

    def _fetch_gdelt_spikes(
        self,
        gpr_spikes: list[dict[str, object]],
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[dict[str, object]]:
        if self.gdelt_fetcher is None:
            return self._fetch_gdelt_daily_spike_windows(gpr_spikes, store=store, run_id=run_id)
        try:
            payload = self.gdelt_fetcher()
        except Exception as exc:  # pragma: no cover - exercised via integration degradation
            LOGGER.error("GDELT fetch failed; geopolitical GDELT candidates skipped: %s", exc)
            return []
        _record_payload(store, run_id, GDELT_SOURCE_ID, "gdelt_geopolitical_volume", payload, "GDELT geopolitical volume sample")
        return parse_gdelt_volume_table(payload)

    def _fetch_gdelt_daily_spike_windows(
        self,
        gpr_spikes: list[dict[str, object]],
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[dict[str, object]]:
        dates = sorted({day for row in gpr_spikes for day in _window_dates(row["date"], self.merge_window_days)})
        rows: list[dict[str, object]] = []
        for day in dates:
            try:
                payload = self.gdelt_daily_fetcher(day)
            except Exception as exc:  # pragma: no cover - exercised via integration degradation
                LOGGER.error("GDELT daily export fetch failed for %s; date skipped: %s", day.isoformat(), exc)
                continue
            source_identifier = f"gdelt_daily_export_{day:%Y%m%d}"
            _record_payload(store, run_id, GDELT_SOURCE_ID, source_identifier, payload, "GDELT daily event export")
            rows.extend(
                parse_gdelt_event_export(
                    payload,
                    source_url=GDELT_DAILY_EXPORT_URL_TEMPLATE.format(date=day),
                    expected_date=day,
                )
            )
        return sorted(rows, key=lambda row: (row["date"], row["event_count"]))

    def _fetch_acled_event_rows(
        self,
        start_year: int,
        end_year: int,
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[dict[str, object]]:
        return _fetch_optional_conflict_rows(
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

    def _fetch_ucdp_event_rows(
        self,
        start_year: int,
        end_year: int,
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[dict[str, object]]:
        return _fetch_optional_conflict_rows(
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

    def _fetch_hdx_hapi_event_rows(
        self,
        start_year: int,
        end_year: int,
        *,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[dict[str, object]]:
        return _fetch_optional_conflict_rows(
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


def parse_gpr_table(payload: str | bytes) -> pd.DataFrame:
    if isinstance(payload, bytes):
        try:
            df = pd.read_excel(io.BytesIO(payload))
        except Exception:
            LOGGER.warning(
                "GPR payload is not valid Excel, falling back to CSV parsing — verify upstream format has not changed",
                exc_info=True,
            )
            try:
                df = pd.read_csv(io.BytesIO(payload))
            except Exception:
                LOGGER.error("GPR payload is neither valid Excel nor valid CSV; cannot parse", exc_info=True)
                raise
    else:
        df = pd.read_csv(io.StringIO(payload))
    lower_columns = {str(column).strip().lower(): column for column in df.columns}
    date_column = lower_columns.get("date") or lower_columns.get("day")
    value_column = lower_columns.get("gpr") or lower_columns.get("gprd") or lower_columns.get("gpr_daily")
    if date_column is None or value_column is None:
        raise ValueError("GPR table must contain date and gpr/GPRD columns")
    out = df[[date_column, value_column]].rename(columns={date_column: "date", value_column: "gpr"})
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["gpr"] = pd.to_numeric(out["gpr"], errors="coerce")
    return out.dropna(subset=["gpr"]).sort_values("date").reset_index(drop=True)


def detect_gpr_spikes(df: pd.DataFrame, *, min_history_days: int, stddev_threshold: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    values = df["gpr"].astype(float)
    rolling_mean = values.shift(1).rolling(min_history_days, min_periods=min_history_days).mean()
    rolling_std = values.shift(1).rolling(min_history_days, min_periods=min_history_days).std(ddof=0)
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
    text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload
    df = pd.read_csv(io.StringIO(text))
    lower_columns = {str(column).strip().lower(): column for column in df.columns}
    date_column = lower_columns.get("date")
    count_column = lower_columns.get("event_count") or lower_columns.get("count")
    if date_column is None or count_column is None:
        raise ValueError("GDELT volume table must contain date and event_count/count columns")
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
                "dominant_theme": str(record.get(theme_column, "geopolitical volume spike")) if theme_column else "geopolitical volume spike",
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
            event_date = dt.datetime.strptime(columns[GDELT_SQLDATE_IDX], "%Y%m%d").date()
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
                "source_url": _source_url_or_export_url(columns[GDELT_SOURCE_URL_IDX], source_url),
            },
        )
        current["event_count"] = int(current["event_count"]) + mentions
    return [row for _, row in sorted(totals.items()) if int(row["event_count"]) > 0]


def parse_acled_events(payload: str | bytes, *, source_url: str) -> list[dict[str, object]]:
    records = _json_records(payload, container_keys=("data",))
    totals: dict[dt.date, dict[str, object]] = {}
    for record in records:
        event_date = _parse_date(record.get("event_date"))
        if event_date is None:
            continue
        current = totals.setdefault(
            event_date,
            {
                "date": event_date,
                "event_count": 0,
                "fatalities": 0,
                "event_types": [],
                "countries": [],
                "source_url": source_url,
            },
        )
        current["event_count"] = int(current["event_count"]) + 1
        current["fatalities"] = int(current["fatalities"]) + _parse_positive_int(str(record.get("fatalities", "0")), default=0)
        _append_unique(current["event_types"], record.get("event_type"))
        _append_unique(current["countries"], record.get("country"))
    return [_summary_row(row, prefix="ACLED") for _, row in sorted(totals.items())]


def parse_ucdp_events(payload: str | bytes, *, source_url: str) -> list[dict[str, object]]:
    records = _json_records(payload, container_keys=("Result", "result", "data"))
    totals: dict[dt.date, dict[str, object]] = {}
    for record in records:
        event_date = _parse_date(record.get("date_start") or record.get("date") or record.get("event_date"))
        if event_date is None:
            continue
        current = totals.setdefault(
            event_date,
            {
                "date": event_date,
                "event_count": 0,
                "fatalities": 0,
                "event_types": [],
                "countries": [],
                "source_url": record.get("source_article") or source_url,
            },
        )
        current["event_count"] = int(current["event_count"]) + 1
        current["fatalities"] = int(current["fatalities"]) + _ucdp_fatalities(record)
        if record.get("type_of_violence"):
            _append_unique(current["event_types"], "organized violence")
        _append_unique(current["countries"], record.get("country"))
    return [_summary_row(row, prefix="UCDP") for _, row in sorted(totals.items())]


def parse_hdx_hapi_conflict_events(payload: str | bytes, *, source_url: str) -> list[dict[str, object]]:
    records = _json_records(payload, container_keys=("data",))
    rows: list[dict[str, object]] = []
    for record in records:
        period_start = _parse_date(record.get("reference_period_start"))
        if period_start is None:
            continue
        event_type = str(record.get("event_type") or "conflict_events")
        location = str(record.get("location_name") or record.get("location_code") or "unknown")
        rows.append(
            {
                "date": period_start,
                "event_count": _parse_positive_int(str(record.get("events", "0")), default=0),
                "fatalities": _parse_positive_int(str(record.get("fatalities", "0")), default=0),
                "dominant_theme": f"HDX HAPI monthly {event_type}: {location}",
                "source_url": source_url,
            }
        )
    return [row for row in rows if int(row["event_count"]) > 0 or int(row["fatalities"]) > 0]


def _decode_gdelt_export(payload: str | bytes) -> str:
    if isinstance(payload, str):
        return payload
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            first_name = archive.namelist()[0]
            return archive.read(first_name).decode("utf-8", errors="replace")
    return payload.decode("utf-8", errors="replace")


def _parse_positive_int(value: str, *, default: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _json_records(payload: str | bytes, *, container_keys: tuple[str, ...]) -> list[dict[str, object]]:
    text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [record for record in parsed if isinstance(record, dict)]
    if isinstance(parsed, dict):
        for key in container_keys:
            value = parsed.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
    return []


def _parse_date(value: object) -> dt.date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


def _ucdp_fatalities(record: dict[str, object]) -> int:
    if "best" in record:
        return _parse_positive_int(str(record["best"]), default=0)
    return sum(
        _parse_positive_int(str(record.get(field, "0")), default=0)
        for field in ("deaths_a", "deaths_b", "deaths_civilians", "deaths_unknown")
    )


def _summary_row(row: dict[str, object], *, prefix: str) -> dict[str, object]:
    event_types = row.pop("event_types")
    countries = row.pop("countries")
    theme = " / ".join(event_types) if event_types else "conflict events"
    location = ", ".join(countries[:3]) if countries else "unknown"
    row["dominant_theme"] = f"{prefix} {theme}: {location}"
    return row


def _append_unique(items: list[str], value: object) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text and text not in items:
        items.append(text)


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


def _signal_candidate(row: dict[str, object], *, source_id: str, event_subtype: str) -> EventCandidate:
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


def _anchor_date(gpr_date: dt.date, gdelt_dates: set[dt.date], window_days: int) -> dt.date:
    nearby = sorted(date for date in gdelt_dates if abs((date - gpr_date).days) <= window_days)
    return gpr_date if gpr_date in nearby or not nearby else gpr_date


def _has_nearby(event_date: dt.date, dates: set[dt.date], window_days: int) -> bool:
    return any(abs((date - event_date).days) <= window_days for date in dates)


def _window_dates(anchor: dt.date, window_days: int) -> list[dt.date]:
    return [anchor + dt.timedelta(days=offset) for offset in range(-window_days, window_days + 1)]


def _fetch_gpr_daily() -> bytes:
    request = Request(GPR_DAILY_URL, headers={"User-Agent": "regime-detection-event-fetch/1.0"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def _fetch_gdelt_daily_export(day: dt.date) -> bytes:
    request = Request(GDELT_DAILY_EXPORT_URL_TEMPLATE.format(date=day), headers={"User-Agent": "regime-detection-event-fetch/1.0"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def _fetch_acled_events(start_year: int, end_year: int) -> str | None:
    token = _acled_access_token()
    if token is None:
        LOGGER.error("ACLED credentials unavailable; set ACLED_API_TOKEN or ACLED_USERNAME/ACLED_PASSWORD to fetch ACLED geopolitical events")
        return None
    params = {
        "_format": "json",
        "event_date": f"{start_year}-01-01|{end_year}-12-31",
        "event_date_where": "BETWEEN",
        "fields": "event_date|event_type|sub_event_type|country|fatalities|source|notes",
        "limit": "5000",
    }
    records: list[dict[str, object]] = []
    page = 1
    while True:
        payload = _http_text(
            f"{ACLED_READ_URL}?{urlencode({**params, 'page': str(page)})}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        page_records = _json_records(payload, container_keys=("data",))
        records.extend(page_records)
        if len(page_records) < 5000:
            break
        page += 1
    return json.dumps({"data": records}, sort_keys=True)


def _acled_access_token() -> str | None:
    token = os.environ.get("ACLED_API_TOKEN", "").strip()
    if token:
        return token
    username = os.environ.get("ACLED_USERNAME", "").strip()
    password = os.environ.get("ACLED_PASSWORD", "").strip()
    if not username or not password:
        return None
    body = urlencode({"username": username, "password": password, "grant_type": "password", "client_id": "acled", "scope": "authenticated"}).encode()
    request = Request(ACLED_TOKEN_URL, data=body, headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "regime-detection-event-fetch/1.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["access_token"])


def _fetch_ucdp_events(start_year: int, end_year: int) -> str | None:
    token = os.environ.get("UCDP_ACCESS_TOKEN", "").strip()
    if not token:
        LOGGER.error("UCDP token unavailable; set UCDP_ACCESS_TOKEN to fetch UCDP GED Candidate geopolitical events")
        return None
    return _fetch_paged_json(
        UCDP_GED_CANDIDATE_URL,
        headers={"x-ucdp-access-token": token},
        result_key="Result",
        extra_params={"StartDate": f"{start_year}-01-01", "EndDate": f"{end_year}-12-31"},
    )


def _fetch_hdx_hapi_conflict_events(start_year: int, end_year: int) -> str | None:
    app_identifier = _hdx_hapi_app_identifier()
    if app_identifier is None:
        LOGGER.error(
            "HDX HAPI app identifier unavailable; set HDX_HAPI_APP_IDENTIFIER "
            "or HDX_HAPI_APP_NAME and HDX_HAPI_APP_EMAIL to fetch conflict events"
        )
        return None
    return _fetch_paged_json(
        HDX_HAPI_CONFLICT_EVENTS_URL,
        headers={},
        result_key="data",
        extra_params={
            "output_format": "json",
            "app_identifier": app_identifier,
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
        },
    )


def _hdx_hapi_app_identifier() -> str | None:
    for env_var in ("HDX_HAPI_APP_IDENTIFIER", "HDX_APP_IDENTIFIER"):
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    app_name = os.environ.get("HDX_HAPI_APP_NAME", "").strip()
    app_email = os.environ.get("HDX_HAPI_APP_EMAIL", "").strip()
    if not app_name or not app_email:
        return None
    return base64.b64encode(f"{app_name}:{app_email}".encode("utf-8")).decode("ascii")


def _fetch_paged_json(
    base_url: str,
    *,
    headers: dict[str, str],
    result_key: str,
    extra_params: dict[str, str],
) -> str:
    records: list[dict[str, object]] = []
    page = 1
    while True:
        params = {**extra_params, "pagesize": str(CONFLICT_API_PAGE_SIZE), "limit": str(CONFLICT_API_PAGE_SIZE), "page": str(page), "offset": str((page - 1) * CONFLICT_API_PAGE_SIZE)}
        payload = json.loads(_http_text(f"{base_url}?{urlencode(params)}", headers=headers))
        page_records = payload.get(result_key, []) if isinstance(payload, dict) else []
        if not isinstance(page_records, list):
            break
        records.extend(record for record in page_records if isinstance(record, dict))
        total_count = _payload_total_count(payload)
        if len(page_records) < CONFLICT_API_PAGE_SIZE:
            if total_count is not None and len(records) < total_count:
                raise RuntimeError(
                    f"{base_url} returned short page before TotalCount was satisfied: "
                    f"records={len(records)} total_count={total_count} page={page}"
                )
            break
        if total_count is not None and len(records) >= total_count:
            break
        page += 1
    return json.dumps({result_key: records}, sort_keys=True)


def _payload_total_count(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in ("TotalCount", "total_count", "count"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _http_text(url: str, *, headers: dict[str, str]) -> str:
    request = Request(url, headers={"User-Agent": "regime-detection-event-fetch/1.0", **headers})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _fetch_optional_conflict_rows(
    fetcher: ConflictFetcher,
    start_year: int,
    end_year: int,
    *,
    store: AcquisitionStore | None,
    run_id: int | None,
    source_id: str,
    source_identifier: str,
    source_url: str,
    parser: Callable[[str | bytes], list[dict[str, object]]],
) -> list[dict[str, object]]:
    try:
        payload = fetcher(start_year, end_year)
    except Exception as exc:  # pragma: no cover - exercised via integration degradation
        LOGGER.error("%s fetch failed; geopolitical candidates skipped: %s", source_id, exc)
        return []
    if payload is None:
        return []
    _record_payload(store, run_id, source_id, source_identifier, payload, f"{source_id} geopolitical event data")
    return [row for row in parser(payload, source_url=source_url) if start_year <= row["date"].year <= end_year]


def _record_payload(
    store: AcquisitionStore | None,
    run_id: int | None,
    source_name: str,
    source_identifier: str,
    payload: str | bytes,
    notes: str,
) -> None:
    if store is None or run_id is None:
        return
    text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload
    store.record_text_artifact(
        run_id=run_id,
        source_name=source_name,
        artifact_kind="text",
        source_identifier=source_identifier,
        content_text=text,
        calendar_assumption="calendar day geopolitical signal",
        timezone="UTC",
        license_note="Public geopolitical-risk signal source",
        notes=notes,
    )
