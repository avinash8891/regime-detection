from __future__ import annotations

import datetime as dt
import io
import logging
from collections.abc import Callable
from urllib.request import Request, urlopen

import pandas as pd

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.event_sources.models import EventCandidate, ValidationResult

LOGGER = logging.getLogger(__name__)
GPR_SOURCE_ID = "gpr:caldara-iacoviello"
GDELT_SOURCE_ID = "gdelt:events-v2"
GPR_DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"


class GPRGDELTSignalGenerator:
    source_id = "gpr-gdelt:geopolitical-signals"
    validator_id = "gpr-gdelt:cross-source"

    def __init__(
        self,
        *,
        gpr_fetcher: Callable[[], str | bytes] | None = None,
        gdelt_fetcher: Callable[[], str | bytes] | None = None,
        min_history_days: int = 252,
        stddev_threshold: float = 3.0,
        merge_window_days: int = 2,
    ) -> None:
        self.gpr_fetcher = gpr_fetcher or _fetch_gpr_daily
        self.gdelt_fetcher = gdelt_fetcher
        self.min_history_days = min_history_days
        self.stddev_threshold = stddev_threshold
        self.merge_window_days = merge_window_days
        self._last_gpr_dates: set[dt.date] = set()
        self._last_gdelt_dates: set[dt.date] = set()

    def generate(
        self,
        *,
        start_year: int,
        end_year: int,
        store: AcquisitionStore | None,
        run_id: int | None,
    ) -> list[EventCandidate]:
        candidates: list[EventCandidate] = []
        gpr_spikes = self._fetch_gpr_spikes(store=store, run_id=run_id)
        gdelt_spikes = self._fetch_gdelt_spikes(store=store, run_id=run_id)
        self._last_gpr_dates = {row["date"] for row in gpr_spikes}
        self._last_gdelt_dates = {row["date"] for row in gdelt_spikes}

        for row in gdelt_spikes:
            if start_year <= row["date"].year <= end_year:
                candidates.append(_gdelt_candidate(row))
        for row in gpr_spikes:
            if start_year <= row["date"].year <= end_year:
                anchor = _anchor_date(row["date"], self._last_gdelt_dates, self.merge_window_days)
                candidates.append(_gpr_candidate(row, anchor))
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
            if candidate.source_id == GPR_SOURCE_ID and _has_nearby(candidate.date, self._last_gdelt_dates, self.merge_window_days):
                validations.append(ValidationResult(key, GDELT_SOURCE_ID, "confirm", candidate.source_url, "GDELT corroborated nearby geopolitical volume spike"))
            elif candidate.source_id == GDELT_SOURCE_ID and _has_nearby(candidate.date, self._last_gpr_dates, self.merge_window_days):
                validations.append(ValidationResult(key, GPR_SOURCE_ID, "confirm", candidate.source_url, "GPR corroborated nearby geopolitical risk spike"))
        return validations

    def _fetch_gpr_spikes(self, *, store: AcquisitionStore | None, run_id: int | None) -> list[dict[str, object]]:
        try:
            payload = self.gpr_fetcher()
        except Exception as exc:  # pragma: no cover - exercised via integration degradation
            LOGGER.error("GPR fetch failed; geopolitical GPR candidates skipped: %s", exc)
            return []
        _record_payload(store, run_id, GPR_SOURCE_ID, "gpr_daily", payload, "GPR daily index")
        return detect_gpr_spikes(parse_gpr_table(payload), min_history_days=self.min_history_days, stddev_threshold=self.stddev_threshold)

    def _fetch_gdelt_spikes(self, *, store: AcquisitionStore | None, run_id: int | None) -> list[dict[str, object]]:
        if self.gdelt_fetcher is None:
            return []
        try:
            payload = self.gdelt_fetcher()
        except Exception as exc:  # pragma: no cover - exercised via integration degradation
            LOGGER.error("GDELT fetch failed; geopolitical GDELT candidates skipped: %s", exc)
            return []
        _record_payload(store, run_id, GDELT_SOURCE_ID, "gdelt_geopolitical_volume", payload, "GDELT geopolitical volume sample")
        return parse_gdelt_volume_table(payload)


def parse_gpr_table(payload: str | bytes) -> pd.DataFrame:
    if isinstance(payload, bytes):
        try:
            df = pd.read_excel(io.BytesIO(payload))
        except Exception:
            df = pd.read_csv(io.BytesIO(payload))
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


def _anchor_date(gpr_date: dt.date, gdelt_dates: set[dt.date], window_days: int) -> dt.date:
    nearby = sorted(date for date in gdelt_dates if abs((date - gpr_date).days) <= window_days)
    return gpr_date if gpr_date in nearby or not nearby else gpr_date


def _has_nearby(event_date: dt.date, dates: set[dt.date], window_days: int) -> bool:
    return any(abs((date - event_date).days) <= window_days for date in dates)


def _fetch_gpr_daily() -> bytes:
    request = Request(GPR_DAILY_URL, headers={"User-Agent": "regime-detection-event-fetch/1.0"})
    with urlopen(request, timeout=30) as response:
        return response.read()


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
