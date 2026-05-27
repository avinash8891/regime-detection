from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportIndexIssue=false

import datetime as dt
import json


def parse_acled_events(
    payload: str | bytes, *, source_url: str
) -> list[dict[str, object]]:
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
        current["fatalities"] = int(current["fatalities"]) + _parse_positive_int(
            str(record.get("fatalities", "0")), default=0
        )
        _append_unique(current["event_types"], record.get("event_type"))
        _append_unique(current["countries"], record.get("country"))
    return [_summary_row(row, prefix="ACLED") for _, row in sorted(totals.items())]


def parse_ucdp_events(
    payload: str | bytes, *, source_url: str
) -> list[dict[str, object]]:
    records = _json_records(payload, container_keys=("Result", "result", "data"))
    totals: dict[dt.date, dict[str, object]] = {}
    for record in records:
        event_date = _parse_date(
            record.get("date_start") or record.get("date") or record.get("event_date")
        )
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


def _parse_positive_int(value: str, *, default: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _json_records(
    payload: str | bytes, *, container_keys: tuple[str, ...]
) -> list[dict[str, object]]:
    text = (
        payload.decode("utf-8", errors="replace")
        if isinstance(payload, bytes)
        else payload
    )
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
