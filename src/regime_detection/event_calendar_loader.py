from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import pandas as pd
import yaml
from opentelemetry.trace import use_span

from regime_detection._loader_utils import (
    column_values,
    is_missing,
    read_csv_dataframe,
)
from regime_detection.observability import (
    capture_exception,
    record_timing,
    start_span,
    tracer,
)
from regime_detection.temporal import parse_date_series

LOGGER = logging.getLogger(__name__)
_TRACER = tracer(__name__)

_SCHEDULED_TYPES = {
    "FOMC",
    "CPI",
    "NFP",
    "election",
    "budget",
    "global_rate_decision",
    "ECB_decision",
    "BOE_decision",
    "BOJ_decision",
}
_V2_MANUAL_TYPES = {
    "budget",
    "election",
    "geopolitical_event",
    "global_rate_decision",
    "ECB_decision",
    "BOE_decision",
    "BOJ_decision",
}
_ALLOWED_TYPES = _SCHEDULED_TYPES | _V2_MANUAL_TYPES | {"ad_hoc"}


def _none_series(index: pd.Index) -> pd.Series:
    return pd.Series([None] * len(index), index=index, dtype="object")


def _matches_event_market(value: object, market: str) -> bool:
    return not is_missing(value) and value in {market, "GLOBAL"}


def _rows_from_yaml_payload(loaded: object) -> list[object]:
    if isinstance(loaded, dict):
        payload = cast(dict[str, object], loaded)
        rows = payload.get("events", [])
        if isinstance(rows, list):
            return list(cast(list[object], rows))
        raise ValueError("event_calendar YAML `events` must be a list")
    if isinstance(loaded, list):
        return list(cast(list[object], loaded))
    raise ValueError("event_calendar YAML must be a list or dict with `events`")


def load_event_calendar(
    source: str | Path | pd.DataFrame,
    *,
    market: str = "US",
) -> pd.DataFrame:
    start_time = time.perf_counter()
    span = start_span(
        _TRACER,
        "load_event_calendar",
        attributes={"market": market, "source_type": type(source).__name__},
    )
    with use_span(span, end_on_exit=True):
        try:
            if isinstance(source, pd.DataFrame):
                frame = _validate_event_df(source, market=market)
                span.set_attribute("rows", len(frame))
                return frame

            path = Path(source)
            span.set_attribute("source_path", str(path))
            if path.suffix.lower() in {".yaml", ".yml"}:
                loaded = yaml.safe_load(path.read_text())
                rows = _rows_from_yaml_payload(loaded)
                frame = _validate_event_df(pd.DataFrame(rows or []), market=market)
                span.set_attribute("rows", len(frame))
                return frame
            if path.suffix.lower() == ".csv":
                frame = _validate_event_df(read_csv_dataframe(path), market=market)
                span.set_attribute("rows", len(frame))
                return frame
            raise ValueError(f"Unsupported event calendar source: {source}")
        except Exception as error:
            capture_exception(
                error,
                logger=LOGGER,
                component="load_event_calendar",
                extra={"market": market},
            )
            span.record_exception(error)
            raise
        finally:
            record_timing("load_event_calendar", start_time)


def _validate_event_df(df: pd.DataFrame, *, market: str) -> pd.DataFrame:
    required = {"date", "market", "type", "importance"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"event_calendar missing required columns: {missing}")

    market_values = column_values(df, "market")
    market_mask = [_matches_event_market(value, market) for value in market_values]
    out = df.loc[market_mask].copy()

    type_values = [str(value) for value in column_values(out, "type")]
    out.loc[:, "type"] = pd.Series(type_values, index=out.index, dtype="object")
    bad_types = sorted(set(type_values) - _ALLOWED_TYPES)
    if bad_types:
        raise ValueError(
            f"event_calendar contains unsupported types for V1: {bad_types}"
        )

    parsed_dates = parse_date_series(
        column_values(out, "date"),
        field_name="date",
        context="event_calendar",
    )
    out.loc[:, "date"] = parsed_dates
    if "window_days" not in out.columns:
        out.loc[:, "window_days"] = _none_series(out.index)
    else:
        out.loc[:, "window_days"] = pd.Series(
            [_parse_window_days(value) for value in column_values(out, "window_days")],
            index=out.index,
            dtype="object",
        )
    if "publication_date" in out.columns:
        out.loc[:, "publication_date"] = parse_date_series(
            column_values(out, "publication_date"),
            field_name="publication_date",
            context="event_calendar",
            nullable=True,
        )
    else:
        out.loc[:, "publication_date"] = _none_series(out.index)
    if "approved_label" not in out.columns:
        out.loc[:, "approved_label"] = _none_series(out.index)
    else:
        approved = [
            None if is_missing(value) else value
            for value in column_values(out, "approved_label")
        ]
        out.loc[:, "approved_label"] = pd.Series(
            approved,
            index=out.index,
            dtype="object",
        )

    normalized_publication_dates: list[date] = []
    for event_type, event_date_value, publication_date_value in zip(
        type_values,
        list(parsed_dates),
        column_values(out, "publication_date"),
        strict=True,
    ):
        if not isinstance(event_date_value, date):
            raise ValueError("event_calendar contains malformed date values")
        if publication_date_value is None or is_missing(publication_date_value):
            if event_type in _SCHEDULED_TYPES:
                # ADR 0002 §52 + ADR 0014 R3: scheduled events default
                # publication_date to date - 90 calendar days when not
                # supplied. ADR 0002 §52 authorizes this for FOMC/CPI/NFP;
                # ADR 0014 R3 extends it to V2 scheduled types
                # (ECB/BOE/BOJ/election/budget/global_rate_decision).
                normalized_publication_dates.append(
                    event_date_value - timedelta(days=90)
                )
            else:
                normalized_publication_dates.append(event_date_value)
            continue
        if not isinstance(publication_date_value, date):
            raise ValueError(
                "event_calendar contains malformed publication_date values"
            )
        normalized_publication_dates.append(publication_date_value)

    out.loc[:, "publication_date"] = pd.Series(
        normalized_publication_dates,
        index=out.index,
        dtype="object",
    )

    return (
        out[
            [
                "date",
                "market",
                "type",
                "importance",
                "publication_date",
                "window_days",
                "approved_label",
            ]
        ]
        .sort_values(["date", "type"])
        .reset_index(drop=True)
    )


def _parse_window_days(value: object) -> list[int] | None:
    if value is None or (isinstance(value, float) and is_missing(value)):
        return None
    if isinstance(value, str):
        parsed = yaml.safe_load(value)
    else:
        parsed = value
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(
            f"event_calendar window_days must be a two-item list: {value!r}"
        )
    parsed_sequence = cast(Sequence[object], parsed)
    if len(parsed_sequence) != 2:
        raise ValueError(
            f"event_calendar window_days must be a two-item list: {value!r}"
        )
    try:
        window = [
            _window_day_int(parsed_sequence[0]),
            _window_day_int(parsed_sequence[1]),
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"event_calendar window_days entries must be integers: {value!r}"
        ) from exc
    # F-037: window_days is an ordered [start_offset, end_offset] pair; a reversed
    # window (start > end) is silently empty downstream and would suppress the event's
    # influence. Reject it at load time, mirroring the load_scheduled_events guard.
    if window[0] > window[1]:
        raise ValueError(
            f"event_calendar window_days must have start <= end: {value!r}"
        )
    return window


def _window_day_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("window_days entries must be integers")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("window_days entries must be integers")
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError("window_days entries must be integers")
