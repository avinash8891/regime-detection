from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from opentelemetry.trace import use_span

from regime_detection.observability import (
    capture_exception,
    record_timing,
    start_span,
    tracer,
)
from regime_detection.temporal import (
    parse_date_series,
    parse_datetime_index,
    parse_datetime_series,
)

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
_PANDAS_READ_CSV = cast(Callable[[Path], pd.DataFrame], cast(Any, pd).read_csv)
_PANDAS_READ_PARQUET = cast(Callable[[Path], pd.DataFrame], cast(Any, pd).read_parquet)


def _read_csv_dataframe(path: Path) -> pd.DataFrame:
    return _PANDAS_READ_CSV(path)


def _column_values(frame: pd.DataFrame, column: str) -> list[object]:
    return list(frame[column])


def _is_missing(value: object) -> bool:
    return bool(cast(Any, pd).isna(value))


def _numeric_value(value: object, *, field_name: str, context: str) -> float:
    if value is None or _is_missing(value) or isinstance(value, bool):
        raise ValueError(f"{context} contains non-numeric {field_name} values")
    try:
        numeric_value = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} contains non-numeric {field_name} values") from exc
    if _is_missing(numeric_value):
        raise ValueError(f"{context} contains non-numeric {field_name} values")
    return numeric_value


def _dated_float_series(
    rows: list[tuple[object, float]],
    *,
    context: str,
    series_name: str,
) -> pd.Series:
    index = parse_datetime_index(
        [raw_date for raw_date, _ in rows],
        field_name="date",
        context=context,
    )
    return pd.Series(
        [numeric_value for _, numeric_value in rows],
        index=index,
        name=series_name,
        dtype=float,
    ).sort_index()


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
                frame = _validate_event_df(_read_csv_dataframe(path), market=market)
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


def _read_source(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Resolve a loader source (parquet/CSV path, parquet dir, or DataFrame)
    to a DataFrame. Shared by every `load_*` helper in this module."""
    if isinstance(source, pd.DataFrame):
        return source
    path = Path(source)
    if path.suffix.lower() == ".parquet" or path.is_dir():
        return _PANDAS_READ_PARQUET(path)
    if path.suffix.lower() == ".csv":
        return _read_csv_dataframe(path)
    raise ValueError(f"Unsupported source: {source}")


def _load_long_form_closes(
    source: str | Path | pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    universe: tuple[str, ...] | list[str] | None,
) -> dict[str, pd.Series]:
    """Shared backend for the v2 long-form parquet/CSV/DataFrame loaders.

    Reads a long-form table with columns `(date, group_col, value_col)`
    (e.g. `(date, symbol, close)` for sector ETFs or `(date, series_id, value)`
    for FRED macros) and returns one date-indexed Series per group.
    """
    df = _read_source(source)

    required_cols = {"date", group_col, value_col}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Source missing required columns: {missing}")

    group_values = _column_values(df, group_col)
    date_values = _column_values(df, "date")
    raw_value_values = _column_values(df, value_col)

    if universe is not None:
        present = {str(value) for value in group_values if not _is_missing(value)}
        absent = [s for s in universe if s not in present]
        if absent:
            raise ValueError(f"Source missing required {group_col}s: {absent}")
        allowed_groups = set(universe)
    else:
        allowed_groups = None

    grouped_rows: dict[str, list[tuple[object, float]]] = {}
    for raw_group, raw_date, raw_value in zip(
        group_values, date_values, raw_value_values, strict=True
    ):
        if _is_missing(raw_group):
            continue
        group_key = str(raw_group)
        if allowed_groups is not None and group_key not in allowed_groups:
            continue
        context = f"Source for {group_col}={group_key!r}"
        grouped_rows.setdefault(group_key, []).append(
            (
                raw_date,
                _numeric_value(raw_value, field_name=value_col, context=context),
            )
        )

    out: dict[str, pd.Series] = {}
    for group_key, rows in grouped_rows.items():
        out[group_key] = _dated_float_series(
            rows,
            context=f"Source for {group_col}={group_key!r}",
            series_name=value_col,
        )
    return out


def load_sector_etf_closes(
    source: str | Path | pd.DataFrame,
    *,
    universe: tuple[str, ...] | list[str] | None = None,
) -> dict[str, pd.Series]:
    """Load close prices for sector ETFs (v2 spec §3.1).

    Source schema: long-form parquet/CSV/DataFrame with columns
    `(date, symbol, close)` (and optionally OHLCV — extra columns are ignored).
    Returns one date-indexed Series per symbol.
    """
    result = _load_long_form_closes(
        source,
        group_col="symbol",
        value_col="close",
        universe=universe,
    )
    if not result:
        LOGGER.warning("load_sector_etf_closes returned 0 symbols from source")
    return result


def load_cross_asset_closes(
    source: str | Path | pd.DataFrame,
    *,
    universe: tuple[str, ...] | list[str] | None = None,
) -> dict[str, pd.Series]:
    """Load close prices for cross-asset proxies (v2 spec §3.1).

    Same source schema as `load_sector_etf_closes`. Kept as a separate
    public function so callers can express their intent explicitly and so
    each side can grow its own validation later without affecting the other.
    """
    result = _load_long_form_closes(
        source,
        group_col="symbol",
        value_col="close",
        universe=universe,
    )
    if not result:
        LOGGER.warning("load_cross_asset_closes returned 0 symbols from source")
    return result


def load_macro_series(
    source: str | Path | pd.DataFrame,
    *,
    series_ids: tuple[str, ...] | list[str] | None = None,
) -> dict[str, pd.Series]:
    """Load FRED macro series (v2 spec §2A / §2B / §2C).

    Source schema: long-form parquet/CSV/DataFrame with columns
    `(date, series_id, value)` (and optionally `realtime_start`,
    `realtime_end`, `logical_name` — extra columns are ignored).
    Returns one Series per canonical key: ``logical_name`` when the column is
    present and non-null for a series, otherwise the FRED ``series_id``.
    Each series appears exactly once — no duplicate aliases.
    """
    df = _read_source(source)
    out = _load_long_form_closes(
        df,
        group_col="series_id",
        value_col="value",
        universe=series_ids,
    )
    out = {key: series.rename(key) for key, series in out.items()}

    if "logical_name" in df.columns:
        allowed_series_ids = set(series_ids) if series_ids is not None else None
        logical_name_values = _column_values(df, "logical_name")
        series_id_values = _column_values(df, "series_id")
        date_values = _column_values(df, "date")
        raw_value_values = _column_values(df, "value")

        logical_rows: dict[str, list[tuple[object, float]]] = {}
        logical_series_ids: set[str] = set()
        for raw_series_id, raw_logical_name, raw_date, raw_value in zip(
            series_id_values,
            logical_name_values,
            date_values,
            raw_value_values,
            strict=True,
        ):
            if _is_missing(raw_logical_name):
                continue
            series_id = str(raw_series_id)
            if allowed_series_ids is not None and series_id not in allowed_series_ids:
                continue
            logical_name = str(raw_logical_name)
            value = _numeric_value(
                raw_value,
                field_name="value",
                context=f"macro logical_name={logical_name!r}",
            )
            if logical_name == "implied_vol_30d":
                value = value / 100.0
            logical_series_ids.add(series_id)
            logical_rows.setdefault(logical_name, []).append((raw_date, value))

        for sid in logical_series_ids:
            out.pop(sid, None)
        for logical_name, rows in logical_rows.items():
            out[logical_name] = _dated_float_series(
                rows,
                context=f"macro logical_name={logical_name!r}",
                series_name=logical_name,
            )

    if not out:
        LOGGER.warning("load_macro_series returned 0 series from source")
    return out


def load_cpi_nowcast_series(source: str | Path | pd.DataFrame) -> pd.Series:
    """Load the Cleveland Fed inflation-nowcast series (v2 §2B, ADR 0006).

    Source schema: wide-form parquet/CSV/DataFrame with columns
    `(date, cpi_nowcast)` — the output of
    `regime_data_fetch.cleveland_fed_nowcast.run_cleveland_fed_nowcast_fetch`.
    Returns a single date-indexed Series for
    `MarketContext.macro_series["cpi_nowcast"]`, which feeds the §2B
    `inflation_surprise_zscore` (the `inflation_shock` single-signal limb).
    """
    df = _read_source(source)
    missing = sorted({"date", "cpi_nowcast"} - set(df.columns))
    if missing:
        raise ValueError(f"cpi_nowcast source missing required columns: {missing}")
    df = df.sort_values("date")
    return pd.Series(
        df["cpi_nowcast"].astype(float).to_numpy(),
        index=parse_datetime_index(
            df["date"], field_name="date", context="cpi_nowcast source"
        ),
        name="cpi_nowcast",
    )


def load_aggregate_forward_eps_revision_series(
    source: str | Path | pd.DataFrame,
) -> pd.Series:
    """Load the 4-week aggregate forward-EPS revision-direction series
    (v2 §2B, documented implementation decision).

    Source schema: the weekly-snapshot accumulator
    (`sp500_eps_weekly_history.parquet`) with columns
    `(observation_date, forward_estimate_value, ...)`. The revision series
    is derived via `compute_eps_revision_direction_4w` — all-NaN until the
    accumulator holds more than `EPS_REVISION_LOOKBACK_WEEKS` rows. Returns a
    date-indexed Series for
    `MarketContext.macro_series["aggregate_forward_eps_revision"]`, which
    feeds the §2B `earnings_expansion` / `earnings_contraction` labels.
    """
    from regime_data_fetch.aggregate_eps import compute_eps_revision_direction_4w

    df = _read_source(source)
    missing = sorted({"observation_date", "forward_estimate_value"} - set(df.columns))
    if missing:
        raise ValueError(
            f"aggregate forward EPS source missing required columns: {missing}"
        )
    return compute_eps_revision_direction_4w(df)


def load_central_bank_text_score(
    *,
    fomc_minutes_source: str | Path | pd.DataFrame | None = None,
    powell_speeches_source: str | Path | pd.DataFrame | None = None,
    max_release_age_days: int | None = None,
    as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load FOMC minutes + Powell speech parquets and score each release.

    Returns a per-release frame with columns ``release_date``,
    ``hawkish_count``, ``dovish_count``, ``total_tokens``, ``net_score``,
    ``source`` — the input to
    ``central_bank_text.to_daily_score_series``. When neither source is
    supplied, returns the empty frame (the engine then sees all-NaN
    daily series and the §2A evidence column is silent).

    Per V2 §2A line 2950 the score feeds ``monetary_pressure.evidence``
    only — this loader has no awareness of rule predicates.
    """
    from regime_detection.central_bank_text import (
        combine_release_frames,
        score_release_frame,
    )

    frames: list[pd.DataFrame] = []
    if fomc_minutes_source is not None:
        df = _read_source(fomc_minutes_source)
        # FOMC parquet column from regime_data_fetch.fomc_minutes is
        # ``release_timestamp`` (datetime). The score scaffold treats it
        # as the release date.
        date_column = (
            "release_timestamp" if "release_timestamp" in df.columns else "release_date"
        )
        frames.append(
            score_release_frame(
                df, date_column=date_column, source_label="fomc_minutes"
            )
        )
    if powell_speeches_source is not None:
        df = _read_source(powell_speeches_source)
        # Powell parquet from regime_data_fetch.powell_speeches uses
        # ``publication_timestamp`` (date-only precision per repo notes).
        date_column = (
            "publication_timestamp"
            if "publication_timestamp" in df.columns
            else "publication_date"
        )
        frames.append(
            score_release_frame(
                df, date_column=date_column, source_label="powell_speech"
            )
        )
    combined = combine_release_frames(*frames)
    if combined.empty:
        return combined
    if max_release_age_days is not None and as_of_date is not None:
        cutoff = (
            pd.Timestamp(as_of_date).date()
            - pd.Timedelta(days=max_release_age_days).to_pytimedelta()
        )
        combined = combined[combined["release_date"] >= cutoff].reset_index(drop=True)
    return combined


def load_news_sentiment_series(
    source: str | Path | pd.DataFrame,
) -> pd.Series:
    """Load the SF Fed Daily News Sentiment Index as a date-indexed Series.

    Source schema: long-form parquet (or CSV/DataFrame) written by
    ``regime_data_fetch.sf_fed_news_sentiment`` with columns
    ``date`` and ``news_sentiment`` (extra columns ``source``,
    ``source_url`` are ignored). Returns a Series indexed by date
    suitable for ``MarketContext.news_sentiment``.

    Used as v2 §1A evidence ONLY — never consumed by the `euphoria`
    rule predicate. The §1A `sentiment_score` (AAII bull-bear 8w-MA)
    remains the canonical input to that rule.
    """
    df = _read_source(source)
    required = {"date", "news_sentiment"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"news_sentiment source missing required columns: {missing}")
    df = df.sort_values("date")
    return pd.Series(
        df["news_sentiment"].astype(float).to_numpy(),
        index=parse_datetime_index(
            df["date"], field_name="date", context="news_sentiment source"
        ),
        name="news_sentiment",
    )


def load_cpi_vintages_first_release(
    source: str | Path | pd.DataFrame,
) -> pd.Series:
    """Load the first-release CPI series from a FRED vintages parquet.

    Spec: V2 §2A line 2956 (cross-ref L2672) — "Original release values
    are point-in-time-correct;
    revised values are not. The engine must use original values for
    historical replay."

    Source schema: long-form ``cpi_all_items_vintages.parquet`` written
    by ``regime_data_fetch.fred`` with realtime params. Each row has at
    minimum:

        date              (the reference date; typically the 1st of the
                           reference month for CPIAUCSL)
        value             (the published level for that reference date
                           in that vintage)
        realtime_start    (the date this value first became public)
        realtime_end      (the date this value was superseded by a
                           revision, or NaT if still current)

    For each reference ``date``, this loader picks the row with the
    **earliest** ``realtime_start`` — the first-release value — and
    returns a Series keyed by that ``realtime_start`` (the *release
    date*, not the reference date). Historical replay then looks up
    each ``as_of_date`` against the release-date index and forward-fills.

    Falls back to a release-date-keyed Series of NaN when the source
    is empty.
    """
    df = _read_source(source)
    required = {"date", "value", "realtime_start"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"cpi_vintages source missing required columns: {missing}")
    if df.empty:
        return pd.Series([], dtype=float, name="cpi_first_release")
    work = df.copy()
    work.loc[:, "date"] = parse_datetime_series(
        _column_values(work, "date"),
        field_name="date",
        context="cpi_vintages source",
    )
    work.loc[:, "realtime_start"] = parse_datetime_series(
        _column_values(work, "realtime_start"),
        field_name="realtime_start",
        context="cpi_vintages source",
    )
    # Earliest realtime_start per reference date = the first release.
    first_releases = (
        work.sort_values(["date", "realtime_start"])
        .drop_duplicates(subset="date", keep="first")
        .reset_index(drop=True)
    )
    # The replay series is keyed by release date and then forward-filled
    # onto trading sessions. If the upstream vintage file contains multiple
    # reference periods with the same earliest realtime_start, keep the most
    # recent reference period available on that release date so the as-of
    # index remains unique and reindex-safe.
    first_releases = (
        first_releases.sort_values(["realtime_start", "date"])
        .drop_duplicates(subset="realtime_start", keep="last")
        .reset_index(drop=True)
    )
    series = pd.Series(
        first_releases["value"].astype(float).to_numpy(),
        index=pd.DatetimeIndex(first_releases["realtime_start"]),
        name="cpi_first_release",
    )
    return series.sort_index()


def _validate_event_df(df: pd.DataFrame, *, market: str) -> pd.DataFrame:
    required = {"date", "market", "type", "importance"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"event_calendar missing required columns: {missing}")

    market_values = _column_values(df, "market")
    market_mask = [(value == market) or (value == "GLOBAL") for value in market_values]
    out = df.loc[market_mask].copy()

    type_values = [str(value) for value in _column_values(out, "type")]
    out.loc[:, "type"] = pd.Series(type_values, index=out.index, dtype="object")
    bad_types = sorted(set(type_values) - _ALLOWED_TYPES)
    if bad_types:
        raise ValueError(
            f"event_calendar contains unsupported types for V1: {bad_types}"
        )

    parsed_dates = parse_date_series(
        _column_values(out, "date"),
        field_name="date",
        context="event_calendar",
    )
    out.loc[:, "date"] = parsed_dates
    if "window_days" not in out.columns:
        out.loc[:, "window_days"] = None
    else:
        out.loc[:, "window_days"] = pd.Series(
            [_parse_window_days(value) for value in _column_values(out, "window_days")],
            index=out.index,
            dtype="object",
        )
    if "publication_date" in out.columns:
        out.loc[:, "publication_date"] = parse_date_series(
            _column_values(out, "publication_date"),
            field_name="publication_date",
            context="event_calendar",
            nullable=True,
        )
    else:
        out.loc[:, "publication_date"] = None
    if "approved_label" not in out.columns:
        out.loc[:, "approved_label"] = None
    else:
        approved = [
            None if _is_missing(value) else value
            for value in _column_values(out, "approved_label")
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
        _column_values(out, "publication_date"),
        strict=True,
    ):
        if not isinstance(event_date_value, date):
            raise ValueError("event_calendar contains malformed date values")
        if publication_date_value is None or _is_missing(publication_date_value):
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
    if value is None or (isinstance(value, float) and _is_missing(value)):
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
        return [
            _window_day_int(parsed_sequence[0]),
            _window_day_int(parsed_sequence[1]),
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"event_calendar window_days entries must be integers: {value!r}"
        ) from exc


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
