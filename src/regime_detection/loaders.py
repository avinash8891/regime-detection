from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yaml

from regime_detection.temporal import (
    parse_date_series,
    parse_datetime_index,
    parse_datetime_series,
)


LOGGER = logging.getLogger(__name__)

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


def load_event_calendar(
    source: str | Path | pd.DataFrame,
    *,
    market: str = "US",
) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return _validate_event_df(source, market=market)

    path = Path(source)
    if path.suffix.lower() in {".yaml", ".yml"}:
        loaded = yaml.safe_load(path.read_text())
        if isinstance(loaded, dict):
            rows = loaded.get("events", [])
        else:
            rows = loaded
        return _validate_event_df(pd.DataFrame(rows or []), market=market)
    if path.suffix.lower() == ".csv":
        return _validate_event_df(pd.read_csv(path), market=market)
    raise ValueError(f"Unsupported event calendar source: {source}")


def _read_source(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Resolve a loader source (parquet/CSV path, parquet dir, or DataFrame)
    to a DataFrame. Shared by every `load_*` helper in this module."""
    if isinstance(source, pd.DataFrame):
        return source
    path = Path(source)
    if path.suffix.lower() == ".parquet" or path.is_dir():
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
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

    if universe is not None:
        present = set(df[group_col].unique())
        absent = [s for s in universe if s not in present]
        if absent:
            raise ValueError(f"Source missing required {group_col}s: {absent}")
        df = df[df[group_col].isin(universe)].copy()

    out: dict[str, pd.Series] = {}
    for key, sub in df.groupby(group_col):
        sub = sub.sort_values("date")
        index = parse_datetime_index(
            sub["date"],
            field_name="date",
            context=f"Source for {group_col}={key!r}",
        )
        try:
            values = pd.to_numeric(sub[value_col], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Source contains non-numeric {value_col} values for {group_col}={key!r}"
            ) from exc
        if values.isna().any():
            raise ValueError(
                f"Source contains non-numeric {value_col} values for {group_col}={key!r}"
            )
        series = pd.Series(
            values.to_numpy(),
            index=index,
            name=value_col,
        )
        out[str(key)] = series
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
    return _load_long_form_closes(
        source, group_col="symbol", value_col="close", universe=universe,
    )


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
    return _load_long_form_closes(
        source, group_col="symbol", value_col="close", universe=universe,
    )


def load_macro_series(
    source: str | Path | pd.DataFrame,
    *,
    series_ids: tuple[str, ...] | list[str] | None = None,
) -> dict[str, pd.Series]:
    """Load FRED macro series (v2 spec §2A / §2B / §2C).

    Source schema: long-form parquet/CSV/DataFrame with columns
    `(date, series_id, value)` (and optionally `realtime_start`,
    `realtime_end`, `logical_name` — extra columns are ignored).
    Returns date-indexed Series by FRED `series_id`. When the fetch-workflow
    `logical_name` column is present, also returns the engine-facing logical
    keys consumed by V2 feature seams.
    """
    out = _load_long_form_closes(
        source, group_col="series_id", value_col="value", universe=series_ids,
    )
    out = {key: series.rename(key) for key, series in out.items()}
    df = _read_source(source)

    if series_ids is not None:
        df = df[df["series_id"].isin(series_ids)].copy()

    if "logical_name" in df.columns:
        logical_df = df.dropna(subset=["logical_name"])
        for key, sub in logical_df.groupby("logical_name"):
            sub = sub.sort_values("date")
            series_key = str(key)
            out[series_key] = pd.Series(
                sub["value"].astype(float).to_numpy(),
                index=parse_datetime_index(
                    sub["date"],
                    field_name="date",
                    context=f"macro logical_name={series_key!r}",
                ),
                name=series_key,
            )

    if "DGS10" in out and "dgs10" not in out:
        out["dgs10"] = out["DGS10"].rename("dgs10")
    if "DGS2" in out and "dgs2" not in out:
        out["dgs2"] = out["DGS2"].rename("dgs2")
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
    missing = sorted(
        {"observation_date", "forward_estimate_value"} - set(df.columns)
    )
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

    Per V2 §2A line 2585 the score feeds ``monetary_pressure.evidence``
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
            "release_timestamp"
            if "release_timestamp" in df.columns
            else "release_date"
        )
        frames.append(
            score_release_frame(df, date_column=date_column, source_label="fomc_minutes")
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
            score_release_frame(df, date_column=date_column, source_label="powell_speech")
        )
    combined = combine_release_frames(*frames)
    if combined.empty:
        return combined
    if max_release_age_days is not None and as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date).date() - pd.Timedelta(days=max_release_age_days).to_pytimedelta()
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
        raise ValueError(
            f"news_sentiment source missing required columns: {missing}"
        )
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

    Spec: V2 §2A lines 2587-2593 — "Original release values are
    point-in-time-correct; revised values are not. The engine must use
    original values for historical replay."

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
        raise ValueError(
            f"cpi_vintages source missing required columns: {missing}"
        )
    if df.empty:
        return pd.Series([], dtype=float, name="cpi_first_release")
    work = df.copy()
    work["date"] = parse_datetime_series(
        work["date"], field_name="date", context="cpi_vintages source"
    )
    work["realtime_start"] = parse_datetime_series(
        work["realtime_start"],
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

    out = df.copy()
    out = out[(out["market"] == market) | (out["market"] == "GLOBAL")].copy()
    out["type"] = out["type"].astype(str)
    bad_types = sorted(set(out["type"]) - _ALLOWED_TYPES)
    if bad_types:
        raise ValueError(f"event_calendar contains unsupported types for V1: {bad_types}")

    out["date"] = parse_date_series(
        out["date"], field_name="date", context="event_calendar"
    )
    if "window_days" not in out.columns:
        out["window_days"] = None
    else:
        out["window_days"] = out["window_days"].apply(_parse_window_days)
    if "publication_date" in out.columns:
        out["publication_date"] = parse_date_series(
            out["publication_date"],
            field_name="publication_date",
            context="event_calendar",
            nullable=True,
        )
    else:
        out["publication_date"] = None
    if "approved_label" not in out.columns:
        out["approved_label"] = None
    else:
        approved = out["approved_label"].where(out["approved_label"].notna(), None)
        out["approved_label"] = approved.astype("object")

    for idx, row in out.iterrows():
        if pd.isna(row["publication_date"]):
            if row["type"] in _SCHEDULED_TYPES:
                out.at[idx, "publication_date"] = row["date"] - timedelta(days=90)
            else:
                out.at[idx, "publication_date"] = row["date"]

    return out[["date", "market", "type", "importance", "publication_date", "window_days", "approved_label"]].sort_values(
        ["date", "type"]
    ).reset_index(drop=True)


def _parse_window_days(value: object) -> list[int] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        parsed = yaml.safe_load(value)
    else:
        parsed = value
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        raise ValueError(f"event_calendar window_days must be a two-item list: {value!r}")
    try:
        return [int(parsed[0]), int(parsed[1])]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"event_calendar window_days entries must be integers: {value!r}"
        ) from exc
