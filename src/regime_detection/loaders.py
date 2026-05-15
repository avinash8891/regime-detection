from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yaml


LOG = logging.getLogger(__name__)

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
        series = pd.Series(
            sub[value_col].to_numpy(),
            index=pd.to_datetime(sub["date"]),
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
    `realtime_end` — extra columns are ignored).
    Returns one date-indexed Series per series_id.
    """
    return _load_long_form_closes(
        source, group_col="series_id", value_col="value", universe=series_ids,
    )


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
        index=pd.to_datetime(df["date"]),
        name="cpi_nowcast",
    )


def load_aggregate_forward_eps_revision_series(
    source: str | Path | pd.DataFrame,
) -> pd.Series:
    """Load the 4-week aggregate forward-EPS revision-direction series
    (v2 §2B, Ambiguity Log #48).

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

    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.date
    if "window_days" not in out.columns:
        out["window_days"] = None
    else:
        out["window_days"] = out["window_days"].apply(_parse_window_days)
    if "publication_date" in out.columns:
        provided_mask = out["publication_date"].notna()
        parsed_publication = pd.to_datetime(out["publication_date"], errors="coerce")
        bad_mask = provided_mask & parsed_publication.isna()
        if bad_mask.any():
            bad_values = sorted({str(value) for value in out.loc[bad_mask, "publication_date"].tolist()})
            raise ValueError(f"event_calendar contains malformed publication_date values: {bad_values}")
        out["publication_date"] = parsed_publication.dt.date.astype("object")
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
