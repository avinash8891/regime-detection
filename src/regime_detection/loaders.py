from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yaml


LOG = logging.getLogger(__name__)

_SCHEDULED_TYPES = {"FOMC", "CPI", "NFP"}
_ALLOWED_TYPES = _SCHEDULED_TYPES | {"ad_hoc"}


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
    if isinstance(source, pd.DataFrame):
        df = source
    else:
        path = Path(source)
        if path.suffix.lower() == ".parquet" or path.is_dir():
            df = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported source: {source}")

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


def _validate_event_df(df: pd.DataFrame, *, market: str) -> pd.DataFrame:
    required = {"date", "market", "type", "importance"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"event_calendar missing required columns: {missing}")

    out = df.copy()
    out = out[out["market"] == market].copy()
    out["type"] = out["type"].astype(str)
    bad_types = sorted(set(out["type"]) - _ALLOWED_TYPES)
    if bad_types:
        raise ValueError(f"event_calendar contains unsupported types for V1: {bad_types}")

    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.date
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

    for idx, row in out.iterrows():
        if pd.isna(row["publication_date"]):
            if row["type"] in _SCHEDULED_TYPES:
                out.at[idx, "publication_date"] = row["date"] - timedelta(days=90)
            else:
                out.at[idx, "publication_date"] = row["date"]

    return out[["date", "market", "type", "importance", "publication_date"]].sort_values(
        ["date", "type"]
    ).reset_index(drop=True)
