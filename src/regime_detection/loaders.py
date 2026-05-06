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
