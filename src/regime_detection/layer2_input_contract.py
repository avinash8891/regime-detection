from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from regime_detection.loaders import load_aggregate_forward_eps_revision_series


PMI_MAX_STALENESS_DAYS = 45
CPI_NOWCAST_MAX_STALENESS_DAYS = 21
EPS_WEEKLY_MAX_STALENESS_DAYS = 21
PMI_MIN_HISTORY_ROWS = 24
EPS_MIN_WEEKLY_ROWS = 5


def validate_layer2_incremental_inputs(*, data_root: Path, as_of_date: dt.date) -> None:
    """Fail fast when manifest-materialized Layer 2 extension inputs regress.

    The engine can technically run with absent optional §2B inputs, but that
    silently darkens inflation/growth evidence. Runners that claim a full
    profile-ready manifest should reject missing or stale PMI, CPI nowcast,
    and EPS weekly-history artifacts before classification.
    """
    _validate_pmi_history(data_root / "pmi" / "us_ism_pmi_history.parquet", as_of_date)
    _validate_cpi_nowcast(
        data_root / "cleveland_fed_nowcast" / "cpi_nowcast.parquet", as_of_date
    )
    _validate_eps_weekly_history(
        data_root / "aggregate_forward_eps" / "sp500_eps_weekly_history.parquet",
        as_of_date,
    )


def _validate_pmi_history(path: Path, as_of_date: dt.date) -> None:
    if not path.exists():
        raise ValueError(f"Layer 2 input contract failed: PMI history missing at {path}")
    frame = pd.read_parquet(path)
    required = {"series_name", "period", "release_timestamp", "value"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Layer 2 input contract failed: PMI history missing columns {missing}")
    if len(frame) < PMI_MIN_HISTORY_ROWS:
        raise ValueError(
            "Layer 2 input contract failed: PMI history has "
            f"{len(frame)} rows, expected at least {PMI_MIN_HISTORY_ROWS}"
        )
    manufacturing = frame[frame["series_name"] == "manufacturing"].copy()
    if manufacturing.empty:
        raise ValueError("Layer 2 input contract failed: PMI manufacturing history is empty")
    release_dates = pd.to_datetime(manufacturing["release_timestamp"], utc=True).dt.date
    valid_dates = [date for date in release_dates if date <= as_of_date]
    if not valid_dates:
        raise ValueError("Layer 2 input contract failed: PMI history has no current rows")
    last_release = max(valid_dates)
    staleness = (as_of_date - last_release).days
    if staleness > PMI_MAX_STALENESS_DAYS:
        raise ValueError(
            "Layer 2 input contract failed: PMI manufacturing stale "
            f"{staleness}d as of {as_of_date}"
        )


def _validate_cpi_nowcast(path: Path, as_of_date: dt.date) -> None:
    if not path.exists():
        raise ValueError(f"Layer 2 input contract failed: CPI nowcast missing at {path}")
    frame = pd.read_parquet(path)
    required = {"date", "cpi_nowcast"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Layer 2 input contract failed: CPI nowcast missing columns {missing}")
    values = frame.dropna(subset=["date", "cpi_nowcast"]).copy()
    if values.empty:
        raise ValueError("Layer 2 input contract failed: CPI nowcast has no values")
    dates = pd.to_datetime(values["date"]).dt.date
    valid_dates = [date for date in dates if date <= as_of_date]
    if not valid_dates:
        raise ValueError("Layer 2 input contract failed: CPI nowcast has no current rows")
    last_date = max(valid_dates)
    staleness = (as_of_date - last_date).days
    if staleness > CPI_NOWCAST_MAX_STALENESS_DAYS:
        raise ValueError(
            "Layer 2 input contract failed: CPI nowcast stale "
            f"{staleness}d as of {as_of_date}"
        )


def _validate_eps_weekly_history(path: Path, as_of_date: dt.date) -> None:
    if not path.exists():
        raise ValueError(f"Layer 2 input contract failed: EPS weekly history missing at {path}")
    frame = pd.read_parquet(path)
    required = {"observation_date", "forward_estimate_value"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Layer 2 input contract failed: EPS weekly history missing columns {missing}"
        )
    if len(frame) < EPS_MIN_WEEKLY_ROWS:
        raise ValueError(
            "Layer 2 input contract failed: EPS weekly history has "
            f"{len(frame)} rows, expected at least {EPS_MIN_WEEKLY_ROWS}"
        )
    revision = load_aggregate_forward_eps_revision_series(path)
    if not revision.notna().any():
        raise ValueError(
            "Layer 2 input contract failed: EPS weekly history has no non-null "
            "aggregate_forward_eps_revision_direction_4w values"
        )
    dates = pd.to_datetime(frame["observation_date"]).dt.date
    valid_dates = [date for date in dates if date <= as_of_date]
    if not valid_dates:
        raise ValueError(
            "Layer 2 input contract failed: EPS weekly history has no current rows"
        )
    last_date = max(valid_dates)
    staleness = (as_of_date - last_date).days
    if staleness > EPS_WEEKLY_MAX_STALENESS_DAYS:
        raise ValueError(
            "Layer 2 input contract failed: EPS weekly history stale "
            f"{staleness}d as of {as_of_date}"
        )
