from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from regime_data_fetch.aggregate_eps_constants import (
    EPS_REVISION_LOOKBACK_WEEKS,
    SOURCE_NAME,
    SOURCE_URL,
)
from regime_data_fetch.aggregate_eps_models import ParsedAggregateEPSWorkbook


def build_aggregate_eps_report(
    *,
    as_of_utc: str,
    workbook_path: Path,
    parsed: ParsedAggregateEPSWorkbook,
    weekly_history: pd.DataFrame,
    revision_available: bool,
    parquet_path: Path,
    weekly_history_path: Path,
    acquisition_db_path: Path | None,
) -> dict[str, object]:
    current_dict = asdict(parsed.current_snapshot)
    current_dict["observation_date"] = (
        parsed.current_snapshot.observation_date.isoformat()
    )
    return {
        "as_of_utc": as_of_utc,
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "source_path": str(workbook_path),
        "workbook_as_of_date": parsed.workbook_as_of_date.isoformat(),
        "public_files_discontinued": parsed.public_files_discontinued,
        "counts": {
            "historical_snapshots": len(parsed.historical_snapshots),
            "current_snapshots": 1,
            "weekly_history_rows": len(weekly_history),
        },
        "current_snapshot": current_dict,
        "limitations": {
            "aggregate_forward_eps_revision_direction_4w_available": revision_available,
            "reason": (
                "Revision direction available — the weekly-snapshot accumulator "
                f"holds {len(weekly_history)} rows (> {EPS_REVISION_LOOKBACK_WEEKS} "
                "required for the 4-week lookback)."
                if revision_available
                else (
                    "The single S&P workbook exposes quarterly history plus one "
                    f"current snapshot. The weekly accumulator holds "
                    f"{len(weekly_history)} row(s); "
                    f"> {EPS_REVISION_LOOKBACK_WEEKS} weekly fetches are required "
                    "before the 4-week revision direction is non-NaN."
                )
            ),
        },
        "paths": {
            "aggregate_eps_parquet": str(parquet_path),
            "aggregate_eps_weekly_history_parquet": str(weekly_history_path),
            "acquisition_db": str(acquisition_db_path) if acquisition_db_path else None,
        },
    }
