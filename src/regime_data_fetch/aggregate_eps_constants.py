from __future__ import annotations

from pathlib import Path

SOURCE_NAME = "S&P Global aggregate forward EPS workbook"
SOURCE_URL = "https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx"
SHEET_NAME = "ESTIMATES&PEs"
WAYBACK_CDX_URL = "https://web.archive.org/cdx/search/cdx"

# Weekly-snapshot accumulator (documented implementation decision path). Each weekly run of
# `run_aggregate_eps_fetch` appends the workbook's current snapshot to this
# parquet, deduped by observation_date. Once at least 5 distinct weekly rows
# have accumulated, `compute_eps_revision_direction_4w` produces a non-NaN
# revision series and the §2B `earnings_expansion` / `earnings_contraction`
# labels unlock. The single S&P workbook only exposes quarterly history +
# one current point, so weekly granularity can only be built by
# accumulating one current-snapshot row per weekly fetch.
WEEKLY_HISTORY_FILENAME = "sp500_eps_weekly_history.parquet"
# Spec §2B: revision direction over 4 weeks. Compare against the row 4 weekly
# observations back, so the first non-NaN value requires 5 rows.
EPS_REVISION_LOOKBACK_WEEKS = 4
# Output sub-directory + Wayback timeline filenames (shared by the live
# fetch, the Wayback backfill, and the accumulator-seeding bridge).
EPS_DIR_NAME = "aggregate_forward_eps"
WAYBACK_DIR_NAME = "aggregate_forward_eps_wayback"
WAYBACK_TIMELINE_FILENAME = "sp500_eps_wayback_timeline.parquet"

SPGLOBAL_EPS_MANUAL_REL_PATH = Path("spglobal_eps") / "sp-500-eps-est.xlsx"
