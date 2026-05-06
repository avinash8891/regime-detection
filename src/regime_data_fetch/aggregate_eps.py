from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


SOURCE_NAME = "S&P Global aggregate forward EPS workbook"
SOURCE_URL = "https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx"
SHEET_NAME = "ESTIMATES&PEs"


class AggregateEPSFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class AggregateEPSSnapshot:
    observation_date: dt.date
    observation_label: str
    estimate_2025e: float | None
    estimate_q4_2025e: float | None
    estimate_2026e: float | None
    price: float | None
    pe_2025e: float | None
    pe_2026e: float | None
    change_vs_prior_observation_2025e: float | None
    change_vs_prior_observation_q4_2025e: float | None
    change_vs_prior_observation_2026e: float | None
    change_vs_prior_observation_price: float | None
    change_vs_prior_observation_pe_2025e: float | None
    change_vs_prior_observation_pe_2026e: float | None


@dataclass(frozen=True)
class ParsedAggregateEPSWorkbook:
    workbook_as_of_date: dt.date
    public_files_discontinued: bool
    historical_snapshots: list[AggregateEPSSnapshot]
    current_snapshot: AggregateEPSSnapshot


def parse_sp500_eps_workbook(workbook_path: Path) -> ParsedAggregateEPSWorkbook:
    wb = load_workbook(workbook_path, read_only=True, data_only=True)
    if SHEET_NAME not in wb.sheetnames:
        raise AggregateEPSFetchError(f"Workbook missing expected sheet {SHEET_NAME!r}")

    ws = wb[SHEET_NAME]
    workbook_as_of_date = _extract_workbook_as_of_date(ws)
    public_files_discontinued = _extract_discontinued_flag(ws)
    table_start_row = _find_observation_header_row(ws)
    current_changes = _find_current_change_row(ws, table_start_row)

    historical: list[AggregateEPSSnapshot] = []
    current_snapshot: AggregateEPSSnapshot | None = None
    for row_idx in range(table_start_row + 1, ws.max_row + 1):
        row = next(ws.iter_rows(min_row=row_idx, max_row=row_idx, values_only=True))
        first = row[0]
        if isinstance(first, dt.datetime):
            historical.append(
                AggregateEPSSnapshot(
                    observation_date=first.date(),
                    observation_label="historical_quarter_end",
                    estimate_2025e=_as_float(row[1]),
                    estimate_q4_2025e=_as_float(row[2]),
                    estimate_2026e=_as_float(row[3]),
                    price=_as_float(row[4]),
                    pe_2025e=_as_float(row[5]),
                    pe_2026e=_as_float(row[6]),
                    change_vs_prior_observation_2025e=None,
                    change_vs_prior_observation_q4_2025e=None,
                    change_vs_prior_observation_2026e=None,
                    change_vs_prior_observation_price=None,
                    change_vs_prior_observation_pe_2025e=None,
                    change_vs_prior_observation_pe_2026e=None,
                )
            )
            continue

        if isinstance(first, str) and first.strip().lower() == "current":
            current_snapshot = AggregateEPSSnapshot(
                observation_date=workbook_as_of_date,
                observation_label="current",
                estimate_2025e=_as_float(row[1]),
                estimate_q4_2025e=_as_float(row[2]),
                estimate_2026e=_as_float(row[3]),
                price=_as_float(row[4]),
                pe_2025e=_as_float(row[5]),
                pe_2026e=_as_float(row[6]),
                change_vs_prior_observation_2025e=current_changes[0],
                change_vs_prior_observation_q4_2025e=current_changes[1],
                change_vs_prior_observation_2026e=current_changes[2],
                change_vs_prior_observation_price=current_changes[3],
                change_vs_prior_observation_pe_2025e=current_changes[4],
                change_vs_prior_observation_pe_2026e=current_changes[5],
            )
            continue

        if current_snapshot is not None:
            break

    if not historical:
        raise AggregateEPSFetchError("Workbook contained no historical aggregate EPS snapshots")
    if current_snapshot is None:
        raise AggregateEPSFetchError("Workbook missing current aggregate EPS snapshot row")

    return ParsedAggregateEPSWorkbook(
        workbook_as_of_date=workbook_as_of_date,
        public_files_discontinued=public_files_discontinued,
        historical_snapshots=historical,
        current_snapshot=current_snapshot,
    )


def run_aggregate_eps_fetch(
    *,
    out_dir: Path,
    workbook_path: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_sp500_eps_workbook(workbook_path)

    rows = [*parsed.historical_snapshots, parsed.current_snapshot]
    df = pd.DataFrame(
        [
            {
                "workbook_as_of_date": parsed.workbook_as_of_date,
                "observation_date": row.observation_date,
                "observation_label": row.observation_label,
                "estimate_2025e": row.estimate_2025e,
                "estimate_q4_2025e": row.estimate_q4_2025e,
                "estimate_2026e": row.estimate_2026e,
                "price": row.price,
                "pe_2025e": row.pe_2025e,
                "pe_2026e": row.pe_2026e,
                "change_vs_prior_observation_2025e": row.change_vs_prior_observation_2025e,
                "change_vs_prior_observation_q4_2025e": row.change_vs_prior_observation_q4_2025e,
                "change_vs_prior_observation_2026e": row.change_vs_prior_observation_2026e,
                "change_vs_prior_observation_price": row.change_vs_prior_observation_price,
                "change_vs_prior_observation_pe_2025e": row.change_vs_prior_observation_pe_2025e,
                "change_vs_prior_observation_pe_2026e": row.change_vs_prior_observation_pe_2026e,
                "source": SOURCE_NAME,
                "source_path": str(workbook_path),
                "public_files_discontinued": parsed.public_files_discontinued,
            }
            for row in rows
        ]
    )

    eps_dir = out_dir / "aggregate_forward_eps"
    eps_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = eps_dir / "sp500_eps_snapshots.parquet"
    df.to_parquet(parquet_path, index=False)

    current_dict = asdict(parsed.current_snapshot)
    current_dict["observation_date"] = parsed.current_snapshot.observation_date.isoformat()
    report = {
        "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "source_path": str(workbook_path),
        "workbook_as_of_date": parsed.workbook_as_of_date.isoformat(),
        "public_files_discontinued": parsed.public_files_discontinued,
        "counts": {
            "historical_snapshots": len(parsed.historical_snapshots),
            "current_snapshots": 1,
        },
        "current_snapshot": current_dict,
        "limitations": {
            "aggregate_forward_eps_revision_direction_4w_available": False,
            "reason": (
                "The captured public workbook exposes quarterly historical observations plus one current snapshot, "
                "not a weekly revision history."
            ),
        },
        "paths": {
            "aggregate_eps_parquet": str(parquet_path),
        },
    }
    report_path = out_dir / "aggregate_eps_fetch_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report_path


def _extract_workbook_as_of_date(ws) -> dt.date:
    for row_idx in range(1, min(30, ws.max_row) + 1):
        row = next(ws.iter_rows(min_row=row_idx, max_row=row_idx, values_only=True))
        first = row[0]
        if isinstance(first, dt.datetime):
            return first.date()
    raise AggregateEPSFetchError("Could not find workbook as-of date in ESTIMATES&PEs sheet")


def _extract_discontinued_flag(ws) -> bool:
    for row_idx in range(1, min(15, ws.max_row) + 1):
        row = next(ws.iter_rows(min_row=row_idx, max_row=row_idx, values_only=True))
        first = row[0]
        if isinstance(first, str) and "public files have been discontinued" in first.lower():
            return True
    return False


def _find_observation_header_row(ws) -> int:
    for row_idx in range(1, ws.max_row + 1):
        row = next(ws.iter_rows(min_row=row_idx, max_row=row_idx, values_only=True))
        if row[0] == "OBSERVATION" and row[1] == "2025E":
            return row_idx
    raise AggregateEPSFetchError("Could not find aggregate EPS observation header row")


def _find_current_change_row(ws, header_row: int) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    for row_idx in range(header_row + 1, min(header_row + 20, ws.max_row) + 1):
        row = next(ws.iter_rows(min_row=row_idx, max_row=row_idx, values_only=True))
        first = row[0]
        if isinstance(first, str) and first.strip().lower() == "change qtr":
            return (
                _as_float(row[1]),
                _as_float(row[2]),
                _as_float(row[3]),
                _as_float(row[4]),
                _as_float(row[5]),
                _as_float(row[6]),
            )
    raise AggregateEPSFetchError("Could not find current aggregate EPS change row")


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
