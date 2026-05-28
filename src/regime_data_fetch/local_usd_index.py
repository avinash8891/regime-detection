from __future__ import annotations

import csv
import datetime as dt
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from regime_data_fetch.acquisition_store import AcquisitionStore

EXPECTED_YAHOO_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]
USD_INDEX_SYMBOL = "^NYICDX"


@dataclass(frozen=True)
class YahooUsdIndexRow:
    date: dt.date
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int


@dataclass(frozen=True)
class YahooUsdIndexLoadResult:
    frame: pd.DataFrame
    quarantined_rows: list[dict[str, object]]
    total_rows_seen: int


def load_yahoo_usd_index_csv(csv_path: Path) -> YahooUsdIndexLoadResult:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_YAHOO_COLUMNS:
            raise ValueError(
                f"Unexpected Yahoo USD index CSV columns: {reader.fieldnames!r}; "
                f"expected {EXPECTED_YAHOO_COLUMNS!r}"
            )

        rows: list[YahooUsdIndexRow] = []
        quarantined_rows: list[dict[str, object]] = []
        seen_dates: set[dt.date] = set()
        total_rows_seen = 0
        for row_number, raw_row in enumerate(reader, start=2):
            total_rows_seen += 1
            if _is_blank_price_row(raw_row):
                quarantined_rows.append(
                    {
                        "row_number": row_number,
                        "reason": "blank_price_row",
                        "raw_row": raw_row,
                    }
                )
                continue
            parsed = _parse_yahoo_usd_index_row(raw_row, row_number=row_number)
            if parsed.date in seen_dates:
                raise ValueError(
                    f"Duplicate USD index row for {parsed.date.isoformat()} at CSV line {row_number}"
                )
            seen_dates.add(parsed.date)
            rows.append(parsed)

    if not rows:
        raise ValueError(f"USD index CSV is empty: {csv_path}")
    if total_rows_seen and (len(quarantined_rows) / total_rows_seen) > 0.01:
        raise ValueError(
            f"Yahoo USD index CSV quarantine rate exceeds 1%: {len(quarantined_rows)}/{total_rows_seen}"
        )

    frame = pd.DataFrame(
        {
            "date": [row.date for row in rows],
            "symbol": USD_INDEX_SYMBOL,
            "open": [row.open for row in rows],
            "high": [row.high for row in rows],
            "low": [row.low for row in rows],
            "close": [row.close for row in rows],
            "adjusted_close": [row.adjusted_close for row in rows],
            "volume": [row.volume for row in rows],
            "source": "yahoo_finance",
        }
    )
    return YahooUsdIndexLoadResult(
        frame=frame.sort_values("date").reset_index(drop=True),
        quarantined_rows=quarantined_rows,
        total_rows_seen=total_rows_seen,
    )


def run_local_usd_index_import(
    *,
    out_dir: Path,
    csv_path: Path,
    acquisition_db_path: Path | None = None,
    artifact_store_root: str | Path | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    store = (
        AcquisitionStore(acquisition_db_path, artifact_store_root=artifact_store_root)
        if acquisition_db_path
        else None
    )
    run_context = (
        store.run(
            fetch_type="usd_index_local",
            params={
                "csv_path": str(csv_path),
                "symbol": USD_INDEX_SYMBOL,
                "source": "yahoo_finance",
            },
        )
        if store
        else nullcontext(None)
    )

    with run_context as fetch_run:
        load_result = load_yahoo_usd_index_csv(csv_path)
        frame = load_result.frame

        if store and fetch_run:
            store.record_file_artifact(
                run_id=fetch_run.run_id,
                source_name="yahoo:^NYICDX",
                artifact_kind="csv_manual",
                source_identifier=f"yahoo:{USD_INDEX_SYMBOL}:{csv_path.name}",
                file_path=csv_path,
                start_date=frame["date"].min().isoformat(),
                end_date=frame["date"].max().isoformat(),
                timezone="UTC",
                license_note="Manual local Yahoo Finance historical export for ^NYICDX",
                notes="Local Yahoo Finance USD index CSV persisted before parquet/report output",
            )

        usd_index_dir = out_dir / "usd_index"
        usd_index_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = usd_index_dir / "nyicdx_daily.parquet"
        frame.to_parquet(parquet_path, index=False)
        quarantine_path = usd_index_dir / "nyicdx_quarantine.jsonl"
        if load_result.quarantined_rows:
            quarantine_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in load_result.quarantined_rows
                )
            )

        report = {
            "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": "yahoo_finance",
            "symbol": USD_INDEX_SYMBOL,
            "csv_path": str(csv_path),
            "counts": {
                "rows": int(len(frame)),
                "quarantined_rows": len(load_result.quarantined_rows),
                "source_rows_seen": load_result.total_rows_seen,
            },
            "coverage": {
                "min_date": frame["date"].min().isoformat(),
                "max_date": frame["date"].max().isoformat(),
            },
            "paths": {
                "parquet": str(parquet_path),
                "quarantine": (
                    str(quarantine_path) if load_result.quarantined_rows else None
                ),
                "acquisition_db": (
                    str(acquisition_db_path) if acquisition_db_path else None
                ),
            },
            "notes": [
                "Optional local Yahoo Finance ^NYICDX import.",
                "This does not replace the approved spec field broad_usd_index sourced from FRED DTWEXBGS.",
            ],
        }
        report_path = out_dir / "usd_index_import_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if store and fetch_run:
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="usd_index_parquet",
                path=parquet_path,
                row_count=len(frame),
                min_date=frame["date"].min().isoformat(),
                max_date=frame["date"].max().isoformat(),
                notes="Normalized local Yahoo Finance ^NYICDX parquet output",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="usd_index_report",
                path=report_path,
                row_count=len(frame),
                min_date=frame["date"].min().isoformat(),
                max_date=frame["date"].max().isoformat(),
                notes="Local USD index import report",
            )
            if load_result.quarantined_rows:
                store.record_output(
                    run_id=fetch_run.run_id,
                    output_kind="usd_index_quarantine",
                    path=quarantine_path,
                    row_count=len(load_result.quarantined_rows),
                    notes="Quarantined blank Yahoo USD index rows",
                )

        return report_path


def _parse_yahoo_usd_index_row(
    raw_row: dict[str, str], *, row_number: int
) -> YahooUsdIndexRow:
    try:
        return YahooUsdIndexRow(
            date=dt.date.fromisoformat(raw_row["Date"]),
            open=float(raw_row["Open"]),
            high=float(raw_row["High"]),
            low=float(raw_row["Low"]),
            close=float(raw_row["Close"]),
            adjusted_close=float(raw_row["Adj Close"]),
            volume=int(raw_row["Volume"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid Yahoo USD index row at CSV line {row_number}: {raw_row!r}"
        ) from exc


def _is_blank_price_row(raw_row: dict[str, str]) -> bool:
    return all(raw_row.get(column, "") == "" for column in EXPECTED_YAHOO_COLUMNS[1:])
