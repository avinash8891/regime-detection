from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from regime_data_fetch.acquisition_store import AcquisitionStore


ECONOMIC_EVENTS_REL = Path("investing_calendar_structured_2016_2026/investing_economic_events_2016-01-01_2026-05-15.csv")
HOLIDAYS_REL = Path("investing_calendar_structured_2016_2026/investing_holidays_2016-01-01_2026-05-15.csv")
CALENDAR_COMBINED_REL = Path("investing_calendar_structured_2016_2026/investing_calendar_combined_2016-01-01_2026-05-15.jsonl")
CALENDAR_FETCH_REPORT_REL = Path("investing_calendar_structured_2016_2026/fetch_report.json")
EARNINGS_REL = Path("investing_earnings_2016_2026/investing_earnings_2016-01-01_2026-05-15.csv")
EARNINGS_JSONL_REL = Path("investing_earnings_2016_2026/investing_earnings_2016-01-01_2026-05-15.jsonl")
EARNINGS_QUARANTINE_REL = Path("investing_earnings_2016_2026/quarantine_earnings_fetch_errors.jsonl")
EARNINGS_FETCH_REPORT_REL = Path("investing_earnings_2016_2026/fetch_report.json")
EARNINGS_RAW_INSTRUMENTS_REL = Path("investing_earnings_2016_2026/raw_instruments")


def run_local_investing_archive_import(
    *,
    out_dir: Path,
    archive_root: Path,
    acquisition_db_path: Path,
    artifact_store_root: str | Path | None = None,
) -> Path:
    required = [
        ECONOMIC_EVENTS_REL,
        HOLIDAYS_REL,
        CALENDAR_COMBINED_REL,
        CALENDAR_FETCH_REPORT_REL,
        EARNINGS_REL,
        EARNINGS_JSONL_REL,
        EARNINGS_FETCH_REPORT_REL,
    ]
    missing = [str(archive_root / rel) for rel in required if not (archive_root / rel).exists()]
    if missing:
        raise SystemExit(f"Missing Investing.com archive file(s): {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    investing_dir = out_dir / "investing"
    investing_dir.mkdir(parents=True, exist_ok=True)
    raw_archive_dir = investing_dir / "raw_archive"
    raw_archive_dir.mkdir(parents=True, exist_ok=True)

    copied_raw_files = _copy_archive_files(archive_root=archive_root, raw_archive_dir=raw_archive_dir)
    economic_events = _read_csv(archive_root / ECONOMIC_EVENTS_REL)
    holidays = _read_csv(archive_root / HOLIDAYS_REL)
    earnings = _read_csv(archive_root / EARNINGS_REL)

    economic_min, economic_max = _date_range(economic_events["occurrence_time_utc"])
    holiday_min, holiday_max = _date_range(holidays["holiday_start_utc"])
    earnings_min, earnings_max = _date_range(earnings["date"])

    economic_path = investing_dir / "economic_events.parquet"
    holidays_path = investing_dir / "holidays.parquet"
    earnings_path = investing_dir / "earnings.parquet"
    economic_events.to_parquet(economic_path, index=False)
    holidays.to_parquet(holidays_path, index=False)
    earnings.to_parquet(earnings_path, index=False)

    store = AcquisitionStore(acquisition_db_path, artifact_store_root=artifact_store_root)
    fetch_run = store.start_fetch_run(
        fetch_type="investing_archive_local",
        params={"archive_root": str(archive_root)},
    )

    raw_records = []
    try:
        for file_path in copied_raw_files:
            rel = file_path.relative_to(raw_archive_dir).as_posix()
            raw_records.append(
                store.record_file_artifact(
                    run_id=fetch_run.run_id,
                    source_name="investing.com:archive",
                    artifact_kind=file_path.suffix.lower().lstrip(".") or "file",
                    source_identifier=rel,
                    file_path=file_path,
                    start_date=_start_for_raw_rel(rel, economic_min, holiday_min, earnings_min),
                    end_date=_end_for_raw_rel(rel, economic_max, holiday_max, earnings_max),
                    timezone="UTC",
                    license_note="Archived Investing.com calendar and earnings export from Provo worktree",
                    notes="Local archived capture imported into the regime data acquisition store",
                )
            )

        output_records = [
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="investing_economic_events_parquet",
                path=economic_path,
                row_count=len(economic_events),
                min_date=economic_min,
                max_date=economic_max,
                artifact_name="investing_economic_events",
                source_name="investing.com",
                artifact_kind="parquet",
                notes="Canonical Investing.com structured economic calendar events",
            ),
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="investing_holidays_parquet",
                path=holidays_path,
                row_count=len(holidays),
                min_date=holiday_min,
                max_date=holiday_max,
                artifact_name="investing_holidays",
                source_name="investing.com",
                artifact_kind="parquet",
                notes="Canonical Investing.com exchange holidays",
            ),
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="investing_earnings_parquet",
                path=earnings_path,
                row_count=len(earnings),
                min_date=earnings_min,
                max_date=earnings_max,
                artifact_name="investing_earnings",
                source_name="investing.com",
                artifact_kind="parquet",
                notes="Canonical Investing.com earnings calendar rows",
            ),
        ]

        first_raw_record_id = next(
            (record.artifact_record_id for record in raw_records if record.artifact_record_id is not None),
            None,
        )
        if first_raw_record_id is not None:
            for output_record in output_records:
                if output_record is not None:
                    store.record_artifact_lineage(
                        output_artifact_record_id=output_record.artifact_record_id,
                        input_artifact_record_id=first_raw_record_id,
                        transform_name="normalize_investing_archive",
                    )

        report = {
            "source": "investing.com:archive",
            "archive_root": str(archive_root),
            "counts": {
                "economic_events_rows": int(len(economic_events)),
                "holiday_rows": int(len(holidays)),
                "earnings_rows": int(len(earnings)),
                "raw_files": len(copied_raw_files),
            },
            "date_range": {
                "economic_events": {"min_date": economic_min, "max_date": economic_max},
                "holidays": {"min_date": holiday_min, "max_date": holiday_max},
                "earnings": {"min_date": earnings_min, "max_date": earnings_max},
            },
            "paths": {
                "economic_events_parquet": str(economic_path),
                "holidays_parquet": str(holidays_path),
                "earnings_parquet": str(earnings_path),
                "raw_archive": {
                    "path": str(raw_archive_dir),
                    "local_path": "data/raw/investing/raw_archive",
                },
                "acquisition_db": str(acquisition_db_path),
            },
        }
        report_path = out_dir / "investing_archive_import_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        store.record_output(
            run_id=fetch_run.run_id,
            output_kind="investing_archive_import_report",
            path=report_path,
            row_count=int(len(economic_events) + len(holidays) + len(earnings)),
            min_date=min(value for value in [economic_min, holiday_min, earnings_min] if value is not None),
            max_date=max(value for value in [economic_max, holiday_max, earnings_max] if value is not None),
            notes="Investing.com archive import report",
        )
        store.finish_fetch_run(
            run_id=fetch_run.run_id,
            status="ok",
            notes=f"economic_events={len(economic_events)};holidays={len(holidays)};earnings={len(earnings)}",
        )
        return report_path
    except Exception as exc:
        store.finish_fetch_run(run_id=fetch_run.run_id, status="failed", notes=str(exc))
        raise


def _copy_archive_files(*, archive_root: Path, raw_archive_dir: Path) -> list[Path]:
    rels = [
        ECONOMIC_EVENTS_REL,
        HOLIDAYS_REL,
        CALENDAR_COMBINED_REL,
        CALENDAR_FETCH_REPORT_REL,
        EARNINGS_REL,
        EARNINGS_JSONL_REL,
        EARNINGS_FETCH_REPORT_REL,
    ]
    if (archive_root / EARNINGS_QUARANTINE_REL).exists():
        rels.append(EARNINGS_QUARANTINE_REL)
    copied: list[Path] = []
    for rel in rels:
        src = archive_root / rel
        dst = raw_archive_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    raw_instruments = archive_root / EARNINGS_RAW_INSTRUMENTS_REL
    if raw_instruments.exists():
        for src in sorted(path for path in raw_instruments.rglob("*") if path.is_file()):
            dst = raw_archive_dir / EARNINGS_RAW_INSTRUMENTS_REL / src.relative_to(raw_instruments)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _date_range(values: pd.Series) -> tuple[str | None, str | None]:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    parsed = parsed.dropna()
    if parsed.empty:
        return None, None
    return parsed.min().date().isoformat(), parsed.max().date().isoformat()


def _start_for_raw_rel(rel: str, economic_min: str | None, holiday_min: str | None, earnings_min: str | None) -> str | None:
    if rel.startswith("investing_calendar_structured_2016_2026/investing_economic_events"):
        return economic_min
    if rel.startswith("investing_calendar_structured_2016_2026/investing_holidays"):
        return holiday_min
    if rel.startswith("investing_earnings_2016_2026/"):
        return earnings_min
    return None


def _end_for_raw_rel(rel: str, economic_max: str | None, holiday_max: str | None, earnings_max: str | None) -> str | None:
    if rel.startswith("investing_calendar_structured_2016_2026/investing_economic_events"):
        return economic_max
    if rel.startswith("investing_calendar_structured_2016_2026/investing_holidays"):
        return holiday_max
    if rel.startswith("investing_earnings_2016_2026/"):
        return earnings_max
    return None
