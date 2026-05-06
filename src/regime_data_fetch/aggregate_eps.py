from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import urllib.request

import pandas as pd
from openpyxl import load_workbook

from regime_data_fetch.acquisition_store import AcquisitionStore


SOURCE_NAME = "S&P Global aggregate forward EPS workbook"
SOURCE_URL = "https://www.spglobal.com/spdji/en/documents/additional-material/sp-500-eps-est.xlsx"
SHEET_NAME = "ESTIMATES&PEs"
WAYBACK_CDX_URL = "https://web.archive.org/cdx/search/cdx"


class AggregateEPSFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class AggregateEPSSnapshot:
    observation_date: dt.date
    observation_label: str
    forward_estimate_label: str | None
    forward_estimate_value: float | None
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


@dataclass(frozen=True)
class EPSWaybackSnapshot:
    timestamp: str
    archive_url: str
    snapshot_date: dt.date


def parse_sp500_eps_workbook(workbook_path: Path) -> ParsedAggregateEPSWorkbook:
    wb = load_workbook(workbook_path, read_only=True, data_only=True)
    if SHEET_NAME not in wb.sheetnames:
        raise AggregateEPSFetchError(f"Workbook missing expected sheet {SHEET_NAME!r}")

    ws = wb[SHEET_NAME]
    workbook_as_of_date = _extract_workbook_as_of_date(ws)
    public_files_discontinued = _extract_discontinued_flag(ws)
    table_start_row, header_labels = _find_observation_header_row(ws)
    current_changes = _find_current_change_row(ws, table_start_row)

    historical: list[AggregateEPSSnapshot] = []
    current_snapshot: AggregateEPSSnapshot | None = None
    for row_idx in range(table_start_row + 1, ws.max_row + 1):
        row = next(ws.iter_rows(min_row=row_idx, max_row=row_idx, values_only=True))
        first = row[0]
        if isinstance(first, dt.datetime):
            label_map = _build_observation_value_map(header_labels, row)
            historical.append(
                AggregateEPSSnapshot(
                    observation_date=first.date(),
                    observation_label="historical_quarter_end",
                    forward_estimate_label=_select_forward_estimate_label(header_labels),
                    forward_estimate_value=_select_forward_estimate_value(label_map, header_labels),
                    estimate_2025e=_value_for_exact_label(label_map, "2025E"),
                    estimate_q4_2025e=_value_for_exact_label(label_map, "Q4 2025E"),
                    estimate_2026e=_value_for_exact_label(label_map, "2026E"),
                    price=_value_for_price(label_map),
                    pe_2025e=_value_for_exact_label(label_map, "2025E P/E"),
                    pe_2026e=_value_for_pe(label_map, "2026"),
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
            label_map = _build_observation_value_map(header_labels, row)
            current_snapshot = AggregateEPSSnapshot(
                observation_date=workbook_as_of_date,
                observation_label="current",
                forward_estimate_label=_select_forward_estimate_label(header_labels),
                forward_estimate_value=_select_forward_estimate_value(label_map, header_labels),
                estimate_2025e=_value_for_exact_label(label_map, "2025E"),
                estimate_q4_2025e=_value_for_exact_label(label_map, "Q4 2025E"),
                estimate_2026e=_value_for_exact_label(label_map, "2026E"),
                price=_value_for_price(label_map),
                pe_2025e=_value_for_exact_label(label_map, "2025E P/E"),
                pe_2026e=_value_for_pe(label_map, "2026"),
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
    acquisition_db_path: Path | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    store = AcquisitionStore(acquisition_db_path) if acquisition_db_path else None
    fetch_run = (
        store.start_fetch_run(
            fetch_type="aggregate_eps",
            params={
                "workbook_path": str(workbook_path),
            },
        )
        if store
        else None
    )

    try:
        if store and fetch_run:
            store.record_file_artifact(
                run_id=fetch_run.run_id,
                source_name=SOURCE_NAME,
                artifact_kind="xlsx_manual",
                source_identifier=str(workbook_path),
                file_path=workbook_path,
                timezone="America/New_York",
                license_note="Manually downloaded workbook; public files reported discontinued by source workbook",
                notes="Manual S&P aggregate EPS workbook snapshot",
            )

        parsed = parse_sp500_eps_workbook(workbook_path)

        rows = [*parsed.historical_snapshots, parsed.current_snapshot]
        df = pd.DataFrame(
            [
                {
                    "workbook_as_of_date": parsed.workbook_as_of_date,
                    "observation_date": row.observation_date,
                    "observation_label": row.observation_label,
                    "forward_estimate_label": row.forward_estimate_label,
                    "forward_estimate_value": row.forward_estimate_value,
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
                "acquisition_db": str(acquisition_db_path) if acquisition_db_path else None,
            },
        }
        report_path = out_dir / "aggregate_eps_fetch_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if store and fetch_run:
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="aggregate_eps_parquet",
                path=parquet_path,
                row_count=len(df),
                min_date=min(df["observation_date"]).isoformat() if not df.empty else None,
                max_date=max(df["observation_date"]).isoformat() if not df.empty else None,
                notes="Aggregate EPS workbook snapshots parquet",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="aggregate_eps_report",
                path=report_path,
                row_count=len(df),
                min_date=min(df["observation_date"]).isoformat() if not df.empty else None,
                max_date=max(df["observation_date"]).isoformat() if not df.empty else None,
                notes="Aggregate EPS fetch report",
            )
            store.finish_fetch_run(run_id=fetch_run.run_id, status="ok")
        return report_path
    except Exception as exc:
        if store and fetch_run:
            store.finish_fetch_run(run_id=fetch_run.run_id, status="failed", notes=str(exc))
        raise


def parse_wayback_cdx_json(cdx_json: str, *, target_url: str) -> list[EPSWaybackSnapshot]:
    try:
        rows = json.loads(cdx_json)
    except json.JSONDecodeError as exc:
        raise AggregateEPSFetchError("Wayback CDX response was not valid JSON") from exc

    if not isinstance(rows, list) or not rows:
        raise AggregateEPSFetchError("Wayback CDX response contained no rows")
    if rows[0] != ["timestamp", "original", "statuscode", "mimetype"]:
        raise AggregateEPSFetchError(f"Unexpected Wayback CDX header: {rows[0]!r}")

    snapshots: list[EPSWaybackSnapshot] = []
    for idx, row in enumerate(rows[1:], start=2):
        if not isinstance(row, list) or len(row) != 4:
            raise AggregateEPSFetchError(f"Wayback CDX row {idx} had unexpected shape")
        timestamp, original, statuscode, mimetype = row
        if statuscode != "200":
            continue
        if original != target_url:
            continue
        if "spreadsheetml.sheet" not in mimetype and "excel" not in mimetype:
            continue
        snapshot_dt = dt.datetime.strptime(timestamp[:8], "%Y%m%d").date()
        snapshots.append(
            EPSWaybackSnapshot(
                timestamp=timestamp,
                archive_url=f"https://web.archive.org/web/{timestamp}if_/{target_url}",
                snapshot_date=snapshot_dt,
            )
        )

    if not snapshots:
        raise AggregateEPSFetchError("Wayback CDX response contained no usable workbook snapshots")
    return snapshots


def fetch_wayback_cdx(target_url: str = SOURCE_URL) -> str:
    query = (
        f"{WAYBACK_CDX_URL}?url={target_url}&output=json"
        "&fl=timestamp,original,statuscode,mimetype&filter=statuscode:200"
    )
    req = urllib.request.Request(query, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_wayback_snapshot_bytes(snapshot: EPSWaybackSnapshot) -> bytes:
    req = urllib.request.Request(snapshot.archive_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def run_wayback_aggregate_eps_fetch(
    *,
    out_dir: Path,
    max_snapshots: int | None = None,
    from_date: dt.date | None = None,
    to_date: dt.date | None = None,
    stop_after_first_success: bool = False,
    cdx_fetcher=fetch_wayback_cdx,
    snapshot_fetcher=fetch_wayback_snapshot_bytes,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots = parse_wayback_cdx_json(cdx_fetcher(), target_url=SOURCE_URL)
    snapshots = _filter_wayback_snapshots(
        snapshots,
        from_date=from_date,
        to_date=to_date,
        max_snapshots=max_snapshots,
    )

    wayback_dir = out_dir / "aggregate_forward_eps_wayback"
    snapshots_dir = wayback_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_index_path = wayback_dir / "wayback_snapshot_index.json"
    snapshot_index_path.write_text(
        json.dumps(
            [
                {
                    "snapshot_date": snapshot.snapshot_date.isoformat(),
                    "timestamp": snapshot.timestamp,
                    "archive_url": snapshot.archive_url,
                }
                for snapshot in snapshots
            ],
            indent=2,
        )
    )
    status_path = wayback_dir / "snapshot_status.jsonl"

    timeline_rows: list[dict[str, object]] = []
    downloaded = 0
    failures = 0
    parsed_ok = 0
    for snapshot in snapshots:
        workbook_path = snapshots_dir / f"{snapshot.timestamp}.xlsx"
        try:
            if workbook_path.exists():
                status = "download_reused"
            else:
                payload = snapshot_fetcher(snapshot)
                workbook_path.write_bytes(payload)
                downloaded += 1
                status = "downloaded"

            parsed = parse_sp500_eps_workbook(workbook_path)
            current = parsed.current_snapshot
            timeline_rows.append(
                {
                    "snapshot_date": snapshot.snapshot_date,
                    "timestamp": snapshot.timestamp,
                    "archive_url": snapshot.archive_url,
                    "workbook_as_of_date": parsed.workbook_as_of_date,
                    "forward_estimate_label": current.forward_estimate_label,
                    "forward_estimate_value": current.forward_estimate_value,
                    "estimate_2025e": current.estimate_2025e,
                    "estimate_q4_2025e": current.estimate_q4_2025e,
                    "estimate_2026e": current.estimate_2026e,
                    "price": current.price,
                    "pe_2025e": current.pe_2025e,
                    "pe_2026e": current.pe_2026e,
                    "change_vs_prior_observation_2025e": current.change_vs_prior_observation_2025e,
                    "change_vs_prior_observation_q4_2025e": current.change_vs_prior_observation_q4_2025e,
                    "change_vs_prior_observation_2026e": current.change_vs_prior_observation_2026e,
                    "change_vs_prior_observation_price": current.change_vs_prior_observation_price,
                    "change_vs_prior_observation_pe_2025e": current.change_vs_prior_observation_pe_2025e,
                    "change_vs_prior_observation_pe_2026e": current.change_vs_prior_observation_pe_2026e,
                    "public_files_discontinued": parsed.public_files_discontinued,
                    "source": "wayback_machine",
                }
            )
            parsed_ok += 1
            _append_wayback_status(
                status_path,
                snapshot=snapshot,
                status="parsed_ok",
                detail=status,
            )
            if stop_after_first_success:
                break
        except Exception as exc:
            failures += 1
            _append_wayback_status(
                status_path,
                snapshot=snapshot,
                status="failed",
                detail=f"{type(exc).__name__}: {exc}",
            )
            continue

    if not timeline_rows:
        raise AggregateEPSFetchError("Wayback EPS backfill produced no parsed timeline rows")

    timeline_df = pd.DataFrame(timeline_rows).sort_values(["snapshot_date", "timestamp"]).reset_index(drop=True)
    timeline_path = wayback_dir / "sp500_eps_wayback_timeline.parquet"
    timeline_df.to_parquet(timeline_path, index=False)

    preview = timeline_df.head(10).copy()
    if "snapshot_date" in preview:
        preview["snapshot_date"] = preview["snapshot_date"].map(lambda x: x.isoformat())
    if "workbook_as_of_date" in preview:
        preview["workbook_as_of_date"] = preview["workbook_as_of_date"].map(lambda x: x.isoformat())

    report = {
        "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "wayback_machine",
        "source_url": SOURCE_URL,
        "counts": {
            "snapshots_listed": len(snapshots),
            "snapshots_downloaded": downloaded,
            "snapshots_failed": failures,
            "snapshots_parsed_ok": parsed_ok,
            "timeline_rows": int(len(timeline_df)),
        },
        "requested": {
            "max_snapshots": max_snapshots,
            "from_date": from_date.isoformat() if from_date else None,
            "to_date": to_date.isoformat() if to_date else None,
            "stop_after_first_success": stop_after_first_success,
        },
        "timeline_preview": preview.to_dict(orient="records"),
        "paths": {
            "snapshots_dir": str(snapshots_dir),
            "snapshot_index_json": str(snapshot_index_path),
            "snapshot_status_jsonl": str(status_path),
            "timeline_parquet": str(timeline_path),
        },
    }
    report_path = out_dir / "aggregate_eps_wayback_fetch_report.json"
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


def _find_observation_header_row(ws) -> tuple[int, list[str]]:
    for row_idx in range(1, ws.max_row + 1):
        row = next(ws.iter_rows(min_row=row_idx, max_row=row_idx, values_only=True))
        if row[0] == "OBSERVATION":
            labels: list[str] = []
            for value in row[1:]:
                label = str(value).strip() if value is not None else ""
                if label == "OBSERVATION":
                    break
                labels.append(label)
            if any(label.endswith("E") for label in labels):
                return row_idx, labels
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


def _build_observation_value_map(labels: list[str], row: tuple[object, ...]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for idx, label in enumerate(labels, start=1):
        if not label:
            continue
        values[label] = _as_float(row[idx]) if idx < len(row) else None
    return values


def _value_for_exact_label(values: dict[str, float | None], label: str) -> float | None:
    return values.get(label)


def _value_for_price(values: dict[str, float | None]) -> float | None:
    return values.get("PRICE") or values.get(" PRICE")


def _value_for_pe(values: dict[str, float | None], year_prefix: str) -> float | None:
    for label, value in values.items():
        normalized = label.replace(" ", "")
        if normalized.startswith(year_prefix) and normalized.endswith("P/E"):
            return value
    return None


def _select_forward_estimate_label(labels: list[str]) -> str | None:
    annual_labels = [
        label
        for label in labels
        if len(label) == 5 and label[:4].isdigit() and label.endswith("E")
    ]
    if not annual_labels:
        return None
    return annual_labels[-1]


def _select_forward_estimate_value(values: dict[str, float | None], labels: list[str]) -> float | None:
    label = _select_forward_estimate_label(labels)
    if label is None:
        return None
    return values.get(label)


def _filter_wayback_snapshots(
    snapshots: list[EPSWaybackSnapshot],
    *,
    from_date: dt.date | None,
    to_date: dt.date | None,
    max_snapshots: int | None,
) -> list[EPSWaybackSnapshot]:
    filtered = [
        snapshot
        for snapshot in snapshots
        if (from_date is None or snapshot.snapshot_date >= from_date)
        and (to_date is None or snapshot.snapshot_date <= to_date)
    ]
    filtered.sort(key=lambda snapshot: (snapshot.snapshot_date, snapshot.timestamp))
    if max_snapshots is not None:
        return filtered[:max_snapshots]
    return filtered


def _append_wayback_status(
    status_path: Path,
    *,
    snapshot: EPSWaybackSnapshot,
    status: str,
    detail: str,
) -> None:
    record = {
        "snapshot_date": snapshot.snapshot_date.isoformat(),
        "timestamp": snapshot.timestamp,
        "archive_url": snapshot.archive_url,
        "status": status,
        "detail": detail,
    }
    with status_path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
