"""Wayback Machine backfill path for the S&P aggregate forward-EPS workbook.

This module contains all Wayback-CDX querying, snapshot downloading,
filtering, and timeline materialisation logic. It is intentionally
independent of the live-fetch path in ``aggregate_eps.py``.

Public entry point: ``run_wayback_aggregate_eps_fetch``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import urllib.request

_LOG = logging.getLogger(__name__)

import pandas as pd

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.aggregate_eps import (
    SOURCE_URL,
    WAYBACK_CDX_URL,
    WAYBACK_DIR_NAME,
    WAYBACK_TIMELINE_FILENAME,
    AggregateEPSFetchError,
    parse_sp500_eps_workbook,
)


@dataclass(frozen=True)
class EPSWaybackSnapshot:
    timestamp: str
    archive_url: str
    snapshot_date: dt.date

    def __post_init__(self) -> None:
        if len(self.timestamp) < 8 or not self.timestamp[:8].isdigit():
            raise ValueError(f"timestamp {self.timestamp!r} is not a valid Wayback timestamp (must start with 8 digits)")
        expected_date = dt.datetime.strptime(self.timestamp[:8], "%Y%m%d").date()
        if self.snapshot_date != expected_date:
            raise ValueError(
                f"snapshot_date {self.snapshot_date} does not match timestamp {self.timestamp!r}"
            )

    @classmethod
    def from_timestamp(cls, timestamp: str, *, target_url: str) -> "EPSWaybackSnapshot":
        snapshot_date = dt.datetime.strptime(timestamp[:8], "%Y%m%d").date()
        archive_url = f"https://web.archive.org/web/{timestamp}if_/{target_url}"
        return cls(timestamp=timestamp, archive_url=archive_url, snapshot_date=snapshot_date)


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
        snapshots.append(EPSWaybackSnapshot.from_timestamp(timestamp, target_url=target_url))

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
    acquisition_db_path: Path | None = None,
    cdx_fetcher=fetch_wayback_cdx,
    snapshot_fetcher=fetch_wayback_snapshot_bytes,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    store = AcquisitionStore(acquisition_db_path) if acquisition_db_path else None
    fetch_run = (
        store.start_fetch_run(
            fetch_type="aggregate_eps_wayback",
            params={
                "max_snapshots": max_snapshots,
                "from_date": from_date.isoformat() if from_date else None,
                "to_date": to_date.isoformat() if to_date else None,
                "stop_after_first_success": stop_after_first_success,
                "source_url": SOURCE_URL,
            },
        )
        if store
        else None
    )

    try:
        cdx_json = cdx_fetcher()
        snapshots = parse_wayback_cdx_json(cdx_json, target_url=SOURCE_URL)
        snapshots = _filter_wayback_snapshots(
            snapshots,
            from_date=from_date,
            to_date=to_date,
            max_snapshots=max_snapshots,
        )

        wayback_dir = out_dir / WAYBACK_DIR_NAME
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

        if store and fetch_run:
            store.record_text_artifact(
                run_id=fetch_run.run_id,
                source_name="wayback:cdx",
                artifact_kind="json",
                source_identifier=SOURCE_URL,
                content_text=cdx_json,
                start_date=from_date.isoformat() if from_date else None,
                end_date=to_date.isoformat() if to_date else None,
                timezone="UTC",
                license_note="Wayback CDX listing for archived S&P aggregate EPS workbook snapshots",
                notes="Wayback CDX listing persisted before filtered snapshot materialization",
            )

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

                if store and fetch_run:
                    store.record_file_artifact(
                        run_id=fetch_run.run_id,
                        source_name="wayback:eps_workbook",
                        artifact_kind="xlsx_wayback",
                        source_identifier=snapshot.timestamp,
                        file_path=workbook_path,
                        effective_date=snapshot.snapshot_date.isoformat(),
                        timezone="UTC",
                        license_note="Archived S&P aggregate EPS workbook snapshot fetched from Wayback Machine",
                        notes=f"Wayback workbook snapshot {status}",
                    )

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
                _LOG.warning(
                    "Wayback EPS snapshot %s (%s) failed — skipping",
                    snapshot.timestamp,
                    snapshot.snapshot_date.isoformat(),
                    exc_info=True,
                )
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
        timeline_path = wayback_dir / WAYBACK_TIMELINE_FILENAME
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
                "acquisition_db": str(acquisition_db_path) if acquisition_db_path else None,
            },
        }
        report_path = out_dir / "aggregate_eps_wayback_fetch_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if store and fetch_run:
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="aggregate_eps_wayback_snapshot_index",
                path=snapshot_index_path,
                row_count=len(snapshots),
                min_date=min(snapshot.snapshot_date for snapshot in snapshots).isoformat() if snapshots else None,
                max_date=max(snapshot.snapshot_date for snapshot in snapshots).isoformat() if snapshots else None,
                notes="Filtered Wayback EPS snapshot index",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="aggregate_eps_wayback_status",
                path=status_path,
                row_count=parsed_ok + failures,
                notes="Wayback EPS per-snapshot status log",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="aggregate_eps_wayback_timeline",
                path=timeline_path,
                row_count=len(timeline_df),
                min_date=min(timeline_df["snapshot_date"]).isoformat() if not timeline_df.empty else None,
                max_date=max(timeline_df["snapshot_date"]).isoformat() if not timeline_df.empty else None,
                notes="Wayback EPS historical snapshot timeline parquet",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="aggregate_eps_wayback_report",
                path=report_path,
                row_count=len(timeline_df),
                min_date=min(timeline_df["snapshot_date"]).isoformat() if not timeline_df.empty else None,
                max_date=max(timeline_df["snapshot_date"]).isoformat() if not timeline_df.empty else None,
                notes="Wayback EPS fetch report",
            )
            store.finish_fetch_run(run_id=fetch_run.run_id, status="ok")
        return report_path
    except Exception as exc:
        _LOG.error("Wayback EPS fetch run failed: %s", exc, exc_info=True)
        if store and fetch_run:
            try:
                store.finish_fetch_run(run_id=fetch_run.run_id, status="failed", notes=str(exc))
            except Exception:
                _LOG.warning(
                    "Could not mark fetch run %d as failed in acquisition store",
                    fetch_run.run_id,
                    exc_info=True,
                )
        raise


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
