from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from regime_data_fetch.aggregate_eps_models import (
    AggregateEPSFetchError,
    EPSWaybackSnapshot,
)


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


def filter_wayback_snapshots(
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


def append_wayback_status(
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
