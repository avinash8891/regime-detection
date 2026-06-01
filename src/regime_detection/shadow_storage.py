"""SQLite-backed shadow-runner storage helpers.

Authoritative spec: ``docs/shadow_runner_spec.md`` (§§1-7). The schemas
and write contract here are a verbatim implementation; the regime engine
spec at ``docs/regime_engine_v2_spec.md`` L28 and L33 explicitly
delegates qualification storage details to that file. All write
operations are keyed on the canonical identity tuple
``(as_of_date, engine_version, config_version)`` per shadow_runner_spec
§3 L93.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from regime_data_fetch.artifact_store import sha256_file
from regime_detection.loaders import load_event_calendar
from regime_shared.pandas_compat import cow_safe_assign


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        suppress = super().__exit__(exc_type, exc_val, exc_tb)
        self.close()
        return suppress


class RunStatus(str, Enum):
    """Status enum for the runs table. Tokens are pinned by spec
    shadow_runner_spec.md §5 L115-119."""

    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE = "failure"


RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY,
    run_timestamp TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    config_version TEXT NOT NULL,
    status TEXT NOT NULL,
    failure_reason TEXT,
    input_archive_path TEXT NOT NULL,
    output_path TEXT,
    output_sha256 TEXT,
    UNIQUE (as_of_date, engine_version, config_version)
)
"""

REPLAY_CHECKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS replay_checks (
    check_id INTEGER PRIMARY KEY,
    check_timestamp TEXT NOT NULL,
    original_run_id INTEGER REFERENCES runs(run_id),
    matches BOOLEAN NOT NULL,
    diff TEXT
)
"""

INCIDENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_id INTEGER PRIMARY KEY,
    incident_date TEXT NOT NULL,
    description TEXT NOT NULL,
    resolution TEXT,
    breaks_qualification BOOLEAN NOT NULL
)
"""


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_shadow_layout(
    output_root: Path,
    *,
    db_name: str = "regime_shadow.db",
    include_reports: bool = False,
) -> dict[str, Path]:
    paths = {
        "db": output_root / db_name,
        "outputs": output_root / "outputs",
        "input_archives": output_root / "input_archives",
    }
    if include_reports:
        paths["reports"] = output_root / "reports"
    output_root.mkdir(parents=True, exist_ok=True)
    paths["outputs"].mkdir(parents=True, exist_ok=True)
    paths["input_archives"].mkdir(parents=True, exist_ok=True)
    if include_reports:
        paths["reports"].mkdir(parents=True, exist_ok=True)
    return paths


def open_shadow_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0, factory=_ClosingConnection)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(RUNS_SCHEMA)
    conn.execute(REPLAY_CHECKS_SCHEMA)
    conn.execute(INCIDENTS_SCHEMA)
    conn.commit()
    return conn


def event_rows_for_yaml(event_df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if event_df is None or event_df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for row in event_df.to_dict(orient="records"):
        out: dict[str, Any] = {}
        for key, value in row.items():
            out[key] = event_value_for_yaml(value)
        rows.append(out)
    return rows


def event_value_for_yaml(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: event_value_for_yaml(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [event_value_for_yaml(item) for item in value]
    if value is None:
        return None
    if bool(pd.isna(value)):
        return None
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(value).date().isoformat()
    return value


def write_archived_inputs(
    *,
    archive_dir: Path,
    market_slice: pd.DataFrame,
    event_df: pd.DataFrame | None,
    macro_series: dict[str, pd.Series] | None = None,
    v2_daily_slice: pd.DataFrame | None = None,
    pit_intervals: pd.DataFrame | None = None,
) -> tuple[Path, Path, Path]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    market_path = archive_dir / "market_data.parquet"
    events_path = archive_dir / "events.yaml"
    macro_path = archive_dir / "macro_series.parquet"
    v2_daily_path = archive_dir / "v2_daily.parquet"
    pit_path = archive_dir / "pit_constituent_intervals.parquet"
    checksums_path = archive_dir / "checksums.json"

    market_slice.to_parquet(market_path, index=False)
    events_path.write_text(
        yaml.safe_dump({"events": event_rows_for_yaml(event_df)}, sort_keys=False),
        encoding="utf-8",
    )
    checksums = {
        "market_data.parquet": sha256_file(market_path),
        "events.yaml": sha256_file(events_path),
    }
    if macro_series:
        _macro_series_frame(macro_series).to_parquet(macro_path, index=False)
        checksums["macro_series.parquet"] = sha256_file(macro_path)
    # F-001: when the runner feeds V2 inputs derived from a daily-OHLCV frame
    # (sector/cross-asset closes, PIT membership, constituent OHLCV), archive the
    # as-of slice so a walk-forward replay can recompute the V2 axes byte-identically.
    # Archive even an empty slice (a non-None frame whose as-of precedes the first v2
    # row) so the archive faithfully records that V2 inputs were supplied-but-empty,
    # rather than leaving the missing file ambiguous with "V2 not provided".
    if v2_daily_slice is not None:
        v2_daily_slice.to_parquet(v2_daily_path, index=False)
        checksums["v2_daily.parquet"] = sha256_file(v2_daily_path)
    # CR-004: archive the explicit PIT membership frame (when the run used one) so a
    # replay reconstructs the SAME constituent universe, instead of silently deriving a
    # different default-from-daily universe and false-failing the gate forever.
    if pit_intervals is not None:
        pit_intervals.to_parquet(pit_path, index=False)
        checksums["pit_constituent_intervals.parquet"] = sha256_file(pit_path)
    checksums_path.write_text(
        json.dumps(
            checksums,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return market_path, events_path, checksums_path


def _macro_series_frame(macro_series: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for logical_name, series in sorted(macro_series.items()):
        clean = series.dropna()
        for observed_date, value in clean.items():
            rows.append(
                {
                    "date": pd.Timestamp(observed_date),
                    "series_id": str(logical_name).upper(),
                    "logical_name": str(logical_name),
                    "value": float(value),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["date", "series_id", "logical_name", "value"],
    )


def load_archived_macro_series(path: Path) -> dict[str, pd.Series] | None:
    if not path.exists():
        return None
    from regime_detection.loaders import load_macro_series

    series = load_macro_series(path)
    # CR-002: the archive (written by _macro_series_frame) stores the exact values the
    # runner already passed to classify — i.e. POST the load-time transform. But
    # load_macro_series re-applies its implied_vol_30d `/100` raw-ingest transform on
    # reload, double-dividing the archived value (100x too small) and breaking the
    # archive→reload identity replay depends on. implied_vol_30d is the only macro key
    # with a load-time transform (loaders.load_macro_series), so undo exactly that one.
    iv = series.get("implied_vol_30d")
    if iv is not None:
        series["implied_vol_30d"] = iv * 100.0
    return series


def insert_run_row(
    *,
    conn: sqlite3.Connection,
    run_timestamp: str,
    as_of_date: date,
    engine_version: str,
    config_version: str,
    archive_dir: Path,
) -> None:
    conn.execute(
        """
        INSERT INTO runs (
            run_timestamp, as_of_date, engine_version, config_version,
            status, failure_reason, input_archive_path, output_path, output_sha256
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_timestamp,
            as_of_date.isoformat(),
            engine_version,
            config_version,
            RunStatus.IN_PROGRESS.value,
            None,
            str(archive_dir),
            None,
            None,
        ),
    )
    conn.commit()


def update_run_row_success(
    *,
    conn: sqlite3.Connection,
    as_of_date: date,
    engine_version: str,
    config_version: str,
    output_path: Path,
) -> None:
    conn.execute(
        """
        UPDATE runs
        SET status = ?, output_path = ?, output_sha256 = ?, failure_reason = NULL
        WHERE as_of_date = ? AND engine_version = ? AND config_version = ?
        """,
        (
            RunStatus.SUCCESS.value,
            str(output_path),
            sha256_file(output_path),
            as_of_date.isoformat(),
            engine_version,
            config_version,
        ),
    )
    conn.commit()


def update_run_row_failure(
    *,
    conn: sqlite3.Connection,
    as_of_date: date,
    engine_version: str,
    config_version: str,
    failure_reason: str,
) -> None:
    conn.execute(
        """
        UPDATE runs
        SET status = ?, failure_reason = ?
        WHERE as_of_date = ? AND engine_version = ? AND config_version = ?
        """,
        (
            RunStatus.FAILURE.value,
            failure_reason,
            as_of_date.isoformat(),
            engine_version,
            config_version,
        ),
    )
    conn.commit()


def load_archived_market_data(path: Path) -> pd.DataFrame:
    archived = pd.read_parquet(path)
    return cow_safe_assign(
        archived,
        {"date": pd.to_datetime(archived["date"]).dt.date},
    )


def load_archived_v2_daily(path: Path) -> pd.DataFrame | None:
    """Load the archived as-of V2 daily-OHLCV slice (F-001), or None if absent
    (V1-only runs do not archive it). Same date-normalization as market_data."""
    if not path.exists():
        return None
    archived = pd.read_parquet(path)
    return cow_safe_assign(
        archived,
        {"date": pd.to_datetime(archived["date"]).dt.date},
    )


def load_archived_pit_intervals(path: Path) -> pd.DataFrame | None:
    """Load the archived explicit PIT membership frame (CR-004), or None if absent
    (the run used the default-from-daily membership). The ticker/start_date/end_date
    columns round-trip faithfully through parquet (date objects, None for open intervals).
    """
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_archived_event_calendar(path: Path) -> pd.DataFrame | None:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not loaded:
        return None
    events = loaded.get("events", []) if isinstance(loaded, dict) else loaded
    if not events:
        return None
    return load_event_calendar(path)


def fetch_run_row(
    *,
    conn: sqlite3.Connection,
    as_of_date: date,
    engine_version: str | None = None,
    config_version: str | None = None,
) -> sqlite3.Row | None:
    """Fetch a runs row by the canonical identity tuple.

    The runs table is keyed on ``(as_of_date, engine_version,
    config_version)`` per shadow_runner_spec.md §3 L93. When only
    ``as_of_date`` is supplied (legacy callers), returns the most
    recently inserted row for that date — deterministic across the
    qualification-breaking restart case where multiple rows can share
    one ``as_of_date``. Pass ``engine_version`` and ``config_version``
    explicitly to query the exact frozen-version row.
    """
    conn.row_factory = sqlite3.Row
    if (engine_version is None) != (config_version is None):
        raise ValueError("engine_version and config_version must be supplied together")
    if engine_version is not None and config_version is not None:
        row = conn.execute(
            """
            SELECT run_id, as_of_date, status, engine_version, config_version,
                   input_archive_path, output_path
            FROM runs
            WHERE as_of_date = ? AND engine_version = ? AND config_version = ?
            """,
            (as_of_date.isoformat(), engine_version, config_version),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT run_id, as_of_date, status, engine_version, config_version,
                   input_archive_path, output_path
            FROM runs
            WHERE as_of_date = ?
            ORDER BY run_id DESC
            LIMIT 1
            """,
            (as_of_date.isoformat(),),
        ).fetchone()
    conn.row_factory = None
    return row


def insert_replay_check(
    *,
    conn: sqlite3.Connection,
    check_timestamp: str,
    original_run_id: int,
    matches: bool,
    diff: dict[str, Any] | None,
) -> None:
    conn.execute(
        """
        INSERT INTO replay_checks (check_timestamp, original_run_id, matches, diff)
        VALUES (?, ?, ?, ?)
        """,
        (
            check_timestamp,
            original_run_id,
            int(matches),
            None if diff is None else json.dumps(diff, sort_keys=True),
        ),
    )
    conn.commit()


def insert_incident(
    *,
    conn: sqlite3.Connection,
    incident_date: date,
    description: str,
    resolution: str | None,
    breaks_qualification: bool,
) -> None:
    conn.execute(
        """
        INSERT INTO incidents (incident_date, description, resolution, breaks_qualification)
        VALUES (?, ?, ?, ?)
        """,
        (
            incident_date.isoformat(),
            description,
            resolution,
            int(breaks_qualification),
        ),
    )
    conn.commit()
