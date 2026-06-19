"""Publish canonical snapshot for a regime-detection manifest.

Closes the manifest-sha-drift gap: after a fetcher rewrites a parquet under
``data/raw/``, this script canonicalizes the on-disk bytes, recomputes
sha256/rows/min_date/max_date, uploads the canonical artifact to S3 (via the
manifest's storage_root), and rewrites the manifest entry. Idempotent.

Modes:
    publish (default)  canonicalize on disk + upload changed bytes + rewrite manifest
    --dry-run          report what publish would do; touch nothing
    --check            in-memory recompute; exit 1 on any drift; touch nothing
    --check-store      with --check, also verify artifacts in the manifest store

The script reuses the project's ``LocalArtifactStore`` / ``S3ArtifactStore``
(see ``src/regime_data_fetch/artifact_store.py``) for uploads so key
normalization and sha verification stay consistent with the rest of the
codebase. No fetcher code is modified.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import pyarrow.parquet as pq

# Allow running as a script without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from regime_data_fetch.artifact_store import (  # noqa: E402
    ArtifactStore,
    build_artifact_store,
    content_addressed_key,
    sha256_bytes,
    sha256_file,
    strip_content_address,
)
from regime_data_fetch.canonical_parquet import (  # noqa: E402
    canonicalize_parquet_bytes,
)
from regime_data_fetch.daily_ohlcv_contract import (  # noqa: E402
    daily_ohlcv_artifact_name,
    parse_daily_ohlcv_artifact_name,
    require_symbol_partition_table,
)

LOGGER = logging.getLogger("publish_canonical_snapshot")

DEFAULT_MANIFEST = Path("manifests/runs/regime_engine_2026-05-17.yaml")
DEFAULT_DATA_ROOT = Path("data/raw")

# Search order for the date-like column to populate min_date / max_date.
DATE_COLUMN_CANDIDATES: tuple[str, ...] = (
    "date",
    "period",
    "observation_date",
    "release_date",
    "start_date",
    "meeting_end_date",
    "speech_date",
)

_DAILY_OHLCV_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
    "symbol",
)


# ---------------------------------------------------------------------------
# YAML I/O with anchor preservation


def _make_ruamel_yaml() -> Any | None:
    try:
        from ruamel.yaml import YAML  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - exercised when ruamel missing
        return None
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096  # avoid line-wrap churn
    y.indent(mapping=2, sequence=2, offset=0)
    return y


def _load_manifest_payload(path: Path) -> tuple[Any, Any | None]:
    """Load manifest. Returns (payload, ruamel_instance_or_None).

    When ruamel is available the payload is a ruamel ``CommentedMap`` /
    ``CommentedSeq`` graph that preserves anchors, comments, and key order
    on round-trip. Otherwise we fall back to ``yaml.safe_load``.
    """
    y = _make_ruamel_yaml()
    if y is not None:
        with path.open("r") as handle:
            payload = y.load(handle)
        return payload, y
    import yaml  # type: ignore[import-untyped]

    LOGGER.warning(
        "ruamel.yaml unavailable; falling back to yaml.safe_dump (anchors will expand)"
    )
    payload = yaml.safe_load(path.read_text())
    return payload, None


def _dump_manifest_atomically(
    payload: Any, ruamel_instance: Any | None, path: Path
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if ruamel_instance is not None:
        with tmp.open("w") as handle:
            ruamel_instance.dump(payload, handle)
    else:
        import yaml  # type: ignore[import-untyped]

        tmp.write_text(yaml.safe_dump(payload, sort_keys=False))
    os.replace(tmp, path)


def _parquet_row_count(payload: bytes) -> int:
    reader = pq.ParquetFile(io.BytesIO(payload))
    return int(reader.metadata.num_rows)


def _parquet_date_range(payload: bytes) -> tuple[str | None, str | None]:
    table = pq.read_table(io.BytesIO(payload))
    columns = list(table.column_names)
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in columns:
            series = table.column(candidate).to_pandas()
            non_null = series.dropna()
            if non_null.empty:
                return None, None
            try:
                import pandas as pd

                parsed = pd.to_datetime(non_null, errors="coerce").dropna()
                if parsed.empty:
                    return None, None
                return (
                    parsed.min().strftime("%Y-%m-%d"),
                    parsed.max().strftime("%Y-%m-%d"),
                )
            except Exception:  # pragma: no cover - defensive
                values = sorted(str(v) for v in non_null.tolist())
                return values[0], values[-1]
    return None, None


def _validate_daily_ohlcv_symbol_contract(
    *,
    artifact: dict[str, Any],
    payload: bytes,
) -> None:
    _validate_daily_ohlcv_contract(
        artifact=artifact,
        payload=payload,
        expected_sessions=None,
    )


def _daily_ohlcv_symbol(artifact: dict[str, Any]) -> str | None:
    return parse_daily_ohlcv_artifact_name(str(artifact.get("name", "")))


def _validate_daily_ohlcv_contract(
    *,
    artifact: dict[str, Any],
    payload: bytes,
    expected_sessions: pd.DatetimeIndex | None,
    active_intervals_by_symbol: (
        dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]] | None
    ) = None,
) -> None:
    expected_symbol = _daily_ohlcv_symbol(artifact)
    if expected_symbol is None:
        return
    table = pq.read_table(io.BytesIO(payload))
    missing_columns = [
        col for col in _DAILY_OHLCV_REQUIRED_COLUMNS if col not in table.column_names
    ]
    if missing_columns:
        raise ValueError(
            f"{artifact['name']} missing required daily OHLCV column(s): {missing_columns}"
        )
    require_symbol_partition_table(
        table, expected_symbol=expected_symbol, source=artifact["name"]
    )
    frame = table.select(list(_DAILY_OHLCV_REQUIRED_COLUMNS)).to_pandas()
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError(f"{artifact['name']} has unparsable/null date row(s)")
    duplicate_dates = dates[dates.duplicated()].dt.strftime("%Y-%m-%d").unique()
    if len(duplicate_dates) > 0:
        examples = ", ".join(sorted(duplicate_dates)[:5])
        raise ValueError(
            f"{artifact['name']} has duplicate date row(s); examples: {examples}"
        )
    value_columns = ["open", "high", "low", "close", "volume", "adjusted_close"]
    null_value_columns = [col for col in value_columns if frame[col].isna().any()]
    if null_value_columns:
        raise ValueError(
            f"{artifact['name']} has null OHLCV value column(s): {null_value_columns}"
        )
    if expected_sessions is None or dates.empty:
        return
    symbol_dates = pd.DatetimeIndex(dates.sort_values().unique())
    expected = _expected_symbol_sessions(
        symbol=expected_symbol,
        symbol_dates=symbol_dates,
        expected_sessions=expected_sessions,
        active_intervals_by_symbol=active_intervals_by_symbol,
    )
    if expected.empty:
        return
    missing = expected.difference(symbol_dates)
    if not missing.empty:
        examples = ", ".join(ts.strftime("%Y-%m-%d") for ts in missing[:5])
        raise ValueError(
            "daily OHLCV calendar coverage gap: "
            f"symbol={expected_symbol} missing {len(missing)} session row(s); "
            f"examples: {examples}"
        )


def _expected_symbol_sessions(
    *,
    symbol: str,
    symbol_dates: pd.DatetimeIndex,
    expected_sessions: pd.DatetimeIndex,
    active_intervals_by_symbol: (
        dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]] | None
    ),
) -> pd.DatetimeIndex:
    if active_intervals_by_symbol is None:
        return expected_sessions[
            (expected_sessions >= symbol_dates.min())
            & (expected_sessions <= symbol_dates.max())
        ]
    intervals = active_intervals_by_symbol.get(symbol)
    if not intervals:
        return pd.DatetimeIndex([])
    active_sessions: list[pd.Timestamp] = []
    for start, end in intervals:
        # PIT end dates are effective-removal dates in the upstream source.
        # Treat them as exclusive so an acquisition/removal day without a final
        # quote does not become a false calendar gap.
        interval_sessions = expected_sessions[
            (expected_sessions >= start)
            & (expected_sessions >= symbol_dates.min())
            & (expected_sessions <= symbol_dates.max())
        ]
        if end is not None:
            interval_sessions = interval_sessions[interval_sessions < end]
        active_sessions.extend(interval_sessions)
    return pd.DatetimeIndex(sorted(set(active_sessions)))


def _validate_parquet_manifest_metadata(
    *,
    artifact: dict[str, Any],
    payload: bytes,
) -> None:
    expected_rows = artifact.get("rows")
    actual_rows = _parquet_row_count(payload)
    if expected_rows is not None and int(expected_rows) != actual_rows:
        raise ValueError(
            f"{artifact['name']} manifest rows={expected_rows} but parquet rows={actual_rows}"
        )
    actual_min, actual_max = _parquet_date_range(payload)
    expected_min = artifact.get("min_date")
    expected_max = artifact.get("max_date")
    if expected_min is not None and actual_min != str(expected_min):
        raise ValueError(
            f"{artifact['name']} manifest min_date={expected_min} but parquet min_date={actual_min}"
        )
    if expected_max is not None and actual_max != str(expected_max):
        raise ValueError(
            f"{artifact['name']} manifest max_date={expected_max} but parquet max_date={actual_max}"
        )


def _semantic_check_artifact(
    *,
    artifact: dict[str, Any],
    local: Path,
    expected_sessions: pd.DatetimeIndex | None,
    active_intervals_by_symbol: (
        dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]] | None
    ),
) -> tuple[str | None, str]:
    if not local.exists() or not _is_parquet(local):
        return None, ""
    try:
        payload = canonicalize_parquet_bytes(local)
        _validate_parquet_manifest_metadata(artifact=artifact, payload=payload)
        _validate_daily_ohlcv_contract(
            artifact=artifact,
            payload=payload,
            expected_sessions=expected_sessions,
            active_intervals_by_symbol=active_intervals_by_symbol,
        )
    except Exception as exc:
        return "SEMANTIC_INVALID", str(exc)
    return "SEMANTIC_OK", ""


def _daily_ohlcv_spy_sessions(
    payload: Any,
    data_root: Path,
) -> pd.DatetimeIndex | None:
    artifacts = payload.get("artifacts", []) if isinstance(payload, dict) else []
    spy_artifact = next(
        (
            artifact
            for artifact in artifacts
            if isinstance(artifact, dict)
            and artifact.get("name") == daily_ohlcv_artifact_name("SPY")
        ),
        None,
    )
    if spy_artifact is None:
        return None
    spy_path = _resolve_local_path(data_root, spy_artifact)
    if not spy_path.exists():
        return None
    table = pq.ParquetFile(spy_path).read(columns=["date"])
    dates = pd.to_datetime(table.column("date").to_pandas(), errors="coerce")
    dates = dates.dropna().dt.normalize()
    if dates.empty:
        return None
    return pd.DatetimeIndex(sorted(dates.unique()))


def _daily_ohlcv_pit_active_intervals(
    payload: Any,
    data_root: Path,
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]] | None:
    artifacts = payload.get("artifacts", []) if isinstance(payload, dict) else []
    pit_artifact = next(
        (
            artifact
            for artifact in artifacts
            if isinstance(artifact, dict)
            and artifact.get("name") == "sp500_pit_constituents"
        ),
        None,
    )
    if pit_artifact is None:
        return None
    pit_path = _resolve_local_path(data_root, pit_artifact)
    if not pit_path.exists():
        return None
    frame = pd.read_parquet(pit_path)
    required = {"ticker", "start_date", "end_date"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"sp500_pit_constituents missing column(s): {sorted(missing)}")
    intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]] = {}
    for row in frame.itertuples(index=False):
        ticker = str(getattr(row, "ticker")).strip().upper()
        if not ticker:
            continue
        start = pd.Timestamp(getattr(row, "start_date")).normalize()
        raw_end = getattr(row, "end_date")
        end = None if pd.isna(raw_end) else pd.Timestamp(raw_end).normalize()
        intervals.setdefault(ticker, []).append((start, end))
    return intervals


# ---------------------------------------------------------------------------
# Per-artifact processing


@dataclass
class ArtifactReport:
    name: str
    local_path: str
    # Local-tree status. One of:
    #   MATCH, DRIFT, MISSING, READ_ERROR,
    #   CANONICALIZE_WOULD_CHANGE_SHA, WOULD_UPLOAD,  (dry-run)
    #   UPDATED, UPLOADED                              (publish)
    local_status: str
    old_sha: str
    new_sha: str | None = None
    # Store-side status (only populated by run_check when a store is supplied):
    #   MATCH, DRIFT, MISSING, UNVERIFIABLE
    store_status: str | None = None
    store_sha: str | None = None
    semantic_status: str | None = None
    note: str = ""
    manifest_updates: dict[str, object] | None = None

    @property
    def is_issue(self) -> bool:
        # UNVERIFIABLE means the store has the object but cannot vouch for
        # its digest (e.g. S3 object missing the sha256 user-metadata header).
        # The local check has already verified content integrity against the
        # manifest, so a missing store digest is logged as a warning rather
        # than failing the run. Real store mismatches (DRIFT/MISSING) still
        # count as issues.
        if self.local_status not in {"MATCH", "UPDATED", "UPLOADED"}:
            return True
        if self.store_status in {"DRIFT", "MISSING"}:
            return True
        if self.semantic_status not in {None, "SEMANTIC_OK"}:
            return True
        return False


def _resolve_local_path(data_root: Path, artifact: dict[str, Any]) -> Path:
    """Resolve manifest ``local_path`` against the data root.

    Manifest ``local_path`` values are repo-relative (e.g.
    ``data/raw/macro/fred.parquet`` or ``configs/events/us_events.yaml``).
    Strategy: if the path begins with the conventional ``data/raw/`` prefix
    the remainder is resolved under ``data_root`` (the dedicated raw data
    tree, defaulting to ``<repo>/data/raw``). Anything else (e.g.
    ``configs/events/us_events.yaml``) is resolved relative to the repo root.
    """
    raw = Path(artifact["local_path"])
    if raw.is_absolute():
        return raw
    parts = raw.parts
    if parts[:2] == ("data", "raw"):
        rel = Path(*parts[2:])
        return data_root / rel
    return _REPO_ROOT / raw


def _is_parquet(path: Path) -> bool:
    return path.suffix.lower() == ".parquet"


def _filter_artifacts(
    artifacts: Iterable[dict[str, Any]], only: list[str] | None
) -> list[dict[str, Any]]:
    if not only:
        return list(artifacts)
    wanted = set(only)
    return [a for a in artifacts if a["name"] in wanted]


# ---------------------------------------------------------------------------
# Mode handlers


def _check_local(
    artifact: dict[str, Any], local: Path
) -> tuple[str, str | None, str | None]:
    """Return ``(status, new_sha, note)`` for a local-tree check.

    ``new_sha`` for parquet artifacts is the *canonicalized* hash, matching
    what ``publish``/``put_bytes`` would have stored — keep this in sync with
    the store's digest scheme so local and store checks compare like with like.
    """
    old_sha = str(artifact["sha256"])
    if not local.exists():
        return "MISSING", None, None
    if _is_parquet(local):
        try:
            canon = canonicalize_parquet_bytes(local)
        except Exception as exc:  # pragma: no cover - defensive
            return "READ_ERROR", None, str(exc)
        new_sha = sha256_bytes(canon)
    else:
        new_sha = sha256_file(local)
    return ("MATCH" if new_sha == old_sha else "DRIFT"), new_sha, None


def _check_store(
    artifact: dict[str, Any],
    store: ArtifactStore,
    *,
    local: Path,
    local_sha: str | None,
) -> tuple[str, str | None]:
    """Return ``(store_status, store_sha)`` for a store-side check.

    Delegates to ``store.check_file`` which does the sha comparison and
    returns a typed verdict (MATCH/DRIFT/MISSING/UNVERIFIABLE). Passes the
    local path/sha as a hint so a local-store backend whose object resolves
    to the same file the caller already hashed can skip the second hash.
    """
    result = store.check_file(
        str(artifact["uri"]),
        expected_sha256=str(artifact["sha256"]),
        known_path=local if local.exists() else None,
        known_sha=local_sha,
    )
    return result.status.name, result.observed_sha


def run_check(
    payload: Any,
    data_root: Path,
    only: list[str] | None,
    *,
    store: ArtifactStore | None = None,
) -> tuple[int, list[ArtifactReport]]:
    reports: list[ArtifactReport] = []
    spy_sessions = _daily_ohlcv_spy_sessions(payload, data_root)
    active_intervals_by_symbol = _daily_ohlcv_pit_active_intervals(payload, data_root)
    for artifact in _filter_artifacts(payload["artifacts"], only):
        local = _resolve_local_path(data_root, artifact)
        local_status, new_sha, note = _check_local(artifact, local)
        semantic_status, semantic_note = _semantic_check_artifact(
            artifact=artifact,
            local=local,
            expected_sessions=spy_sessions,
            active_intervals_by_symbol=active_intervals_by_symbol,
        )
        if semantic_note:
            note = "; ".join(part for part in (note, semantic_note) if part)
        store_status: str | None = None
        store_sha: str | None = None
        if store is not None:
            store_status, store_sha = _check_store(
                artifact, store, local=local, local_sha=new_sha
            )
        reports.append(
            ArtifactReport(
                name=artifact["name"],
                local_path=str(local),
                local_status=local_status,
                old_sha=str(artifact["sha256"]),
                new_sha=new_sha,
                store_status=store_status,
                store_sha=store_sha,
                semantic_status=semantic_status,
                note=note or "",
            )
        )
    exit_code = 0 if all(not r.is_issue for r in reports) else 1
    return exit_code, reports


def run_dry_run(
    payload: Any, data_root: Path, only: list[str] | None
) -> list[ArtifactReport]:
    reports: list[ArtifactReport] = []
    for artifact in _filter_artifacts(payload["artifacts"], only):
        local = _resolve_local_path(data_root, artifact)
        old_sha = str(artifact["sha256"])
        if not local.exists():
            reports.append(
                ArtifactReport(artifact["name"], str(local), "MISSING", old_sha)
            )
            continue
        if _is_parquet(local):
            current_sha = sha256_file(local)
            canon = canonicalize_parquet_bytes(local)
            new_sha = sha256_bytes(canon)
            if new_sha == old_sha and current_sha == old_sha:
                status = "MATCH"
            elif new_sha == old_sha and current_sha != old_sha:
                # On-disk bytes drift, but canonicalization would restore.
                status = "CANONICALIZE_WOULD_CHANGE_SHA"
            elif new_sha != old_sha:
                status = "WOULD_UPLOAD"
            else:  # pragma: no cover - unreachable
                status = "MATCH"
        else:
            new_sha = sha256_file(local)
            status = "MATCH" if new_sha == old_sha else "WOULD_UPLOAD"
        reports.append(
            ArtifactReport(artifact["name"], str(local), status, old_sha, new_sha)
        )
    return reports


def run_publish(
    payload: Any,
    data_root: Path,
    only: list[str] | None,
    *,
    skip_upload: bool,
    store: ArtifactStore | None,
) -> list[ArtifactReport]:
    reports: list[ArtifactReport] = []
    artifacts = payload["artifacts"]
    selected_names = set(a["name"] for a in _filter_artifacts(artifacts, only))
    for artifact in artifacts:
        if artifact["name"] not in selected_names:
            continue
        local = _resolve_local_path(data_root, artifact)
        old_sha = str(artifact["sha256"])
        if not local.exists():
            reports.append(
                ArtifactReport(artifact["name"], str(local), "MISSING", old_sha)
            )
            continue

        if _is_parquet(local):
            canon = canonicalize_parquet_bytes(local)
            _validate_daily_ohlcv_symbol_contract(artifact=artifact, payload=canon)
            new_sha = sha256_bytes(canon)
            disk_sha = sha256_file(local)
            if disk_sha != new_sha:
                # Rewrite on-disk file with canonical bytes (atomic).
                tmp = local.with_suffix(local.suffix + ".canon.tmp")
                tmp.write_bytes(canon)
                os.replace(tmp, local)
        else:
            new_sha = sha256_file(local)
            canon = None

        if new_sha == old_sha:
            reports.append(
                ArtifactReport(artifact["name"], str(local), "MATCH", old_sha, new_sha)
            )
            continue

        # Content changed: mint a NEW content-addressed key so the new object never
        # overwrites the one an older lockfile still pins. Derive the logical key
        # from local_path (stable) — not the current uri, which may already carry a
        # prior sha.
        new_key = content_addressed_key(
            strip_content_address(str(artifact["uri"])), new_sha
        )
        if not skip_upload and store is not None:
            store.put_file(local, new_key)

        # Update artifact in place to preserve YAML anchors / structure.
        manifest_updates: dict[str, object] = {"sha256": new_sha, "uri": new_key}
        artifact["sha256"] = new_sha
        artifact["uri"] = new_key
        if _is_parquet(local):
            if canon is None:
                raise RuntimeError(f"canonical parquet bytes missing for {local}")
            rows = _parquet_row_count(canon)
            artifact["rows"] = rows
            manifest_updates["rows"] = rows
            min_d, max_d = _parquet_date_range(canon)
            if min_d is not None:
                artifact["min_date"] = min_d
                manifest_updates["min_date"] = min_d
            if max_d is not None:
                artifact["max_date"] = max_d
                manifest_updates["max_date"] = max_d
        status = "UPDATED" if skip_upload or store is None else "UPLOADED"
        reports.append(
            ArtifactReport(
                artifact["name"],
                str(local),
                status,
                old_sha,
                new_sha,
                manifest_updates=manifest_updates,
            )
        )
    return reports


# ---------------------------------------------------------------------------
# Targeted manifest patching


_ARTIFACT_NAME_RE = re.compile(r"^(?P<indent>\s*)-\s+name:\s+(?P<name>.+?)\s*$")


def _patch_manifest_artifact_metadata(
    *,
    manifest_path: Path,
    updates_by_name: dict[str, dict[str, object]],
) -> None:
    """Patch changed artifact scalar metadata without round-tripping YAML.

    The run manifests contain thousands of artifact entries and YAML anchors.
    A full YAML dump can rewrite unrelated ``null`` / blank-null spellings and
    anchors. This patcher limits writes to scalar fields inside artifact blocks
    that actually changed.
    """
    if not updates_by_name:
        return

    original = manifest_path.read_text()
    lines = original.splitlines(keepends=True)
    spans = _artifact_block_spans(lines)
    missing = set(updates_by_name) - set(spans)
    if missing:
        names = ", ".join(sorted(missing))
        raise RuntimeError(f"Manifest missing artifact block(s): {names}")

    patched = lines[:]
    for name, updates in updates_by_name.items():
        start, end = spans[name]
        _patch_artifact_block_lines(patched, start=start, end=end, updates=updates)

    rendered = "".join(patched)
    if rendered == original:
        return
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(rendered)
    os.replace(tmp, manifest_path)


def _artifact_block_spans(lines: list[str]) -> dict[str, tuple[int, int]]:
    spans: dict[str, tuple[int, int]] = {}
    current_name: str | None = None
    current_start: int | None = None
    current_indent: str | None = None

    for index, line in enumerate(lines):
        match = _ARTIFACT_NAME_RE.match(line.rstrip("\n"))
        if match is None:
            continue
        indent = match.group("indent")
        if (
            current_name is not None
            and current_start is not None
            and indent == current_indent
        ):
            spans[current_name] = (current_start, index)
        current_name = _parse_yaml_scalar(match.group("name"))
        current_start = index
        current_indent = indent

    if current_name is not None and current_start is not None:
        spans[current_name] = (current_start, len(lines))
    return spans


def _patch_artifact_block_lines(
    lines: list[str],
    *,
    start: int,
    end: int,
    updates: dict[str, object],
) -> None:
    remaining = dict(updates)
    for index in range(start + 1, end):
        key, current_value = _split_yaml_key_value(lines[index])
        if key not in remaining:
            continue
        value = remaining.pop(key)
        newline = "\n" if lines[index].endswith("\n") else ""
        prefix = lines[index].split(":", 1)[0]
        lines[index] = f"{prefix}: {_format_yaml_scalar(value, current_value)}{newline}"
    if remaining:
        keys = ", ".join(sorted(remaining))
        artifact_name = lines[start].strip()
        raise RuntimeError(
            f"Manifest artifact block {artifact_name!r} missing field(s): {keys}"
        )


def _split_yaml_key_value(line: str) -> tuple[str | None, str]:
    stripped_newline = line.rstrip("\n")
    if ":" not in stripped_newline:
        return None, ""
    key_part, value_part = stripped_newline.split(":", 1)
    key = key_part.strip()
    if not key or key.startswith("- "):
        return None, ""
    return key, value_part.strip()


def _parse_yaml_scalar(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _format_yaml_scalar(value: object, current_value: str) -> str:
    if value is None:
        return "" if current_value == "" else "null"
    if isinstance(value, int):
        return str(value)
    rendered = str(value)
    if current_value.startswith("'") and current_value.endswith("'"):
        return "'" + rendered.replace("'", "''") + "'"
    if current_value.startswith('"') and current_value.endswith('"'):
        return '"' + rendered.replace('"', '\\"') + '"'
    return rendered


# ---------------------------------------------------------------------------
# CLI


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish canonical snapshot for a regime-detection manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument(
        "--check-store",
        action="store_true",
        help="check mode only: also verify manifest-store artifacts",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="publish mode only: canonicalize + rewrite manifest, skip S3 upload",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="process only artifacts with the given name (repeatable)",
    )
    return parser


def _format_report_lines(
    reports: list[ArtifactReport],
    *,
    limit: int | None,
    pick: Callable[[ArtifactReport], tuple[str, str | None]],
) -> str:
    """Render a report list. ``pick`` extracts the ``(status, sha)`` pair to
    surface for each report; the three thin wrappers below cover the three
    sides the script actually uses (local / store / semantic).
    """
    lines: list[str] = []
    for r in reports[:limit] if limit else reports:
        status, new_sha = pick(r)
        new = new_sha[:12] if new_sha else "-"
        lines.append(
            f"  {status:<32} {r.name:<40} old={r.old_sha[:12]} new={new}"
            + (f"  ({r.note})" if r.note else "")
        )
    return "\n".join(lines)


def _format_local_report(
    reports: list[ArtifactReport], *, limit: int | None = None
) -> str:
    return _format_report_lines(
        reports, limit=limit, pick=lambda r: (r.local_status, r.new_sha)
    )


def _format_store_report(
    reports: list[ArtifactReport], *, limit: int | None = None
) -> str:
    return _format_report_lines(
        reports,
        limit=limit,
        pick=lambda r: (f"STORE_{r.store_status or '-'}", r.store_sha),
    )


def _format_semantic_report(
    reports: list[ArtifactReport], *, limit: int | None = None
) -> str:
    return _format_report_lines(
        reports,
        limit=limit,
        pick=lambda r: (r.semantic_status or "-", r.new_sha),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_arg_parser().parse_args(argv)

    manifest_path = (
        args.manifest if args.manifest.is_absolute() else (_REPO_ROOT / args.manifest)
    )
    data_root = (
        args.data_root
        if args.data_root.is_absolute()
        else (_REPO_ROOT / args.data_root)
    )

    payload, ruamel_instance = _load_manifest_payload(manifest_path)
    if not isinstance(payload, dict) or "artifacts" not in payload:
        LOGGER.error("manifest does not contain artifacts list: %s", manifest_path)
        return 2

    if args.check:
        store = (
            build_artifact_store(str(payload["storage_root"]))
            if args.check_store
            else None
        )
        exit_code, reports = run_check(payload, data_root, args.only, store=store)

        def _local(status: str) -> list[ArtifactReport]:
            return [r for r in reports if r.local_status == status]

        def _store(status: str) -> list[ArtifactReport]:
            return [r for r in reports if r.store_status == status]

        local_match = _local("MATCH")
        local_drift = _local("DRIFT")
        local_missing = _local("MISSING")
        store_match = _store("MATCH")
        store_drift = _store("DRIFT")
        store_missing = _store("MISSING")
        store_unverifiable = _store("UNVERIFIABLE")
        semantic_invalid = [
            r for r in reports if r.semantic_status == "SEMANTIC_INVALID"
        ]
        LOGGER.info(
            "check: %d checked | local: %d MATCH, %d DRIFT, %d MISSING | "
            "store: %d MATCH, %d DRIFT, %d MISSING, %d UNVERIFIABLE | "
            "semantic: %d INVALID",
            len(reports),
            len(local_match),
            len(local_drift),
            len(local_missing),
            len(store_match),
            len(store_drift),
            len(store_missing),
            len(store_unverifiable),
            len(semantic_invalid),
        )
        for label, bucket in (
            ("DRIFT", local_drift),
            ("MISSING", local_missing),
        ):
            if bucket:
                LOGGER.info("LOCAL %s:", label)
                LOGGER.info(_format_local_report(bucket))
        for label, bucket in (
            ("DRIFT", store_drift),
            ("MISSING", store_missing),
            ("UNVERIFIABLE", store_unverifiable),
        ):
            if bucket:
                LOGGER.info("STORE %s:", label)
                LOGGER.info(_format_store_report(bucket))
        if semantic_invalid:
            LOGGER.info("SEMANTIC INVALID:")
            LOGGER.info(_format_semantic_report(semantic_invalid))
        return exit_code

    if args.dry_run:
        reports = run_dry_run(payload, data_root, args.only)
        status_counts = Counter(r.local_status for r in reports)
        LOGGER.info("dry-run summary: %s", dict(status_counts))
        non_match = [
            r
            for r in reports
            if r.local_status != "MATCH" and r.local_status != "MISSING"
        ]
        if non_match:
            LOGGER.info("changes preview:")
            LOGGER.info(_format_local_report(non_match))
        return 0

    # publish mode
    store: ArtifactStore | None = None
    if not args.skip_upload:
        store = build_artifact_store(str(payload["storage_root"]))

    reports = run_publish(
        payload, data_root, args.only, skip_upload=args.skip_upload, store=store
    )
    changed = [r for r in reports if r.local_status in {"UPDATED", "UPLOADED"}]
    LOGGER.info("publish: %d processed, %d changed", len(reports), len(changed))
    if changed:
        LOGGER.info(_format_local_report(changed))
        del ruamel_instance
        updates_by_name = {
            report.name: report.manifest_updates
            for report in changed
            if report.manifest_updates is not None
        }
        _patch_manifest_artifact_metadata(
            manifest_path=manifest_path,
            updates_by_name=updates_by_name,
        )
        LOGGER.info("manifest rewritten: %s", manifest_path)
    else:
        LOGGER.info("no changes; manifest left untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
