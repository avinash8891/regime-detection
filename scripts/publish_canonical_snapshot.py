"""Publish canonical snapshot for a regime-detection manifest.

Closes the manifest-sha-drift gap: after a fetcher rewrites a parquet under
``data/raw/``, this script canonicalizes the on-disk bytes, recomputes
sha256/rows/min_date/max_date, uploads the canonical artifact to S3 (via the
manifest's storage_root), and rewrites the manifest entry. Idempotent.

Modes:
    publish (default)  canonicalize on disk + upload changed bytes + rewrite manifest
    --dry-run          report what publish would do; touch nothing
    --check            in-memory recompute; exit 1 on any drift; touch nothing

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.types as pat

# Allow running as a script without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from regime_data_fetch.artifact_store import (  # noqa: E402
    ArtifactStore,
    build_artifact_store,
    sha256_bytes,
    sha256_file,
)


LOGGER = logging.getLogger("publish_canonical_snapshot")

DEFAULT_MANIFEST = Path("manifests/runs/regime_engine_2026-05-17.yaml")
DEFAULT_DATA_ROOT = Path("data/raw")

# Search order for the date-like column to populate min_date / max_date.
DATE_COLUMN_CANDIDATES: tuple[str, ...] = (
    "date",
    "period",
    "release_date",
    "start_date",
    "meeting_end_date",
    "speech_date",
)

# Canonical parquet writer settings. These are the *stability requirement*:
# any change to pyarrow's default ``created_by`` string embeds a new
# version in the parquet footer and changes the sha256. Pin pyarrow
# version in dev/CI to keep canonical bytes stable across machines.
_PARQUET_COMPRESSION = "snappy"
_PARQUET_COERCE_TIMESTAMPS = "us"


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

    LOGGER.warning("ruamel.yaml unavailable; falling back to yaml.safe_dump (anchors will expand)")
    payload = yaml.safe_load(path.read_text())
    return payload, None


def _dump_manifest_atomically(payload: Any, ruamel_instance: Any | None, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if ruamel_instance is not None:
        with tmp.open("w") as handle:
            ruamel_instance.dump(payload, handle)
    else:
        import yaml  # type: ignore[import-untyped]

        tmp.write_text(yaml.safe_dump(payload, sort_keys=False))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Parquet canonicalization


def _canonicalize_parquet_bytes(source: Path) -> bytes:
    """Return canonical parquet bytes for a file.

    Canonical form: rows sorted ascending by all columns (pyarrow sort_by),
    schema metadata stripped (no pandas round-trip embeds), snappy
    compression, coerce_timestamps='us', no int96 timestamps. The function
    is a fixed point: canonicalize(canonicalize(x)) == canonicalize(x).
    """
    table = pq.ParquetFile(source).read()
    table = table.replace_schema_metadata(None)
    if any(pat.is_dictionary(field.type) for field in table.schema):
        table = table.from_arrays(
            [
                column.combine_chunks().dictionary_decode()
                if pat.is_dictionary(field.type)
                else column
                for field, column in zip(table.schema, table.itercolumns(), strict=True)
            ],
            names=table.column_names,
        )
    if table.num_rows > 0:
        sort_keys = [(name, "ascending") for name in table.column_names]
        indices = pc.sort_indices(table, sort_keys=sort_keys)
        table = table.take(indices)
    buf = io.BytesIO()
    pq.write_table(
        table,
        buf,
        compression=_PARQUET_COMPRESSION,
        coerce_timestamps=_PARQUET_COERCE_TIMESTAMPS,
        use_deprecated_int96_timestamps=False,
    )
    return buf.getvalue()


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


# ---------------------------------------------------------------------------
# Per-artifact processing


@dataclass
class ArtifactReport:
    name: str
    local_path: str
    status: str  # MATCH, MISSING, CANONICALIZE_WOULD_CHANGE_SHA, WOULD_UPLOAD,
                 # UPDATED, UPLOADED, UNCHANGED_AFTER_CANONICALIZE
    old_sha: str
    new_sha: str | None = None
    note: str = ""
    manifest_updates: dict[str, object] | None = None


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


def _is_yaml(path: Path) -> bool:
    return path.suffix.lower() in {".yaml", ".yml"}


def _filter_artifacts(
    artifacts: Iterable[dict[str, Any]], only: list[str] | None
) -> list[dict[str, Any]]:
    if not only:
        return list(artifacts)
    wanted = set(only)
    return [a for a in artifacts if a["name"] in wanted]


# ---------------------------------------------------------------------------
# Mode handlers


def run_check(
    payload: Any, data_root: Path, only: list[str] | None
) -> tuple[int, list[ArtifactReport]]:
    reports: list[ArtifactReport] = []
    drift = 0
    for artifact in _filter_artifacts(payload["artifacts"], only):
        local = _resolve_local_path(data_root, artifact)
        old_sha = str(artifact["sha256"])
        if not local.exists():
            reports.append(
                ArtifactReport(artifact["name"], str(local), "MISSING", old_sha)
            )
            continue
        if _is_parquet(local):
            try:
                canon = _canonicalize_parquet_bytes(local)
            except Exception as exc:  # pragma: no cover - defensive
                reports.append(
                    ArtifactReport(
                        artifact["name"],
                        str(local),
                        "READ_ERROR",
                        old_sha,
                        note=str(exc),
                    )
                )
                drift += 1
                continue
            new_sha = sha256_bytes(canon)
        elif _is_yaml(local):
            new_sha = sha256_file(local)
        else:
            new_sha = sha256_file(local)
        status = "MATCH" if new_sha == old_sha else "DRIFT"
        if status == "DRIFT":
            drift += 1
        reports.append(
            ArtifactReport(artifact["name"], str(local), status, old_sha, new_sha)
        )
    return (0 if drift == 0 else 1), reports


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
            canon = _canonicalize_parquet_bytes(local)
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
            canon = _canonicalize_parquet_bytes(local)
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

        # Manifest entry must be updated. Upload first if applicable.
        if not skip_upload and store is not None:
            key = str(artifact["uri"])
            store.put_file(local, key, overwrite=True)

        # Update artifact in place to preserve YAML anchors / structure.
        manifest_updates: dict[str, object] = {"sha256": new_sha}
        artifact["sha256"] = new_sha
        if _is_parquet(local):
            assert canon is not None
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
        if current_name is not None and current_start is not None and indent == current_indent:
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
        raise RuntimeError(f"Manifest artifact block {artifact_name!r} missing field(s): {keys}")


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


def _format_report(reports: list[ArtifactReport], *, limit: int | None = None) -> str:
    lines: list[str] = []
    for r in reports[:limit] if limit else reports:
        new = r.new_sha[:12] if r.new_sha else "-"
        lines.append(
            f"  {r.status:<32} {r.name:<40} old={r.old_sha[:12]} new={new}"
            + (f"  ({r.note})" if r.note else "")
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_arg_parser().parse_args(argv)

    manifest_path = args.manifest if args.manifest.is_absolute() else (_REPO_ROOT / args.manifest)
    data_root = args.data_root if args.data_root.is_absolute() else (_REPO_ROOT / args.data_root)

    payload, ruamel_instance = _load_manifest_payload(manifest_path)
    if not isinstance(payload, dict) or "artifacts" not in payload:
        LOGGER.error("manifest does not contain artifacts list: %s", manifest_path)
        return 2

    if args.check:
        exit_code, reports = run_check(payload, data_root, args.only)
        drift = [r for r in reports if r.status == "DRIFT"]
        missing = [r for r in reports if r.status == "MISSING"]
        matched = [r for r in reports if r.status == "MATCH"]
        LOGGER.info(
            "check: %d checked, %d MATCH, %d DRIFT, %d MISSING",
            len(reports),
            len(matched),
            len(drift),
            len(missing),
        )
        if drift:
            LOGGER.info("DRIFT:")
            LOGGER.info(_format_report(drift))
        return exit_code

    if args.dry_run:
        reports = run_dry_run(payload, data_root, args.only)
        from collections import Counter

        status_counts = Counter(r.status for r in reports)
        LOGGER.info("dry-run summary: %s", dict(status_counts))
        non_match = [r for r in reports if r.status != "MATCH" and r.status != "MISSING"]
        if non_match:
            LOGGER.info("changes preview:")
            LOGGER.info(_format_report(non_match))
        return 0

    # publish mode
    store: ArtifactStore | None = None
    if not args.skip_upload:
        store = build_artifact_store(str(payload["storage_root"]))

    reports = run_publish(
        payload, data_root, args.only, skip_upload=args.skip_upload, store=store
    )
    changed = [r for r in reports if r.status in {"UPDATED", "UPLOADED"}]
    LOGGER.info("publish: %d processed, %d changed", len(reports), len(changed))
    if changed:
        LOGGER.info(_format_report(changed))
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
