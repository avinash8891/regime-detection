from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Iterable

from regime_data_fetch.artifact_manifest import (
    ArtifactManifest,
    ManifestArtifact,
    strip_data_raw_prefix,
    write_manifest,
)
from regime_data_fetch.artifact_store import build_artifact_store


def emit_manifest_for_report_paths(
    *,
    report_paths: Iterable[Path],
    out_dir: Path,
    artifact_store_root: str,
    manifest_path: Path,
    artifact_set: str,
    required_for: list[str],
    repo_root: Path | None = None,
) -> ArtifactManifest:
    store = build_artifact_store(artifact_store_root)
    artifacts: list[ManifestArtifact] = []
    seen_local_paths: set[str] = set()
    for report_path in report_paths:
        payload = _load_report_payload(report_path)
        if payload is None:
            continue
        exportable_for_report = 0
        for name, path, local_path_override in _iter_existing_report_files(payload):
            local_path = _local_path_for(
                path=path,
                out_dir=out_dir,
                repo_root=repo_root,
                local_path_override=local_path_override,
            )
            if local_path is None:
                continue
            exportable_for_report += 1
            if local_path in seen_local_paths:
                continue
            seen_local_paths.add(local_path)
            key = _store_key_for(local_path)
            stored = store.put_file(path, key)
            artifacts.append(
                ManifestArtifact.from_dict(
                    {
                        "name": name,
                        "stage": "canonical",
                        "uri": stored.uri,
                        "local_path": local_path,
                        "sha256": stored.sha256,
                        "schema_version": None,
                        "rows": None,
                        "min_date": None,
                        "max_date": None,
                        "required_for": required_for,
                    }
                )
            )
        if exportable_for_report == 0 and payload.get("materializable") is not False:
            raise ValueError(f"no exportable artifact files in report: {report_path}")
    if not artifacts:
        raise ValueError("no existing artifact files found in report paths")
    manifest = ArtifactManifest(
        artifact_set=artifact_set,
        created_at_utc=dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        storage_root=artifact_store_root,
        artifacts=artifacts,
    )
    write_manifest(manifest, manifest_path)
    return manifest


def _load_report_payload(report_path: Path) -> dict[str, object] | None:
    if not report_path.exists() or report_path.suffix.lower() != ".json":
        return None
    payload = json.loads(report_path.read_text())
    return payload if isinstance(payload, dict) else None


def _iter_existing_report_files(payload: dict[str, object]) -> Iterable[tuple[str, Path, str | None]]:
    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        return
    for name, value in sorted(paths.items()):
        if name in {"acquisition_db"}:
            continue
        entry = _parse_report_path_entry(value)
        if entry is None:
            continue
        path, local_path_override = entry
        if path.exists() and path.is_file():
            yield name, path, local_path_override
        elif path.exists() and path.is_dir():
            for child in sorted(item for item in path.rglob("*") if item.is_file()):
                child_name = f"{name}_{child.relative_to(path).as_posix().replace('/', '_')}"
                child_local_path = None
                if local_path_override is not None:
                    child_local_path = str(Path(local_path_override) / child.relative_to(path))
                yield child_name, child, child_local_path


def _parse_report_path_entry(value: object) -> tuple[Path, str | None] | None:
    if isinstance(value, str):
        return Path(value), None
    if not isinstance(value, dict):
        return None
    path_value = value.get("path")
    local_path_value = value.get("local_path")
    if not isinstance(path_value, str) or not isinstance(local_path_value, str):
        return None
    return Path(path_value), _normalize_manifest_local_path(local_path_value)


def _local_path_for(
    *,
    path: Path,
    out_dir: Path,
    repo_root: Path | None = None,
    local_path_override: str | None = None,
) -> str | None:
    if local_path_override is not None:
        return _normalize_manifest_local_path(local_path_override)
    path = path.resolve()
    out_dir = out_dir.resolve()
    try:
        relative = path.relative_to(out_dir)
    except ValueError:
        if repo_root is None:
            return None
        try:
            repo_relative = path.relative_to(repo_root.resolve())
        except ValueError:
            return None
        if repo_relative == Path("configs") / "events" / "us_events.yaml":
            return str(Path("data") / "raw" / "event_calendar" / "us_events.yaml")
        return str(repo_relative)
    return str(Path("data") / "raw" / relative)


def _store_key_for(local_path: str) -> str:
    return str(Path("canonical") / strip_data_raw_prefix(Path(local_path)))


def _normalize_manifest_local_path(local_path: str) -> str:
    normalized = Path(local_path)
    if normalized.is_absolute() or normalized == Path("..") or ".." in normalized.parts:
        raise ValueError(f"manifest local_path must be relative within the repo: {local_path}")
    return str(normalized)
