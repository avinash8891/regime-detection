from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Iterable

from regime_data_fetch.artifact_manifest import ArtifactManifest, ManifestArtifact, write_manifest
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
        for name, path in _iter_existing_report_files(report_path):
            local_path = _local_path_for(path=path, out_dir=out_dir, repo_root=repo_root)
            if local_path is None:
                continue
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


def _iter_existing_report_files(report_path: Path) -> Iterable[tuple[str, Path]]:
    if not report_path.exists() or report_path.suffix.lower() != ".json":
        return
    payload = json.loads(report_path.read_text())
    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        return
    for name, value in sorted(paths.items()):
        if not isinstance(value, str):
            continue
        path = Path(value)
        if path.exists() and path.is_file():
            yield name, path
        elif path.exists() and path.is_dir():
            for child in sorted(item for item in path.rglob("*") if item.is_file()):
                child_name = f"{name}_{child.relative_to(path).as_posix().replace('/', '_')}"
                yield child_name, child


def _local_path_for(*, path: Path, out_dir: Path, repo_root: Path | None = None) -> str | None:
    path = path.resolve()
    out_dir = out_dir.resolve()
    try:
        relative = path.relative_to(out_dir)
    except ValueError:
        if repo_root is None:
            return None
        try:
            return str(path.relative_to(repo_root.resolve()))
        except ValueError:
            return None
    return str(Path("data") / "raw" / relative)


def _store_key_for(local_path: str) -> str:
    path = Path(local_path)
    if path.parts[:2] == ("data", "raw"):
        relative = Path(*path.parts[2:])
    else:
        relative = path
    return str(Path("canonical") / relative)
