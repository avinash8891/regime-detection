from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from regime_data_fetch.artifact_manifest import (
    DATA_RAW_PREFIX,
    ManifestArtifact,
    load_manifest,
    strip_data_raw_prefix,
)
from regime_data_fetch.artifact_store import build_artifact_store


@dataclass(frozen=True)
class MaterializedArtifact:
    name: str
    destination: Path
    sha256: str


def materialize_manifest(
    *,
    manifest_path: Path,
    local_root: Path,
    repo_root: Path | None = None,
    store_root: str | None = None,
    required_for: str | None = None,
) -> list[MaterializedArtifact]:
    manifest = load_manifest(manifest_path)
    artifacts = manifest.required_for(required_for) if required_for else manifest.artifacts
    if required_for and not artifacts:
        raise ValueError(f"manifest has no artifacts required for {required_for}")

    store = build_artifact_store(store_root or manifest.storage_root)
    staging_parent = local_root.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".manifest-materialize-", dir=staging_parent
    ) as tmp_dir:
        staging_root = Path(tmp_dir)
        staged: list[tuple[ManifestArtifact, Path, Path]] = []
        for index, artifact in enumerate(artifacts):
            destination = destination_for(artifact, local_root, repo_root=repo_root)
            staged_path = staging_root / "staged" / str(index) / destination.name
            store.get_file(
                artifact.uri,
                staged_path,
                expected_sha256=artifact.sha256,
            )
            staged.append((artifact, destination, staged_path))

        promoted: list[tuple[Path, Path | None]] = []
        backup_root = staging_root / "backups"
        try:
            # Promote only after every artifact has passed checksum verification.
            # Existing files move to the same temp tree first so a mid-run failure
            # can roll back without leaving a mixed old/new manifest directory.
            for index, (_artifact, destination, staged_path) in enumerate(staged):
                destination.parent.mkdir(parents=True, exist_ok=True)
                backup_path: Path | None = None
                if destination.exists():
                    backup_path = backup_root / str(index) / destination.name
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    destination.replace(backup_path)
                staged_path.replace(destination)
                promoted.append((destination, backup_path))
        except Exception:
            for destination, backup_path in reversed(promoted):
                if destination.exists():
                    destination.unlink()
                if backup_path is not None and backup_path.exists():
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    backup_path.replace(destination)
            raise

    return [
        MaterializedArtifact(
            name=artifact.name,
            destination=destination,
            sha256=artifact.sha256,
        )
        for artifact, destination, _staged_path in staged
    ]


def destination_for(artifact: ManifestArtifact, local_root: Path, *, repo_root: Path | None = None) -> Path:
    local_path = Path(artifact.local_path)
    if local_path.parts[: len(DATA_RAW_PREFIX)] == DATA_RAW_PREFIX:
        return local_root / strip_data_raw_prefix(local_path)
    if repo_root is None:
        return local_root / local_path
    return repo_root / local_path


def materialize_if_requested(
    *,
    manifest_path: Path | None,
    local_root: Path,
    repo_root: Path | None = None,
    store_root: str | None,
    required_for: str,
) -> list[MaterializedArtifact]:
    if manifest_path is None:
        return []
    return materialize_manifest(
        manifest_path=manifest_path,
        local_root=local_root,
        repo_root=repo_root,
        store_root=store_root,
        required_for=required_for,
    )
