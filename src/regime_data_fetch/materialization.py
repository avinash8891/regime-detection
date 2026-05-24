from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from regime_data_fetch.artifact_manifest import (
    DATA_RAW_PREFIX,
    ManifestArtifact,
    load_manifest,
    strip_data_raw_prefix,
)
from regime_data_fetch.artifact_store import build_artifact_store
from regime_data_fetch.artifact_store import sha256_file

# Sentinel SHA for documented placeholder entries (empty-string digest).
# These are not fetchable artifacts; the OHLCV tree they represent is
# discovered structurally by the resolver. Skipping avoids a guaranteed
# sha mismatch that would mislead operators into thinking the store is corrupt.
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


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
    # Skip documented placeholder entries (empty-string sha sentinel). These
    # are not real fetchable artifacts; their role is to mark a structural
    # contract the resolver discovers another way (e.g. the 762-symbol OHLCV
    # tree exposed via its own per-symbol lockfile).
    artifacts = [a for a in artifacts if a.sha256 != _EMPTY_SHA256]

    effective_store_root = _resolve_store_root(
        store_root=store_root,
        manifest_storage_root=manifest.storage_root,
        manifest_path=manifest_path,
    )
    store = build_artifact_store(effective_store_root)
    staging_parent = local_root.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".manifest-materialize-", dir=staging_parent
    ) as tmp_dir:
        staging_root = Path(tmp_dir)
        staged: list[tuple[ManifestArtifact, Path, Path]] = []
        materialized: list[tuple[ManifestArtifact, Path]] = []
        for index, artifact in enumerate(artifacts):
            destination = destination_for(artifact, local_root, repo_root=repo_root)
            if destination.exists() and sha256_file(destination) == artifact.sha256:
                materialized.append((artifact, destination))
                continue
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
                materialized.append((_artifact, destination))
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
        for artifact, destination in materialized
    ]


def _resolve_store_root(
    *,
    store_root: str | None,
    manifest_storage_root: str,
    manifest_path: Path,
) -> str:
    """Pick the effective artifact-store root and make relative paths portable.

    An explicit ``--artifact-store`` override always wins. Otherwise the
    manifest's ``storage_root`` field is used. When that field is a relative
    local path (no URI scheme, not absolute), we anchor it to the manifest
    file's own directory rather than the process cwd. That keeps a manifest
    movable between checkouts: a checked-in
    ``storage_root: .context/regime-artifact-store-20260517`` resolves the
    same way no matter which workspace the operator invokes the runner from.
    Absolute paths, ``file://`` URIs, and ``s3://`` URIs are passed through
    untouched.
    """
    candidate = store_root or manifest_storage_root
    parsed = urlparse(candidate)
    if parsed.scheme:
        return candidate
    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        return candidate
    return str((manifest_path.resolve().parent / candidate_path).resolve())


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
