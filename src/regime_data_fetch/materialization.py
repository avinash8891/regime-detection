from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from regime_data_fetch.artifact_manifest import ManifestArtifact, load_manifest
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
    store_root: str | None = None,
    required_for: str | None = None,
) -> list[MaterializedArtifact]:
    manifest = load_manifest(manifest_path)
    artifacts = manifest.required_for(required_for) if required_for else manifest.artifacts
    if required_for and not artifacts:
        raise ValueError(f"manifest has no artifacts required for {required_for}")

    store = build_artifact_store(store_root or manifest.storage_root)
    materialized: list[MaterializedArtifact] = []
    for artifact in artifacts:
        destination = destination_for(artifact, local_root)
        store.get_file(artifact.uri, destination, expected_sha256=artifact.sha256)
        materialized.append(
            MaterializedArtifact(
                name=artifact.name,
                destination=destination,
                sha256=artifact.sha256,
            )
        )
    return materialized


def destination_for(artifact: ManifestArtifact, local_root: Path) -> Path:
    local_path = Path(artifact.local_path)
    if local_path.parts[:2] == ("data", "raw"):
        relative = Path(*local_path.parts[2:])
    else:
        relative = local_path
    return local_root / relative


def materialize_if_requested(
    *,
    manifest_path: Path | None,
    local_root: Path,
    store_root: str | None,
    required_for: str,
) -> list[MaterializedArtifact]:
    if manifest_path is None:
        return []
    return materialize_manifest(
        manifest_path=manifest_path,
        local_root=local_root,
        store_root=store_root,
        required_for=required_for,
    )
