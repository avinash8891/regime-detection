from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from regime_data_fetch.artifact_manifest import (
    ArtifactManifest,
    ManifestArtifact,
    write_manifest,
)
from regime_data_fetch.artifact_store import sha256_file
from regime_data_fetch.materialization import materialize_manifest


def _store_uri(root: Path, key: str) -> str:
    return (root.resolve() / key).as_uri()


def test_materialize_manifest_copies_artifacts_into_local_raw_root(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    raw_source = store_root / "canonical" / "macro" / "fred_macro_series.parquet"
    raw_source.parent.mkdir(parents=True)
    raw_source.write_bytes(b"macro")
    sentiment_source = store_root / "canonical" / "sentiment" / "aaii_sentiment.parquet"
    sentiment_source.parent.mkdir(parents=True)
    sentiment_source.write_bytes(b"sentiment")

    manifest = ArtifactManifest(
        artifact_set="regime_engine_2026-05-15",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root=str(store_root),
        artifacts=[
            ManifestArtifact.from_dict(
                {
                    "name": "macro",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root, "canonical/macro/fred_macro_series.parquet"
                    ),
                    "local_path": "data/raw/macro/fred_macro_series.parquet",
                    "sha256": "27d66c0dcef19a926429158d80111b954a5c23d076833347da3e27b91e4b423d",
                    "required_for": ["profile_engine"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "aaii",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root, "canonical/sentiment/aaii_sentiment.parquet"
                    ),
                    "local_path": "data/raw/sentiment/aaii_sentiment.parquet",
                    "sha256": "df82d7244147796b6a6486dc21c4502c0c4d67a8aecc64d86ab310926503c5e3",
                    "required_for": ["profile_engine"],
                }
            ),
        ],
    )
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, manifest_path)

    materialized = materialize_manifest(
        manifest_path=manifest_path, local_root=tmp_path / "data" / "raw"
    )

    assert [item.name for item in materialized] == ["macro", "aaii"]
    assert (
        tmp_path / "data" / "raw" / "macro" / "fred_macro_series.parquet"
    ).read_bytes() == b"macro"
    assert (
        tmp_path / "data" / "raw" / "sentiment" / "aaii_sentiment.parquet"
    ).read_bytes() == b"sentiment"


def test_materialize_manifest_rejects_hash_mismatch_before_partial_success(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    source = store_root / "canonical" / "macro" / "fred_macro_series.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"bad")
    manifest = ArtifactManifest(
        artifact_set="regime_engine_2026-05-15",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root=str(store_root),
        artifacts=[
            ManifestArtifact.from_dict(
                {
                    "name": "macro",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root, "canonical/macro/fred_macro_series.parquet"
                    ),
                    "local_path": "data/raw/macro/fred_macro_series.parquet",
                    "sha256": "dc2f66a495b3469d8a9e3e7da70c9176d82768e9742117b92eedd8c8c5a84a3c",
                    "required_for": ["profile_engine"],
                }
            )
        ],
    )
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, manifest_path)

    with pytest.raises(Exception, match="sha256 mismatch"):
        materialize_manifest(
            manifest_path=manifest_path, local_root=tmp_path / "data" / "raw"
        )

    assert not (
        tmp_path / "data" / "raw" / "macro" / "fred_macro_series.parquet"
    ).exists()


def test_materialize_manifest_restores_repo_relative_paths_to_repo_root(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    source = store_root / "canonical" / "configs" / "events" / "us_events.yaml"
    source.parent.mkdir(parents=True)
    source.write_text("events: []\n")
    manifest = ArtifactManifest(
        artifact_set="regime_engine_2026-05-15",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root=str(store_root),
        artifacts=[
            ManifestArtifact.from_dict(
                {
                    "name": "events",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root, "canonical/configs/events/us_events.yaml"
                    ),
                    "local_path": "configs/events/us_events.yaml",
                    "sha256": "fb9b2e59663fe1741488ac52428726f13fd98f2f28cc73db64f022c8ae629999",
                    "required_for": ["profile_engine"],
                }
            )
        ],
    )
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, manifest_path)

    materialize_manifest(
        manifest_path=manifest_path,
        local_root=tmp_path / "repo" / "data" / "raw",
        repo_root=tmp_path / "repo",
    )

    assert (
        tmp_path / "repo" / "configs" / "events" / "us_events.yaml"
    ).read_text() == "events: []\n"


def test_materialize_manifest_refreshes_existing_drift_after_source_verification(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    source = store_root / "canonical" / "macro" / "fred_macro_series.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"macro")
    manifest = ArtifactManifest(
        artifact_set="regime_engine_2026-05-15",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root=str(store_root),
        artifacts=[
            ManifestArtifact.from_dict(
                {
                    "name": "macro",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root, "canonical/macro/fred_macro_series.parquet"
                    ),
                    "local_path": "data/raw/macro/fred_macro_series.parquet",
                    "sha256": "27d66c0dcef19a926429158d80111b954a5c23d076833347da3e27b91e4b423d",
                    "required_for": ["profile_engine"],
                }
            )
        ],
    )
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, manifest_path)
    destination = tmp_path / "data" / "raw" / "macro" / "fred_macro_series.parquet"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"macro")

    materialize_manifest(
        manifest_path=manifest_path, local_root=tmp_path / "data" / "raw"
    )
    assert destination.read_bytes() == b"macro"

    destination.write_bytes(b"local drift")
    materialize_manifest(
        manifest_path=manifest_path, local_root=tmp_path / "data" / "raw"
    )
    assert destination.read_bytes() == b"macro"


def test_materialize_manifest_reruns_into_existing_local_root_directory(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    source = store_root / "canonical" / "macro" / "fred_macro_series.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"macro")
    manifest = ArtifactManifest(
        artifact_set="regime_engine_2026-05-15",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root=str(store_root),
        artifacts=[
            ManifestArtifact.from_dict(
                {
                    "name": "macro",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root, "canonical/macro/fred_macro_series.parquet"
                    ),
                    "local_path": "data/raw/macro/fred_macro_series.parquet",
                    "sha256": "27d66c0dcef19a926429158d80111b954a5c23d076833347da3e27b91e4b423d",
                    "required_for": ["profile_engine"],
                }
            )
        ],
    )
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, manifest_path)
    local_root = tmp_path / "data" / "raw"
    local_root.mkdir(parents=True)

    first = materialize_manifest(manifest_path=manifest_path, local_root=local_root)
    second = materialize_manifest(manifest_path=manifest_path, local_root=local_root)

    destination = local_root / "macro" / "fred_macro_series.parquet"
    assert [item.destination for item in first] == [destination]
    assert [item.destination for item in second] == [destination]
    assert destination.read_bytes() == b"macro"


def test_materialize_manifest_does_not_replace_existing_files_until_all_artifacts_verify(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    first_source = store_root / "canonical" / "macro" / "fred_macro_series.parquet"
    first_source.parent.mkdir(parents=True)
    first_source.write_bytes(b"new macro")
    second_source = store_root / "canonical" / "sentiment" / "aaii_sentiment.parquet"
    second_source.parent.mkdir(parents=True)
    second_source.write_bytes(b"bad sentiment")
    local_root = tmp_path / "data" / "raw"
    first_destination = local_root / "macro" / "fred_macro_series.parquet"
    first_destination.parent.mkdir(parents=True)
    first_destination.write_bytes(b"old macro")
    manifest = ArtifactManifest(
        artifact_set="regime_engine_2026-05-15",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root=str(store_root),
        artifacts=[
            ManifestArtifact.from_dict(
                {
                    "name": "macro",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root, "canonical/macro/fred_macro_series.parquet"
                    ),
                    "local_path": "data/raw/macro/fred_macro_series.parquet",
                    "sha256": sha256_file(first_source),
                    "required_for": ["profile_engine"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "aaii",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root, "canonical/sentiment/aaii_sentiment.parquet"
                    ),
                    "local_path": "data/raw/sentiment/aaii_sentiment.parquet",
                    "sha256": "0" * 64,
                    "required_for": ["profile_engine"],
                }
            ),
        ],
    )
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, manifest_path)

    with pytest.raises(Exception, match="sha256 mismatch"):
        materialize_manifest(manifest_path=manifest_path, local_root=local_root)

    assert first_destination.read_bytes() == b"old macro"
    assert not (local_root / "sentiment" / "aaii_sentiment.parquet").exists()


def test_materialize_regime_data_cli_materializes_manifest(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    source = store_root / "canonical" / "macro" / "fred_macro_series.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"macro")
    manifest = ArtifactManifest(
        artifact_set="regime_engine_2026-05-15",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root=str(store_root),
        artifacts=[
            ManifestArtifact.from_dict(
                {
                    "name": "macro",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root, "canonical/macro/fred_macro_series.parquet"
                    ),
                    "local_path": "data/raw/macro/fred_macro_series.parquet",
                    "sha256": "27d66c0dcef19a926429158d80111b954a5c23d076833347da3e27b91e4b423d",
                    "required_for": ["profile_engine"],
                }
            )
        ],
    )
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, manifest_path)
    local_root = tmp_path / "data" / "raw"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/materialize_regime_data.py",
            "--manifest",
            str(manifest_path),
            "--local-root",
            str(local_root),
            "--required-for",
            "profile_engine",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.startswith(
        "macro\t27d66c0dcef19a926429158d80111b954a5c23d076833347da3e27b91e4b423d\t"
    )
    assert "fred_macro_series.parquet" in result.stdout
    assert (local_root / "macro" / "fred_macro_series.parquet").read_bytes() == b"macro"


def test_materialize_manifest_handles_workspace_portable_uris(
    tmp_path: Path,
) -> None:
    """A manifest authored on one workspace must still materialize on another.

    Simulates the nicosia/vaduz split: the manifest's ``file://`` URI encodes
    an absolute path that lives under a different parent prefix than the
    local store root, but the trailing ``<store-name>/...`` segment matches.
    Also exercises the relative-key URI shape (scheme-less) the portable
    manifests now ship with.
    """
    foreign_root = tmp_path / "alice" / "regime-artifact-store-20260517"
    local_root = tmp_path / "bob" / "regime-artifact-store-20260517"
    local_source = local_root / "canonical" / "macro" / "fred_macro_series.parquet"
    local_source.parent.mkdir(parents=True)
    local_source.write_bytes(b"macro")
    sentiment_source = local_root / "canonical" / "sentiment" / "aaii_sentiment.parquet"
    sentiment_source.parent.mkdir(parents=True)
    sentiment_source.write_bytes(b"sentiment")

    foreign_uri = (
        foreign_root / "canonical" / "macro" / "fred_macro_series.parquet"
    ).as_uri()

    manifest = ArtifactManifest(
        artifact_set="regime_engine_2026-05-17",
        created_at_utc="2026-05-17T12:00:00Z",
        storage_root=str(local_root),
        artifacts=[
            ManifestArtifact.from_dict(
                {
                    "name": "macro",
                    "stage": "canonical",
                    "uri": foreign_uri,
                    "local_path": "data/raw/macro/fred_macro_series.parquet",
                    "sha256": sha256_file(local_source),
                    "required_for": ["profile_engine"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "aaii",
                    "stage": "canonical",
                    "uri": "canonical/sentiment/aaii_sentiment.parquet",
                    "local_path": "data/raw/sentiment/aaii_sentiment.parquet",
                    "sha256": sha256_file(sentiment_source),
                    "required_for": ["profile_engine"],
                }
            ),
        ],
    )
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, manifest_path)

    materialized = materialize_manifest(
        manifest_path=manifest_path,
        local_root=tmp_path / "data" / "raw",
    )

    assert [item.name for item in materialized] == ["macro", "aaii"]
    assert (
        tmp_path / "data" / "raw" / "macro" / "fred_macro_series.parquet"
    ).read_bytes() == b"macro"
    assert (
        tmp_path / "data" / "raw" / "sentiment" / "aaii_sentiment.parquet"
    ).read_bytes() == b"sentiment"


def test_materialize_manifest_resolves_relative_storage_root_against_manifest_dir(
    tmp_path: Path,
) -> None:
    """A relative ``storage_root`` must anchor to the manifest file's parent.

    Without this, operators get different store paths depending on cwd. Here
    we put the manifest in a subdirectory and run the test from a different
    cwd to prove the resolver follows the manifest's location.
    """
    project = tmp_path / "project"
    project.mkdir()
    store_root = project / ".context" / "regime-store"
    source = store_root / "canonical" / "macro" / "fred_macro_series.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"macro")

    manifest = ArtifactManifest(
        artifact_set="regime_engine_2026-05-17",
        created_at_utc="2026-05-17T12:00:00Z",
        storage_root=".context/regime-store",
        artifacts=[
            ManifestArtifact.from_dict(
                {
                    "name": "macro",
                    "stage": "canonical",
                    "uri": "canonical/macro/fred_macro_series.parquet",
                    "local_path": "data/raw/macro/fred_macro_series.parquet",
                    "sha256": sha256_file(source),
                    "required_for": ["profile_engine"],
                }
            ),
        ],
    )
    # Manifest lives at the project root so its sibling ``.context/`` store
    # resolves correctly.
    manifest_path = project / "regime.yaml"
    write_manifest(manifest, manifest_path)

    # Run from a foreign cwd; relative store_root must NOT resolve there.
    foreign_cwd = tmp_path / "elsewhere"
    foreign_cwd.mkdir()
    import os

    original_cwd = Path.cwd()
    os.chdir(foreign_cwd)
    try:
        materialized = materialize_manifest(
            manifest_path=manifest_path,
            local_root=foreign_cwd / "data" / "raw",
        )
    finally:
        os.chdir(original_cwd)

    assert [item.name for item in materialized] == ["macro"]
    assert (
        foreign_cwd / "data" / "raw" / "macro" / "fred_macro_series.parquet"
    ).read_bytes() == b"macro"
