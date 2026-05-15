from __future__ import annotations

from pathlib import Path

import pytest

from regime_data_fetch.artifact_manifest import ArtifactManifest, ManifestArtifact, write_manifest
from regime_data_fetch.materialization import materialize_manifest


def test_materialize_manifest_copies_artifacts_into_local_raw_root(tmp_path: Path) -> None:
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
                    "uri": "canonical/macro/fred_macro_series.parquet",
                    "local_path": "data/raw/macro/fred_macro_series.parquet",
                    "sha256": "27d66c0dcef19a926429158d80111b954a5c23d076833347da3e27b91e4b423d",
                    "required_for": ["profile_engine_30d"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "aaii",
                    "stage": "canonical",
                    "uri": "canonical/sentiment/aaii_sentiment.parquet",
                    "local_path": "data/raw/sentiment/aaii_sentiment.parquet",
                    "sha256": "df82d7244147796b6a6486dc21c4502c0c4d67a8aecc64d86ab310926503c5e3",
                    "required_for": ["profile_engine_30d"],
                }
            ),
        ],
    )
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, manifest_path)

    materialized = materialize_manifest(manifest_path=manifest_path, local_root=tmp_path / "data" / "raw")

    assert [item.name for item in materialized] == ["macro", "aaii"]
    assert (tmp_path / "data" / "raw" / "macro" / "fred_macro_series.parquet").read_bytes() == b"macro"
    assert (tmp_path / "data" / "raw" / "sentiment" / "aaii_sentiment.parquet").read_bytes() == b"sentiment"


def test_materialize_manifest_rejects_hash_mismatch_before_partial_success(tmp_path: Path) -> None:
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
                    "uri": "canonical/macro/fred_macro_series.parquet",
                    "local_path": "data/raw/macro/fred_macro_series.parquet",
                    "sha256": "dc2f66a495b3469d8a9e3e7da70c9176d82768e9742117b92eedd8c8c5a84a3c",
                    "required_for": ["profile_engine_30d"],
                }
            )
        ],
    )
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, manifest_path)

    with pytest.raises(Exception, match="sha256 mismatch"):
        materialize_manifest(manifest_path=manifest_path, local_root=tmp_path / "data" / "raw")

    assert not (tmp_path / "data" / "raw" / "macro" / "fred_macro_series.parquet").exists()


def test_materialize_manifest_restores_repo_relative_paths_to_repo_root(tmp_path: Path) -> None:
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
                    "uri": "canonical/configs/events/us_events.yaml",
                    "local_path": "configs/events/us_events.yaml",
                    "sha256": "fb9b2e59663fe1741488ac52428726f13fd98f2f28cc73db64f022c8ae629999",
                    "required_for": ["profile_engine_30d"],
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

    assert (tmp_path / "repo" / "configs" / "events" / "us_events.yaml").read_text() == "events: []\n"
