from __future__ import annotations

from pathlib import Path

import pytest

from regime_data_fetch.artifact_manifest import (
    ArtifactManifest,
    ManifestArtifact,
    ManifestValidationError,
    load_manifest,
    write_manifest,
)


def _artifact(**overrides: object) -> ManifestArtifact:
    values: dict[str, object] = {
        "name": "fred_macro_series",
        "stage": "canonical",
        "uri": "canonical/macro/fred_macro_series.parquet",
        "local_path": "data/raw/macro/fred_macro_series.parquet",
        "sha256": "a" * 64,
        "schema_version": "fred_macro_series.v1",
        "rows": 10,
        "min_date": "2026-05-01",
        "max_date": "2026-05-15",
        "required_for": ["profile_engine_30d"],
    }
    values.update(overrides)
    return ManifestArtifact.from_dict(values)


def test_manifest_round_trips_yaml(tmp_path: Path) -> None:
    manifest = ArtifactManifest(
        artifact_set="regime_engine_2026-05-15",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root="file:///tmp/regime-data",
        artifacts=[_artifact()],
    )
    path = tmp_path / "manifest.yaml"

    write_manifest(manifest, path)
    loaded = load_manifest(path)

    assert loaded == manifest


def test_manifest_rejects_duplicate_local_paths() -> None:
    manifest = ArtifactManifest(
        artifact_set="regime_engine_2026-05-15",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root="file:///tmp/regime-data",
        artifacts=[
            _artifact(name="macro", uri="canonical/macro.parquet"),
            _artifact(name="macro_copy", uri="canonical/macro-copy.parquet"),
        ],
    )

    with pytest.raises(ManifestValidationError, match="duplicate local_path"):
        manifest.validate()


def test_manifest_rejects_duplicate_artifact_uris() -> None:
    manifest = ArtifactManifest(
        artifact_set="regime_engine_2026-05-15",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root="file:///tmp/regime-data",
        artifacts=[
            _artifact(name="macro", local_path="data/raw/macro/a.parquet"),
            _artifact(name="macro_copy", local_path="data/raw/macro/b.parquet"),
        ],
    )

    with pytest.raises(ManifestValidationError, match="duplicate artifact uri"):
        manifest.validate()


def test_manifest_ignores_forward_compatible_unknown_fields() -> None:
    payload = _artifact().to_dict()
    payload["future_field"] = {"ignored": True}
    manifest = ArtifactManifest.from_dict(
        {
            "artifact_set": "regime_engine_2026-05-15",
            "created_at_utc": "2026-05-15T12:00:00Z",
            "storage_root": "file:///tmp/regime-data",
            "artifacts": [payload],
            "future_top_level_field": ["ignored"],
        }
    )

    assert manifest.artifacts[0].name == "fred_macro_series"


def test_manifest_rejects_unknown_stage() -> None:
    with pytest.raises(ManifestValidationError, match="unknown artifact stage"):
        _artifact(stage="scratch")


def test_manifest_rejects_bad_sha256() -> None:
    with pytest.raises(ManifestValidationError, match="sha256"):
        _artifact(sha256="bad")


def test_manifest_rejects_absolute_local_path() -> None:
    with pytest.raises(ManifestValidationError, match="local_path must be relative"):
        _artifact(local_path="/tmp/data/raw/macro.parquet")


def test_manifest_required_artifacts_for_use_case() -> None:
    manifest = ArtifactManifest(
        artifact_set="regime_engine_2026-05-15",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root="file:///tmp/regime-data",
        artifacts=[
            _artifact(
                name="macro", required_for=["profile_engine_30d", "v2_calibration"]
            ),
            _artifact(
                name="aaii",
                uri="canonical/sentiment/aaii_sentiment.parquet",
                local_path="data/raw/sentiment/aaii_sentiment.parquet",
                required_for=["profile_engine_30d"],
            ),
        ],
    )

    assert [a.name for a in manifest.required_for("v2_calibration")] == ["macro"]
    assert [a.name for a in manifest.required_for("profile_engine_30d")] == [
        "macro",
        "aaii",
    ]
