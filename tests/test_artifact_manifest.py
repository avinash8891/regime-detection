from __future__ import annotations

from pathlib import Path
from typing import get_args, get_type_hints

import pytest

from regime_data_fetch.artifact_manifest import (
    ArtifactManifest,
    ArtifactStage,
    ManifestArtifact,
    ManifestValidationError,
    has_data_raw_prefix,
    load_manifest,
    prepend_data_raw_prefix,
    strip_data_raw_prefix,
    write_manifest,
)


def _file_uri(key: str) -> str:
    return f"file:///tmp/regime-data/{key}"


def _artifact(**overrides: object) -> ManifestArtifact:
    values: dict[str, object] = {
        "name": "fred_macro_series",
        "stage": "canonical",
        "uri": _file_uri("canonical/macro/fred_macro_series.parquet"),
        "local_path": "data/raw/macro/fred_macro_series.parquet",
        "sha256": "a" * 64,
        "schema_version": "fred_macro_series.v1",
        "rows": 10,
        "min_date": "2026-05-01",
        "max_date": "2026-05-15",
        "required_for": ["profile_engine"],
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
            _artifact(name="macro", uri=_file_uri("canonical/macro.parquet")),
            _artifact(name="macro_copy", uri=_file_uri("canonical/macro-copy.parquet")),
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


def test_manifest_rejects_duplicate_artifact_names() -> None:
    manifest = ArtifactManifest(
        artifact_set="regime_engine_2026-05-15",
        created_at_utc="2026-05-15T12:00:00Z",
        storage_root="file:///tmp/regime-data",
        artifacts=[
            _artifact(
                name="macro",
                uri=_file_uri("canonical/a.parquet"),
                local_path="a.parquet",
            ),
            _artifact(
                name="macro",
                uri=_file_uri("canonical/b.parquet"),
                local_path="b.parquet",
            ),
        ],
    )

    with pytest.raises(ManifestValidationError, match="duplicate artifact name"):
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


def test_manifest_artifact_stage_is_a_closed_type() -> None:
    assert set(get_args(ArtifactStage)) == {
        "raw_capture",
        "normalized",
        "canonical",
        "run_inputs",
    }
    assert get_type_hints(ManifestArtifact)["stage"] == ArtifactStage


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
            _artifact(name="macro", required_for=["profile_engine", "v2_calibration"]),
            _artifact(
                name="aaii",
                uri=_file_uri("canonical/sentiment/aaii_sentiment.parquet"),
                local_path="data/raw/sentiment/aaii_sentiment.parquet",
                required_for=["profile_engine"],
            ),
        ],
    )

    assert [a.name for a in manifest.required_for("v2_calibration")] == ["macro"]
    assert [a.name for a in manifest.required_for("profile_engine")] == [
        "macro",
        "aaii",
    ]


def test_data_raw_prefix_helpers_round_trip_writer_and_reader() -> None:
    # The writer prepends the prefix to a repo-relative path; the reader strips
    # it back. Both must agree, so prepend then strip is the identity.
    repo_relative = Path("daily_ohlcv") / "symbol=SPY" / "ohlcv.parquet"
    manifest_path = prepend_data_raw_prefix(repo_relative)

    assert manifest_path == Path("data/raw/daily_ohlcv/symbol=SPY/ohlcv.parquet")
    assert has_data_raw_prefix(manifest_path) is True
    assert strip_data_raw_prefix(manifest_path) == repo_relative


def test_data_raw_prefix_detection_rejects_non_prefixed_paths() -> None:
    configs_path = Path("configs/events/us_events.yaml")

    assert has_data_raw_prefix(configs_path) is False
    # strip is a no-op when the prefix is absent.
    assert strip_data_raw_prefix(configs_path) == configs_path
