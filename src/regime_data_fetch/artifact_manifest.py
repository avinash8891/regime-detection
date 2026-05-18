from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml


ArtifactStage = Literal["raw_capture", "normalized", "canonical", "run_inputs"]
VALID_STAGES: frozenset[ArtifactStage] = frozenset(
    ("raw_capture", "normalized", "canonical", "run_inputs")
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATA_RAW_PREFIX: tuple[str, ...] = ("data", "raw")


def strip_data_raw_prefix(path: Path) -> Path:
    """Strip the conventional ``data/raw/`` prefix from a manifest-relative path."""
    if path.parts[: len(DATA_RAW_PREFIX)] == DATA_RAW_PREFIX:
        return Path(*path.parts[len(DATA_RAW_PREFIX) :])
    return path


class ManifestValidationError(ValueError):
    pass


def _parse_stage(value: object) -> ArtifactStage:
    stage = str(value)
    if stage not in VALID_STAGES:
        raise ManifestValidationError(f"unknown artifact stage: {stage}")
    return cast(ArtifactStage, stage)


@dataclass(frozen=True)
class ManifestArtifact:
    name: str
    stage: ArtifactStage
    uri: str
    local_path: str
    sha256: str
    schema_version: str | None
    rows: int | None
    min_date: str | None
    max_date: str | None
    required_for: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ManifestArtifact:
        required = ["name", "stage", "uri", "local_path", "sha256"]
        missing = [field for field in required if not payload.get(field)]
        if missing:
            raise ManifestValidationError(
                f"manifest artifact missing required fields: {missing}"
            )

        required_for = payload.get("required_for", [])
        if not isinstance(required_for, list) or not all(
            isinstance(item, str) for item in required_for
        ):
            raise ManifestValidationError("required_for must be a list[str]")

        artifact = cls(
            name=str(payload["name"]),
            stage=_parse_stage(payload["stage"]),
            uri=str(payload["uri"]),
            local_path=str(payload["local_path"]),
            sha256=str(payload["sha256"]),
            schema_version=(
                str(payload["schema_version"])
                if payload.get("schema_version") is not None
                else None
            ),
            rows=int(payload["rows"]) if payload.get("rows") is not None else None,
            min_date=str(payload["min_date"])
            if payload.get("min_date") is not None
            else None,
            max_date=str(payload["max_date"])
            if payload.get("max_date") is not None
            else None,
            required_for=tuple(required_for),
        )
        artifact.validate()
        return artifact

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "uri": self.uri,
            "local_path": self.local_path,
            "sha256": self.sha256,
            "schema_version": self.schema_version,
            "rows": self.rows,
            "min_date": self.min_date,
            "max_date": self.max_date,
            "required_for": list(self.required_for),
        }

    def validate(self) -> None:
        if self.stage not in VALID_STAGES:
            raise ManifestValidationError(f"unknown artifact stage: {self.stage}")
        if not SHA256_RE.fullmatch(self.sha256):
            raise ManifestValidationError(
                f"sha256 must be 64 lowercase hex chars for {self.name}"
            )
        local_path = Path(self.local_path)
        if local_path.is_absolute() or ".." in local_path.parts:
            raise ManifestValidationError(
                f"local_path must be relative and stay within data root: {self.local_path}"
            )
        if self.rows is not None and self.rows < 0:
            raise ManifestValidationError(f"rows must be non-negative for {self.name}")


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_set: str
    created_at_utc: str
    storage_root: str
    artifacts: list[ManifestArtifact]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactManifest:
        required = ["artifact_set", "created_at_utc", "storage_root", "artifacts"]
        missing = [field for field in required if not payload.get(field)]
        if missing:
            raise ManifestValidationError(
                f"manifest missing required fields: {missing}"
            )
        raw_artifacts = payload["artifacts"]
        if not isinstance(raw_artifacts, list):
            raise ManifestValidationError("artifacts must be a list")
        manifest = cls(
            artifact_set=str(payload["artifact_set"]),
            created_at_utc=str(payload["created_at_utc"]),
            storage_root=str(payload["storage_root"]),
            artifacts=[ManifestArtifact.from_dict(item) for item in raw_artifacts],
        )
        manifest.validate()
        return manifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_set": self.artifact_set,
            "created_at_utc": self.created_at_utc,
            "storage_root": self.storage_root,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    def validate(self) -> None:
        if not self.artifacts:
            raise ManifestValidationError("manifest must include at least one artifact")
        local_paths: set[str] = set()
        names: set[str] = set()
        uris: set[str] = set()
        for artifact in self.artifacts:
            artifact.validate()
            if artifact.local_path in local_paths:
                raise ManifestValidationError(
                    f"duplicate local_path in manifest: {artifact.local_path}"
                )
            if artifact.name in names:
                raise ManifestValidationError(
                    f"duplicate artifact name in manifest: {artifact.name}"
                )
            if artifact.uri in uris:
                raise ManifestValidationError(
                    f"duplicate artifact uri in manifest: {artifact.uri}"
                )
            local_paths.add(artifact.local_path)
            names.add(artifact.name)
            uris.add(artifact.uri)

    def required_for(self, use_case: str) -> list[ManifestArtifact]:
        return [
            artifact for artifact in self.artifacts if use_case in artifact.required_for
        ]


def load_manifest(path: Path) -> ArtifactManifest:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ManifestValidationError("manifest file must contain a mapping")
    return ArtifactManifest.from_dict(payload)


def write_manifest(manifest: ArtifactManifest, path: Path) -> None:
    manifest.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False))
