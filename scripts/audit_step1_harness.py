#!/usr/bin/env python3
"""Step 1 audit harness for regime-detection trust-audit.

Captures the manifest/materialization provenance bundle defined in
``docs/superpowers/specs/2026-05-21-regime-detection-trust-audit-design.md``
§Step 2a, plus cross-worktree provenance per §Cross-worktree scope.

Drop-in usage::

    from regime_data_fetch.materialization import materialize_if_requested
    from regime_data_fetch.manifest_inputs import resolve_runner_input_paths
    from scripts.audit_step1_harness import emit_step1_provenance

    materialized = materialize_if_requested(...)  # currently discards return
    resolved = resolve_runner_input_paths(...)

    emit_step1_provenance(
        runner_name="profile_engine",
        manifest_path=args.manifest,
        materialized_artifacts=materialized,
        resolved_from_manifest=resolved.resolved_from_manifest,
        cli_overrides=resolved.cli_overrides,
        materialize_called_by_runner=True,
        output_path=Path("audit/step1/profile_engine.json"),
    )

The harness is **fail-open**: emission errors are logged but never raised,
so audit instrumentation cannot break a production runner. See CLAUDE.md
global rule "Instrumentation must NEVER block business logic."
"""

from __future__ import annotations

import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from regime_data_fetch.materialization import MaterializedArtifact

from regime_data_fetch.artifact_store import sha256_file as _sha256_file

logger = logging.getLogger(__name__)

_PARALLEL_HASH_THRESHOLD = 50
_DEFAULT_HASH_WORKERS = 8

# Directories whose source files are hashed for cross-worktree drift detection.
# These are the trust-critical engine surfaces per the audit spec.
_CROSS_WORKTREE_SOURCE_DIRS: tuple[str, ...] = (
    "src/regime_detection",
    "src/regime_data_fetch",
)

# Known bypass classes from spec §Step 2a "Bypass paths confirmed in the codebase".
_BYPASS_CLASSES: frozenset[str] = frozenset(
    {
        "whole_manifest_bypass",
        "per_field_cli_override",
        "runner_level_skip",
        "materialize_but_dont_bind",
        "cross_worktree_drift",
    }
)


@dataclass(frozen=True)
class MaterializedArtifactRecord:
    """JSON-serializable view of MaterializedArtifact."""

    name: str
    destination: str
    sha256: str

    @classmethod
    def from_materialized(
        cls, artifact: "MaterializedArtifact"
    ) -> "MaterializedArtifactRecord":
        return cls(
            name=artifact.name,
            destination=str(artifact.destination),
            sha256=artifact.sha256,
        )


@dataclass(frozen=True)
class Step1ProvenanceBundle:
    """Step 1 provenance bundle per audit spec §Step 2a + §Cross-worktree scope.

    Field set is the minimum required by the spec; do not add classifier
    self-reports here. Those are "semantic breadcrumbs" per the spec and
    must not be conflated with non-semantic provenance.
    """

    # Runner identity
    runner_name: str
    repo_root: str
    git_head: str

    # Manifest-level provenance
    manifest_path: str | None
    manifest_path_provided: bool
    materialize_called_by_runner: bool

    # Per-artifact provenance (from MaterializedArtifact return)
    materialized_artifacts: list[MaterializedArtifactRecord] = field(
        default_factory=list
    )

    # Resolver-level provenance
    resolved_from_manifest: list[str] = field(default_factory=list)
    cli_overrides: list[str] = field(default_factory=list)

    # Bypass markers (populated when runner does not follow standard manifest path)
    bypass_classes: list[str] = field(default_factory=list)

    # Cross-worktree provenance: sha256 per tracked source file
    source_file_sha256: dict[str, str] = field(default_factory=dict)


def emit_step1_provenance(
    *,
    runner_name: str,
    manifest_path: Path | None,
    materialized_artifacts: "Sequence[MaterializedArtifact] | None",
    resolved_from_manifest: Sequence[str] | None,
    cli_overrides: Sequence[str] | None,
    materialize_called_by_runner: bool,
    output_path: Path,
    repo_root: Path | None = None,
    bypass_classes: Sequence[str] | None = None,
) -> None:
    """Write the Step 1 provenance bundle for one runner invocation.

    Fail-open: emission errors are logged at ERROR but never raised. Callers
    can drop this in around existing materialize/resolve calls without risk
    of breaking the runner.
    """
    try:
        bundle = _build_bundle(
            runner_name=runner_name,
            manifest_path=manifest_path,
            materialized_artifacts=materialized_artifacts,
            resolved_from_manifest=resolved_from_manifest,
            cli_overrides=cli_overrides,
            materialize_called_by_runner=materialize_called_by_runner,
            repo_root=repo_root,
            bypass_classes=bypass_classes,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(bundle), indent=2, sort_keys=True))
        logger.info(
            "step1_provenance_emitted",
            extra={"runner": runner_name, "output": str(output_path)},
        )
    except Exception:  # noqa: BLE001 — fail-open instrumentation per spec
        logger.exception("step1_provenance_emit_failed", extra={"runner": runner_name})


def _build_bundle(
    *,
    runner_name: str,
    manifest_path: Path | None,
    materialized_artifacts: "Sequence[MaterializedArtifact] | None",
    resolved_from_manifest: Sequence[str] | None,
    cli_overrides: Sequence[str] | None,
    materialize_called_by_runner: bool,
    repo_root: Path | None,
    bypass_classes: Sequence[str] | None,
) -> Step1ProvenanceBundle:
    resolved_repo_root = repo_root or _detect_repo_root()
    declared_bypasses = list(bypass_classes or [])
    inferred_bypasses = _infer_bypass_classes(
        manifest_path=manifest_path,
        materialize_called_by_runner=materialize_called_by_runner,
        cli_overrides=cli_overrides,
    )
    merged_bypasses = sorted(set(declared_bypasses + inferred_bypasses))
    _validate_bypass_classes(merged_bypasses)

    return Step1ProvenanceBundle(
        runner_name=runner_name,
        repo_root=str(resolved_repo_root),
        git_head=_git_head(resolved_repo_root),
        manifest_path=str(manifest_path) if manifest_path is not None else None,
        manifest_path_provided=manifest_path is not None,
        materialize_called_by_runner=materialize_called_by_runner,
        materialized_artifacts=[
            MaterializedArtifactRecord.from_materialized(a)
            for a in (materialized_artifacts or [])
        ],
        resolved_from_manifest=list(resolved_from_manifest or []),
        cli_overrides=list(cli_overrides or []),
        bypass_classes=merged_bypasses,
        source_file_sha256=_hash_cross_worktree_sources(resolved_repo_root),
    )


def _infer_bypass_classes(
    *,
    manifest_path: Path | None,
    materialize_called_by_runner: bool,
    cli_overrides: Sequence[str] | None,
) -> list[str]:
    """Infer bypass class from harness inputs. Spec §Step 2a defines five classes."""
    inferred: list[str] = []
    if manifest_path is None:
        inferred.append("whole_manifest_bypass")
    if not materialize_called_by_runner:
        inferred.append("runner_level_skip")
    if cli_overrides:
        inferred.append("per_field_cli_override")
    return inferred


def _validate_bypass_classes(classes: Sequence[str]) -> None:
    unknown = sorted(set(classes) - _BYPASS_CLASSES)
    if unknown:
        # Don't raise — log and continue. Unknown bypass class is a finding,
        # not a runtime error.
        logger.warning(
            "step1_provenance_unknown_bypass_class", extra={"unknown_classes": unknown}
        )


def _detect_repo_root() -> Path:
    """Walk up from CWD to find the repo root (directory containing pyproject.toml)."""
    current = Path.cwd()
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current


def _git_head(repo_root: Path) -> str:
    """Return git HEAD SHA. Returns 'unknown' on failure rather than raising."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        logger.exception("step1_provenance_git_head_failed")
    return "unknown"


def _hash_cross_worktree_sources(repo_root: Path) -> dict[str, str]:
    """Compute sha256 for every tracked source file under the spec-defined directories."""
    files = _enumerate_source_files(repo_root)
    if not files:
        return {}
    if len(files) > _PARALLEL_HASH_THRESHOLD:
        return _hash_parallel(repo_root, files)
    return {str(f.relative_to(repo_root)): _sha256_file(f) for f in files}


def _enumerate_source_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for rel_dir in _CROSS_WORKTREE_SOURCE_DIRS:
        directory = repo_root / rel_dir
        if not directory.exists():
            continue
        files.extend(p for p in directory.rglob("*.py") if p.is_file())
    return sorted(files)


def _hash_parallel(repo_root: Path, files: list[Path]) -> dict[str, str]:
    with ThreadPoolExecutor(max_workers=_DEFAULT_HASH_WORKERS) as pool:
        digests = list(pool.map(_sha256_file, files))
    return {str(f.relative_to(repo_root)): d for f, d in zip(files, digests)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Emit a Step 1 audit provenance bundle for a runner invocation."
    )
    parser.add_argument("--runner-name", required=True)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--materialize-called",
        action="store_true",
        help="Set when the runner invoked materialize_manifest_from_args.",
    )
    parser.add_argument("--resolved-from-manifest", nargs="*", default=[])
    parser.add_argument("--cli-overrides", nargs="*", default=[])
    parser.add_argument(
        "--bypass-class",
        action="append",
        default=[],
        help="Explicit bypass class to record.",
    )
    args = parser.parse_args()

    emit_step1_provenance(
        runner_name=args.runner_name,
        manifest_path=args.manifest_path,
        materialized_artifacts=None,
        resolved_from_manifest=args.resolved_from_manifest,
        cli_overrides=args.cli_overrides,
        materialize_called_by_runner=args.materialize_called,
        output_path=args.output,
        bypass_classes=args.bypass_class,
    )
