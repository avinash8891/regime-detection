"""Unit tests for the Step 1 audit harness.

Uses realistic production identifiers per CLAUDE.md global rule "NEVER use
toy names in unit tests." Runner names come from the spec's §Required
runner set; artifact names come from real manifest fixtures.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from regime_data_fetch.materialization import MaterializedArtifact
from scripts.audit_step1_harness import (
    Step1ProvenanceBundle,
    _BYPASS_CLASSES,
    _CROSS_WORKTREE_SOURCE_DIRS,
    _hash_cross_worktree_sources,
    _infer_bypass_classes,
    _sha256_file,
    emit_step1_provenance,
)


# Production identifiers from the spec's §Required runner set — not toy names.
PROFILE_ENGINE = "profile_engine"
AUDIT_LAYER2_30D = "audit_layer2_30d"
RUN_SHADOW_REGIME = "run_shadow_regime"
RUN_HISTORICAL_WALKFORWARD = "run_historical_walkforward"


def _read_bundle(output_path: Path) -> dict:
    return json.loads(output_path.read_text())


def _make_materialized_artifact(name: str, sha: str, tmp: Path) -> MaterializedArtifact:
    return MaterializedArtifact(name=name, destination=tmp / name, sha256=sha)


@pytest.mark.unit
def test_full_manifest_path_records_no_bypass(tmp_path: Path) -> None:
    """When manifest is provided and resolver runs, no bypass markers fire."""
    output = tmp_path / "profile_engine.json"
    artifacts = [
        _make_materialized_artifact(
            "daily_ohlcv_partitioned",
            "a" * 64,
            tmp_path,
        ),
        _make_materialized_artifact(
            "macro_parquet",
            "b" * 64,
            tmp_path,
        ),
    ]
    emit_step1_provenance(
        runner_name=PROFILE_ENGINE,
        manifest_path=tmp_path / "manifest.yaml",
        materialized_artifacts=artifacts,
        resolved_from_manifest=["daily_dir", "constituent_tree", "macro_parquet"],
        cli_overrides=[],
        materialize_called_by_runner=True,
        output_path=output,
        repo_root=tmp_path,
    )

    bundle = _read_bundle(output)
    assert bundle["runner_name"] == PROFILE_ENGINE
    assert bundle["manifest_path_provided"] is True
    assert bundle["materialize_called_by_runner"] is True
    assert bundle["bypass_classes"] == []
    assert len(bundle["materialized_artifacts"]) == 2
    assert bundle["materialized_artifacts"][0]["name"] == "daily_ohlcv_partitioned"
    assert bundle["materialized_artifacts"][0]["sha256"] == "a" * 64
    assert bundle["resolved_from_manifest"] == [
        "daily_dir",
        "constituent_tree",
        "macro_parquet",
    ]
    assert bundle["cli_overrides"] == []


@pytest.mark.unit
def test_no_manifest_records_whole_manifest_bypass(tmp_path: Path) -> None:
    """Whole-manifest bypass marker fires when manifest_path is None.

    Spec §Step 2a bypass class 1 — materialize_if_requested(manifest_path=None)
    at src/regime_data_fetch/materialization.py:150-151 returns [] and skips
    sha verification entirely.
    """
    output = tmp_path / "shadow.json"
    emit_step1_provenance(
        runner_name=RUN_SHADOW_REGIME,
        manifest_path=None,
        materialized_artifacts=None,
        resolved_from_manifest=None,
        cli_overrides=None,
        materialize_called_by_runner=False,
        output_path=output,
        repo_root=tmp_path,
    )

    bundle = _read_bundle(output)
    assert bundle["manifest_path_provided"] is False
    assert bundle["materialize_called_by_runner"] is False
    assert "whole_manifest_bypass" in bundle["bypass_classes"]
    assert "runner_level_skip" in bundle["bypass_classes"]


@pytest.mark.unit
def test_cli_overrides_records_per_field_bypass(tmp_path: Path) -> None:
    """Per-field CLI override marker fires when cli_overrides is non-empty.

    Spec §Step 2a bypass class 2 — resolve_runner_input_paths at
    src/regime_data_fetch/manifest_inputs.py:269-289 accepts CLI values
    that bypass manifest resolution per field.
    """
    output = tmp_path / "audit.json"
    emit_step1_provenance(
        runner_name=AUDIT_LAYER2_30D,
        manifest_path=tmp_path / "manifest.yaml",
        materialized_artifacts=[],
        resolved_from_manifest=["daily_dir"],
        cli_overrides=["event_calendar", "macro_parquet"],
        materialize_called_by_runner=True,
        output_path=output,
        repo_root=tmp_path,
    )

    bundle = _read_bundle(output)
    assert "per_field_cli_override" in bundle["bypass_classes"]
    assert bundle["cli_overrides"] == ["event_calendar", "macro_parquet"]


@pytest.mark.unit
def test_materialize_but_dont_bind_recorded_explicitly(tmp_path: Path) -> None:
    """4th bypass class is not inferred — caller must declare it explicitly.

    Spec §Step 2a bypass class 4 — scripts/run_historical_walkforward.py:386-401
    calls materialize_if_requested but passes original CLI paths into
    run_walkforward instead of manifest-resolved paths. The harness cannot
    infer this from its inputs alone; the runner author must declare it.
    """
    output = tmp_path / "historical.json"
    emit_step1_provenance(
        runner_name=RUN_HISTORICAL_WALKFORWARD,
        manifest_path=tmp_path / "manifest.yaml",
        materialized_artifacts=[_make_materialized_artifact("market_data", "c" * 64, tmp_path)],
        resolved_from_manifest=[],
        cli_overrides=[],
        materialize_called_by_runner=True,
        output_path=output,
        repo_root=tmp_path,
        bypass_classes=["materialize_but_dont_bind"],
    )

    bundle = _read_bundle(output)
    assert "materialize_but_dont_bind" in bundle["bypass_classes"]


@pytest.mark.unit
def test_cross_worktree_hash_stable_across_calls(tmp_path: Path) -> None:
    """Repeated calls on an unchanged worktree produce identical sha256 dict.

    This is the no-drift case: hashing the same files twice gives the same
    digest, so the harness will not produce spurious drift findings.
    """
    fake_root = tmp_path
    src = fake_root / "src" / "regime_detection"
    src.mkdir(parents=True)
    (src / "engine.py").write_text("from __future__ import annotations\n\n# real module\n")
    (src / "feature_store.py").write_text("from __future__ import annotations\n")

    first = _hash_cross_worktree_sources(fake_root)
    second = _hash_cross_worktree_sources(fake_root)
    assert first == second
    assert "src/regime_detection/engine.py" in first
    assert "src/regime_detection/feature_store.py" in first
    assert all(len(digest) == 64 for digest in first.values())


@pytest.mark.unit
def test_cross_worktree_hash_changes_when_source_changes(tmp_path: Path) -> None:
    """Drift case: modifying a source file produces a different sha256.

    Spec §Cross-worktree scope: differences across worktrees in the same
    logical file are recorded as BROKEN_WIRING × MISMATCH findings.
    """
    fake_root = tmp_path
    src = fake_root / "src" / "regime_detection"
    src.mkdir(parents=True)
    target = src / "engine.py"
    target.write_text("v1\n")
    before = _hash_cross_worktree_sources(fake_root)

    target.write_text("v2 — divergent\n")
    after = _hash_cross_worktree_sources(fake_root)

    rel = "src/regime_detection/engine.py"
    assert before[rel] != after[rel]


@pytest.mark.unit
def test_emit_is_fail_open_when_output_dir_unwritable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Spec + CLAUDE.md global rule: instrumentation must never block business logic.

    A write failure must be logged, not raised.
    """
    output = Path("/proc/cannot-write-here/profile_engine.json")
    # Should not raise.
    emit_step1_provenance(
        runner_name=PROFILE_ENGINE,
        manifest_path=None,
        materialized_artifacts=None,
        resolved_from_manifest=None,
        cli_overrides=None,
        materialize_called_by_runner=False,
        output_path=output,
        repo_root=tmp_path,
    )
    assert any(
        "step1_provenance_emit_failed" in record.message
        for record in caplog.records
        if record.levelname == "ERROR"
    )


@pytest.mark.unit
def test_bypass_classes_enum_matches_spec() -> None:
    """The five bypass classes must match the spec §Step 2a enumeration."""
    assert _BYPASS_CLASSES == frozenset(
        {
            "whole_manifest_bypass",
            "per_field_cli_override",
            "runner_level_skip",
            "materialize_but_dont_bind",
            "cross_worktree_drift",
        }
    )


@pytest.mark.unit
def test_source_dirs_match_spec() -> None:
    """Cross-worktree provenance covers exactly the trust-critical engine surfaces."""
    assert _CROSS_WORKTREE_SOURCE_DIRS == (
        "src/regime_detection",
        "src/regime_data_fetch",
    )


@pytest.mark.unit
def test_infer_bypass_classes_combinations() -> None:
    """All inference paths produce the documented bypass class set."""
    assert _infer_bypass_classes(
        manifest_path=Path("/m.yaml"),
        materialize_called_by_runner=True,
        cli_overrides=[],
    ) == []
    assert "whole_manifest_bypass" in _infer_bypass_classes(
        manifest_path=None,
        materialize_called_by_runner=True,
        cli_overrides=[],
    )
    assert "runner_level_skip" in _infer_bypass_classes(
        manifest_path=Path("/m.yaml"),
        materialize_called_by_runner=False,
        cli_overrides=[],
    )
    assert "per_field_cli_override" in _infer_bypass_classes(
        manifest_path=Path("/m.yaml"),
        materialize_called_by_runner=True,
        cli_overrides=["event_calendar"],
    )


@pytest.mark.integration
def test_cli_entrypoint_emits_bundle(tmp_path: Path) -> None:
    """End-to-end: invoking the script as a subprocess produces a valid JSON bundle.

    This tests the integration path (subprocess invocation, not just import)
    per CLAUDE.md global rule "Test the INTEGRATION path, not just the unit."
    """
    output = tmp_path / "cli_bundle.json"
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.audit_step1_harness",
            "--runner-name",
            PROFILE_ENGINE,
            "--output",
            str(output),
            "--materialize-called",
            "--resolved-from-manifest",
            "daily_dir",
            "macro_parquet",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    bundle = _read_bundle(output)
    assert bundle["runner_name"] == PROFILE_ENGINE
    assert bundle["materialize_called_by_runner"] is True
    assert bundle["resolved_from_manifest"] == ["daily_dir", "macro_parquet"]
    # Cross-worktree provenance covers real source dirs because subprocess
    # ran from the actual repo root.
    assert "src/regime_detection/engine.py" in bundle["source_file_sha256"]
