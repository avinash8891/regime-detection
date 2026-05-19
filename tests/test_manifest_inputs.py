from __future__ import annotations

from pathlib import Path
from typing import get_type_hints
from urllib.parse import urlparse

import pytest
import yaml

from regime_data_fetch.artifact_manifest import load_manifest
from regime_data_fetch.manifest_inputs import (
    ARTIFACT_BY_FIELD,
    MANIFEST_INPUT_FLAGS,
    MANIFEST_INPUT_SPECS,
    ManifestInputResolutionError,
    OptionalRunnerInputPaths,
    REQUIRED_INPUT_FIELDS,
    RequiredRunnerInputPaths,
    RunnerInputPaths,
    resolve_runner_input_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MANIFEST = REPO_ROOT / "manifests" / "runs" / "regime_engine_2026-05-17.yaml"
OHLCV_ONLY_MANIFEST = (
    REPO_ROOT
    / "manifests"
    / "runs"
    / "profile_ready_daily_ohlcv_762_2016_20260515.yaml"
)
# Runners whose contract REQUIRES the macro/PIT/event/sentiment bundle and
# therefore can only be satisfied by the merged engine manifest. The OHLCV-only
# manifest must NOT claim these runners in any artifact's ``required_for``.
ENGINE_RUNNERS_REQUIRING_FULL_BUNDLE = (
    "profile_engine",
    "v2_calibration",
    "historical_walkforward",
    "audit_layer2_30d",
)
# SHA-256 of the empty string; used in the committed manifest as a sentinel
# for TODO placeholder artifacts whose canonical store entry has not been
# generated yet (see manifests/runs/regime_engine_2026-05-17.yaml header).
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
# Constituent-OHLCV placeholder name: the real per-symbol manifest has not
# been regenerated yet (see profile_ready_daily_ohlcv_762_2016_20260515.md).
# The resolver discovers it structurally via local_path, not by name, so it
# is permitted to skip the ARTIFACT_BY_FIELD round-trip.
OHLCV_PLACEHOLDER_NAME = "daily_ohlcv_762"
# TODO-placeholder artifacts whose canonical fetchers have not been wired
# yet. These are documented in the manifest header and in README.md as
# operator-action items; they are allowed to skip the ARTIFACT_BY_FIELD
# round-trip until their fetchers and resolver fields land together.
KNOWN_TODO_PLACEHOLDER_NAMES = frozenset(
    {
        "sp500_eps_weekly_history",
    }
)
# Audit-only provenance artifacts tracked in the manifest but not consumed
# as runner inputs. They are materialized for audit/reproducibility but do
# not need an ARTIFACT_BY_FIELD resolver mapping.
AUDIT_ONLY_ARTIFACT_NAMES = frozenset(
    {
        "event_candidates",
        "event_validations",
        "event_quarantine",
    }
)


SHA = "0" * 64


def test_runner_input_required_fields_are_backed_by_required_dataclass() -> None:
    required_annotations = get_type_hints(RequiredRunnerInputPaths)
    assert required_annotations == {
        "daily_dir": Path,
        "constituent_tree": Path,
        "macro_parquet": Path,
        "pit_parquet": Path,
        "event_calendar": Path,
    }
    assert REQUIRED_INPUT_FIELDS == frozenset(required_annotations)
    assert set(get_type_hints(RunnerInputPaths)) >= {
        "required",
        "optional",
        "resolved_from_manifest",
        "cli_overrides",
    }


def _artifact(name: str, local_path: str, required_for: list[str] | None = None):
    return {
        "name": name,
        "stage": "canonical",
        "uri": f"s3://bucket/{local_path}",
        "local_path": local_path,
        "sha256": SHA,
        "schema_version": None,
        "rows": 1,
        "min_date": None,
        "max_date": None,
        "required_for": required_for or ["profile_engine"],
    }


def _write_manifest(tmp_path: Path, artifacts: list[dict]) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "artifact_set": "test",
                "created_at_utc": "2026-05-17T00:00:00Z",
                "storage_root": "s3://bucket/root",
                "artifacts": artifacts,
            },
            sort_keys=False,
        )
    )
    return path


def test_resolve_runner_input_paths_uses_manifest_artifact_names(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            _artifact(
                "constituent_ohlcv_AAPL",
                "data/raw/daily_ohlcv_762/symbol=AAPL/ohlcv.parquet",
            ),
            _artifact("fred_macro_series", "data/raw/macro/fred_macro_series.parquet"),
            _artifact(
                "sp500_pit_constituents",
                "data/raw/pit_constituents/sp500_ticker_intervals.parquet",
            ),
            _artifact("event_calendar_us", "data/raw/event_calendar/us_events.yaml"),
            _artifact("ism_pmi_history", "data/raw/pmi/us_ism_pmi_history.parquet"),
            _artifact(
                "sf_fed_news_sentiment",
                "data/raw/news_sentiment/sf_fed_news_sentiment.parquet",
            ),
            _artifact("aaii_sentiment", "data/raw/sentiment/aaii_sentiment.parquet"),
            _artifact("fomc_minutes", "data/raw/fomc_minutes/fomc_minutes.parquet"),
            _artifact(
                "powell_speeches",
                "data/raw/powell_speeches/powell_speeches.parquet",
            ),
            _artifact(
                "cpi_all_items_vintages",
                "data/raw/macro_vintages/cpi_all_items_vintages.parquet",
            ),
        ],
    )
    data_root = tmp_path / "materialized" / "data" / "raw"

    resolved = resolve_runner_input_paths(
        manifest_path=manifest_path,
        data_root=data_root,
        runner_name="profile_engine",
        cli_values={},
        cli_overrides=set(),
        repo_root=tmp_path,
    )

    assert resolved.daily_dir == data_root / "daily_ohlcv_762"
    assert resolved.constituent_tree == data_root / "daily_ohlcv_762"
    assert resolved.macro_parquet == data_root / "macro" / "fred_macro_series.parquet"
    assert resolved.pit_parquet == (
        data_root / "pit_constituents" / "sp500_ticker_intervals.parquet"
    )
    assert resolved.event_calendar == data_root / "event_calendar" / "us_events.yaml"
    assert resolved.pmi_path == data_root / "pmi" / "us_ism_pmi_history.parquet"
    assert resolved.news_sentiment_parquet == (
        data_root / "news_sentiment" / "sf_fed_news_sentiment.parquet"
    )
    assert "news_sentiment_parquet" in resolved.resolved_from_manifest


def test_resolve_runner_input_paths_accepts_emitted_partition_tree_names(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            _artifact(
                "daily_ohlcv_parquet_symbol=SPY_ohlcv.parquet",
                "data/raw/daily_ohlcv_762/symbol=SPY/ohlcv.parquet",
            ),
            _artifact("fred_macro_series", "data/raw/macro/fred_macro_series.parquet"),
            _artifact(
                "sp500_pit_constituents",
                "data/raw/pit_constituents/sp500_ticker_intervals.parquet",
            ),
            _artifact("event_calendar_us", "data/raw/event_calendar/us_events.yaml"),
        ],
    )
    data_root = tmp_path / "materialized" / "data" / "raw"

    resolved = resolve_runner_input_paths(
        manifest_path=manifest_path,
        data_root=data_root,
        runner_name="profile_engine",
        cli_values={},
        cli_overrides=set(),
        repo_root=tmp_path,
    )

    assert resolved.daily_dir == data_root / "daily_ohlcv_762"
    assert resolved.constituent_tree == data_root / "daily_ohlcv_762"


def test_resolve_runner_input_paths_respects_cli_override(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            _artifact(
                "constituent_ohlcv_AAPL",
                "data/raw/daily_ohlcv_762/symbol=AAPL/ohlcv.parquet",
            ),
            _artifact("fred_macro_series", "data/raw/macro/fred_macro_series.parquet"),
            _artifact(
                "sp500_pit_constituents",
                "data/raw/pit_constituents/sp500_ticker_intervals.parquet",
            ),
            _artifact("event_calendar_us", "data/raw/event_calendar/us_events.yaml"),
            _artifact(
                "sf_fed_news_sentiment",
                "data/raw/news_sentiment/sf_fed_news_sentiment.parquet",
            ),
        ],
    )
    override_path = tmp_path / "manual" / "news.parquet"

    resolved = resolve_runner_input_paths(
        manifest_path=manifest_path,
        data_root=tmp_path / "data" / "raw",
        runner_name="profile_engine",
        cli_values={"news_sentiment_parquet": override_path},
        cli_overrides={"news_sentiment_parquet"},
    )

    assert resolved.news_sentiment_parquet == override_path
    assert "news_sentiment_parquet" in resolved.cli_overrides
    assert "news_sentiment_parquet" not in resolved.resolved_from_manifest


def test_resolve_runner_input_paths_fails_without_constituent_tree(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            _artifact("fred_macro_series", "data/raw/macro/fred_macro_series.parquet"),
            _artifact(
                "sp500_pit_constituents",
                "data/raw/pit_constituents/sp500_ticker_intervals.parquet",
            ),
        ],
    )

    with pytest.raises(
        ManifestInputResolutionError,
        match="missing required daily OHLCV artifacts",
    ):
        resolve_runner_input_paths(
            manifest_path=manifest_path,
            data_root=tmp_path / "data" / "raw",
            runner_name="profile_engine",
            cli_values={},
            cli_overrides=set(),
        )


@pytest.mark.unit
def test_committed_manifest_materializes_from_fresh_workspace() -> None:
    """Manifest-shape check on the real committed lockfile.

    Validates that ``manifests/runs/regime_engine_2026-05-17.yaml`` can be
    materialized from any checkout without rewriting paths:

    1. Every artifact carries a non-null ``uri`` (or is flagged with the
       empty-string sentinel + TODO comment).
    2. URIs are workspace-agnostic — no absolute ``/Users/...`` paths and no
       ``file:///...`` URIs that embed an absolute workspace path.
    3. ``storage_root`` is relative, so the loader can anchor it to the
       manifest file's parent directory.
    4. Every artifact name (except the OHLCV-tree placeholder, which the
       resolver discovers structurally) round-trips through
       ``ARTIFACT_BY_FIELD`` — i.e. the resolver knows every artifact in the
       manifest, or that name belongs to the constituent-OHLCV partition
       contract documented in ``manifest_inputs._is_constituent_ohlcv_artifact``.
    """
    assert COMMITTED_MANIFEST.exists(), (
        f"committed manifest missing: {COMMITTED_MANIFEST}"
    )
    manifest = load_manifest(COMMITTED_MANIFEST)

    # (4) storage_root must be workspace-agnostic. Two shapes satisfy the
    # invariant: a relative local path (anchored to the manifest's parent dir
    # by _resolve_store_root) or a URI with a scheme (s3:// / file://) which
    # is passed through unchanged. An absolute local filesystem path would
    # pin the lockfile to one operator's workspace and is rejected.
    parsed_storage = urlparse(manifest.storage_root)
    if not parsed_storage.scheme:
        assert not Path(manifest.storage_root).is_absolute(), (
            f"storage_root must be relative (or a URI with a scheme) for "
            f"workspace-agnostic resolution; got {manifest.storage_root!r}"
        )

    known_artifact_names = set(ARTIFACT_BY_FIELD.values())
    placeholder_uris: list[str] = []

    for artifact in manifest.artifacts:
        # (1) uri must be present and non-empty.
        assert artifact.uri, f"artifact {artifact.name} has empty uri"

        # (2) uri must be workspace-agnostic.
        assert not artifact.uri.startswith("/Users/"), (
            f"artifact {artifact.name} uri embeds an absolute home path: "
            f"{artifact.uri}"
        )
        if artifact.uri.startswith("file:///"):
            # A file:// URI inside a committed manifest would pin the lockfile
            # to one operator's workspace — reject outright.
            pytest.fail(
                f"artifact {artifact.name} uses a file:// uri that embeds an "
                f"absolute workspace path: {artifact.uri}"
            )
        # No bare absolute POSIX paths either (covers '/private/...' etc).
        assert not artifact.uri.startswith("/"), (
            f"artifact {artifact.name} uri is an absolute filesystem path: "
            f"{artifact.uri}"
        )

        # (5) Every name must either be in ARTIFACT_BY_FIELD or be the
        # documented OHLCV-tree placeholder / a per-symbol partition name.
        if artifact.name in known_artifact_names:
            pass  # resolver knows this name
        elif artifact.name == OHLCV_PLACEHOLDER_NAME:
            pass  # documented placeholder for the 762-symbol tree
        elif artifact.name.startswith(
            ("constituent_ohlcv_", "daily_ohlcv_parquet_", "daily_ohlcv_762_")
        ):
            pass  # per-symbol partition contract (manifest_inputs._is_constituent_ohlcv_artifact)
        elif (
            artifact.name in KNOWN_TODO_PLACEHOLDER_NAMES
            and artifact.sha256 == EMPTY_SHA256
        ):
            pass  # documented TODO placeholder; fetcher + resolver land together
        elif artifact.name in AUDIT_ONLY_ARTIFACT_NAMES:
            pass  # provenance-only; materialized for audit, not a runner input
        else:
            pytest.fail(
                f"manifest artifact {artifact.name!r} is not known to the "
                f"resolver (ARTIFACT_BY_FIELD round-trip failed). Either add "
                f"a field mapping in manifest_inputs.ARTIFACT_BY_FIELD or "
                f"remove the artifact from the manifest."
            )

        # Track TODO sentinels so we can keep the documented set explicit.
        if artifact.sha256 == EMPTY_SHA256:
            placeholder_uris.append(artifact.name)

    # (1 cont.) Placeholders are allowed but must remain a known set so a new
    # accidental TODO entry breaks the test loudly.
    assert set(placeholder_uris) <= (
        {OHLCV_PLACEHOLDER_NAME} | KNOWN_TODO_PLACEHOLDER_NAMES
    ), (
        f"unexpected TODO-placeholder artifacts in committed manifest: "
        f"{sorted(placeholder_uris)}"
    )


@pytest.mark.unit
def test_ohlcv_only_manifest_does_not_claim_engine_runners() -> None:
    """Contract test: the OHLCV-only lockfile must not advertise itself as a
    valid input for runners that require the full macro/PIT/event bundle.

    Historical bug: every artifact in
    ``manifests/runs/profile_ready_daily_ohlcv_762_2016_20260515.yaml``
    carried ``required_for: [profile_engine, v2_calibration,
    historical_walkforward, audit_layer2_30d]``, so the runner ran far enough
    to materialize 5/1086 OHLCV parquets before failing inside
    ``resolve_runner_input_paths`` with
    ``ManifestInputResolutionError: manifest missing required artifact
    fred_macro_series for macro_parquet``. This test prevents the false claim
    from being reintroduced.
    """
    assert OHLCV_ONLY_MANIFEST.exists(), (
        f"OHLCV-only manifest missing: {OHLCV_ONLY_MANIFEST}"
    )
    manifest = load_manifest(OHLCV_ONLY_MANIFEST)
    for runner in ENGINE_RUNNERS_REQUIRING_FULL_BUNDLE:
        matching = manifest.required_for(runner)
        assert matching == [], (
            f"OHLCV-only manifest must not claim runner {runner!r}; the "
            f"runner requires the merged engine manifest "
            f"(regime_engine_YYYY-MM-DD.yaml). Offenders: "
            f"{[a.name for a in matching][:5]}"
        )


@pytest.mark.unit
def test_ohlcv_only_manifest_fails_fast_for_profile_engine(tmp_path: Path) -> None:
    """The honest failure mode when an operator points ``profile_engine.py`` at
    the OHLCV-only manifest is an early ``ManifestInputResolutionError`` from
    ``resolve_runner_input_paths`` with ``manifest has no artifacts required
    for profile_engine`` — i.e. before materialization side effects.
    """
    with pytest.raises(
        ManifestInputResolutionError,
        match="manifest has no artifacts required for profile_engine",
    ):
        resolve_runner_input_paths(
            manifest_path=OHLCV_ONLY_MANIFEST,
            data_root=tmp_path / "data" / "raw",
            runner_name="profile_engine",
            cli_values={},
            cli_overrides=set(),
        )


@pytest.mark.unit
def test_merged_manifest_satisfies_profile_engine_contract() -> None:
    """Contract test for the runner the live profile_engine.py call enforces:
    every field in ``REQUIRED_INPUT_FIELDS`` that maps through
    ``ARTIFACT_BY_FIELD`` must be present in the merged manifest with
    ``profile_engine`` in ``required_for``, and a constituent-OHLCV
    partition must be reachable for ``_constituent_tree_root``.
    """
    from regime_data_fetch.manifest_inputs import _is_constituent_ohlcv_artifact

    merged = load_manifest(COMMITTED_MANIFEST)
    required_artifact_names = {
        ARTIFACT_BY_FIELD[field]
        for field in REQUIRED_INPUT_FIELDS
        if field in ARTIFACT_BY_FIELD
    }
    runner_artifacts = merged.required_for("profile_engine")
    runner_artifact_names = {a.name for a in runner_artifacts}
    missing = required_artifact_names - runner_artifact_names
    assert not missing, (
        f"merged manifest does not declare 'profile_engine' for required "
        f"artifacts: {sorted(missing)}"
    )
    assert any(_is_constituent_ohlcv_artifact(a) for a in runner_artifacts), (
        "merged manifest declares 'profile_engine' but carries no "
        "constituent OHLCV partition artifacts for daily_dir / constituent_tree"
    )


# ---------------------------------------------------------------------------
# Registry-driven coverage tests.
#
# These tests are the class-of-issue guard for the manifest-router gap that
# the 2026-05-18 audit found (`cpi_nowcast_parquet` was in ARTIFACT_BY_FIELD
# but missing from MANIFEST_INPUT_FLAGS). They assert that the unified
# MANIFEST_INPUT_SPECS registry stays in lock-step with every derived
# structure — adding a spec entry must propagate to the artifact-name map,
# the CLI-flag map, the dataclass fields, and the runner argparse surface.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    MANIFEST_INPUT_SPECS,
    ids=lambda spec: spec.field,
)
def test_manifest_input_spec_propagates_to_artifact_map(spec) -> None:
    """Every spec with an artifact_name must round-trip through
    ARTIFACT_BY_FIELD. Catches the 2026-05-18 audit's class-of-issue bug:
    declaring a manifest field with an artifact but no router wiring."""
    if spec.artifact_name is None:
        assert spec.field not in ARTIFACT_BY_FIELD
    else:
        assert ARTIFACT_BY_FIELD[spec.field] == spec.artifact_name


@pytest.mark.parametrize(
    "spec",
    MANIFEST_INPUT_SPECS,
    ids=lambda spec: spec.field,
)
def test_manifest_input_spec_propagates_to_flag_map(spec) -> None:
    """Every spec must register its CLI flag in MANIFEST_INPUT_FLAGS so the
    runner-side ``apply_manifest_input_paths`` loop can copy the resolved
    path onto args."""
    assert MANIFEST_INPUT_FLAGS[spec.field] == spec.cli_flag


@pytest.mark.parametrize(
    "spec",
    MANIFEST_INPUT_SPECS,
    ids=lambda spec: spec.field,
)
def test_manifest_input_spec_has_dataclass_field(spec) -> None:
    """Every spec must be backed by an attribute on either
    RequiredRunnerInputPaths or OptionalRunnerInputPaths — the resolver
    will fail to construct RunnerInputPaths otherwise."""
    required_fields = {f.name for f in __import__("dataclasses").fields(RequiredRunnerInputPaths)}
    optional_fields = {f.name for f in __import__("dataclasses").fields(OptionalRunnerInputPaths)}
    if spec.is_required:
        assert spec.field in required_fields
    else:
        assert spec.field in optional_fields


def test_artifact_by_field_has_no_drift_against_registry() -> None:
    """The flat ARTIFACT_BY_FIELD dict must equal the derived view of
    MANIFEST_INPUT_SPECS — no manual key can sneak in outside the
    registry."""
    derived = {
        spec.field: spec.artifact_name
        for spec in MANIFEST_INPUT_SPECS
        if spec.artifact_name is not None
    }
    assert ARTIFACT_BY_FIELD == derived


def test_manifest_input_flags_has_no_drift_against_registry() -> None:
    """The flat MANIFEST_INPUT_FLAGS dict must equal the derived view of
    MANIFEST_INPUT_SPECS. This is the gate that would have caught the
    cpi_nowcast_parquet drift in the audit."""
    derived = {spec.field: spec.cli_flag for spec in MANIFEST_INPUT_SPECS}
    assert MANIFEST_INPUT_FLAGS == derived


def test_register_manifest_input_args_covers_every_optional_spec() -> None:
    """The runner-side argparse helper must register a flag for every
    optional spec — preventing a runner from silently consuming
    ``getattr(args, field, None)`` without a corresponding --flag."""
    import argparse

    from scripts._v2_calibration_helpers import register_manifest_input_args

    parser = argparse.ArgumentParser()
    register_manifest_input_args(parser, include_required_paths=False)
    registered_flags = {
        flag
        for action in parser._actions
        for flag in action.option_strings
    }
    optional_flags = {
        spec.cli_flag for spec in MANIFEST_INPUT_SPECS if not spec.is_required
    }
    missing = optional_flags - registered_flags
    assert not missing, (
        f"register_manifest_input_args dropped optional flags: {sorted(missing)}"
    )


def test_apply_manifest_input_defaults_covers_every_spec_with_default_relpath(
    tmp_path: Path,
) -> None:
    """Every spec with a default_relpath must produce an args attribute
    when apply_manifest_input_defaults runs. Catches drift where a spec
    declares a relpath but the helper skips it."""
    import argparse

    from scripts._v2_calibration_helpers import apply_manifest_input_defaults

    args = argparse.Namespace()
    for spec in MANIFEST_INPUT_SPECS:
        setattr(args, spec.field, None)

    apply_manifest_input_defaults(args, tmp_path)

    for spec in MANIFEST_INPUT_SPECS:
        if spec.default_relpath is None:
            continue
        expected = tmp_path.joinpath(*spec.default_relpath)
        assert getattr(args, spec.field) == expected, (
            f"apply_manifest_input_defaults skipped {spec.field}"
        )
