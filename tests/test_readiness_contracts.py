from __future__ import annotations

from pathlib import Path

import tomllib
import yaml

import scripts.profile_engine_reporting as profile_engine_reporting


def _workflow_triggers(payload: dict[object, object]) -> object:
    return payload.get("on", payload.get(True))


def test_pyright_strict_scope_matches_readiness_allowlist_exactly() -> None:
    with Path("pyproject.toml").open("rb") as handle:
        payload = tomllib.load(handle)

    include_paths = payload["tool"]["pyright"]["include"]

    assert include_paths == [
        "src/regime_detection",
        "src/regime_data_fetch",
        "src/regime_shared",
        "scripts/_fetch_regime_engine_v1_args.py",
        "scripts/_v2_calibration_helpers.py",
        "scripts/approve_group_b_candidate.py",
        "scripts/audit_layer2_30d.py",
        "scripts/audit_step1_harness.py",
        "scripts/build_walkforward_report.py",
        "scripts/consolidate_regime_acquisition.py",
        "scripts/detect_flaky_tests.py",
        "scripts/fetch_aaii_sentiment.py",
        "scripts/fetch_regime_engine_v1_data.py",
        "scripts/materialize_constituent_ohlcv_tree.py",
        "scripts/materialize_regime_data.py",
        "scripts/normalize_s3_daily_to_sqlite_layout.py",
        "scripts/profile_engine.py",
        "scripts/profile_engine_reporting.py",
        "scripts/profile_engine_timers.py",
        "scripts/publish_canonical_snapshot.py",
        "scripts/run_historical_walkforward.py",
        "scripts/run_shadow_deadman_check.py",
        "scripts/run_shadow_regime.py",
        "scripts/run_shadow_replay_check.py",
        "scripts/run_v2_calibration.py",
        "scripts/run_v2_shadow_ab_gate.py",
        "scripts/run_v2_walkforward_gate.py",
        "scripts/upload_missing_ohlcv_to_manifest.py",
        "scripts/validate_agents_md.py",
        "scripts/validate_central_bank_text_lexicon.py",
        "scripts/verify_fixtures.py",
    ]


def test_ci_is_split_between_pr_fast_and_full_verification() -> None:
    ci_payload = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    full_payload = yaml.safe_load(
        Path(".github/workflows/full-verification.yml").read_text()
    )

    ci_on = _workflow_triggers(ci_payload)
    full_on = _workflow_triggers(full_payload)

    assert ci_on == {"pull_request": None}
    assert "push" in full_on
    assert full_on["push"]["branches"] == ["main"]


def test_pr_fast_ci_keeps_default_pytest_scope() -> None:
    ci_payload = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    commands = [
        step["run"]
        for step in ci_payload["jobs"]["checks"]["steps"]
        if isinstance(step, dict) and "run" in step
    ]

    pytest_runs = [command for command in commands if "python -m pytest" in command]
    assert pytest_runs
    assert all('-m ""' not in command for command in pytest_runs)


def test_validate_agents_guardrail_understands_split_ci() -> None:
    validator = Path("scripts/validate_agents_md.py").read_text()

    assert "SHARED_COMMANDS" in validator
    assert "PR_CI_COMMANDS" in validator
    assert "FULL_VERIFICATION_COMMANDS" in validator
    assert ".github/workflows/full-verification.yml" in validator


def test_release_workflow_runs_all_pytest_markers() -> None:
    release_payload = yaml.safe_load(Path(".github/workflows/release.yml").read_text())
    commands = [
        step["run"]
        for step in release_payload["jobs"]["release"]["steps"]
        if isinstance(step, dict) and "run" in step
    ]

    pytest_runs = [command for command in commands if "python -m pytest" in command]
    assert pytest_runs
    assert any('-m ""' in command for command in pytest_runs)


def test_labels_workflow_can_checkout_with_declared_permissions() -> None:
    labels_payload = yaml.safe_load(Path(".github/workflows/labels.yml").read_text())

    assert labels_payload["permissions"]["contents"] == "read"
    assert labels_payload["permissions"]["issues"] == "write"


def test_devcontainer_uses_ruff_python_formatter() -> None:
    devcontainer = yaml.safe_load(Path(".devcontainer/devcontainer.json").read_text())
    settings = devcontainer["customizations"]["vscode"]["settings"]

    assert "python.formatting.provider" not in settings
    assert settings["[python]"]["editor.defaultFormatter"] == "charliermarsh.ruff"


def test_profile_engine_uses_public_reporting_helpers() -> None:
    for helper_name in (
        "input_status",
        "profile_input_seam_values",
        "format_stage_rows",
        "build_json_report",
        "write_json_report",
    ):
        assert hasattr(profile_engine_reporting, helper_name)

    profile_engine_source = Path("scripts/profile_engine.py").read_text()

    assert "from scripts.profile_engine_reporting import (" in profile_engine_source
    for private_name in (
        "_input_status",
        "_profile_input_seam_values",
        "_format_stage_rows",
        "_build_json_report",
        "_write_json_report",
    ):
        assert private_name not in profile_engine_source


def test_audit_layer2_uses_public_profile_engine_helpers() -> None:
    audit_source = Path("scripts/audit_layer2_30d.py").read_text()

    assert "from scripts.profile_engine import (" in audit_source
    for public_name in (
        "build_required_sessions",
        "load_constituent_ohlcv_from_tree",
        "load_optional_aaii_sentiment",
        "load_optional_central_bank_text_releases",
        "load_optional_cpi_first_release",
        "load_event_calendar_input",
        "load_optional_news_sentiment",
    ):
        assert public_name in audit_source

    for private_name in (
        "_build_required_sessions",
        "_load_constituent_ohlcv_from_tree",
        "_load_optional_aaii_sentiment",
        "_load_optional_central_bank_text_releases",
        "_load_optional_cpi_first_release",
        "_load_event_calendar",
        "_load_optional_news_sentiment",
    ):
        assert private_name not in audit_source
