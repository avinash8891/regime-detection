from __future__ import annotations

from pathlib import Path

import tomllib
import yaml


def _workflow_triggers(payload: dict[object, object]) -> object:
    return payload.get("on", payload.get(True))


def test_pyright_strict_slice_covers_runtime_and_guardrail_scripts() -> None:
    with Path("pyproject.toml").open("rb") as handle:
        payload = tomllib.load(handle)

    include_paths = set(payload["tool"]["pyright"]["include"])

    assert {
        "src/regime_detection/engine.py",
        "src/regime_detection/models.py",
        "src/regime_detection/axis_series.py",
        "src/regime_detection/observability.py",
        "src/regime_detection/loaders.py",
        "scripts/detect_flaky_tests.py",
        "scripts/validate_agents_md.py",
    } <= include_paths


def test_pyright_strict_slice_documents_pandas_stub_policy() -> None:
    policy = Path("docs/pyright_pandas_stub_policy.md")

    assert policy.exists()
    text = policy.read_text()
    assert "reportUnknownMemberType" in text
    assert "business logic" in text
    assert "blanket ignores" in text


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
