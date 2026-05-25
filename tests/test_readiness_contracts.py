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
        "src/regime_detection/observability.py",
        "src/regime_detection/loaders.py",
        "scripts/detect_flaky_tests.py",
        "scripts/validate_agents_md.py",
    } <= include_paths


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
