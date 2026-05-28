from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

AuditStep1Fixtures = dict[str, dict[str, dict[str, Any]]]


def expand_audit_step1_fixtures(fixture_root: Path) -> AuditStep1Fixtures:
    runner_names = json.loads((fixture_root / "runner_names.json").read_text())
    return {
        "current": _expand_group(
            template_path=fixture_root / "current_template.json",
            runner_names=runner_names,
        ),
        "historical": _expand_group(
            template_path=fixture_root / "historical_template.json",
            runner_names=runner_names,
        ),
    }


def _expand_group(
    *,
    template_path: Path,
    runner_names: list[str],
) -> dict[str, dict[str, Any]]:
    template = json.loads(template_path.read_text())
    defaults = template["defaults"]
    overrides_by_runner = template.get("runner_overrides", {})

    expanded: dict[str, dict[str, Any]] = {}
    for runner_name in runner_names:
        payload = copy.deepcopy(defaults)
        payload.update(copy.deepcopy(overrides_by_runner.get(runner_name, {})))
        payload["runner_name"] = runner_name
        expanded[runner_name] = payload
    return expanded
