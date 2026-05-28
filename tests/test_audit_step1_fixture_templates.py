from __future__ import annotations

import json
from pathlib import Path

from audit_step1_fixture_templates import expand_audit_step1_fixtures

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "audit_step1"
EXPECTED_RUNNERS = {
    "approve_group_b_candidate",
    "audit_layer2_30d",
    "build_walkforward_report",
    "materialize_regime_data",
    "profile_engine",
    "run_historical_walkforward",
    "run_shadow_regime",
    "run_v2_calibration",
    "run_v2_walkforward_gate",
}


def test_audit_step1_fixtures_are_template_generated() -> None:
    fixture_files = {
        path.relative_to(FIXTURE_ROOT).as_posix()
        for path in FIXTURE_ROOT.rglob("*")
        if path.is_file()
    }

    assert fixture_files == {
        "current_template.json",
        "historical_template.json",
        "runner_names.json",
    }

    generated = expand_audit_step1_fixtures(FIXTURE_ROOT)

    assert set(generated) == {"current", "historical"}
    assert set(generated["current"]) == EXPECTED_RUNNERS
    assert set(generated["historical"]) == EXPECTED_RUNNERS
    assert all(
        payload["runner_name"] == runner_name
        for group in generated.values()
        for runner_name, payload in group.items()
    )

    runner_names = json.loads((FIXTURE_ROOT / "runner_names.json").read_text())
    assert runner_names == sorted(EXPECTED_RUNNERS)
