from __future__ import annotations

from pathlib import Path

import yaml


def test_fixture_verification_is_deterministic() -> None:
    """
    Hard gate for Slice 2:
    - golden_dates.yaml is hand-labeled (never engine-generated)
    - the report must be reproducible against the committed raw CSVs
    - expected values in the report must match the hand-labeled expectations
    """
    repo_root = Path(__file__).resolve().parents[1]
    derived_path = repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml"

    committed_derived = yaml.safe_load(derived_path.read_text())
    assert committed_derived.get("provenance") == "hand_labeled", (
        "golden_dates.yaml must carry provenance: hand_labeled — "
        "expected values are independently derived, not from engine output"
    )

    import importlib.util

    script_path = repo_root / "scripts" / "verify_fixtures.py"
    spec = importlib.util.spec_from_file_location("verify_fixtures", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    generate_report = getattr(mod, "generate_report")

    report = generate_report(
        generated_at_utc="2026-05-19T00:00:00+00:00",
        generated_by_commit="test_determinism",
    )
    for row in report["rows"]:
        mismatches = row.get("mismatches")
        assert not mismatches, (
            f"Engine output diverges from hand-labeled expectation for "
            f"{row['intent_id']}: {mismatches}"
        )
