from __future__ import annotations

from pathlib import Path

import yaml


def test_fixture_verification_is_deterministic() -> None:
    """
    Hard gate for Slice 2:
    - committed derived/report fixtures must be exactly what verify_fixtures produces
      against the committed raw CSVs.
    """
    repo_root = Path(__file__).resolve().parents[1]
    derived_path = repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml"
    report_path = repo_root / "tests" / "fixtures" / "verification" / "golden_dates_report.yaml"

    committed_derived = yaml.safe_load(derived_path.read_text())
    committed_report = yaml.safe_load(report_path.read_text())

    # Load the script as a module from its path (scripts/ isn't a Python package).
    import importlib.util

    script_path = repo_root / "scripts" / "verify_fixtures.py"
    spec = importlib.util.spec_from_file_location("verify_fixtures", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    generate_docs = getattr(mod, "generate_docs")

    regen_derived, regen_report = generate_docs(
        generated_at_utc=committed_derived["generated_at_utc"]
    )

    assert committed_derived == regen_derived
    assert committed_report == regen_report
