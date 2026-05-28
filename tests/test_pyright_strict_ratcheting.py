from __future__ import annotations

import tomllib
from pathlib import Path


def test_business_logic_modules_are_in_pyright_strict_include() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    include = set(pyproject["tool"]["pyright"]["include"])

    assert {
        "src/regime_detection/engine.py",
        "src/regime_detection/models.py",
        "src/regime_detection/legacy_v1_wire.py",
        "src/regime_detection/model_status.py",
        "src/regime_detection/evidence_payloads.py",
        "src/regime_detection/classification_status.py",
        "src/regime_detection/axis_output_models.py",
        "src/regime_detection/strategy_models.py",
        "src/regime_detection/coverage_models.py",
        "src/regime_detection/wire_models.py",
        "src/regime_detection/axis_series.py",
        "src/regime_detection/feature_store.py",
        "src/regime_detection/timeline.py",
        "src/regime_detection/classification_coverage.py",
        "src/regime_detection/rule_provenance.py",
    }.issubset(include)


def test_pyright_strict_include_has_no_file_level_suppressions() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    include = pyproject["tool"]["pyright"]["include"]

    suppressed_paths = []
    for included_path in include:
        lines = Path(included_path).read_text().splitlines()
        if any(line.strip().startswith("# pyright:") for line in lines[:10]):
            suppressed_paths.append(included_path)

    assert suppressed_paths == []
