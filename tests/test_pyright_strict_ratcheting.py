from __future__ import annotations

import tomllib
from pathlib import Path


def test_business_logic_modules_are_in_pyright_strict_include() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    include = set(pyproject["tool"]["pyright"]["include"])

    assert {
        "src/regime_detection/engine.py",
        "src/regime_detection/models.py",
        "src/regime_detection/axis_series.py",
        "src/regime_detection/feature_store.py",
        "src/regime_detection/timeline.py",
        "src/regime_detection/classification_coverage.py",
        "src/regime_detection/rule_provenance.py",
    }.issubset(include)
