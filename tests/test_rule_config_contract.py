from __future__ import annotations

import re
from pathlib import Path


def test_rule_modules_do_not_hide_missing_config_fields_with_getattr() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src" / "regime_detection"
    offenders: list[str] = []
    for path in src_root.glob("*_rules.py"):
        text = path.read_text()
        for match in re.finditer(r"getattr\s*\(\s*config\b", text):
            lineno = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(src_root)}:{lineno}")
    offenders.sort()
    assert offenders == []
