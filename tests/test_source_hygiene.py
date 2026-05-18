from __future__ import annotations

import re
from pathlib import Path


SOURCE_ROOTS = (Path("src/regime_detection"), Path("src/regime_data_fetch"))
SOURCE_SUFFIXES = {".py", ".yaml"}
FORBIDDEN_SOURCE_PATTERNS = re.compile(
    r"TODO\([^)]*ticket=|TD-|audit M[0-9]|spec_code_data_audit|"
    r"Ambiguity Log|Implementation Ambiguity Log|Slice[s]? [0-9]|slice-[0-9]|"
    r"documented implementation[^\n]*#|implementation phase\.[0-9]|"
    r"source-data audit[^\n]*M[0-9]|Log #[0-9]"
)


def test_source_comments_do_not_embed_task_or_audit_references() -> None:
    offenders: list[str] = []
    for root in SOURCE_ROOTS:
        for path in sorted(root.rglob("*")):
            if path.suffix not in SOURCE_SUFFIXES:
                continue
            for line_number, line in enumerate(path.read_text().splitlines(), start=1):
                if FORBIDDEN_SOURCE_PATTERNS.search(line):
                    offenders.append(f"{path}:{line_number}: {line.strip()}")

    assert offenders == []


def test_axis_builders_do_not_use_common_typing_escape_hatch() -> None:
    assert not Path("src/regime_detection/axis_builders/common.py").exists()
