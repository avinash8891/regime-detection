from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS_PATH = ROOT / "AGENTS.md"
CI_PATH = ROOT / ".github" / "workflows" / "ci.yml"

CANONICAL_COMMANDS = (
    "python -m black --check src tests scripts",
    "python -m ruff check .",
    "python -m pyright",
    'python -m pytest -q -m "" -n 8 --cov=src --cov-report=term-missing --cov-fail-under=80',
)


def _require_contains(text: str, needle: str, *, source: str) -> None:
    if needle not in text:
        raise SystemExit(f"{source} is missing required command: {needle}")


def main() -> int:
    agents_text = AGENTS_PATH.read_text(encoding="utf-8")
    ci_text = CI_PATH.read_text(encoding="utf-8")

    for command in CANONICAL_COMMANDS:
        _require_contains(agents_text, command, source="AGENTS.md")
        _require_contains(ci_text, command, source=".github/workflows/ci.yml")

    print("AGENTS.md command references match CI automation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
