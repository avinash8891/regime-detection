from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


def test_autoresearch_script_reports_full_multi_digit_test_counts(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "python3"
    fake_python.write_text(
        """#!/bin/sh
if [ "$1" = "-m" ] && [ "$2" = "py_compile" ]; then
  exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pytest" ]; then
  printf '%s\\n' 'collected 1338 items'
  printf '%s\\n' '1328 passed, 10 skipped in 12.34s'
  exit 0
fi
exit 2
""",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    result = subprocess.run(
        ["bash", "autoresearch.sh"],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
    )

    assert "METRIC collected_tests=1338" in result.stdout
    assert "METRIC passed_tests=1328" in result.stdout
    assert "METRIC skipped_tests=10" in result.stdout
