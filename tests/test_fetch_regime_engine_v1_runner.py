from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from regime_data_fetch.pmi import DEFAULT_MANUAL_PMI_HISTORY_DIR


def test_fetch_runner_subprocess_loads_env_file_for_pmi_history_dir(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / "runner.env"
    env_file.write_text(f"REGIME_PMI_HISTORY_DIR={DEFAULT_MANUAL_PMI_HISTORY_DIR}\n")
    out_dir = tmp_path / "data" / "raw"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/fetch_regime_engine_v1_data.py",
            "--fetch",
            "pmi",
            "--env-file",
            str(env_file),
            "--out-dir",
            str(out_dir),
            "--end",
            "2026-05-07",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((out_dir / "pmi_fetch_report.json").read_text())
    assert report["selected_source"] == "manual_investing_history"
