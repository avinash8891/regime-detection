#!/usr/bin/env python3
"""Write the §10 reproducible shadow strategy-metrics report (F-014, ADR 0025).

Reads the shadow ledger under ``--output-root`` and writes
``reports/shadow_strategy_metrics.json`` with the six §10 success metrics plus the
no-regime baseline. Pure reducer over the ledger — deterministic on re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from regime_detection.shadow_strategy_metrics import (  # noqa: E402
    compute_shadow_strategy_metrics,
)


def build_shadow_metrics_report(output_root: Path) -> Path:
    metrics = compute_shadow_strategy_metrics(output_root)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "shadow_strategy_metrics.json"
    report_path.write_text(
        json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the §10 reproducible shadow strategy-metrics report."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report_path = build_shadow_metrics_report(args.output_root)
    print(json.dumps({"report_path": str(report_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
