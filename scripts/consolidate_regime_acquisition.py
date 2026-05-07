#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from regime_data_fetch.acquisition_consolidation import consolidate_acquisition_dbs


def main() -> int:
    target_db = REPO_ROOT / "data" / "raw" / "acquisition" / "acquisition.db"
    report = consolidate_acquisition_dbs(target_db_path=target_db)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
