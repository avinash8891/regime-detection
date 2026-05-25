#!/usr/bin/env python3
"""Weekly updater for the AAII Investor Sentiment Survey parquet.

On first run (no existing parquet): seeds from data/raw/sentiment/aaii_sentiment_historical.cfb.
On subsequent runs: scrapes the AAII HTML table for new rows and appends them.

Usage:
    python scripts/fetch_aaii_sentiment.py
    python scripts/fetch_aaii_sentiment.py --out-dir data/raw
    python scripts/fetch_aaii_sentiment.py --url https://www.aaii.com/sentimentsurvey/sent_results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from regime_data_fetch.aaii_sentiment import AAII_SENTIMENT_URL, run_sentiment_fetch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch/update AAII sentiment survey data.")
    ap.add_argument(
        "--out-dir",
        default="data/raw",
        help="Root raw data directory (default: data/raw).",
    )
    ap.add_argument(
        "--url", default=AAII_SENTIMENT_URL, help="AAII sentiment page URL."
    )
    ap.add_argument(
        "--acquisition-db",
        default=None,
        help="Optional SQLite path for acquisition/provenance recording.",
    )
    ap.add_argument(
        "--artifact-store",
        default=None,
        help="Optional artifact-store root for acquisition artifacts.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    report_path = run_sentiment_fetch(
        out_dir=out_dir,
        url=args.url,
        acquisition_db_path=Path(args.acquisition_db) if args.acquisition_db else None,
        artifact_store_root=(
            args.artifact_store if args.acquisition_db and args.artifact_store else None
        ),
    )

    report = json.loads(report_path.read_text())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
