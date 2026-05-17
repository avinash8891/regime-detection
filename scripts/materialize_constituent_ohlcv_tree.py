from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from regime_data_fetch.constituent_ohlcv_tree import (  # noqa: E402
    materialize_constituent_ohlcv_tree,
)
from regime_data_fetch.universe import FIXED_UNIVERSE_TREE_NAME  # noqa: E402


def _parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD date, got {value!r}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a canonical constituent OHLCV tree for a PIT overlap window."
    )
    parser.add_argument(
        "--source-tree",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "daily_ohlcv",
        help="Source OHLCV tree, accepting either symbol=X/ohlcv.parquet or partition files.",
    )
    parser.add_argument(
        "--out-tree",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / FIXED_UNIVERSE_TREE_NAME,
        help=f"Canonical output tree. Defaults to data/raw/{FIXED_UNIVERSE_TREE_NAME}.",
    )
    parser.add_argument(
        "--pit-parquet",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "raw"
        / "pit_constituents"
        / "sp500_ticker_intervals.parquet",
        help="PIT constituent interval parquet.",
    )
    parser.add_argument("--start", required=True, type=_parse_date)
    parser.add_argument("--end", required=True, type=_parse_date)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument(
        "--allow-missing-symbols",
        action="store_true",
        help="Write available symbols and record missing symbols instead of failing.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = materialize_constituent_ohlcv_tree(
        source_tree=args.source_tree,
        output_tree=args.out_tree,
        pit_parquet_path=args.pit_parquet,
        start_date=args.start,
        end_date=args.end,
        report_path=args.report_path,
        allow_missing_symbols=args.allow_missing_symbols,
    )
    print(f"constituent_tree={result.output_tree}")
    print(f"requested_symbols={result.requested_symbols}")
    print(f"written_symbols={result.written_symbols}")
    print(f"missing_symbols={len(result.missing_symbols)}")
    print(f"manifest={result.manifest_path}")
    print(f"aggregate_sha256={result.aggregate_sha256}")
    print(f"report={result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
