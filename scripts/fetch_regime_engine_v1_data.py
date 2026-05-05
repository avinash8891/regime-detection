#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd

import sys

# Allow running as a script without requiring PYTHONPATH/installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from regime_data_fetch.universe import build_or_load_us_universe_10b_cache
from regime_data_fetch.alpaca_daily import fetch_daily_bars_alpaca, verify_min_start_date


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)

def _load_env_file(path: Path) -> None:
    """Minimal dotenv loader: KEY=VALUE lines, supports optional quotes.

    Does not overwrite existing env vars.
    """
    if not path.exists():
        raise SystemExit(f"env file not found: {path}")
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if not k:
            continue
        if os.environ.get(k, "").strip():
            continue
        os.environ[k] = v


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch regime-engine-v1 raw market data (daily bars).")
    ap.add_argument("--market-data-hub-root", required=True, help="Path to the market-data-hub repo.")
    ap.add_argument("--out-dir", default="data/raw", help="Output directory for Parquet + reports.")
    ap.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD).")
    ap.add_argument("--end", default=dt.date.today().isoformat(), help="End date (YYYY-MM-DD).")
    ap.add_argument("--min-cap-b", type=float, default=10.0, help="Universe filter threshold in $B.")
    ap.add_argument("--adjustment", default="raw", help="Alpaca adjustment: raw|split|dividend|all.")
    ap.add_argument("--list-symbols", action="store_true", help="Only print symbol counts and exit.")
    ap.add_argument("--build-universe", action="store_true", help="Force-refresh the 10B+ universe cache (network: yfinance).")
    ap.add_argument("--env-file", default=None, help="Optional .env file to load (for Alpaca creds).")
    ap.add_argument(
        "--universe-json",
        default=None,
        help="Optional path to a JSON list[str] of symbols to fetch (use this for the 762-symbol final universe).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    # ---- Universe (stocks only) ----
    uni_dir = out_dir / "universe"
    if args.universe_json:
        universe_path = Path(args.universe_json)
        stocks = json.loads(universe_path.read_text())
        if not isinstance(stocks, list) or not all(isinstance(s, str) for s in stocks):
            raise SystemExit("--universe-json must be a JSON list[str]")
        uni_cache_path = str(universe_path)
    else:
        uni = build_or_load_us_universe_10b_cache(
            market_data_hub_root=args.market_data_hub_root,
            out_dir=uni_dir,
            min_cap_b=args.min_cap_b,
            allow_update=args.build_universe or (not args.list_symbols),
        )
        stocks = uni.symbols
        uni_cache_path = str(uni.cache_path)

    anchors = ["SPY", "RSP"]
    vix_candidates = ["VIX", "^VIX"]

    all_symbols = stocks + anchors + vix_candidates

    if args.list_symbols:
        print(json.dumps(
            {
                "stocks_count": len(stocks),
                "note": (
                    "If this is not 762, run again with --build-universe to refresh the cache "
                    "(uses yfinance once, may rate-limit) OR pass --universe-json pointing to your 762 list."
                ),
                "anchors": anchors,
                "vix_candidates": vix_candidates,
                "universe_source": uni_cache_path,
            },
            indent=2,
            default=str,
        ))
        return 0

    # ---- Fetch daily bars from Alpaca ----
    # NOTE: This requires env vars to be set. We do not log secrets.
    if args.env_file:
        _load_env_file(Path(args.env_file))

    for k in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"):
        if not os.environ.get(k, "").strip():
            raise SystemExit(f"Missing required env var: {k}")

    res = fetch_daily_bars_alpaca(
        symbols=all_symbols,
        start_date=start,
        end_date=end,
        adjustment=args.adjustment,
    )
    df = res.df

    # ---- VIX requirement: Alpaca only ----
    have_vix = any((df["symbol"] == s).any() for s in vix_candidates)
    if not have_vix:
        raise SystemExit(
            "VIX not returned by Alpaca for symbols {VIX,^VIX}. "
            "If Alpaca does not support the index on your feed/account, "
            "we need a substitute decision (e.g. VIXY or alternate source)."
        )

    # Prefer VIX over ^VIX if both exist; otherwise use whichever exists.
    vix_symbol = "VIX" if (df["symbol"] == "VIX").any() else "^VIX"

    # ---- Verify earliest date vs requirement ----
    checks = {}
    for sym in ["SPY", "RSP", vix_symbol]:
        min_date, ok = verify_min_start_date(df, symbol=sym, required_start=start)
        checks[sym] = {"min_date": str(min_date) if min_date else None, "ok": ok}

    # ---- Write Parquet dataset ----
    # Partitioning keeps each symbol separate and makes later reads cheaper.
    parquet_dir = out_dir / "daily_ohlcv"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_dir, index=False, partition_cols=["symbol"])

    report = {
        "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "requested": {
            "start": str(start),
            "end": str(end),
            "adjustment": args.adjustment,
            "universe_min_cap_b": args.min_cap_b,
        },
        "counts": {
            "rows": int(len(df)),
            "symbols_requested": int(len(all_symbols)),
            "symbols_returned": int(df["symbol"].nunique()) if not df.empty else 0,
            "missing_symbols": int(len(res.missing_symbols)),
        },
        "vix_symbol_used": vix_symbol,
        "min_date_checks": checks,
        "missing_symbols_sample": res.missing_symbols[:50],
        "paths": {
            "universe_source": uni_cache_path,
            "parquet_dir": str(parquet_dir),
        },
    }
    (out_dir / "fetch_report.json").write_text(json.dumps(report, indent=2))

    # Print a small summary (safe to show in logs).
    print(json.dumps(report["counts"], indent=2))
    print(json.dumps(report["min_date_checks"], indent=2))
    print(str(out_dir / "fetch_report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
