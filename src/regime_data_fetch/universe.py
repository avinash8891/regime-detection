from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import time


@dataclass(frozen=True)
class UniverseBuildResult:
    cache_path: Path
    symbols: list[str]


def load_symbols_from_us_universe_cache(cache_path: str | os.PathLike[str]) -> list[str]:
    """Load canonical stock symbols from a market-data-hub us_universe cache JSON.

    Expected shape (see market-data-hub/universes/us_universe.py):
        {
          "updated_at": "...",
          "stock_count": 762,
          "stocks": { "AAPL": ["NASDAQ", "Apple Inc", "Technology", 3500.0], ... }
        }
    """
    p = Path(cache_path)
    raw = json.loads(p.read_text())
    stocks = raw.get("stocks", {})
    syms = sorted(stocks.keys())
    return syms


def build_or_load_us_universe_10b_cache(
    *,
    market_data_hub_root: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    min_cap_b: float = 10.0,
    allow_update: bool = True,
) -> UniverseBuildResult:
    """Ensure the market-data-hub US universe cache exists locally, then load symbols.

    This uses market-data-hub's own `universes/us_universe.py` update logic, but
    forces its LOG_DIR into our repo-local `out_dir` so we don't depend on
    /var/log/trading permissions or remote state.
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: market-data-hub's us_universe.py reads LOG_DIR at import time.
    old_log_dir = os.environ.get("LOG_DIR")
    os.environ["LOG_DIR"] = str(out_dir_p)

    import sys

    hub_root = str(Path(market_data_hub_root))
    if hub_root not in sys.path:
        sys.path.insert(0, hub_root)

    try:
        from universes import us_universe  # type: ignore
    finally:
        # Avoid leaking env mutations outside this helper.
        if old_log_dir is None:
            os.environ.pop("LOG_DIR", None)
        else:
            os.environ["LOG_DIR"] = old_log_dir

    cache_path = Path(os.environ["LOG_DIR"]) / "us_universe_cache.json"
    if not cache_path.exists():
        if allow_update:
            _refresh_us_universe_cache_from_yfinance_seeds(
                cache_path=cache_path,
                seeds=us_universe._all_seeds(),  # type: ignore[attr-defined]
                min_cap_b=min_cap_b,
            )
        else:
            # Return the static fallback list (typically smaller) without
            # forcing a network universe refresh.
            return UniverseBuildResult(cache_path=cache_path, symbols=list(us_universe.US_LIST))

    symbols = load_symbols_from_us_universe_cache(cache_path)
    return UniverseBuildResult(cache_path=cache_path, symbols=symbols)


def _refresh_us_universe_cache_from_yfinance_seeds(
    *,
    cache_path: Path,
    seeds: list[str],
    min_cap_b: float,
    max_workers: int = 16,
) -> None:
    """Build a minimal us_universe cache JSON by fetching market caps via yfinance.

    This is intentionally faster than market-data-hub's reference updater, and
    only needs to be good enough to reproduce the 10B+ symbol set for data fetch.
    """
    import yfinance as yf

    min_cap = float(min_cap_b) * 1e9

    def fetch_cap(sym: str) -> tuple[str, float | None]:
        try:
            t = yf.Ticker(sym)
            fi = getattr(t, "fast_info", None)
            cap = None
            if fi is not None:
                cap = getattr(fi, "market_cap", None)
            if cap is None:
                info = t.info
                cap = info.get("marketCap")
            if cap is None:
                return sym, None
            return sym, float(cap)
        except Exception:
            return sym, None

    # Threaded to reduce wall time; yfinance is IO-bound.
    qualified: dict[str, float] = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_cap, s): s for s in seeds}
        for fut in as_completed(futs):
            sym, cap = fut.result()
            if cap is not None and cap >= min_cap:
                qualified[sym] = round(cap / 1e9, 1)

    stocks: dict[str, list] = {}
    for sym, cap_b in qualified.items():
        stocks[sym] = ["UNKNOWN", sym, "Unknown", cap_b]

    payload = {
        "updated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "stock_count": len(stocks),
        "min_market_cap_B": float(min_cap_b),
        "stocks": stocks,
        "build_seconds": round(time.time() - started, 3),
        "source": "yfinance_parallel_minimal",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(cache_path)
