from __future__ import annotations

import json
import os
from pathlib import Path


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
