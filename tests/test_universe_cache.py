from __future__ import annotations

import json
from pathlib import Path

from regime_data_fetch.universe import load_symbols_from_us_universe_cache


def test_load_symbols_from_us_universe_cache(tmp_path: Path) -> None:
    cache = tmp_path / "us_universe_cache.json"
    payload = {
        "updated_at": "2026-02-25T12:00:00+00:00",
        "stock_count": 3,
        "min_market_cap_B": 10.0,
        "stocks": {
            "AAPL": ["NASDAQ", "Apple", "Technology", 3500.0],
            "MSFT": ["NASDAQ", "Microsoft", "Technology", 3000.0],
            "BRK.B": ["NYSE", "Berkshire", "Financials", 900.0],
        },
    }
    cache.write_text(json.dumps(payload))
    syms = load_symbols_from_us_universe_cache(cache)
    assert syms == ["AAPL", "BRK.B", "MSFT"]
