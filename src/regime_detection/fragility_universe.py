"""Network fragility universe constants (v2 spec §3.1).

The 22-ETF universe spans US sector ETFs, cross-asset proxies, and
regional banks (KRE). Kept here, not in regime_data_fetch, because the
engine must not depend on the fetcher.
"""
from __future__ import annotations

from typing import Final

# V2 §3.1 — 11 GICS sector ETFs.
SECTOR_ETFS: Final[tuple[str, ...]] = (
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
)

# V2 §3.1 — 10 cross-asset proxies (equity index, intl equity, rates/credit, FX/commodities).
CROSS_ASSET_SYMBOLS: Final[tuple[str, ...]] = (
    "QQQ", "IWM", "EFA", "EEM", "TLT", "HYG", "LQD", "GLD", "USO", "UUP",
)

# V2 §3.1 — regional banks signal (Layer 2C credit/funding).
REGIONAL_BANKS_SYMBOL: Final[str] = "KRE"

NETWORK_FRAGILITY_UNIVERSE: Final[tuple[str, ...]] = (
    *SECTOR_ETFS,
    *CROSS_ASSET_SYMBOLS,
    REGIONAL_BANKS_SYMBOL,
)

assert len(NETWORK_FRAGILITY_UNIVERSE) == 22, (
    "Network fragility universe must be 22 symbols per v2 spec §3.1."
)
