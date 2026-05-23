"""Network fragility universe constants (v2 spec §3.1).

The 24-asset universe is the union of:
  - 11 GICS sector ETFs (XLB-XLY)
  - 1 broad-market index (SPY)
  - 12 cross-asset proxies (QQQ, IWM, EFA, EEM, TLT, IEF, GLD, HYG, LQD,
    USO, DBC, UUP)

SPY is split out as `INDEX_SYMBOL` because the engine reads its close from
`MarketContext.spy_ohlcv` (v1 path) — not from the v2 `cross_asset_closes`
dict. The remaining 12 cross-asset names ship through `cross_asset_closes`.

Kept here, not in regime_data_fetch, because the engine must not depend on
the fetcher.
"""
from __future__ import annotations

from typing import Final


# V2 §3.1 lines 3406-3417 — 11 GICS sector ETFs.
SECTOR_ETFS: Final[tuple[str, ...]] = (
    "XLB",   # Materials
    "XLC",   # Communications
    "XLE",   # Energy
    "XLF",   # Financials
    "XLI",   # Industrials
    "XLK",   # Technology
    "XLP",   # Consumer Staples
    "XLRE",  # Real Estate
    "XLU",   # Utilities
    "XLV",   # Healthcare
    "XLY",   # Consumer Discretionary
)

# V2 §3.1 line 3419 — broad-market index, broken out because its OHLCV
# flows through MarketContext.spy_ohlcv on the V1 path.
INDEX_SYMBOL: Final[str] = "SPY"

# V2 §3.1 lines 3420-3431 — 12 cross-asset proxies (13 cross_asset_etfs in the
# spec yaml minus SPY which lives in INDEX_SYMBOL).
CROSS_ASSET_SYMBOLS: Final[tuple[str, ...]] = (
    "QQQ",   # Tech-heavy
    "IWM",   # Small cap
    "EFA",   # Developed ex-US
    "EEM",   # Emerging markets
    "TLT",   # Long Treasuries
    "IEF",   # Intermediate Treasuries
    "GLD",   # Gold
    "HYG",   # High yield bonds
    "LQD",   # Investment grade bonds
    "USO",   # Oil
    "DBC",   # Broad commodities
    "UUP",   # Dollar
)

# Full 24-asset universe per v2 §3.1 line 3434 ("24 assets total. Above the
# 20-asset preferred floor.").
NETWORK_FRAGILITY_UNIVERSE: Final[tuple[str, ...]] = (
    *SECTOR_ETFS,
    INDEX_SYMBOL,
    *CROSS_ASSET_SYMBOLS,
)

assert len(NETWORK_FRAGILITY_UNIVERSE) == 24, (
    "Network fragility universe must be 24 symbols per v2 spec §3.1."
)
