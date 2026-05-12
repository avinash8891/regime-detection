"""v2 §1D Layer 1 V2 Breadth features — evidence-only compute (Slice 2.3).

Scope-restricted slice: ships only the §1D feature that does NOT require a
point-in-time (PIT) constituent-membership data pipeline. Per v2 §10 absolute
rule ("when the spec is ambiguous or silent, stop and ask. Do not invent.")
and §1D lines 198–205 ("V2 PIT breadth must not silently fall back to biased
current constituents"), the PIT-dependent features are deferred until the
PIT membership ingestion slice lands.

Features shipped here:

- ``sector_breadth``  v2 §1D lines 228–229
  ``% of {XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY} with
  return_21d > 0``. Denominator is the spec-fixed 11 (count of US GICS
  sector ETFs).

Deferred features (require PIT constituent universe — Implementation
Ambiguity Log entries #21–#26):

- ``pct_above_200dma`` (§1D lines 207–210) — `mean(member.close > member.sma_200)`.
- ``ad_line`` / ``ad_line_slope_20d`` (§1D lines 213–216).
- ``nh_nl_ratio`` (§1D lines 218–221).
- ``upvol_downvol_ratio`` (§1D lines 223–226).
- ``breadth_thrust`` (§1D lines 231–237).
- All new V2 breadth labels (§1D lines 239–246).

Implementation choices that resolve ambiguities:

- **Sector universe constant**: reuses ``SECTOR_ETFS`` from
  ``regime_detection.fragility_universe`` (single source of truth — V2 §3.1
  already pins this same set). No re-listed strings.
- **return_21d window**: ``close[t] / close[t - lookback] - 1`` (point-to-point
  total return over the lookback). NaN until ``t >= lookback``.
- **Strict ``> 0`` rule**: a sector with exactly ``return_21d == 0`` is NOT
  counted in the numerator (spec text "with return_21d > 0").
- **Missing-sector policy**: if any of the 11 sector closes are absent from
  ``sector_etf_closes``, every output session is NaN (do NOT fall back to a
  partial-denominator mean). Rationale: §1D line 229 explicitly writes
  "divided by 11"; silently rebasing the denominator would change the
  feature's semantics. See Implementation Ambiguity Log entry #27.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_detection.config import BreadthV2Config
from regime_detection.fragility_universe import SECTOR_ETFS


@dataclass(frozen=True)
class BreadthV2Features:
    """v2 §1D — per-session continuous breadth features (slice 2.3)."""

    sector_breadth: pd.Series

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("sector_breadth",)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: getattr(self, name) for name in self.feature_names}
        )


def compute_breadth_v2_features(
    *,
    sector_etf_closes: dict[str, pd.Series],
    config: BreadthV2Config,
) -> BreadthV2Features:
    """Compute the v2 §1D sector_breadth feature from sector ETF closes.

    Parameters
    ----------
    sector_etf_closes
        Mapping ``symbol -> close series`` for the 11 US GICS sector ETFs
        (``SECTOR_ETFS``). All series must share a common DatetimeIndex.
    config
        ``BreadthV2Config`` instance — supplies ``sector_breadth_lookback_days``
        (no magic numbers in the function body).

    Returns
    -------
    BreadthV2Features
        Frozen dataclass with ``sector_breadth: pd.Series`` aligned to the
        first present sector's index.
    """
    lookback = config.sector_breadth_lookback_days
    sector_universe = SECTOR_ETFS
    expected_universe_size = len(sector_universe)

    # Resolve a reference index from the first present sector. If NO sectors
    # are present we cannot return a typed series at all — callers must guard
    # via feature_store. Treat as "fail loud" here.
    present_symbols = [s for s in sector_universe if s in sector_etf_closes]
    if not present_symbols:
        raise ValueError(
            "sector_etf_closes must contain at least one of the 11 SECTOR_ETFS "
            f"(got: {sorted(sector_etf_closes.keys())})."
        )
    reference_index = sector_etf_closes[present_symbols[0]].index

    # Missing-sector policy (Ambiguity Log entry #27): if any of the 11
    # sectors is absent, the entire output series is NaN. Do NOT rebase the
    # denominator to the present subset.
    missing = [s for s in sector_universe if s not in sector_etf_closes]
    if missing:
        nan_series = pd.Series(
            np.full(len(reference_index), np.nan),
            index=reference_index,
            name="sector_breadth",
        )
        return BreadthV2Features(sector_breadth=nan_series)

    # Build a (n_sessions x 11) frame of returns_lookback_days for each sector.
    returns_frame = pd.DataFrame(
        {
            symbol: sector_etf_closes[symbol].astype(float).pct_change(
                periods=lookback
            )
            for symbol in sector_universe
        },
        index=reference_index,
    )

    # Strictly > 0 (spec line 229: "return_21d > 0").
    positive_mask = returns_frame > 0.0
    # Count positives per row, divide by the spec-fixed denominator (11).
    sector_breadth = (
        positive_mask.sum(axis=1).astype(float) / float(expected_universe_size)
    )
    # NaN cold-start: any NaN return in the row (t < lookback or missing
    # data) must propagate to NaN, not be silently treated as "not positive".
    has_any_nan = returns_frame.isna().any(axis=1)
    sector_breadth = sector_breadth.where(~has_any_nan)
    sector_breadth.name = "sector_breadth"

    return BreadthV2Features(sector_breadth=sector_breadth)
