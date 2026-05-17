"""v2 §1D Layer 1 V2 Breadth features.

Computes sector breadth and, when PIT constituent intervals plus constituent
OHLCV are supplied, PIT-derived breadth features. The current free PIT source
is explicitly marked with a survivorship-bias warning; callers surface this as
``pit_constituent_biased_research`` rather than pretending it is a true vendor
PIT feed.

Features shipped here:

- ``sector_breadth``  v2 §1D lines 228–229
  ``% of {XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY} with
  return_21d > 0``. Denominator is the spec-fixed 11 (count of US GICS
  sector ETFs).

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

from bisect import bisect_left, bisect_right
import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_detection.config import BreadthV2Config
from regime_detection.fragility_universe import SECTOR_ETFS


# PIT feature names in spec order (v2 §1D lines 207-237).
_PIT_FEATURE_NAMES: tuple[str, ...] = (
    "pct_above_50dma",
    "pct_above_200dma",
    "ad_line",
    "ad_line_slope_20d",
    "nh_nl_ratio",
    "upvol_downvol_ratio",
    "breadth_thrust",
)

# Bias-warning provenance: local duplicates avoid an acquisition-layer
# dependency while preserving the published PIT provenance contract exactly.
_PIT_BIAS_WARNING_CODE = "survivorship_biased_constituent_universe"
_PIT_BIAS_SOURCE = "fja05680/sp500"
_PIT_BIAS_SOURCE_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"

# breadth_thrust 10-session rolling mean window (v2 §1D line 231-237).
_BREADTH_THRUST_WINDOW = 10

# ad_line_slope_20d lookback (v2 §1D line 216).
_AD_LINE_SLOPE_LOOKBACK = 20


@dataclass(frozen=True)
class BreadthV2Features:
    """v2 §1D — per-session continuous breadth features.

    ``sector_breadth`` is always present. Optional PIT fields are ``None`` on
    the v1+v2-sector-only callsite and
    materialised when both ``pit_constituent_intervals`` and
    ``constituent_ohlcv`` are threaded through ``compute_breadth_v2_features``.
    """

    sector_breadth: pd.Series
    bias_warnings: pd.DataFrame | None = None
    pct_above_50dma: pd.Series | None = None
    pct_above_200dma: pd.Series | None = None
    ad_line: pd.Series | None = None
    ad_line_slope_20d: pd.Series | None = None
    nh_nl_ratio: pd.Series | None = None
    upvol_downvol_ratio: pd.Series | None = None
    breadth_thrust: pd.Series | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = ["sector_breadth"]
        for pit_name in _PIT_FEATURE_NAMES:
            if getattr(self, pit_name) is not None:
                names.append(pit_name)
        return tuple(names)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {name: getattr(self, name) for name in self.feature_names}
        )


_BIAS_WARNING_COLUMNS = ("warning_code", "feature_name", "source", "source_url")


def make_bias_warnings_frame(rows: Iterable[Mapping[str, str]]) -> pd.DataFrame:
    """Build the canonical 4-column bias-warnings frame.

    Each row must have exactly the keys: warning_code, feature_name,
    source, source_url. Raises ValueError on key mismatch (missing or extra).
    Empty input returns a 4-column DataFrame with 0 rows.
    """
    expected = set(_BIAS_WARNING_COLUMNS)
    materialized = []
    for idx, row in enumerate(rows):
        got = set(row)
        if got != expected:
            missing = expected - got
            extra = got - expected
            raise ValueError(
                f"bias_warnings row {idx} key mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        materialized.append({k: row[k] for k in _BIAS_WARNING_COLUMNS})
    if not materialized:
        return pd.DataFrame({col: pd.Series(dtype=object) for col in _BIAS_WARNING_COLUMNS})
    return pd.DataFrame(materialized, columns=list(_BIAS_WARNING_COLUMNS))


def compute_breadth_v2_features(
    *,
    sector_etf_closes: dict[str, pd.Series],
    config: BreadthV2Config,
    pit_constituent_intervals: pd.DataFrame | None = None,
    constituent_ohlcv: dict[str, pd.DataFrame] | None = None,
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

    # PIT-aware §1D features (Slice 2.8c). Both new kwargs must be supplied
    # for the seven survivorship-biased features to materialise; otherwise
    # the v1+v2-sector-only callsite is preserved (all-None PIT fields).
    if pit_constituent_intervals is None or constituent_ohlcv is None:
        return BreadthV2Features(sector_breadth=sector_breadth)

    pit_features = _compute_pit_features(
        reference_index=reference_index,
        pit_constituent_intervals=pit_constituent_intervals,
        constituent_ohlcv=constituent_ohlcv,
        config=config,
    )
    bias_warnings = make_bias_warnings_frame(
        [
            {
                "warning_code": _PIT_BIAS_WARNING_CODE,
                "feature_name": feat,
                "source": _PIT_BIAS_SOURCE,
                "source_url": _PIT_BIAS_SOURCE_URL,
            }
            for feat in _PIT_FEATURE_NAMES
        ]
    )
    return BreadthV2Features(
        sector_breadth=sector_breadth,
        bias_warnings=bias_warnings,
        **pit_features,
    )


# ---------------------------------------------------------------------------
# PIT feature helpers (Slice 2.8c). Each takes a (sessions × members)
# adjusted_close DataFrame plus the precomputed members_by_session mapping
# and returns a per-session pd.Series aligned to ``reference_index``.
# ---------------------------------------------------------------------------


def _normalize_interval_dates(intervals: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``start_date``/``end_date`` columns to ``datetime.date`` objects.

    The on-disk PIT parquet stores ISO date strings; ``read_pit_intervals``
    converts them to ``dt.date``. Test fixtures construct the frame directly
    with ISO strings, so we normalize defensively before calling
    ``members_on`` (which compares to ``dt.date``).
    """
    def _to_date(value: object) -> dt.date | None:
        if value is None:
            return None
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, dt.date):
            return value
        return dt.date.fromisoformat(str(value))

    out = intervals.copy()
    out["start_date"] = out["start_date"].map(_to_date)
    out["end_date"] = out["end_date"].map(_to_date)
    return out


def _compute_pit_features(
    *,
    reference_index: pd.DatetimeIndex,
    pit_constituent_intervals: pd.DataFrame,
    constituent_ohlcv: dict[str, pd.DataFrame],
    config: BreadthV2Config,
) -> dict[str, pd.Series]:
    intervals = _normalize_interval_dates(pit_constituent_intervals)
    all_member_tickers = sorted(set(intervals["ticker"].tolist()))
    session_days = list(reference_index.date)
    ticker_to_col = {ticker: idx for idx, ticker in enumerate(all_member_tickers)}

    # Build the (sessions × members) adjusted_close and volume frames by
    # filling a preallocated dense matrix. This preserves the exact
    # ticker-missing => all-NaN semantics without paying DataFrame-of-Series
    # alignment costs for every member on every run.
    adj_close_frame = _build_member_frame(
        reference_index=reference_index,
        all_member_tickers=all_member_tickers,
        constituent_ohlcv=constituent_ohlcv,
        column_name="adjusted_close",
    )
    volume_frame = _build_member_frame(
        reference_index=reference_index,
        all_member_tickers=all_member_tickers,
        constituent_ohlcv=constituent_ohlcv,
        column_name="volume",
    )

    # PIT membership mask: derive directly from the interval rows. This keeps
    # the exact members_on semantics but avoids per-session set construction.
    membership_array = np.zeros(
        (len(reference_index), len(all_member_tickers)),
        dtype=bool,
    )
    for row in intervals.itertuples(index=False):
        ticker = str(row.ticker)
        col_idx = ticker_to_col.get(ticker)
        if col_idx is None:
            continue
        start_pos = bisect_left(session_days, row.start_date)
        if start_pos >= len(session_days):
            continue
        if row.end_date is None:
            end_pos_exclusive = len(session_days)
        else:
            end_pos_exclusive = bisect_right(session_days, row.end_date)
        if start_pos >= end_pos_exclusive:
            continue
        membership_array[start_pos:end_pos_exclusive, col_idx] = True
    membership_mask = pd.DataFrame(
        membership_array,
        index=reference_index,
        columns=all_member_tickers,
        dtype=bool,
    )

    # Per-ticker daily direction sign on adjusted_close (Ambiguity Log #56):
    # +1 advance, -1 decline, 0 unchanged. Diff produces NaN at t=0 → mapped
    # to NaN (no prior close → excluded from advance / decline counts).
    diff_frame = adj_close_frame.diff()
    advance_mask = diff_frame > 0.0
    decline_mask = diff_frame < 0.0
    unchanged_mask = diff_frame == 0.0
    # Note: NaN values yield False on all three comparisons — exactly the
    # "no prior close → excluded" behavior required.

    pct_above_50dma = _compute_pct_above_sma(
        adj_close_frame=adj_close_frame,
        membership_mask=membership_mask,
        sma_window=config.sma_lookback_50,
    )
    pct_above_200dma = _compute_pct_above_sma(
        adj_close_frame=adj_close_frame,
        membership_mask=membership_mask,
        sma_window=config.sma_lookback_200,
    )
    ad_line, ad_line_slope_20d = _compute_ad_line(
        advance_mask=advance_mask,
        decline_mask=decline_mask,
        membership_mask=membership_mask,
    )
    nh_nl_ratio = _compute_nh_nl_ratio(
        adj_close_frame=adj_close_frame,
        membership_mask=membership_mask,
        window=config.nh_nl_lookback_sessions,
    )
    upvol_downvol_ratio = _compute_upvol_downvol_ratio(
        advance_mask=advance_mask,
        decline_mask=decline_mask,
        membership_mask=membership_mask,
        volume_frame=volume_frame,
    )
    breadth_thrust = _compute_breadth_thrust(
        advance_mask=advance_mask,
        decline_mask=decline_mask,
        unchanged_mask=unchanged_mask,
        membership_mask=membership_mask,
    )

    return {
        "pct_above_50dma": pct_above_50dma,
        "pct_above_200dma": pct_above_200dma,
        "ad_line": ad_line,
        "ad_line_slope_20d": ad_line_slope_20d,
        "nh_nl_ratio": nh_nl_ratio,
        "upvol_downvol_ratio": upvol_downvol_ratio,
        "breadth_thrust": breadth_thrust,
    }


def _build_member_frame(
    *,
    reference_index: pd.DatetimeIndex,
    all_member_tickers: list[str],
    constituent_ohlcv: dict[str, pd.DataFrame],
    column_name: str,
) -> pd.DataFrame:
    data = np.full(
        (len(reference_index), len(all_member_tickers)),
        np.nan,
        dtype=float,
    )
    for col_idx, ticker in enumerate(all_member_tickers):
        frame = constituent_ohlcv.get(ticker)
        if frame is None:
            continue
        aligned = frame[column_name].reindex(reference_index)
        data[:, col_idx] = aligned.to_numpy(dtype=float, na_value=np.nan)
    return pd.DataFrame(
        data,
        index=reference_index,
        columns=all_member_tickers,
        dtype=float,
    )


def _compute_pct_above_sma(
    *,
    adj_close_frame: pd.DataFrame,
    membership_mask: pd.DataFrame,
    sma_window: int,
) -> pd.Series:
    """v2 §1D pct_above_{N}dma — strict ``>`` against the per-ticker SMA.

    NaN-SMA tickers are excluded from BOTH numerator AND denominator
    (Ambiguity Log #58); zero-denominator → NaN.
    """
    sma_frame = adj_close_frame.rolling(sma_window, min_periods=sma_window).mean()
    # A ticker is "valid" at session D if it's a member AND has a defined SMA
    # AND has a defined close. (Defined close is implied by defined SMA but
    # we guard explicitly.)
    valid_mask = (
        membership_mask
        & sma_frame.notna()
        & adj_close_frame.notna()
    )
    above_mask = valid_mask & (adj_close_frame > sma_frame)
    above_count = above_mask.sum(axis=1).astype(float)
    valid_count = valid_mask.sum(axis=1).astype(float)
    ratio = above_count / valid_count.where(valid_count > 0)
    ratio.name = f"pct_above_{sma_window}dma"
    return ratio


def _compute_ad_line(
    *,
    advance_mask: pd.DataFrame,
    decline_mask: pd.DataFrame,
    membership_mask: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """v2 §1D ad_line / ad_line_slope_20d (Ambiguity Log #56, #57).

    ad_line[0] = 0 anchor; ad_line[t] = ad_line[t-1] + (advances - declines)
    where advances/declines count PIT members with strict direction.
    """
    advances = (advance_mask & membership_mask).sum(axis=1).astype(float)
    declines = (decline_mask & membership_mask).sum(axis=1).astype(float)
    delta = advances - declines
    # Anchor at 0 on the first session (Ambiguity Log #57).
    delta.iloc[0] = 0.0
    ad_line = delta.cumsum()
    ad_line.name = "ad_line"

    slope = (ad_line - ad_line.shift(_AD_LINE_SLOPE_LOOKBACK)) / float(
        _AD_LINE_SLOPE_LOOKBACK
    )
    slope.name = "ad_line_slope_20d"
    return ad_line, slope


def _compute_nh_nl_ratio(
    *,
    adj_close_frame: pd.DataFrame,
    membership_mask: pd.DataFrame,
    window: int,
) -> pd.Series:
    """v2 §1D nh_nl_ratio over a 252-session inclusive window.

    A ticker contributes only when it has at least ``window`` non-NaN
    sessions in the trailing inclusive window. ``ratio = nh / max(nh+nl, 1)``;
    when no member has sufficient history, the value is NaN.
    """
    rolling_max = adj_close_frame.rolling(window, min_periods=window).max()
    rolling_min = adj_close_frame.rolling(window, min_periods=window).min()

    sufficient = rolling_max.notna() & rolling_min.notna()
    valid_mask = membership_mask & sufficient
    new_high_mask = valid_mask & (adj_close_frame == rolling_max)
    new_low_mask = valid_mask & (adj_close_frame == rolling_min)

    new_highs = new_high_mask.sum(axis=1).astype(float)
    new_lows = new_low_mask.sum(axis=1).astype(float)
    valid_count = valid_mask.sum(axis=1).astype(float)

    denom = (new_highs + new_lows).where(lambda s: s > 0, other=1.0)
    ratio = new_highs / denom
    # Where no member has sufficient history, surface NaN (not 0/1).
    ratio = ratio.where(valid_count > 0)
    ratio.name = "nh_nl_ratio"
    return ratio


def _compute_upvol_downvol_ratio(
    *,
    advance_mask: pd.DataFrame,
    decline_mask: pd.DataFrame,
    membership_mask: pd.DataFrame,
    volume_frame: pd.DataFrame,
) -> pd.Series:
    """v2 §1D upvol_downvol_ratio (Ambiguity Log #56).

    Direction uses adjusted_close (strict ``>`` / ``<``); volume is RAW
    shares from constituent_ohlcv[ticker]['volume'].
    """
    advance_vol_mask = advance_mask & membership_mask
    decline_vol_mask = decline_mask & membership_mask
    upvol = volume_frame.where(advance_vol_mask, other=0.0).sum(axis=1)
    downvol = volume_frame.where(decline_vol_mask, other=0.0).sum(axis=1)
    ratio = upvol / downvol.where(downvol > 0, other=1.0)
    ratio.name = "upvol_downvol_ratio"
    return ratio


def _compute_breadth_thrust(
    *,
    advance_mask: pd.DataFrame,
    decline_mask: pd.DataFrame,
    unchanged_mask: pd.DataFrame,
    membership_mask: pd.DataFrame,
) -> pd.Series:
    """v2 §1D breadth_thrust — 10-session MA of pct_advancing.

    ``pct_advancing = advances / max(advances + declines + unchanged, 1)``
    where the denominator counts PIT members with a valid prior close
    (Ambiguity Log #56).
    """
    advances = (advance_mask & membership_mask).sum(axis=1).astype(float)
    declines = (decline_mask & membership_mask).sum(axis=1).astype(float)
    unchanged = (unchanged_mask & membership_mask).sum(axis=1).astype(float)
    valid = advances + declines + unchanged
    pct_advancing = advances / valid.where(valid > 0, other=1.0)
    thrust = pct_advancing.rolling(
        _BREADTH_THRUST_WINDOW, min_periods=_BREADTH_THRUST_WINDOW
    ).mean()
    thrust.name = "breadth_thrust"
    return thrust
