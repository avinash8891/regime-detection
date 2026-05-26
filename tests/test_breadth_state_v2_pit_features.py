"""TDD RED-phase tests for V2 Slice 2.8c — PIT-aware §1D breadth features.

These tests are deliberately written BEFORE production code lands. They will
fail (TypeError / AttributeError / ValidationError) until the GREEN-phase
subagent extends ``compute_breadth_v2_features`` to accept the new
``pit_constituent_intervals`` and ``constituent_ohlcv`` kwargs and surface the
seven new PIT features on ``BreadthV2Features``.

Spec pins exercised:
    Ambiguity Log #54 — PIT features read ``adjusted_close`` (NOT raw close).
    Ambiguity Log #55 — NH/NL uses 252 trailing NYSE sessions, inclusive.
    Ambiguity Log #56 — STRICT inequality for advance/decline; equality = neither.
    Ambiguity Log #57 — ``ad_line[0] = 0`` anchor at the first computation session.
    Ambiguity Log #58 — NaN-SMA / insufficient-history tickers are excluded from
                        BOTH numerator AND denominator; zero-denominator → NaN.
    Ambiguity Log #59 — PIT universe at D selects members; each member's
                        technical state uses its OWN full adjusted_close history.

Real S&P 500 tickers only (no toy names). Hand-computed expected values on
every assertion (no ``is not None`` placeholders).
"""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import pandas as pd
import pytest

from regime_detection.breadth_state_v2 import (
    BreadthV2Features,
    _normalize_interval_dates,
    compute_breadth_v2_features,
    make_bias_warnings_frame,  # noqa: F401  (asserted as importable in GREEN)
)
from regime_detection.config import BreadthV2Config
from regime_detection.fragility_universe import SECTOR_ETFS

# ---------------------------------------------------------------------------
# Constants — imported directly from pit_constituents to enforce that the
# breadth-state PIT bias warnings carry the same provenance as the parquet
# rows written by run_pit_constituents_fetch. Hardcoding string literals here
# would let the two homes drift silently (AGENTS rule B).
# ---------------------------------------------------------------------------

from regime_shared.pit_provenance import (  # noqa: E402
    BIAS_WARNING as _PIT_BIAS_WARNING,
    SOURCE_NAME as _PIT_SOURCE,
    SOURCE_URL as _PIT_SOURCE_URL,
)

_PIT_FEATURE_NAMES = (
    "pct_above_50dma",
    "pct_above_200dma",
    "ad_line",
    "ad_line_slope_20d",
    "nh_nl_ratio",
    "upvol_downvol_ratio",
    "breadth_thrust",
)
_AVAILABLE_SECTOR_FEATURE_NAMES = (
    "available_sector_breadth",
    "available_sector_count",
    "missing_sector_count",
    "missing_sector_symbols",
)

# Realistic S&P 500 tickers used across tests (no toy names).
AAPL = "AAPL"
MSFT = "MSFT"
NVDA = "NVDA"
GOOGL = "GOOGL"
AMZN = "AMZN"
IBM = "IBM"
JNJ = "JNJ"
KO = "KO"
XOM = "XOM"
PG = "PG"
JPM = "JPM"
V = "V"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bdate_index(n: int, start: str = "2023-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _make_ohlcv_frame(
    adjusted_close: list[float],
    volume: list[int],
    start: str = "2023-01-02",
) -> pd.DataFrame:
    """Build a per-ticker OHLCV frame matching the read_constituent_ohlcv shape.

    Index is a pandas business-day DatetimeIndex (UTC-naive) starting at
    ``start``. Columns are ``[open, high, low, close, volume, adjusted_close]``.
    Open/high/low/close are filler equal to adjusted_close; only ``volume`` and
    ``adjusted_close`` are consumed by the PIT compute under spec pins
    #54 and #56.
    """
    assert len(adjusted_close) == len(volume)
    index = _bdate_index(len(adjusted_close), start=start)
    adj = pd.Series(adjusted_close, index=index, dtype=float)
    vol = pd.Series(volume, index=index, dtype="int64")
    return pd.DataFrame(
        {
            "open": adj.copy(),
            "high": adj.copy(),
            "low": adj.copy(),
            "close": adj.copy(),
            "volume": vol,
            "adjusted_close": adj,
        }
    )


def _make_pit_intervals(rows: list[tuple[str, str, str | None]]) -> pd.DataFrame:
    """Build the parquet-shape PIT intervals DataFrame.

    Each input tuple is ``(ticker, start_iso, end_iso_or_None)``. The output
    columns match ``pit_constituents.py`` lines 121-138: ticker, start_date,
    end_date, source, source_url, bias_warning.
    """
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "start_date": start_iso,
                "end_date": end_iso,
                "source": _PIT_SOURCE,
                "source_url": _PIT_SOURCE_URL,
                "bias_warning": _PIT_BIAS_WARNING,
            }
            for (ticker, start_iso, end_iso) in rows
        ]
    )


def _make_sector_closes(n: int, start: str = "2023-01-02") -> dict[str, pd.Series]:
    """Build a non-pathological all-11 sector ETF close fixture.

    Each sector follows a slightly different monotone-rising path so that
    sector_breadth is well-defined past the lookback warmup. The PIT features
    are independent of this input, but ``compute_breadth_v2_features`` requires
    all 11 to be present for ``sector_breadth`` not to be all-NaN.
    """
    index = _bdate_index(n, start=start)
    out: dict[str, pd.Series] = {}
    for i, symbol in enumerate(SECTOR_ETFS):
        rate = 0.0005 + i * 0.00005
        arr = 100.0 * np.exp(np.arange(n) * rate)
        out[symbol] = pd.Series(arr, index=index, name=symbol)
    return out


@pytest.fixture
def v2_breadth_config() -> BreadthV2Config:
    """Default production config. ``nh_nl_lookback_sessions=252`` lands in 2.8c.

    If construction fails with a ValidationError that's the RED signal for the
    GREEN subagent to add the new field with default 252.
    """
    return BreadthV2Config(sector_breadth_lookback_days=21)


# =============================================================================
# Group A — PIT features absence (4 tests)
# =============================================================================


def _assert_all_pit_none(out: BreadthV2Features) -> None:
    assert out.pct_above_50dma is None
    assert out.pct_above_200dma is None
    assert out.ad_line is None
    assert out.ad_line_slope_20d is None
    assert out.nh_nl_ratio is None
    assert out.upvol_downvol_ratio is None
    assert out.breadth_thrust is None


def test_normalize_interval_dates_is_clean_under_copy_on_write_warning_mode() -> None:
    intervals = pd.DataFrame(
        {
            "ticker": [AAPL],
            "start_date": ["2020-01-01"],
            "end_date": [None],
        }
    )

    with pd.option_context("mode.copy_on_write", "warn"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            out = _normalize_interval_dates(intervals)

    assert out.loc[0, "start_date"] == dt.date(2020, 1, 1)
    assert out.loc[0, "end_date"] is None


def test_pit_features_none_when_pit_intervals_not_supplied(v2_breadth_config):
    closes = _make_sector_closes(n=60)
    out = compute_breadth_v2_features(
        sector_etf_closes=closes, config=v2_breadth_config
    )
    _assert_all_pit_none(out)
    # sector_breadth path unchanged — values exist past warmup.
    assert out.sector_breadth.dropna().shape[0] > 0


def test_pit_features_none_when_only_intervals_supplied(v2_breadth_config):
    closes = _make_sector_closes(n=60)
    intervals = _make_pit_intervals(
        [(AAPL, "2023-01-02", None), (MSFT, "2023-01-02", None)]
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=None,
    )
    _assert_all_pit_none(out)


def test_pit_features_none_when_only_ohlcv_supplied(v2_breadth_config):
    closes = _make_sector_closes(n=60)
    ohlcv = {
        AAPL: _make_ohlcv_frame([100.0] * 60, [1000] * 60),
        MSFT: _make_ohlcv_frame([100.0] * 60, [1000] * 60),
    }
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=None,
        constituent_ohlcv=ohlcv,
    )
    _assert_all_pit_none(out)


def test_pit_features_none_when_no_member_present_in_ohlcv(v2_breadth_config):
    """Both kwargs present but PIT intervals reference tickers NOT in ohlcv.

    Per Ambiguity Log #58, denominator collapses to zero on every session →
    feature value is NaN on every session. The series IS emitted (not None)
    because both kwargs are present.
    """
    n = 60
    closes = _make_sector_closes(n=n)
    intervals = _make_pit_intervals(
        [(GOOGL, "2023-01-02", None), (AMZN, "2023-01-02", None)]
    )
    # OHLCV contains only AAPL — none of the PIT members.
    ohlcv = {AAPL: _make_ohlcv_frame([100.0] * n, [1000] * n)}
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    # Series emitted, all NaN.
    assert out.pct_above_50dma is not None
    assert out.pct_above_50dma.isna().all()


# =============================================================================
# Group B — pct_above_50dma (3 tests)
# =============================================================================


def test_pct_above_50dma_two_tickers_one_above_one_below(v2_breadth_config):
    """AAPL monotonically rising, MSFT monotonically falling. At session 55,
    AAPL strictly above its 50d SMA and MSFT strictly below → 1/2 = 0.5.
    """
    n = 60
    closes = _make_sector_closes(n=n)
    rising = [100.0 + i for i in range(n)]  # 100, 101, ..., 159
    falling = [200.0 - i for i in range(n)]  # 200, 199, ..., 141
    ohlcv = {
        AAPL: _make_ohlcv_frame(rising, [1000] * n),
        MSFT: _make_ohlcv_frame(falling, [1000] * n),
    }
    intervals = _make_pit_intervals(
        [(AAPL, "2023-01-02", None), (MSFT, "2023-01-02", None)]
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    # Hand-computed at index 54 (session 55, 1-indexed): both have a defined
    # 50d SMA (sessions 5..54 inclusive), AAPL is above, MSFT below.
    assert out.pct_above_50dma.iloc[54] == pytest.approx(0.5)


def test_pct_above_50dma_uses_adjusted_close_not_raw_close(v2_breadth_config):
    """Ambiguity Log #54: only adjusted_close is consumed.

    adjusted_close is flat at 100 (no movement post-split-adjustment). Raw
    ``close`` has a synthetic split-day drop to 50, which an incorrect
    implementation reading raw close might use to compute a low SMA, then
    declare the post-split close "above" SMA. With the spec-correct
    adjusted_close path, the SMA equals 100 and the ticker is at-SMA, so the
    strict ``>`` comparison fails → numerator 0 → pct_above_50dma == 0.0.
    """
    n = 60
    closes = _make_sector_closes(n=n)
    adj = [100.0] * n
    frame = _make_ohlcv_frame(adj, [1000] * n)
    # Inject a split-shape divergence into raw close (NOT into adjusted_close).
    frame.loc[frame.index[30], "close"] = 50.0
    ohlcv = {AAPL: frame}
    intervals = _make_pit_intervals([(AAPL, "2023-01-02", None)])
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    # SMA-on-adjusted at session 55 = 100; adjusted_close == 100; strict > fails.
    assert out.pct_above_50dma.iloc[54] == pytest.approx(0.0)


def test_pct_above_50dma_newly_listed_ticker_excluded_from_both(
    v2_breadth_config,
):
    """Ambiguity Log #58: NaN-SMA tickers are dropped from BOTH
    numerator AND denominator. Zero-denominator → NaN.
    """
    n = 60
    closes = _make_sector_closes(n=n)
    aapl_adj = [100.0 + i for i in range(n)]  # rising — above its SMA past warmup
    nvda_adj = [np.nan] * 49 + [200.0 + i for i in range(n - 49)]
    ohlcv = {
        AAPL: _make_ohlcv_frame(aapl_adj, [1000] * n),
        NVDA: _make_ohlcv_frame(nvda_adj, [1000] * n),
    }
    # NVDA "joined" on session 50 (index 49); AAPL has been a member from day 0.
    intervals = _make_pit_intervals(
        [
            (AAPL, "2023-01-02", None),
            (NVDA, _bdate_index(n)[49].strftime("%Y-%m-%d"), None),
        ]
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    # Session 50 (index 50): NVDA has only 2 sessions → NaN SMA → dropped from
    # BOTH. AAPL above SMA → 1/1 = 1.0.
    assert out.pct_above_50dma.iloc[50] == pytest.approx(1.0)
    # Session 49 (index 49): AAPL is a member with defined SMA; NVDA just
    # joined and has NaN SMA. Per Ambiguity Log #59, AAPL's own history is
    # used regardless of past membership, so AAPL contributes → still 1.0.
    # The pathological-zero case is exercised in the no-member-in-ohlcv test.
    # Here we instead assert that at session 5 (only 6 sessions of data, no 50d
    # SMA possible for either ticker) the denominator collapses → NaN.
    assert np.isnan(out.pct_above_50dma.iloc[5])


def test_pct_above_50dma_honors_non_session_interval_boundaries(
    v2_breadth_config,
):
    """Interval boundaries are calendar dates, not guaranteed NYSE sessions.

    Weekend start/end dates should include the first NYSE session after the
    start boundary and exclude sessions after the last NYSE session on or
    before the end boundary.
    """
    n = 60
    closes = _make_sector_closes(n=n)
    rising = [100.0 + i for i in range(n)]
    falling = [200.0 - i for i in range(n)]
    ohlcv = {
        AAPL: _make_ohlcv_frame(rising, [1000] * n),
        MSFT: _make_ohlcv_frame(falling, [1000] * n),
    }
    idx = _bdate_index(n)
    # AAPL active from Saturday before idx[49] through Sunday after idx[54].
    # MSFT active across the full window. At idx[54], both contribute -> 0.5.
    intervals = _make_pit_intervals(
        [
            (
                AAPL,
                (idx[49] - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                (idx[54] + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
            ),
            (MSFT, "2023-01-02", None),
        ]
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    assert out.pct_above_50dma.iloc[54] == pytest.approx(0.5)
    # After the weekend-capped end boundary, AAPL drops out, leaving only MSFT
    # below its SMA -> 0/1 = 0.0.
    assert out.pct_above_50dma.iloc[55] == pytest.approx(0.0)


# =============================================================================
# Group C — pct_above_200dma (1 test)
# =============================================================================


def test_pct_above_200dma_three_tickers_two_above_one_below(v2_breadth_config):
    n = 220
    closes = _make_sector_closes(n=n)
    rising_a = [100.0 + i for i in range(n)]
    rising_b = [150.0 + i * 0.5 for i in range(n)]
    falling = [400.0 - i for i in range(n)]
    ohlcv = {
        AAPL: _make_ohlcv_frame(rising_a, [1000] * n),
        MSFT: _make_ohlcv_frame(rising_b, [1000] * n),
        IBM: _make_ohlcv_frame(falling, [1000] * n),
    }
    intervals = _make_pit_intervals(
        [
            (AAPL, "2023-01-02", None),
            (MSFT, "2023-01-02", None),
            (IBM, "2023-01-02", None),
        ]
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    assert out.pct_above_200dma.iloc[215] == pytest.approx(2.0 / 3.0)


# =============================================================================
# Group D — ad_line + ad_line_slope_20d (3 tests)
# =============================================================================


def test_ad_line_anchors_at_zero_first_session(v2_breadth_config):
    """Ambiguity Log #57: ``ad_line[0] = 0`` at the first computation session."""
    n = 30
    closes = _make_sector_closes(n=n)
    rising_a = [100.0 + i for i in range(n)]
    rising_b = [200.0 + i for i in range(n)]
    ohlcv = {
        AAPL: _make_ohlcv_frame(rising_a, [1000] * n),
        MSFT: _make_ohlcv_frame(rising_b, [1000] * n),
    }
    intervals = _make_pit_intervals(
        [(AAPL, "2023-01-02", None), (MSFT, "2023-01-02", None)]
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    assert out.ad_line.iloc[0] == pytest.approx(0.0)


def test_ad_line_strict_inequality_unchanged_days_count_as_neither(
    v2_breadth_config,
):
    """Ambiguity Log #56: strict ``>`` for advance, strict ``<`` for decline.
    Unchanged → neither. Three tickers: +1, -1, unchanged → delta = 0.
    """
    n = 30
    closes = _make_sector_closes(n=n)
    aapl = [100.0] + [101.0] * (n - 1)  # advances once at t=1, then flat
    msft = [100.0] + [99.0] * (n - 1)  # declines once at t=1, then flat
    ibm = [100.0] * n  # always unchanged
    ohlcv = {
        AAPL: _make_ohlcv_frame(aapl, [1000] * n),
        MSFT: _make_ohlcv_frame(msft, [1000] * n),
        IBM: _make_ohlcv_frame(ibm, [1000] * n),
    }
    intervals = _make_pit_intervals(
        [
            (AAPL, "2023-01-02", None),
            (MSFT, "2023-01-02", None),
            (IBM, "2023-01-02", None),
        ]
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    # At t=1: advances=1, declines=1, unchanged=1 → delta = 0.
    # ad_line[1] = ad_line[0] + 0 = 0.
    assert out.ad_line.iloc[1] == pytest.approx(0.0)


def test_ad_line_slope_20d_formula(v2_breadth_config):
    """30 sessions, 2 tickers, all advancing each day.

    delta = +2 every session ≥ 1; ad_line[t] = 2*t for t ≥ 1, ad_line[0] = 0.
    Slope at t=20: (ad_line[20] - ad_line[0]) / 20 = (40 - 0)/20 = 2.0.
    Slope at t=21: (ad_line[21] - ad_line[1]) / 20 = (42 - 2)/20 = 2.0.
    Slope at t<20: NaN (insufficient lookback).
    """
    n = 30
    closes = _make_sector_closes(n=n)
    a = [100.0 + i for i in range(n)]
    b = [200.0 + i for i in range(n)]
    ohlcv = {
        AAPL: _make_ohlcv_frame(a, [1000] * n),
        MSFT: _make_ohlcv_frame(b, [1000] * n),
    }
    intervals = _make_pit_intervals(
        [(AAPL, "2023-01-02", None), (MSFT, "2023-01-02", None)]
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    assert out.ad_line_slope_20d.iloc[21] == pytest.approx(2.0)
    assert out.ad_line_slope_20d.iloc[20] == pytest.approx(2.0)
    assert np.isnan(out.ad_line_slope_20d.iloc[19])


# =============================================================================
# Group E — nh_nl_ratio (2 tests)
# =============================================================================


def test_nh_nl_ratio_252_session_window_inclusive(v2_breadth_config):
    """Ambiguity Log #55: 252 trailing sessions, inclusive.

    AAPL monotonically rising → at session t=251 (index 251) it equals
    ``max(adj[0..251])`` → new_high. MSFT flat (constant) → never new high,
    never new low (max == min == close at every t, but the spec defines
    new_high as ``adjusted_close[t] == max(window)`` — under a flat series the
    close equals max AND equals min, but spec pin treats flat as neither high
    nor low for ratio purposes per #56's strict-direction intent). For this
    test we use a clean non-flat MSFT: a tiny non-monotonic series that is
    never at its own 252-session max OR min at t=251.
    """
    n = 260
    closes = _make_sector_closes(n=n)
    aapl = [100.0 + i for i in range(n)]
    # MSFT: high early, then gradually lower but never below the early
    # minimum nor above the early maximum at t=251.
    msft = [200.0] + [150.0 + i * 0.01 for i in range(n - 1)]
    # msft[0] = 200 is the all-time max; msft[1..] is bounded in [150, 152.6].
    # At t=251: msft[251] = 150 + 250*0.01 = 152.5; max over [0..251] = 200;
    # min over [0..251] = 150.01 → neither new high nor new low.
    ohlcv = {
        AAPL: _make_ohlcv_frame(aapl, [1000] * n),
        MSFT: _make_ohlcv_frame(msft, [1000] * n),
    }
    intervals = _make_pit_intervals(
        [(AAPL, "2023-01-02", None), (MSFT, "2023-01-02", None)]
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    # At t=251: new_highs=1 (AAPL), new_lows=0 → ratio = 1 / max(1+0, 1) = 1.0.
    assert out.nh_nl_ratio.iloc[251] == pytest.approx(1.0)
    # At t=250 (one short of 252 sessions of history) → insufficient history
    # → both tickers excluded → denominator collapses to zero → NaN.
    assert np.isnan(out.nh_nl_ratio.iloc[250])


def test_nh_nl_ratio_zero_when_no_new_high_or_low(v2_breadth_config):
    """All tickers strictly inside their 252-session range at t=251 →
    0 new highs, 0 new lows, ratio = 0 / max(0, 1) = 0.0.
    """
    n = 260
    closes = _make_sector_closes(n=n)
    # Bracket pattern: highest at t=0, lowest at t=1, then mid-range forever.
    a = [200.0, 50.0] + [125.0] * (n - 2)
    b = [180.0, 60.0] + [120.0] * (n - 2)
    ohlcv = {
        AAPL: _make_ohlcv_frame(a, [1000] * n),
        MSFT: _make_ohlcv_frame(b, [1000] * n),
    }
    intervals = _make_pit_intervals(
        [(AAPL, "2023-01-02", None), (MSFT, "2023-01-02", None)]
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    assert out.nh_nl_ratio.iloc[251] == pytest.approx(0.0)


def test_nh_nl_ratio_flat_series_counts_toward_both_new_highs_and_new_lows(
    v2_breadth_config,
):
    """Ambiguity Log #61: a ticker whose adjusted_close is constant across
    the full 252-session window satisfies BOTH equality predicates
    (adj_close[D] == rolling_max AND adj_close[D] == rolling_min). It must
    contribute to BOTH new_highs AND new_lows.

    Setup: two tickers, both perfectly flat at different price levels for the
    full 260-session window. At t=251 (first session with 252 sessions of
    history), both tickers fire as new_high AND as new_low. So
    new_highs = 2, new_lows = 2, ratio = 2 / max(2 + 2, 1) = 0.5.

    The 0.5 value is the load-bearing observation: if either ticker were
    counted toward only one side (the spec-rejected interpretations
    Option Y / Option Z in Log #61), the ratio would be either 0.0 (both
    suppressed) or 1.0 (both biased to high). Only Option X — count toward
    both — yields 0.5.
    """
    n = 260
    closes = _make_sector_closes(n=n)
    ohlcv = {
        AAPL: _make_ohlcv_frame([100.0] * n, [1000] * n),
        MSFT: _make_ohlcv_frame([200.0] * n, [1000] * n),
    }
    intervals = _make_pit_intervals(
        [(AAPL, "2023-01-02", None), (MSFT, "2023-01-02", None)]
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    assert out.nh_nl_ratio.iloc[251] == pytest.approx(0.5)


# =============================================================================
# Group F — upvol_downvol_ratio (1 test)
# =============================================================================


def test_upvol_downvol_ratio_uses_strict_direction_and_raw_volume(
    v2_breadth_config,
):
    """At t=1: AAPL advances on 1000 volume, MSFT declines on 500 volume.
    upvol = 1000, downvol = 500, ratio = 1000 / max(500, 1) = 2.0.
    """
    n = 30
    closes = _make_sector_closes(n=n)
    aapl_adj = [100.0, 101.0] + [101.0] * (n - 2)
    aapl_vol = [800, 1000] + [800] * (n - 2)
    msft_adj = [100.0, 99.0] + [99.0] * (n - 2)
    msft_vol = [400, 500] + [400] * (n - 2)
    ohlcv = {
        AAPL: _make_ohlcv_frame(aapl_adj, aapl_vol),
        MSFT: _make_ohlcv_frame(msft_adj, msft_vol),
    }
    intervals = _make_pit_intervals(
        [(AAPL, "2023-01-02", None), (MSFT, "2023-01-02", None)]
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    assert out.upvol_downvol_ratio.iloc[1] == pytest.approx(2.0)


# =============================================================================
# Group G — breadth_thrust feature (1 test)
# =============================================================================


def test_breadth_thrust_is_10_session_ma_of_pct_advancing(v2_breadth_config):
    """12 sessions, 2 tickers, all advancing each session t>=1 →
    pct_advancing[t>=1] = 2/2 = 1.0. The 10-session rolling mean over a window
    of all-1.0 values at session 10 (inclusive of t=1..10) = 1.0.
    """
    n = 12
    closes = _make_sector_closes(n=n)
    a = [100.0 + i for i in range(n)]
    b = [200.0 + i for i in range(n)]
    ohlcv = {
        AAPL: _make_ohlcv_frame(a, [1000] * n),
        MSFT: _make_ohlcv_frame(b, [1000] * n),
    }
    intervals = _make_pit_intervals(
        [(AAPL, "2023-01-02", None), (MSFT, "2023-01-02", None)]
    )
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    assert out.breadth_thrust.iloc[10] == pytest.approx(1.0)


# =============================================================================
# Group H — bias_warnings emission (3 tests)
# =============================================================================


def _full_pit_inputs(n: int = 60) -> tuple[dict, pd.DataFrame]:
    a = [100.0 + i for i in range(n)]
    b = [200.0 + i for i in range(n)]
    ohlcv = {
        AAPL: _make_ohlcv_frame(a, [1000] * n),
        MSFT: _make_ohlcv_frame(b, [1000] * n),
    }
    intervals = _make_pit_intervals(
        [(AAPL, "2023-01-02", None), (MSFT, "2023-01-02", None)]
    )
    return ohlcv, intervals


def test_bias_warnings_emitted_when_pit_features_computed(v2_breadth_config):
    closes = _make_sector_closes(n=60)
    ohlcv, intervals = _full_pit_inputs(n=60)
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    assert isinstance(out.bias_warnings, pd.DataFrame)
    pit_warnings = out.bias_warnings[
        out.bias_warnings["warning_code"] == _PIT_BIAS_WARNING
    ]
    assert len(pit_warnings) == 7


def test_bias_warnings_all_use_survivorship_biased_constituent_universe_code(
    v2_breadth_config,
):
    closes = _make_sector_closes(n=60)
    ohlcv, intervals = _full_pit_inputs(n=60)
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    pit_warnings = out.bias_warnings[
        out.bias_warnings["warning_code"] == _PIT_BIAS_WARNING
    ]
    assert set(pit_warnings["warning_code"]) == {_PIT_BIAS_WARNING}
    assert set(pit_warnings["source"]) == {_PIT_SOURCE}
    assert set(pit_warnings["source_url"]) == {_PIT_SOURCE_URL}


def test_bias_warnings_feature_names_cover_all_seven_pit_features(
    v2_breadth_config,
):
    closes = _make_sector_closes(n=60)
    ohlcv, intervals = _full_pit_inputs(n=60)
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    pit_warnings = out.bias_warnings[
        out.bias_warnings["warning_code"] == _PIT_BIAS_WARNING
    ]
    assert set(pit_warnings["feature_name"]) == set(_PIT_FEATURE_NAMES)


# =============================================================================
# Group I — feature_names + to_frame expansion (2 tests)
# =============================================================================


def test_feature_names_expand_to_include_pit_features_when_present(
    v2_breadth_config,
):
    closes = _make_sector_closes(n=60)
    ohlcv, intervals = _full_pit_inputs(n=60)
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    assert (
        out.feature_names
        == ("sector_breadth",) + _AVAILABLE_SECTOR_FEATURE_NAMES + _PIT_FEATURE_NAMES
    )


def test_to_frame_columns_match_feature_names(v2_breadth_config):
    closes = _make_sector_closes(n=60)
    ohlcv, intervals = _full_pit_inputs(n=60)
    out = compute_breadth_v2_features(
        sector_etf_closes=closes,
        config=v2_breadth_config,
        pit_constituent_intervals=intervals,
        constituent_ohlcv=ohlcv,
    )
    frame = out.to_frame()
    assert list(frame.columns) == list(out.feature_names)
    assert len(frame) == len(out.sector_breadth)
