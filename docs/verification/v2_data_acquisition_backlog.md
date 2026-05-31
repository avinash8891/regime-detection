# V2 Data-Acquisition Backlog

Two remaining V2 gaps are **data-acquisition tasks, not engine/code gaps**. The
engine classifies correctly; what is missing is real historical market data that
cannot be fabricated (golden/regression fixtures require real data, and synthetic
data is forbidden). They are tracked here with the exact inputs required and the
guards that keep the repository honest until the data lands.

## 1. Pre-2019 V2 fixture history (for the four pre-2019 §9.4 golden dates)

**Blocked:** 4 of the 9 §9.4 golden dates cannot be classified live because the
V2 daily-OHLCV fixture (`tests/fixtures/raw/v2/daily_ohlcv.csv`) starts
`2019-01-02`.

| §9.4 date | Event | Tests |
|---|---|---|
| 2010-05-06 | Flash Crash | systemic_stress, correlation_to_one |
| 2011-08-08 | US downgrade | credit_stress, funding_squeeze |
| 2015-08-24 | China devaluation | rising_fragility, bull→correlation_to_one |
| 2018-10-10 | Q4-2018 stress | bull→narrowing_breadth→bear_stress |

**Data required** (real, point-in-time, back to ~2009 so each date has its
required lookback — e.g. 252-session percentiles, 250-session Hurst):
- Daily OHLCV + volume for SPY, RSP, QQQ, the 11 GICS sector ETFs (XLB, XLC,
  XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY), and the VIX (real index rows —
  the current fixture's VIX-proxy gap is the explicit unsupported reason).
- Cross-asset closes used by credit/funding + inflation/growth: HYG, LQD, TLT,
  KRE, DBC.
- FRED macro series back to ~2009: SOFR (or pre-2018 proxy), IORB, NFCI, broad
  USD index, CPIAUCSL, PMI manufacturing, 2y/10y yields.
- PIT SPX constituent intervals + constituent OHLCV covering the window (see §2).

**Source of truth:** the same Alpaca/local archived parquet pipeline named in ADR
0020 (F-049). XLC and XLRE did not exist before 2015/2018 respectively — the
sector-breadth "available denominator" proxy already handles missing sectors, so
pre-listing sectors are expected to be absent, not fabricated.

**Honest guard (in place):** `tests/test_fixture_verification.py`
`_V2_LIVE_FIXTURE_UNSUPPORTED_GOLDEN_DATES` lists these four dates with the
reason "V2 daily OHLCV fixture must include real VIX rows", and
`test_v2_golden_dates_classify_expected_fields` asserts the supported/unsupported
split exactly — so the gap is explicit and cannot be silently skipped. When the
data lands, move each date out of that dict and it will be classified live.

## 2. True point-in-time SPX constituent feed

**Blocked:** PIT constituent breadth currently sources from the free
`fja05680/sp500` membership approximation, documented as an approximation
(`pit_constituents.py:17-27`, `pit_provenance.py`) and tagged with a
`survivorship_biased_constituent_universe` bias warning on every PIT feature.

**Data required:** a vendor point-in-time membership feed (CRSP / Compustat /
FactSet / Norgate) with effective add/remove dates and delisted symbols. The
on-disk interval schema (`ticker / start_date / end_date`) is already
vendor-compatible (`pit_constituents.py:145-157`), so this is a **sourcing swap,
not a code rewrite** — `members_on` reconstructs per-date membership from
intervals unchanged.

**Honest guards (in place):**
- The `survivorship_biased_constituent_universe` bias warning is attached to all
  seven PIT breadth features until a true feed replaces the approximation.
- The V2PRE-010 / §1D line 327 fail-closed ingestion gate
  (`regime_data_fetch.pit_constituents.read_pit_intervals` +
  `is_survivorship_biased_universe`, param `allow_survivorship_biased_breadth`,
  default `False`) rejects a loaded universe that is survivorship-biased — a
  current-only snapshot with no removed/delisted members — unless biased research
  mode is explicitly requested. The real fja05680 universe includes delistings
  and passes; the V1 ETF-proxy fallback (no PIT universe loaded) is unaffected.

## Not on this list (resolved or out-of-scope)

- The §9.4 `2020-08-15` Saturday date is re-anchored to `2020-08-14` and now
  classifies live (see `golden_dates_v2.yaml` provenance note).
- §5.3 vol-crush exposure response is a downstream strategy-layer contract, not
  this engine's responsibility (ADR 0020, F-053).
