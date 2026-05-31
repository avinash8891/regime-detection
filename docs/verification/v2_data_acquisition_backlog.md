# V2 Data-Acquisition Backlog

Remaining V2 data items. The engine classifies correctly; these are about real
historical market data (which cannot be fabricated — golden/regression fixtures
require real data, and synthetic data is forbidden).

## 1. Pre-2019 V2 fixture history — RESOLVED (2026-05)

**Status:** ✅ Done. All four pre-2019 §9.4 golden dates (Flash Crash 2010-05-06,
US downgrade 2011-08-08, China devaluation 2015-08-24, Q4-2018 stress 2018-10-10)
now classify live. The V2 daily-OHLCV fixture (`tests/fixtures/raw/v2/daily_ohlcv.csv`)
was extended back to `2009-01-02` with real **Yahoo `/v8/finance/chart`** daily
bars (fetched per-symbol in yearly chunks on a local residential host — Yahoo's
API rate-limits datacenter IPs; Alpaca has no pre-2016 data), including the real
`^VIX` index mapped to `VIX`. The placeholder PIT membership intervals in
`tests/conftest.py::v2_pit_constituent_intervals` now start at each sector ETF's
real inception (XLRE 2015-10-08, XLC 2018-06-19, others 2009-01-02) so pre-2019
dates have active members for the PIT-breadth → clustering → transition-score
chain. `_V2_LIVE_FIXTURE_UNSUPPORTED_GOLDEN_DATES` is now empty;
`test_v2_golden_dates_classify_expected_fields` confirms all nine classify with
their expected fields present. See `tests/fixtures/raw/v2/PROVENANCE.md`.

**Open follow-up — FRED macro not yet extended.** The committed FRED fixture
(`fred_macro_series.csv`) still starts 2016 and carries `hy_oas`, `ig_bbb_oas`,
`iorb`, `broad_usd_index`, `nfci`, `sofr`. The locally-fetched
`.context/v2_fred_macro_2009_2018.csv` uses a *different* series set
(`fedfunds`, `ioer_legacy`, `2y_yield`, `10y_yield`, `cpi_all_items`,
`implied_vol_30d`, plus `broad_usd_index`/`nfci`/`sofr`) and does **not** include
`hy_oas`/`ig_bbb_oas`, so it cannot be appended without reconciling the logical
names the conftest macro loader expects. Pre-2016 credit/funding/monetary axes
therefore emit `unknown` for the pre-2016 golden dates (the §9.4 test is
presence-based, so they still pass). `SOFR` (2018-04+) and `IORB` (2021+) are
genuinely younger than these dates regardless of fetch.

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
