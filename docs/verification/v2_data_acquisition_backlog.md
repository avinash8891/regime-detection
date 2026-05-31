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

**FRED macro — partially extended; the rest is API-limited, not deferrable.**
`broad_usd_index` (DTWEXBGS) and `nfci` (NFCI) were fetched from the FRED API and
now span 2009-01-02..2026 in `fred_macro_series.csv`. The remaining fixture
series have genuine availability floors the FRED **API** cannot backfill:
- `sofr` (SOFR) starts 2018-04; `iorb` (IORB) starts 2021-07 — these post-date the
  pre-2019 golden dates regardless of any fetch.
- `hy_oas`/`ig_bbb_oas` (ICE BofA OAS, `BAMLH0A0HYM2`/`BAMLC0A4CBBB`) are capped by
  FRED's **API** to a ~3-year rolling window under ICE license redistribution
  (`/fred/series` reports observation_start 2023-05-30). The full 1996+ history
  exists only on the FRED **website CSV**, which is unreachable from the build
  environment. Production fetches via the same API, so the fixture's OAS coverage
  already matches production — this is not a fixture-specific gap.

Consequently the pre-2016 credit OAS metrics are unavailable, so `credit_funding`
emits `unknown` for the pre-2016 golden dates. The §9.4 test is presence-based and
passes; the four pre-2019 dates classify live. Fully real pre-2016 credit_stress
would require the OAS website CSV (or another licensed source) — out of scope of
the FRED API.

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
