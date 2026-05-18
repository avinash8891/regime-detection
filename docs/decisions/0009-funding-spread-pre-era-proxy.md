# Decision 0009: V2 §2C funding-spread pre-era proxy — DFF-IOER splice

**Status:** accepted — implemented alongside the DFF/IOER FRED fetch addition.

## Context

The V2 §2C credit/funding unknown gate (spec lines 2122-2126 / 3253) fires when
SOFR or IORB is stale beyond the global freshness budget. For the 2016–2021
backtest window, this caused the gate to fire on *every session* because SOFR
and IORB do not exist before their creation dates:

- **SOFR** (Secured Overnight Financing Rate): first published **2018-04-03** by
  the NY Fed. No data exists before that date — it is not a gap; the instrument
  was not yet created.
- **IORB** (Interest on Reserve Balances): first published **2021-07-29** when
  the Fed consolidated IOER and IOBR into a single rate. No data exists before
  that date.

Both confirmed directly against the FRED API (series `SOFR`, `IORB`). The
original staleness check used raw SOFR/IORB series, which return the
`_STALENESS_SENTINEL = 10**9` for pre-creation sessions, triggering "unknown"
for all 2016–2021 credit/funding outputs. This in turn blocked three of seven
`inflation_growth` rule predicates (goldilocks, recession_scare, recovery_growth
all require a specific credit_funding active label) for the full 2016–2021
window — explaining the observed 53% `unknown` rate in the `inflation_growth`
axis over that period.

## Decision

Add **DFF** (FRED `DFF`) and **IOER** (FRED `IOER`) as pre-era
funding-spread proxies:

- **DFF**: Effective Federal Funds Rate, available daily from 1954+.
- **IOER**: Interest on Excess Reserves, available 2008-10-09 through 2021-07-28
  (the direct predecessor of IORB, same policy-floor semantics).

The `sofr_iorb_spread` feature becomes a **spliced series** in
`credit_funding.compute_credit_funding_features`, using the best available
source for each session:

1. `SOFR - IORB` when both are available (Jul 2021+) — authoritative
2. `SOFR - IOER` when SOFR exists but IORB does not (Apr 2018 – Jul 2021)
3. `DFF - IOER` when neither SOFR nor IORB exists (Oct 2008 – Apr 2018)

The staleness gate in `axis_builders/credit_funding.py` is updated to check
the staleness of the **spliced series** (`features.sofr_iorb_spread`) rather
than the raw SOFR/IORB series. For historical sessions (2016+) where
DFF-IOER fills the splice, the staleness is 0–1 days, so the gate passes.

A `funding_spread_fedfunds_ioer_proxy` bias-warning row is emitted on the
feature-store output whenever the splice is active (i.e., when the spliced
series has more non-NaN values than the raw SOFR-IORB series alone).

## Why this is NOT a patch-on-patch

DFF and IOER are **not approximations** of SOFR and IORB. They are the
*predecessor instruments* that measured the same economic quantity (overnight
bank funding cost relative to the Fed's policy floor) during the era before
SOFR and IORB existed. The signal is conceptually identical:

| Era | Source | Signal |
|-----|--------|--------|
| Oct 2008 – Apr 2018 | DFF – IOER | Fed funds rate minus excess-reserve floor |
| Apr 2018 – Jul 2021 | SOFR – IOER | Overnight repo rate minus excess-reserve floor |
| Jul 2021 – present | SOFR – IORB | Overnight repo rate minus reserve-balance floor |

This is the same design pattern as ADR 0007 (TLT-vs-HYG/LQD proxy covers the
era before real-OAS data exists on FRED) and ADR 0006 (Cleveland Fed nowcast
substitutes for the survey-based CPI consensus estimate).

## What this does NOT change

- The `sofr_iorb_spread` and `sofr_iorb_slope_21d` **feature names** are
  unchanged. Downstream rule predicates (`evaluate_funding_squeeze`,
  `evaluate_deleveraging`) continue to read the same fields.
- The rule predicates themselves are unchanged.
- `credit_funding_state_proxy` (TLT-vs-HYG/LQD) remains a separate parallel
  metric and is unaffected.
- The original staleness gate behavior is preserved for live sessions (2021+),
  where SOFR and IORB are fully available and the splice does not activate.

## Consequences

- `credit_funding_effective_state` now emits real labels (not `unknown`) for
  2016–2021 sessions when DFF and IOER are materialized.
- The `inflation_growth` goldilocks/recession_scare/recovery_growth predicates
  can fire for the full 2016–2026 backtest window.
- DFF and IOER are fetched as part of the standard `--fetch macro` step
  via `V2_FRED_SERIES` in `regime_data_fetch.fetch_workflow`.
- The `fred_macro_series.parquet` manifest artifact must be re-materialized
  (re-fetched from FRED) to include the new series before running `profile_engine`.
