# ADR 0011 — Inflation/Growth Rule Coverage Fix

**Status:** Accepted
**Date:** 2026-05-20
**Context:** v2 §2B inflation_growth axis was classifying ~51% of sessions as
"unknown" (no_rule_fired / no_rule_fired_missing_feature), creating a massive
coverage gap in the regime timeline.

## Problem

Three structural issues in the §2B rule predicates prevented classification:

1. **Disinflation over-gated on yields** — Required `treasury_10y_yield_slope_21d < 0`
   in addition to `cpi_6m_change_pct_slope_21d < 0`. Disinflation (falling CPI) is
   an objective measure; bond yields are a lagging confirmation signal. Many periods
   of genuinely falling CPI were blocked because yields hadn't responded yet.

2. **Goldilocks had a CPI stability paradox** — Required either
   `abs(cpi_drift) ≤ 0.005` or `cpi_slope ≤ 0`. Environments with moderate
   inflation (e.g., CPI 6m = 2-3%) and mild upward drift were rejected, even
   though "manageable low inflation + growth" IS goldilocks.

3. **Cross-axis credit dependency created a coverage cliff** — When
   `credit_funding_active_label` was None (axis unbuilt/stale), goldilocks,
   recession_scare, and recovery_growth were completely blocked. ~33% of sessions
   had no credit classification, blocking 3 of 6 rules.

## Decision

### Fix 1: Yield-independent disinflation (`disinflation_yield_independent: true`)

Disinflation fires when `cpi_6m_change_pct_slope_21d < 0 AND PMI > 45`,
without requiring `yield_slope < 0`. Rationale: CPI is the primary inflation
measure. Yields often lag by weeks/months during disinflationary transitions.

Config: `disinflation_yield_independent` (bool, default true).

### Fix 2: Benign CPI ceiling for goldilocks (`cpi_goldilocks_benign_ceiling: 0.04`)

Added a third CPI path: when `cpi_6m_change_pct < 0.04` (4% annualized),
CPI is treated as benign regardless of drift/slope direction. This captures
"mild reflation with strong growth" — economically goldilocks.

Config: `cpi_goldilocks_benign_ceiling` (float, default 0.04).

### Fix 3: Credit-independent fallback (`allow_credit_independent_fallback: true`)

When `credit_funding_active_label` is None (axis unbuilt), goldilocks and
recovery_growth use their existing non-credit conditions without the
credit_calm gate. Recession_scare uses a stricter SPY threshold (-7% vs -5%)
to compensate for missing credit confirmation.

Configs:
- `allow_credit_independent_fallback` (bool, default true)
- `spy_recession_credit_independent_threshold` (float, default -0.07)

## Impact

| Metric | Before | After |
|--------|--------|-------|
| inflation_growth stale % | 51.4% | 36.1% |
| Sessions classified | 1170/2408 | 1539/2408 |
| disinflation coverage | 513 sessions | 947 sessions |

Year-by-year improvement concentrated in 2017-2019 and 2022-2024 where
CPI was declining but yields were rising (Fed tightening cycles).

## Remaining Gaps (36.1%)

- **spread_widening + mild equity decline** (338 sessions): SPY between 0%
  and -5% during credit stress. Would need a relaxed recession_scare or a
  new "credit_watch" label.
- **Credit stale + CPI not falling** (275 sessions): Genuinely ambiguous —
  no credit signal and no clear inflation direction.
- **credit_calm + PMI ≤ 50 or SPY ≤ 0** (170 sessions): Below goldilocks
  gates but not stressed enough for recession_scare.

## Consequences

- Disinflation now has higher precedence than goldilocks. Sessions that were
  goldilocks with `cpi_slope < 0` are now correctly classified as disinflation.
- The NaN-falsifies contract for `inflation_surprise_zscore` and
  `aggregate_forward_eps_revision_direction_4w` is preserved — those features
  remain optional.
- All changes are config-gated and can be reverted per-environment.
