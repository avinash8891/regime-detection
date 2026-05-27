# ADR 0019 - Valid-Data Rule Partitions

**Status:** Accepted
**Date:** 2026-05-26

## Context

The 2016-01-04 through 2026-05-15 profile run exposed valid-data rule
fallthroughs in three layer-2 axes. The first attempted fix replaced those
fallthroughs with broad fallback labels. That was not sufficient: a fallback
label can hide an incomplete taxonomy the same way `unknown` did.

This ADR supersedes that fallback approach. A valid rule input row must be
classified by a named predicate. The final tail branch in a rule engine is now
a diagnostic (`unknown` with `unpartitioned_*_rule_space`), not a market
state.

## Decision

### Credit/funding

Keep the severe precedence unchanged:

```text
deleveraging > funding_squeeze > credit_stress
```

Then partition non-severe credit/funding states by spread level and direction:

| Label | Predicate | Rationale |
|---|---|---|
| `spread_widening` | HY slope > 0 and (IG slope > 0 or HY percentile >= 0.50) | HY-led widening from an elevated spread level is deterioration even if IG lags. |
| `credit_divergence` | HY percentile < 0.50 and HY slope > 0 and IG slope <= 0 | Low-spread HY-only softening without broad credit confirmation. |
| `credit_recovery` | HY percentile >= 0.50 and HY slope < 0 | Elevated or stressed spreads are repairing unless severe rules already fired. |
| `credit_calm` | HY percentile < 0.50 and HY slope <= 0 | Low spread level with non-widening HY. |

### Inflation/growth

Add missing macro states using existing CPI, PMI, equity, sector, and credit
inputs:

| Label | Predicate | Rationale |
|---|---|---|
| `contractionary_disinflation` | CPI slope <= 0 and PMI <= 45 | Demand contraction with easing inflation pressure. |
| `recovery_growth_unconfirmed` | PMI > 50, PMI slope > 0, cyclical/defensive slope > 0, CPI slope <= 0, and credit is not calm | Growth repair that lacks calm-credit confirmation. |
| `late_cycle_inflation_stress` | CPI slope >= 0, PMI > 50, and either SPY <= 0 or credit is stressed/widening | Expansion with non-declining inflation and market/credit stress. |
| `macro_neutral` | Core macro inputs are finite and no directional macro/earnings predicate fired | Explicit neutral/no-impulse state, not a catch-all stress label. |

### Network fragility

Add named states for the uncovered network shapes:

| Label | Predicate | Rationale |
|---|---|---|
| `decorrelated_calm` | Average correlation percentile < 0.30 and dispersion percentile <= 0.70 | Broad market is decorrelated without stock-picker dispersion. |
| `idiosyncratic_crisis` | Average correlation percentile < 0.30, dispersion percentile > 0.70, and volatility is `crisis_vol` | High dispersion in a crisis-volatility tape is not benign stock picking. |
| `rotation_watch` | 0.60 < average correlation percentile <= 0.75 and effective-rank stability >= 0.05 | Upper-normal correlation with unstable factor structure, below concentration. |

`unknown` remains available for stale/unavailable/insufficient data and for
the explicit unpartitioned-rule diagnostic. It is not a valid market state.

## Consequences

- The profile target is no active `unknown` caused by `no_rule_fired` on valid
  inputs.
- Any future rule-space escape should be treated as a classifier bug and
  reproduced with a failing test before adding another label.
- New labels must be added to label literals, risk ranks, hysteresis config,
  and profile evidence tests together.

## Measured Impact

Profile artifact:
`.context/profile_engine_2016_to_latest_rule_partitions.json`

Measured on the 2016-01-04 through 2026-05-15 selected window:

| Axis | Status | Active `unknown` | New explicit labels |
|---|---|---:|---|
| `network_fragility` | `classified=2607` | 0 | `decorrelated_calm=327`, `rotation_watch=32` |
| `credit_funding_state_proxy` | `classified=2607` | 0 | `credit_divergence=43`, `credit_recovery=393` |
| `credit_funding_effective_state` | `classified=2607` | 0 | `credit_divergence=63`, `credit_recovery=351` |
| `inflation_growth_state` | `classified=2607` | 0 | `contractionary_disinflation=37`, `late_cycle_inflation_stress=111`, `recovery_growth_unconfirmed=59`, `macro_neutral=2` |

The direct OAS-backed `credit_funding_state` still has 2357 active `unknown`
sessions, all reported as data-quality statuses:

| Status | Sessions |
|---|---:|
| `stale_data` | 1854 |
| `data_unavailable` | 352 |
| `insufficient_history` | 151 |

Profile stdout contained zero `no_rule_fired`, zero `unpartitioned_`, zero
HMM non-monotonic seed warnings, and zero pandas chained-assignment warnings.
