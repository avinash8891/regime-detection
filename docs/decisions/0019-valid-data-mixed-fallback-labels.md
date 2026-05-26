# ADR 0019 - Valid-Data Mixed Fallback Labels

**Status:** Accepted
**Date:** 2026-05-26

## Context

The 2016-01-04 through 2026-05-15 profile run showed that several layer-2
axes emitted `unknown` even when their rule inputs were present, finite, and
fresh enough to classify:

| Axis | Profile symptom |
|---|---:|
| `credit_funding_state_proxy` | 187 active `unknown` sessions |
| `credit_funding_effective_state` | 180 active `unknown` sessions, reported as `no_rule_fired` |
| `inflation_growth_state` | 113 active `unknown` sessions |
| `network_fragility` | 83 active `unknown` sessions |

That mixed two different conditions into one label:

- data-quality failure: stale data, unavailable data, insufficient history, or
  non-finite inputs;
- valid-data rule fallthrough: all inputs are usable, but no stress, calm,
  concentration, growth, inflation, or earnings predicate dominates.

Those conditions require different operator action. Data-quality failures
should remain visible as `unknown`. Valid mixed markets should be classified
as valid states so profile output can distinguish "the data is bad" from
"the market signal is balanced."

## Decision

Add explicit valid-data fallback labels for the rule engines that had governed
rule inputs but no neutral mixed state:

| Axis | Valid-data fallback | Fallback reason |
|---|---|---|
| Credit/funding | `credit_mixed` | `no_dominant_credit_funding_signal` |
| Inflation/growth | `macro_mixed` | no dominant macro/inflation/growth/earnings signal |
| Network fragility | `network_mixed` | `no_dominant_network_fragility_signal` |

`unknown` is reserved for data-quality failures and backward-compatible model
fields that still need to represent unavailable classifier state. A rule
engine that receives complete, finite, fresh inputs must return a named
classified label.

The fallback labels are deliberately low risk-rank and non-sticky:

| Label | Risk rank | Hysteresis |
|---|---:|---:|
| `credit_mixed` | 0 | 0 sessions |
| `macro_mixed` | 1 | 0 sessions |
| `network_mixed` | 0 | 0 sessions |

## Measured Impact

Profile artifact:
`.context/profile_engine_2016_to_latest_full_coverage.json`

Measured on the 2016-01-04 through 2026-05-15 selected window:

| Axis | Active `unknown` after change | Mixed fallback count |
|---|---:|---:|
| `network_fragility` | 0 | `network_mixed`: 83 |
| `credit_funding_state_proxy` | 0 | `credit_mixed`: 187 |
| `credit_funding_effective_state` | 0 | `credit_mixed`: 185 |
| `inflation_growth_state` | 0 | `macro_mixed`: 113 |

The direct OAS-backed `credit_funding_state` still reports data-quality
unknowns because the real spread source does not provide full usable history
for the whole selected window:

| Status | Sessions |
|---|---:|
| `stale_data` | 1854 |
| `data_unavailable` | 352 |
| `insufficient_history` | 151 |
| `classified` | 250 |

That is expected data-quality behavior, not rule fallthrough.

## Consequences

- `no_rule_fired` should be zero for these axes when inputs are complete and
  finite.
- Profile reports can now treat any remaining `unknown` as an actionable data
  availability, freshness, or warmup problem.
- Future axes should include an explicit valid-data fallback before wiring
  profile output. Adding a new rule is not the right fix for balanced markets
  unless the new rule is theoretically justified as a distinct regime.
