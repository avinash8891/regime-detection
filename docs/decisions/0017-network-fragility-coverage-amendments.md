# ADR 0017 — Network Fragility Coverage Amendments

**Status:** Accepted
**Date:** 2026-05-23
**Context:** A spec/code audit found two production-shipped
`network_fragility` rule amendments that were implemented in code and config
but not fully reflected in `docs/regime_engine_v2_spec.md` §3.5:

- `diversified_normal_relaxed_inner_band` lets mid-range correlation
  (`0.30 <= avg_pairwise_corr_percentile_504d <= 0.60`) classify as
  `diversified_normal` even when effective-rank stability is not below the
  5% threshold.
- `concentration_absorption_ratio_min` adds
  `absorption_ratio_top3 > 0.90` as a fourth `correlation_concentration`
  limb.

Both were introduced during the May 2026 rule-coverage passes that reduced
unexplained `unknown` output and closed audit gaps.

## Problem

The original §3.5 rule text was too narrow in two places.

### R1 — Mid-correlation normal markets

The spec required `diversified_normal` to satisfy both:

```text
0.0 <= avg_pairwise_corr_percentile_504d <= 0.75
AND effective_rank stable (21d std < 5% of mean)
```

That made ordinary factor rotation in a moderate-correlation market fall
through to `unknown` whenever effective rank moved more than 5% over 21
sessions. But moderate correlation is not, by itself, fragility. In the last
2016-present profile, the shipped relaxed inner band classified 147 sessions
that would otherwise have remained `unknown`.

### R2 — Top-3 eigenvalue concentration

The spec defined `absorption_ratio_top3` as a §3.2 feature but did not consume
it in any §3.5 label rule. That left a direct concentration measure available
only as evidence. A market can have top-3 eigenvalue dominance without the
single largest eigenvalue percentile or effective-rank percentile being the
first threshold to cross.

In the last 2016-present profile, the absorption-only limb changed 0 sessions,
so this is currently a dormant coverage amendment. It is still ratified so the
feature has a governed classifier role instead of being unexplained evidence.

## Decision

### R1 — Ratify `diversified_normal` relaxed inner band

`diversified_normal` is now:

```text
0.0 <= avg_pairwise_corr_percentile_504d <= 0.75
AND (
  effective_rank stable (21d std < 5% of mean)
  OR 0.30 <= avg_pairwise_corr_percentile_504d <= 0.60
)
```

Rationale:

- `0.30–0.60` is the neutral middle of the correlation distribution.
- Effective-rank instability inside that middle band often reflects sector or
  factor rotation, not systemic fragility.
- This reduces `unknown/no_rule_fired` without converting high-correlation or
  low-correlation edge cases into benign labels.
- The stability condition remains required outside the inner band, preserving
  discipline near correlation extremes.

### R2 — Ratify `absorption_ratio_top3 > 0.90`

`correlation_concentration` is now:

```text
avg_pairwise_corr_percentile_504d > 0.75
OR largest_eigenvalue_share_percentile_504d > 0.75
OR effective_rank_percentile_504d < 0.25
OR absorption_ratio_top3 > 0.90
```

Rationale:

- `absorption_ratio_top3` measures concentration across the dominant few
  eigenvectors, not just the single largest eigenvector.
- The threshold is deliberately high (`0.90`) so the limb fires only on strong
  top-3 dominance.
- The feature already exists in §3.2 and the feature store; leaving it
  evidence-only made the rule set incomplete.

## Measured impact

Profile artifact:
`.context/profile_engine_2010-01-04_2026-05-15_warmup_detection_from_2016_after_effective_summary.json`

Measured on the 2016-01-04 through 2026-05-15 slice:

| Amendment | Sessions affected |
|---|---:|
| Relaxed `diversified_normal` inner band | 147 |
| Absorption-only `correlation_concentration` limb | 0 |

The profile also showed no 2016+ data-quality unknowns for network
fragility. Remaining `unknown` labels were rule fallthroughs
(`no_rule_fired` / `no_rule_fired_hysteresis`), not missing feature data.

## Code/config anchors

| Concept | Location |
|---|---|
| Relaxed inner-band predicate | `src/regime_detection/network_fragility_rules.py::evaluate_diversified_normal` |
| Relaxed inner-band config | `src/regime_detection/configs/core3-v2.0.0.yaml` (`diversified_normal_relaxed_inner_band`, `diversified_normal_inner_band_lo`, `diversified_normal_inner_band_hi`) |
| Absorption concentration predicate | `src/regime_detection/network_fragility_rules.py::evaluate_correlation_concentration` |
| Absorption threshold config | `src/regime_detection/configs/core3-v2.0.0.yaml` (`concentration_absorption_ratio_min`) |
| Spec text updated | `docs/regime_engine_v2_spec.md` §3.5 |

## 2026-05-23 universe amendment

Accepted follow-up: add `DBC` and `IEF` together to the §3.1 network-fragility
universe. `DBC` gives the correlation matrix a broad commodity sleeve rather
than relying only on the narrow oil proxy (`USO`). `IEF` adds intermediate
Treasury duration alongside `TLT`.

The pair is intentionally shipped together. In the 2016-01-04 to 2026-05-15
A/B run, `add_IEF_23` reduced unknowns but shifted the COVID active-label mix
from `correlation_to_one` into `systemic_stress` because hysteresis entered the
March 2020 stress block from a lower-rank label. `add_DBC_IEF_24` kept the
COVID systemic count aligned with baseline while still reducing unknowns:

| Variant | Active unknown | COVID correlation_to_one | COVID systemic_stress |
|---|---:|---:|---:|
| baseline_22 | 95 | 23 | 22 |
| add_IEF_23 | 75 | 5 | 33 |
| add_DBC_IEF_24 | 76 | 21 | 22 |

## Consequences

- The spec now matches shipped behavior for the two network-fragility coverage
  amendments and the paired DBC/IEF universe amendment.
- Future calibration may tune the inner-band bounds or absorption threshold,
  but removing either behavior requires a follow-up ADR because both now
  define classifier semantics.
- Reviewers should no longer flag these two code paths as ungoverned drift.
