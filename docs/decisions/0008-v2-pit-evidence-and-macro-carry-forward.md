# Decision 0008: V2 PIT evidence and macro carry-forward semantics

**Status:** accepted — implemented in the 30-session profile fix.

## Context

The 30-session regime profile exposed three classes of missing evidence:

- FRED macro series were reindexed to the NYSE calendar without carrying the
  latest available observation forward. Normal publication gaps inside 5-year
  rolling windows made monetary z-score features all-NaN.
- Credit/funding required same-session SOFR/IORB rows, so ordinary publication
  lag could force `unknown` even when the most recent observation was fresh.
- HMM and GMM were fit on the final training window and all earlier emitted
  rows were masked to avoid future leak. That avoided lookahead, but it also
  made multi-day profile outputs mostly blank.

## Decision

1. **Macro observation alignment is latest-known-as-of, not same-session-only.**
   Daily, weekly, and monthly macro inputs may be sparse on NYSE sessions.
   Feature math carries the latest known observation forward before rolling
   calculations. Staleness remains enforced at the classifier boundary.

2. **Funding-spread freshness is staleness-based.** Credit/funding carries the
   funding-spread seam (`sofr_iorb_spread`) forward for feature math. The
   unknown gate fires only when the latest observation of that seam is older
   than the global freshness budget, not merely absent on the current NYSE
   session. The seam is a spliced series (see ADR 0009): SOFR-IORB for
   Jul 2021+, SOFR-IOER for Apr 2018–Jul 2021, FEDFUNDS-IOER for Oct 2008–
   Apr 2018. For sessions before any era of the splice, SOFR and IORB are
   genuinely non-existent (created Apr 2018 and Jul 2021 respectively), so the
   gate would have incorrectly fired "unknown" for the entire 2016–2021 window
   under the original raw-SOFR/IORB staleness check.

3. **Trainable evidence layers emit point-in-time evidence across a window.**
   HMM and GMM outputs for session `t` must be produced from a model trained
   only on data available through `t`. A multi-day `classify_window` must not
   reuse a final-date fit for earlier sessions and must not blank warmed rows
   solely because the final-date fit would leak.

4. **`hmm_probability_shift[t]` needs five pre-window warmed sessions.** Runners
   that request an N-session emitted window must materialize enough additional
   history for `top_state_prob[t-5]` on the first emitted session.

5. **Monetary `unknown` is data absence, not a persistent regime.** The
   monetary-pressure axis uses `unknown: 0` de-escalation. Once current-session
   monetary features recover, the active label should move immediately to the
   rule-derived label rather than holding a stale quality-gap label.

## Consequences

- Multi-day profiles should show populated `monetary_pressure_state`,
  `credit_funding_state`, `cluster`, and `hmm_probability_shift` when their
  sources are present and fresh.
- FRED publication gaps no longer poison long rolling windows.
- HMM/GMM runtime is higher than final-fit masking because point-in-time evidence
  requires repeated fits. Future optimization should cache PIT model snapshots
  or fit on the configured retrain cadence, but it must preserve the same
  no-lookahead contract.
- Existing V1 behavior remains unchanged when V2 seams/config blocks are absent.

