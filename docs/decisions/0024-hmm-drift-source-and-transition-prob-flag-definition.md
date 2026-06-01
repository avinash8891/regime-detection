# 0024 — HMM §6.1 drift source and transition-prob flag definition

- Status: accepted
- Date: 2026-06-02
- Findings: F-047, F-052 (spec_review.md), milestone M3
- Code: `src/regime_detection/hmm_state.py`
  (`HMMParameterDrift`, `compute_hmm_features`, `compute_hmm_parameter_drift`)

The V2 spec §6.1 (lines 4434-4468) defines an operator-side HMM calibration-drift
review. Two operational details were left ambiguous by the spec text; this ADR pins
them to the shipped behavior and the tests that lock it.

## F-047 — what "prior version" means

**Decision: "prior version" = the immediately preceding in-call PIT refit checkpoint.**

`compute_hmm_features` performs PIT-monotonic refits at each retrain-cadence
checkpoint within a single call. The drift report compares each refit's
de-standardized state means and transition matrix to the **previous checkpoint in
the same call** (`previous_raw_means` / `previous_transmat`), not to a persisted
artifact from an earlier process run. Consequences, pinned by tests:

- With a single checkpoint there is no prior, so `parameter_drift` is `None`
  (`test_compute_hmm_features_parameter_drift_is_none_with_single_checkpoint`).
- With multiple checkpoints the report is well-formed and finite, aligned across
  all states
  (`test_compute_hmm_features_reports_parameter_drift_across_refit_checkpoints`).

We chose in-call checkpoint comparison over loading a persisted prior artifact
because the engine is a pure, replayable classifier: drift is computed from the same
inputs every replay, with no cross-process state to version or reconcile. A persisted
prior would couple the deterministic classifier to an external artifact store and
break byte-identical replay. If a future operational need requires cross-refit-cycle
drift across process runs, that belongs in the shadow/operator layer, not the engine.

## F-052 — the 30% transition-prob flag is an ABSOLUTE move

**Decision: `transition_prob_review_flag` fires on a maximum ABSOLUTE aligned
transition-probability change > 0.30 (a "30 percentage point" move), not a relative
change.**

Transition-matrix entries are bounded in `[0, 1]`. A relative threshold (`|Δp|/p`)
explodes for near-zero probabilities — a 0.001 → 0.002 move is +100% relative but
operationally meaningless — making the flag unstable and noise-dominated. The
absolute reading is the only stable one across the full probability range. This is a
review-only flag and does not block deployment (state-mean drift is the blocking
signal). Pinned by `test_hmm_parameter_drift_transition_review_flag_is_independent`
(a 0.38 absolute shift → `max_transition_prob_shift == 0.38`, flag True) and
`test_hmm_parameter_drift_below_thresholds_raises_no_alert`.

## Consequences

- The `HMMParameterDrift` docstring states both decisions inline and cites this ADR.
- F-039 (the drift WARNING) emits on either flag; these definitions fix what that
  WARNING means.
- Revisit only if the spec §6.1 text is amended to mandate persisted-artifact drift
  or a relative transition-prob threshold.
