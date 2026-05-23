# ADR 0013 — Evidence-Strategy Layer Governance Closeouts

**Status:** Accepted (R1, R2, R3)
**Date:** 2026-05-22
**Context:** A comment-audit verification pass of
`src/regime_detection/_config_evidence_strategy.py` surfaced four un-ADRed
config knobs in the evidence-strategy layer (transition score, HMM,
change-point). Investigation determined three of them needed governance
sign-off without behavior change, and one was a live behavioral bug. This
ADR records all four resolutions in a single artifact so the file's
governance gaps are closed in one pass.

## Flag inventory (with investigation findings)

### R1 — `TransitionScoreConfig.cooldown_window_days` (LIVE BUG, fixed)

- **Defined:** `src/regime_detection/_config_evidence_strategy.py` (was
  `Field(default=3, ge=0)`; now `Field(default=5, ge=0)`).
- **Shipped:** `src/regime_detection/configs/core3-v2.0.0.yaml:278` was
  `cooldown_window_days: 3`; now `5`.
- **Consumed:** `src/regime_detection/transition_risk_series.py:160-164`
  via `build_transition_risk_history(cooldown_window_days=...)`, then at
  line 302: `days_since_axis_switch.le(cooldown_window_days)`.
- **Authoritative spec:** v1 §9.4 (`docs/regime_engine_v1_final_spec.md`
  lines 948-952): `` `post_switch_cooldown`: days_since_axis_switch <= 5
  ... The label fires across the entire 6-session window (days 0–5
  inclusive). ``
- **Legacy classifier:** `src/regime_detection/transition_risk.py:55-57`
  hardcodes `days_since_axis_switch <= 5` with a comment explicitly citing
  v1 §9.4.
- **The bug:** the series classifier was reading a config knob defaulted
  (in code AND YAML) to `3`, silently shortening the v1 5-day cooldown
  window to 3 days. The legacy non-series classifier kept the spec-correct
  5-day window via its hardcoded value. The two production paths emitted
  *different* `post_switch_cooldown` labels for days 4–5.
- **Test gap:** `tests/test_transition_risk.py:188` only exercises the
  hardcoded `<= 5` path. No test caught the series-path divergence.
- **Resolution:** tighten code default and YAML to `5` to match v1 §9.4.
  The legacy hardcoded `<= 5` in `transition_risk.py` is preserved as a
  second source of truth; a follow-up refactor to plumb the config knob
  through both paths is logged as a TODO below.

### R2 — `HMMConfig.model_version = "hmm_4state_v1.0"` (governance gap, ratified)

- **Defined:** `_config_evidence_strategy.py` HMMConfig.
- **Shipped:** YAML at `core3-v2.0.0.yaml:485` ships `n_states: 4`.
- **Spec text:** `docs/regime_engine_v2_spec.md` §6.1 line 4073-4074:
  `3 states (recommended): calm_bull, choppy_normal, stress_crash.
  Optionally 4 states (split bull into trending vs euphoric) once 3-state
  version validates.`
- **Validation evidence:**
  - `tests/test_hmm_state.py:85,139,293` — production runs all assert
    `n_states == 4`.
  - `docs/verification/hmm_state_label_map.yaml:24` — operator-shipped
    label artifact uses `n_states: 4`.
- **Authorization:** spec §6.1 line 4074 explicitly authorizes the
  4-state split contingent on 3-state validation. ADR 0013 R2 records
  that the validation work (test suite + committed label map) is complete
  and the 4-state choice is the V2 ship default.
- **Resolution:** annotate `HMMConfig.model_version` with an ADR 0013 R2
  pointer; no behavior change. Spec §6.1 example JSON at line 4106 still
  shows `hmm_3state_v1.0` as the canonical example — that stays as the
  illustrative recommendation, with the operator default re-pinned via
  this ADR.

### R3 — `ChangePointConfig.student_t_*` priors (library detail, ratified)

- **Defined:** `_config_evidence_strategy.py` ChangePointConfig fields
  `student_t_alpha = 0.1`, `student_t_beta = 0.01`, `student_t_kappa = 1.0`,
  `student_t_mu = 0.0`.
- **Shipped:** `core3-v2.0.0.yaml:537-540` matches the defaults.
- **Consumed:** `src/regime_detection/change_point.py:184-189` passes the
  values unchanged into the `bayesian_changepoint_detection` library's
  `_StudentT` prior.
- **Spec text:** `docs/regime_engine_v2_spec.md:4259` cites
  "BOCPD (Bayesian Online Change Point Detection, Adams & MacKay 2007)"
  and line 2122-2123 mentions "the canonical Adams-MacKay conjugate prior
  for Gaussian-with-unknown-mean-and-variance" but does **not** pin numeric
  priors.
- **Authorization:** ADR 0013 R3 ratifies these as the library convention
  defaults (Adams & MacKay 2007 §3.2 worked example) so calibration §9.1
  has an explicit baseline to retune from.
- **Resolution:** annotate the four fields with an ADR 0013 R3 pointer;
  no behavior change.

### R4 — `ChangePointConfig.score_window_days=5` / `break_threshold=0.5` (no action; already spec-resolved)

- **Defined:** `_config_evidence_strategy.py` ChangePointConfig.
- **Spec text:** `docs/regime_engine_v2_spec.md` **Implementation Ambiguity
  Log entries #64 (lines 2135-2150) and #65 (lines 2152-2164)** explicitly
  pin both numbers with rationale, including the calibration-exposure
  language: *"Window length exposed as `ChangePointConfig.score_window_days
  = 5` so calibration can retune without code changes."*
- **Resolution:** no ADR needed. The Ambiguity Log entries are the
  in-spec pointer of record. This ADR R4 only annotates the field
  comments to make the spec reference more discoverable from code.

## What changed under this ADR

| File | Change |
|---|---|
| `src/regime_detection/_config_evidence_strategy.py` (TransitionScoreConfig) | `cooldown_window_days` default `3` → `5`; comment cites ADR 0013 R1 + v1 §9.4. |
| `src/regime_detection/_config_evidence_strategy.py` (HMMConfig) | `model_version` field annotated with ADR 0013 R2 + spec §6.1 line 4074 pointer. |
| `src/regime_detection/_config_evidence_strategy.py` (ChangePointConfig) | `score_window_days` / `break_threshold` annotated with Ambiguity Log #64/#65 pointers (R4). `student_t_*` priors annotated with ADR 0013 R3 + Adams & MacKay 2007 §3.2 citation. |
| `src/regime_detection/configs/core3-v2.0.0.yaml:278` | `cooldown_window_days` `3` → `5` (R1). |

## Behavior impact

- **R1 (superseded by final-state refactor):** `cooldown_window_days`
  remains the source of truth for recent-switch detection, but the transition
  risk layer no longer emits a standalone `post_switch_cooldown` label.
  Cooldown is now a triggered rule that can produce final state `watch` when
  the pressure score is otherwise stable.
- **R2, R3, R4 (no change):** annotation only.

## Test plan / follow-ups

1. **Backfill a final-state cooldown test** asserting
   `cooldown_window_days=5` records `post_switch_cooldown` in
   `triggered_rules` and emits `watch` only when the score band is stable.
2. **Delete legacy classifier references** from any remaining tests/docs as
   part of the transition-risk final-state cleanup.
3. **Re-run the regime-detection profile** to capture the R1 coverage
   delta on `post_switch_cooldown` firings (days 4–5).
4. **Amend spec §6.1 line 4106 example JSON** to mention that 4-state is
   the V2 ship default per ADR 0013 R2 (the 3-state example can remain as
   the introductory illustration).

## Consequences

- R1 is a behavior change that affects backtest output starting at any
  axis switch. Capture the diff in fixtures before merging.
- R2 normalizes the gap between spec text and shipped artifacts; future
  audits will not flag the 4-state default as unauthorized.
- R3 closes the StudentT-prior gap; future calibration work can cite the
  defaults as the baseline.
- R4 is documentation-only and stops the next audit from re-discovering
  Ambiguity Log #64/#65.
