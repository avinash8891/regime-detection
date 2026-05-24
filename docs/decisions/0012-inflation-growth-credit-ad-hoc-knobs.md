# ADR 0012 — Disposition of un-ADRed Inflation/Growth Knobs (Resolution: R2)

**Status:** Accepted (R2)
**Date:** 2026-05-22
**Resolution:** R2 — ratify Knob A (`cpi_3m_acceleration_threshold`) with a
backing test; tighten Knob B (`spy_recession_credit_confirmed_threshold`)
default back to `-0.05` to match spec, deferring the mild-decline coverage
gap to a future `credit_watch` label ADR.
**Context:** A comment-audit of `src/regime_detection/_config_layer2.py` surfaced
two config knobs that are live in production, consumed by the §2B classifier,
and change classifier output relative to the spec — but were not ratified by
ADR 0011 ("Inflation/Growth Rule Coverage Fix") or any subsequent decision.
This ADR forces the disposition decision (ratify or remove) and records the
rationale either way.

## Problem

ADR 0011 ratified three §2B knobs (`disinflation_yield_independent`,
`cpi_goldilocks_benign_ceiling`, `allow_credit_independent_fallback` +
`spy_recession_credit_independent_threshold`). Two further knobs in the same
config — both consumed by `inflation_growth_rules.evaluate_*` — have no
governance trail:

### Knob A — `cpi_3m_acceleration_threshold`

- **Defined:** `src/regime_detection/_config_layer2.py:141`
  (`Field(default=0.02, gt=0.0)`).
- **Shipped:** `src/regime_detection/configs/core3-v2.0.0.yaml:423` (`0.02`).
- **Consumed:** `src/regime_detection/inflation_growth_rules.py:93-100`
  (Limb 3 of `evaluate_inflation_shock`).
- **Behavior with default:** Adds a third OR-limb to `inflation_shock`:
  `cpi_3m_change_pct > 0.02 AND treasury_10y_yield_slope_21d > 0`.
- **Spec text (regime_engine_v2_spec.md:3045-3050):** lists only TWO
  OR-limbs (`inflation_surprise_zscore` and the commodity composite).
- **Test coverage:** none — no test in `tests/` exercises the rapid-onset limb.

### Knob B — `spy_recession_credit_confirmed_threshold`

- **Defined:** `src/regime_detection/_config_layer2.py:148`
  (`Field(default=-0.03, lt=0.0)`).
- **Shipped:** `src/regime_detection/configs/core3-v2.0.0.yaml:426` (`-0.03`).
- **Consumed:**
  - `src/regime_detection/inflation_growth_rules.py:150` in
    `evaluate_recession_scare` (credit-confirmed branch).
  - `src/regime_detection/inflation_growth_rules.py:212` in
    `evaluate_risk_off_mild`.
- **Behavior with default:** When credit is in `{spread_widening, credit_stress}`,
  `recession_scare` fires on `spy_21d_return < -0.03` instead of the spec's
  `-0.05` — i.e. a milder equity decline qualifies.
- **Spec text (regime_engine_v2_spec.md:3057-3061):** single SPY threshold
  of `-0.05`.
- **ADR 0011 Remaining Gaps section line 70-72** explicitly identifies this
  scenario as **unresolved**: *"spread_widening + mild equity decline …
  Would need a relaxed recession_scare or a new 'credit_watch' label."*
  The code shipped the relaxed-recession_scare option without an ADR
  recording the choice.

## Decision space

| Option | Knob A | Knob B | Impact |
|---|---|---|---|
| **R1 — Ratify both** | Keep with current defaults | Keep with current default `-0.03` | No behavior change; spec & ADR catalog updated to match code. |
| **R2 — Ratify A, remove B** | Keep | Tighten default to `-0.05`; route mild-equity-decline case to `risk_off_mild` or new `credit_watch` label | Recession_scare matches spec verbatim. Mild-decline coverage requires the new-label path ADR 0011 flagged. |
| **R3 — Remove both** | Strip Limb 3 from `evaluate_inflation_shock`; remove knob | Same as R2 for B | Classifier matches spec verbatim. May regress `inflation_shock` and `recession_scare` coverage on historical runs. |

## Recommendation

**R2** — ratify Knob A (`cpi_3m_acceleration_threshold`) with the addition of
a backing test that exercises Limb 3, and tighten Knob B's default to `-0.05`
to match the spec, deferring the mild-equity-decline coverage gap to a
follow-up ADR (e.g. a `credit_watch` label).

Rationale:

- Knob A fills a real signal gap: a sharp 3-month CPI acceleration with rising
  yields is a textbook inflation_shock onset that the commodity-composite limb
  misses. But shipping it without a test is a gap — the unit-test backfill
  is one PR away.
- Knob B is exactly the scenario ADR 0011 deferred. Continuing to ship
  `-0.03` without an ADR amendment normalizes the practice of code overriding
  governance. Tightening to `-0.05` and forcing the mild-decline question
  through the proper label-design path (credit_watch) keeps governance ahead
  of code.

## Implementation if R2 is accepted

1. Add `tests/test_inflation_growth.py::test_inflation_shock_rapid_onset_limb`
   asserting Limb 3 fires when `cpi_3m_change_pct > 0.02 AND
   treasury_10y_yield_slope_21d > 0`.
2. Change `_config_layer2.py:148` default to `-0.05` and update the comment.
3. Change `core3-v2.0.0.yaml:426` to `-0.05` (or remove the override).
4. Add `#### Inflation Shock Rapid-Onset Limb` subsection in
   `regime_engine_v2_spec.md` §2B describing Limb 3, with a pointer to this
   ADR.
5. Re-run the regime-detection profile to measure coverage delta from
   tightening Knob B and capture the result in this ADR's `## Impact` section.

## Consequences

- **If R1 (ratify both):** any future divergence between code and spec
  re-affirms the precedent that code can run ahead of governance. Should be
  weighed against the cost of process discipline.
- **If R2:** one production-shipped config value (`-0.03` → `-0.05`) changes,
  which may shift historical `recession_scare` firings. Capture the diff in
  fixtures.
- **If R3:** larger behavior shift; recommend a coverage-impact backtest
  before merging.

## Resolution log (2026-05-22)

R2 was selected and implemented in the same audit pass that surfaced the
governance gap:

| Action | Location | Change |
|---|---|---|
| Backing test for Knob A | `tests/test_inflation_growth.py` | Added `test_inflation_shock_rapid_onset_limb_fires`, `…_silent_at_threshold`, `…_silent_when_yields_flat` (3 tests after the existing single-signal-limb suite). |
| Knob B default tightened | `src/regime_detection/_config_layer2.py:148` | `Field(default=-0.05, lt=0.0)` (was `-0.03`); comment rewritten to cite ADR 0012 R2 + the known credit_watch gap. |
| Knob B YAML tightened | `src/regime_detection/configs/core3-v2.0.0.yaml:426` | `spy_recession_credit_confirmed_threshold: -0.05` (was `-0.03`); comment updated. |
| Spec §2B `inflation_shock` rule body | `docs/regime_engine_v2_spec.md` (post-edit) | Added the third OR-limb verbatim with `# ADR 0012 Fix A` pointer. |
| Spec §2B `recession_scare` rule body | `docs/regime_engine_v2_spec.md` (post-edit) | Removed the "Disposition pending" placeholder; added inline note that the spread_widening + mild-equity-decline gap stays open pending a future credit_watch ADR. |
| Knob A code comment | `src/regime_detection/_config_layer2.py:139-145` | Replaced "Disposition pending" placeholder with the ratified rationale + test pointer. |

## Open follow-up (out of scope for this ADR)

The 338-session spread_widening + mild-equity-decline coverage gap remains
unresolved. A future ADR (e.g. 0013) should propose the `credit_watch` label
that ADR 0011 Remaining Gaps line 70-72 sketched. Until then,
`spy_recession_credit_confirmed_threshold = -0.05` is the strict spec
default and those sessions stay `unknown`.

## Consequences (measured)

- **Knob A:** No behavior change — code already shipped `0.02`; this ADR
  only formalizes governance and adds test coverage.
- **Knob B:** Behavior change. With the default tightened from `-0.03` to
  `-0.05`, `recession_scare` no longer fires on the `[-0.05, -0.03)` SPY
  band when credit is confirmed stressed. Sessions in that band now stay
  in `unknown` until a future `credit_watch` label is introduced. The
  coverage delta has NOT been measured in this audit pass; the next
  regime-detection profile run should capture it.
