# Spec-Compliance Resolution — Design & Todo List

**Date:** 2026-05-31
**Source review:** `spec_review.md` (root) — 55 confirmed findings (1 critical · 8 high · 25 medium · 21 low), 5 ambiguities, regression delta 40 fixed / 6 partial / 1 still-open / 1 regressed.
**Scope:** verify and resolve every confirmed finding + the 5 ambiguities to an ideal, long-term standard. No patches, no shortcuts, no partial "done".
**Framing:** one engine, two phases — see `CLAUDE.md`. "V1" / "V2" are phases, not systems.

> Full per-finding evidence (spec quote, code lines, why-it-fails) lives in `spec_review.md`
> §2 / §2a, referenced here by `F-0xx`. This document is the **plan**, not a re-statement
> of the review — one home per concept (`AGENTS.md` rule B).

---

## 1. Locked decisions

| # | Decision | Resolution |
|---|---|---|
| Q1 | V1/V2 identity | **Resolved:** one engine, currently phase 2 (`__version__ = 2.0.0`). Phase-1 spec literals (`v1.0.0`, `core3-v1.0.0.yaml`) are stale examples; the binding rule is the package↔engine_version coupling test. |
| Q2 | Spec-doc edit authority | **Granted**, with **per-change sign-off** at execution time. Precedent: ADR 0020. |
| Q3 | Phase-1 fixture re-freeze | **(i) Justified regeneration** — see §4. |
| Q4 | Cadence | **Option A** — plan to disk now; execute later; **one PR per milestone** (M0…M6), each independently validated and CI-green before the next opens. |

---

## 2. Per-finding lifecycle (the unit of work)

Every finding passes all five gates. A finding is **not done** until gate 5 produces evidence.

1. **Reproduce (TDD red).** Write a test that encodes the *spec clause* — real production
   constants/enums, exact-value assertions (never `is not None`; a test that would still pass
   with the body replaced by a constant is worthless — `AGENTS.md` G). It MUST fail against
   current code first. For `test_gap` findings the red→green test *is* the deliverable.
2. **Root-cause fix (ideal, not patch).** Prefer **subtraction**: delete/consolidate, extend an
   existing module, fix a wrong data model rather than branch around it. A guard that adds >5
   lines triggers a "is this a symptom?" stop (`AGENTS.md` C). Rewrite where the design is wrong.
3. **Independent validator (fresh subagent, read-only).** A *different* agent — reusing the
   existing `spec-code-reviewer` / `backtesting-reviewer` agent types, **not** new infra — gets
   only the spec clause + the diff and is told to **refute**: matches the cited clause? root-cause
   or patch? preserves phase-1 byte-identity? is the test meaningful? Returns PASS / BLOCK + evidence.
   BLOCK ⇒ fix is not done; iterate.
4. **Verification-before-completion gate.** Targeted local checks during dev; the **authoritative**
   verification is the milestone PR's **GitHub Actions run green** (`AGENTS.md` 6), reported with
   the run URL + test summary. No "done"/"passing" claim without pasted evidence (`AGENTS.md` K).
5. **Milestone PR boundary.** `/code-review` on the whole milestone diff; stop and wait for
   "continue" (`AGENTS.md` 4). Next milestone opens only after this one is green and merged.

### Reuse / subtraction rules (apply to every fix)
- Grep for the concept before writing any function >15 lines (`AGENTS.md` 3). Reuse the existing
  leak-safe pattern rather than re-deriving (e.g. F-010 reuses the `reindex(..., method='ffill')`
  pattern already in `feature_store.py:344`).
- Delete-before-add bias. F-053 is a pure deletion (dead scalar path). F-013 is a one-line removal.
- One home per concept: a fix that needs a helper at a second call site **promotes** the helper out
  of `_private`, never duplicates it.

---

## 3. Independent validation & verification model

- **Independent validator** = a fresh agent per finding (gate 3), plus a milestone-level
  `/code-review`. Reuses existing agent types; produces a written PASS/BLOCK with cited evidence.
- **Verification authority** = GitHub Actions on the PR (`AGENTS.md` 6). Local runs are debugging
  only. "47/48 passing" is reported as ambiguous, never as "tests pass".
- **No self-certification.** The agent that wrote a fix does not get to declare it correct; gate 3 +
  CI do. This is the `verification-before-completion` discipline: evidence before assertions.

---

## 4. Phase-1 byte-identity & re-freeze policy (Q3 = justified regeneration)

Phase-1 ("V1") wire output is a frozen, byte-identical replay contract. A fix that changes a
phase-1 frozen output is allowed only when ALL hold:

1. The fix proves the **old** output was *wrong* per the cited spec clause (correctness, not preference).
2. The fixture is regenerated via the **committed generator** with a written justification under
   `docs/verification/` (date, finding id, old→new value, spec clause).
3. The `tests/_v1_frozen_models.py` shim is updated if the wire *shape* changes; `test_v1_frozen_replay`
   stays byte-identical thereafter.
4. **Explicit per-date sign-off** from the spec owner before merge.

Silent relaxation of any golden/fixture assertion to make a red test green is **banned**
(`AGENTS.md` review guidelines; P1). Re-freeze candidates are flagged per finding below; most M1
fixes are phase-2 and verified by the **V2** golden suite, not the V1 frozen fixtures.

---

## 5. Sequencing refinement — oracle-first (NEEDS YOUR SIGN-OFF)

The synthesis ordered capital-protection (M1) before golden-date integrity (M4). Planning surfaced a
bootstrapping problem: **a milestone can only be "verified green" if the regression oracle actually
asserts.** Today the V2 golden suite checks field *presence*, not *value* (F-008), and the §9.4
`sequence` assertion is skipped (F-031). So I recommend a small resequence:

- Pull **F-008** (golden value comparison) and **F-031** (sequence assertion) to the **front**, in
  **M0**, as the "definition of correct". The fixtures already carry **hand-labeled** expected values
  (spec-derived ground truth, e.g. `network_fragility: systemic_stress` on a crash date). Turning the
  comparison real will **correctly go red** exactly where the M1/M2 behavioral bugs live — those red
  golden assertions then become the acceptance criteria for M1/M2. This is the opposite of a shortcut:
  it makes the oracle trustworthy *before* we trust it.

Net change: M0 gains F-008 + F-031 (oracle integrity) alongside the doc reconciliations. The behavioral
milestones then have a real oracle. **If you prefer the original order, say so and I'll keep M4 intact.**

**Reviewer-driven refinements (independent review, 2026-06-01):**
- **Data-floor resolution (Q2 = replace, C).** Golden dates must have *complete* data for every field
  they assert. Any §9.4 date missing a required feed (credit/OAS before 2023-05-15 — `us_downgrade_2011`,
  `svb_2023`) is **swapped** for a different date that exercises the same behavior with all feeds present
  (for credit-stress: a post-2023-05 episode). Hand-label the replacements and amend §9.4. No
  `unknown`-watering and no skips — every golden assertion lands on data the engine fully has.
- **A green F-008 does NOT exempt M1.** "Goes red where bugs live" is a probability, not a guarantee:
  some M1 bugs change `score_components`, not top-level labels (e.g. F-002), so a label-only golden
  suite can miss them. M1 correctness is therefore verified by each finding's **dedicated property test**
  (the §9 G2 gate); F-008 is a *regression backstop*, not M1's sole oracle. A green M0 oracle does not
  cancel M1 execution.
- **Hard sequencing rule:** M1 does not open until the F-008/F-031 oracle is written and its per-date
  red/green status is recorded.

---

## 6. Milestone plan & todo

Legend — **Reuse/Sub:** `sub`=deletion/consolidation, `reuse`=extend existing, `doc`=spec/comment, `add`=new code (justify).
**RF?** = phase-1 re-freeze risk (per §4).

### M0 — Foundations: spec reconciliation + oracle integrity (no behavioral code change)

| ID | Sev | Ideal fix (distilled) | Test | RF? | Reuse/Sub |
|---|---|---|---|---|---|
| F-035 | low | Amend §2.4/§11: under the unified package, `engine_version = regime-engine-v<package-version>`; keep `test_version_coupling_*` as the binding check. No code change. | existing coupling test | no | doc |
| F-036 | low | Amend §2.4.1: default config follows the package major. No code change. | existing default-config test | no | doc |
| F-044 | low | Document in §11/§11.1 that null `data_quality.reason` is omitted via `exclude_none` (matches frozen fixtures). Verify against `v1_frozen_outputs/*.json` first. | byte-identity replay (unchanged) | no | doc |
| Amb #1 | — | Amend §7.1: inflation/growth & credit/funding are implemented as real phase-2 axes (cross-ref §2A/§2B); the "not_implemented_v1 placeholder" text is historical. Pin: assert V1 `structural_causal_state` key set == {event_calendar, monetary_pressure}. | new key-set assertion | no | doc+test |
| Amb #5 | — | Reconcile §11.1 monetary_pressure sketch to the §2A label-triple shape (code already follows §2A). | shape assertion on V2 wire | no | doc |
| **F-008** | high | **Per Q2:** first swap any data-incomplete §9.4 date for a complete-data date exercising the same behavior (credit-stress → a post-2023-05 episode); hand-label + amend §9.4. Then make `test_v2_golden_dates_classify_expected_fields` assert `dumped[field] == expected` (scalar axes; transition_risk_minimum via risk-ordering `>=`; transition_evidence via named-key membership). Becomes the M1/M2 regression backstop. | strengthen the test on complete-data dates (red on real bugs) | no | reuse |
| **F-031** | medium | Replace the `if field=="sequence": continue` with a real bull→narrowing_breadth→bear_stress ordering assertion on the 2018-10-10 window. | strengthen existing test | no | reuse |

### M1 — Capital-protection correctness (fail-closed routing, conservatism, cold-start)

| ID | Sev | Ideal fix (distilled) | Test | RF? | Reuse/Sub |
|---|---|---|---|---|---|
| F-005 | high | In `cohort_routing.py`, evaluate the core-axis-unknown fail-closed guard **before** the specialist walk (crisis_specialist may still pre-empt): crisis → data_outage-on-unknown → other specialists. **Depends on F-015 cohort-identity decision (pulled to M0).** | cohort test asserts the **blocked-modes invariant** (partial outage ⇒ leverage/short-vol blocked), not the cohort string | no (V2) | reuse |
| F-002 | high | **Q1 = surface (B).** In the emitted (2016+) path, a missing per-session model value ⇒ that date's transition risk is `insufficient_data` with a reason; **never** silently renormalize `model_instability` away (`transition_score.py:248-258`). Warmup (pre-2016) is not emitted ⇒ its degradation is moot; hard-raise stays reserved for the seam-not-wired build error (`transition_risk_series.py:153-162`, already enforced). Update `test_compose_transition_score_marks_model_instability_missing_on_cold_start` to assert emitted-path surfacing. | param test: seam ENABLED + a per-session value missing on an emitted date ⇒ `insufficient_data`, not a renormalized score; fails if reverted | no (V2) | reuse |
| F-004 | high | **Q4 = keep + authorize (B). → moved to M0 (doc-only).** Keep `cold_start_corr_to_one_enabled` default true; write a new ADR documenting the warmup fallback predicate (acts only in pre-2016 warmup) and cite it in §3.5. No behavior change. | ADR exists + §3.5 cross-ref; config default unchanged | no | doc |
| F-006 | high | In `_build_sentiment_score_series`, mask `sentiment_score` to NaN where <4 distinct AAII weekly readings exist at/before the session (running distinct-publication count). `evaluate_euphoria` already falsifies on NaN. | euphoria cannot fire on <4 readings; warm case unaffected | no (V2) | reuse |
| F-012 | medium | **Q4 = keep + authorize (B). → moved to M0 (doc-only).** Keep `cold_start_liquidity_gap_enabled` default true; cover it in the same new ADR (warmup-only fallback) and cite in §1E. No behavior change. | ADR + §1E cross-ref; config default unchanged | no | doc |
| F-013 | medium | **DECISION (sign-off):** remove `leverage_allowed = False` (line 139) so recovery_attempt mutates only the five §10.4 fields, OR reconcile §10.4 JSON to include it. | recovery_attempt modifier field-set == §10.4 | V1: **no** (verifier: branch not in V1 byte-contract); V2-golden: verify | sub |
| F-051 | low | **DECISION (sign-off):** amend §2C Unknown-Gate wording to the min-of-proxy-pairs reality, OR gate SOFR/IORB staleness independently post-2021. | funding-spread staleness gate matches ratified rule | no (V2) | doc/config |
| Amb #2 | — | **DECISION (sign-off):** `risk_off_mild` has no §2B operational rule yet outranks benign labels. Pin the exact predicate in §2B/an ADR, OR remove it from the precedence walker so it can't shadow goldilocks/recovery_growth. | precedence test: risk_off_mild only fires on the pinned predicate | no (V2) | doc/code |

### M2 — Output, archive & replay contracts (reproducibility foundation)

| ID | Sev | Ideal fix (distilled) | Test | RF? | Reuse/Sub |
|---|---|---|---|---|---|
| F-001 | **crit** | Parameterize `run_replay_check` with `db_name` (accept `regime_walkforward.db`) + thread archived macro; add a walk-forward replay driver that samples N as-of dates, recomputes from archived inputs, writes `reports/replay_verification.json` with `all_passed` + verified dates + engine/config_version; `_replay_gate_reasons` enumerates the dates and asserts version == frozen runs-table pair (mirror `_single_golden_gate_reasons`). | gate fails on a seeded replay mismatch; passes on match; version-mismatch fails | no | reuse |
| F-003 | high | Pass (as-of-sliced) `macro_series` to `write_archived_inputs` in `run_historical_walkforward.py` (mirror `run_shadow_regime.py:263-267`). | archived inputs include macro; replay of a macro-dependent label reproduces | no | reuse |
| F-019 | medium | Add `run_timestamp` to `_load_runs_from_db` SELECT + `_per_date_provenance` + summary CSV; embed run_timestamp/input_archive_path in the per-date JSON sidecar (or reference the canonical runs row by run_id). | per-artifact provenance present + asserted | no | reuse |
| F-042 | low | Add `isinstance(request.config, RegimeConfig)` guard in `classify_request` before `build_market_context` (raise TypeError). | non-RegimeConfig override fails loudly at boundary | no | reuse |

### M3 — Gate enforcement (promotion/shipping paths actually fire)

> **M3 status (2026-06-02): 7 of 8 findings shipped** — F-009+F-016 (baseline
> materiality), F-020 (label-contract NaN-leakage), F-017 (deadman interior-gap exit),
> F-039 (HMM drift WARNING), F-050 (crash-window crisis-label red flag), F-022 (§9.1
> gate offline-promotion scope, ADR 0023), F-018 (mid-window config-hash reset), F-047
> +F-052 (HMM drift decisions, ADR 0024), F-014 (reproducible §10 strategy-metrics
> report, ADR 0025). **F-007 DEFERRED** — see note below.

| ID | Sev | Ideal fix (distilled) | Test | RF? | Reuse/Sub |
|---|---|---|---|---|---|
| F-007 | high | **DEFERRED (2026-06-02).** Add a golden-runner that executes the 10 golden dates through `RegimeEngine` at the frozen config, pre+post batch, writing the `_single_golden_gate_reasons` shape; document as the required §7 step feeding `--golden-results`. **Why deferred:** a no-duplication producer must reuse the careful V1(<2020, core3-v1.0.0)/V2(≥2020, synthetic-kwargs) split golden classification in `tests/conftest.py:_classify_all_golden_rows`, which is built from session-scoped pytest fixtures (`raw_market_data`, `v2_market_df_for_asof`, `synthetic_v2_kwargs_for_market_data`, `event_calendar_df`). Reusing it without duplication requires extracting that pipeline into an importable module shared by conftest and the runner — a high-blast-radius refactor of the file the whole suite imports. Duplicating the pipeline into the runner instead would violate the code-reuse rule and risk golden-classification divergence. Needs a focused session with the full suite green as the safety net. The gate itself is sound (contract-validates pre/post shape, 10 dates, all_passed, frozen engine/config); the gap is the absent producer, so deferral does not regress any shipped behavior. | gate consumes real produced golden JSON; fails if a date regresses | no | add (wires existing engine) |
| F-009 | high | Test only — baseline worse on every metric ⇒ `status=='fail'`, `materially_worse_than_baseline` in reasons. | new failure-path test | no | reuse |
| F-016 | medium | Add per-metric materiality epsilon; ties (within ε) are non-improving and can't rescue an all-worse run; require ≥1 materially-improved dimension. | tie-on-one-metric still fails the gate | no | reuse |
| F-017 | medium | In `run_deadman_check`, when `qualification.qualifies` is False with a contiguity reason, return a non-ok status (`window_gap`) and make `main()` exit nonzero; optionally insert a breaking incident for the earliest gap. | interior gap ⇒ nonzero exit | no | reuse |
| F-018 | medium | Have `run_shadow_regime` record engine+config version (and config hash) and auto-insert a breaking incident / fail-fast when it differs from the version/hash that started the current window. | config change mid-window resets qualification | no | reuse |
| F-020 | medium | Add a label-contract check: success-row label columns must be valid labels ∪ {unknown, insufficient_history}; anything else ⇒ nan_leakage/contract_violation. | stray label string ⇒ gate fail | no | reuse |
| F-022 | medium | Wire `evaluate_v2_gate` into the walk-forward/promotion runner so a failing `GateResult` blocks promotion + logs metrics; OR document it as offline-only (it currently claims runner consumption). | failing GateResult blocks; passing allows | no | reuse |
| F-014 | medium | Add a reproducible shadow strategy-metrics report (return, max_dd, Sharpe, false-switch, detection lag, wrong-env-avoided) from the ledger + no-regime baseline; or document the external reproducible producer and link it. | metrics report reproducible from ledger | no | add |
| F-039 | low | **Expose part shipped (commit 55beb07: `HmmOutput.parameter_drift`).** Remaining: log a WARNING when `state_mean_drift_alert` / `transition_prob_review_flag` is True so the quarterly review receives the >20%/>30% alerts. | assert the WARNING fires via `caplog.at_level(WARNING)` / `pytest.warns` | no | reuse |
| F-047 | low | Document (module/ADR) that "prior version" = previous in-call refit checkpoint, OR load the persisted prior versioned artifact and compare. **DECISION.** | drift-source semantics pinned | no | doc/code |
| F-052 | low | Pin the §6.1 30%-transition-prob flag definition (absolute pp) in spec/ADR, OR switch to relative. **DECISION.** | definition pinned + tested | no | doc/code |
| F-050 | low | Add a configured crash-window date set (e.g. §9.4 dates) + a `crisis_label_missing_in_crash_window` red flag. | red flag fires when no crisis label in a crash window | no | reuse |

> **M4 status (2026-06-02): COMPLETE (12/12).** F-024 F-025 F-026 F-027 F-028 F-029
> F-030 F-032 F-033 F-034 F-054 F-055 all shipped + pushed.
> **M5 status (2026-06-02): COMPLETE (10/10).** F-010 F-011 F-021 F-037 F-038 F-040
> F-041 F-043 F-048 F-049 all shipped + pushed. F-010 RF-checked: V2-golden value-asserts
> unchanged, no re-freeze.
> **M6 status (2026-06-02): COMPLETE (7/7).** F-015 F-023 F-045 F-046 F-053 Amb#3 Amb#4
> all shipped + pushed.

### M4 — Golden-date & walk-forward test integrity

> F-008, F-031 moved to M0 (oracle-first, §5). Remaining here strengthen existing assertions /
> add failure-path coverage. Several are currently-correct behavior → tests go green; they harden
> the suite so later regressions are caught.

| ID | Sev | Ideal fix (distilled) | Test | RF? | Reuse/Sub |
|---|---|---|---|---|---|
| F-026 | medium | Cover `v2_dependency_payload_contracts`-only replay drift ⇒ `matches=False` caught by replay_mismatch (breaks_qualification stays 0). | new replay test | no | reuse |
| F-027 | medium | Drop an interior NYSE session ⇒ `missing_sessions` failure reason + dropped date listed. | new failure-path test | no | reuse |
| F-028 | medium | A `status=='failure'` row ⇒ `run_failures_present` + failure_count. | new failure-path test | no | reuse |
| F-029 | medium | Interior `status=='failure'` session ⇒ `qualifies` False, count, `non_success_run`. | new reset test | no | reuse |
| F-030 | medium | Summary CSV missing a required column ⇒ `missing_report_columns` + listed. | new failure-path test | no | reuse |
| F-024 | medium | Add the same-risk-rank `oas_confirmed` test (branch is correct on read). | new coverage test | no | reuse |
| F-025 | medium | `classify_series` over a multi-session window: every emitted date == pointwise `classify`. | extend trivial single-day test | no | reuse |
| F-032 | medium | Assert `trend_direction._RISK_RANK` == §3.6 dict (mirror volatility test). | new unit test | no | reuse |
| F-033 | medium | Compute `_compute_adx_14` on a known series and compare to inline `ewm(alpha=1/14,…)`, first 13 NaN. | new unit test | no | reuse |
| F-034 | medium | Cover §10.3 precedence when sideways_chop co-fires with a transition scenario; optionally order blocks by §10.3 rank with comments. | new precedence test | no | reuse |
| F-054 | low | Add modifiers_applied ordering coverage for a co-firing pair both ranked between default_neutral and crisis. | new ordering test | no | reuse |
| F-055 | low | Document the followthrough breakout_level 20d-preferred tie-break + a fixture locking it. | new fixture/test | no | doc+test |

### M5 — Schema, edge-cases & reporting

| ID | Sev | Ideal fix (distilled) | Test | RF? | Reuse/Sub |
|---|---|---|---|---|---|
| F-010 | medium | Use `reindex(spy_index, method='ffill')` for NFCI and first/latest CPI (mirror AAII/EPS leak-safe pattern). | non-NYSE-stamped obs still forward-fills | yes (V2 golden) | reuse |
| F-011 | medium | **Q5 = doc right, fix code.** Verified first-hand: §6.6 (line 700) and §6.8 (line 741) both pin `min_periods=50`; code `breadth_state.py:70` uses `63` and its comment (lines 65-67) *misquotes* §6.8 as "requires complete rolling windows". Fix code → `rolling(63, min_periods=50)` and correct the comment. No emitted-output change (full window by 2016). | assert first non-NaN at session index 49 (50th obs); pin against §6.8 | no | reuse |
| F-021 | medium | `_build_markdown`: add Data-Source/Archive-Policy, Incidents/Anomalies/Reruns (explicit empty), and transition-risk evidence sections (read the columns already written). | report contains §10 sections | no | reuse |
| F-037 | low | In `loaders._parse_window_days`, raise ValueError when `parsed[0] > parsed[1]` (mirror the `load_scheduled_events` check). | reversed window_days rejected | no | reuse |
| F-038 | low | Move `_validate_v2_request_input_contracts` ahead of `build_market_context` in `classify_request`. | missing V2 input raises the boundary error first | no | reuse |
| F-040 | low | Override the label triple on `NetworkFragilityOutput` to the closed `NetworkFragilityLabel` Literal (mirror sibling axes). | out-of-set label rejected by the model | no | reuse |
| F-041 | low | **DECISION:** declare `shadow_storage` temporal columns as `TIMESTAMP/DATE/BOOLEAN` to match §3 verbatim (SQLite keeps the ISO strings), OR soften the "verbatim implementation" docstring to note TEXT/ISO-8601 storage with identical round-trip. | round-trip semantics unchanged | no | doc/schema |
| F-043 | low | Document the V2-owned modifier names (transition_weakening etc.) + their precedence rank relative to the six §10.4 scenarios. **DECISION.** | modifiers_applied vocabulary fully specified | no | doc |
| F-049 | low | Add a clarifying comment that the V1 branch's state-machine result is intentionally discarded except state/evidence (NOT a deletion — it's load-bearing for byte-identity). | byte-identity replay (unchanged) | no | doc |
| F-048 | low | Document/justify the `event_calendar` profile-manifest requirement in §2.1, OR relax the gate to require non-empty provenance. **DECISION.** | gate matches documented contract | no | doc/code |

### M6 — Spec-set integrity, scaffolding scope, subtraction & docs

| ID | Sev | Ideal fix (distilled) | Test | RF? | Reuse/Sub |
|---|---|---|---|---|---|
| F-015 | medium | **Q3 = formalize (A). → decided at M0.** Amend spec §5.1 (+ cohort list / Ambiguity Log) to add `data_outage_specialist` as the official 10th cohort: trigger = any core risk axis `unknown`; precedence = pre-empts other specialists but `crisis_specialist` pre-empts it; its blocked trade modes. Code already emits it ⇒ now in-spec (F-005 depends on this). | property test: every emitted `active_cohort` ∈ the closed 10-cohort set | no | doc |
| F-023 | medium | Expand `V1_CONTRACT_GUARD_PATHS` to every V1-owned module (or invert to a V2-owned allowlist) + document the §14.3 scoping rationale. | a V2 import in any V1 file fails the hygiene test | no | reuse |
| F-045 | low | Update Ambiguity Log #9 to cite `axis_builders/network_fragility.build_network_fragility_axis_series` (current home of the KeyError/None logic). | — (doc) | no | doc |
| F-046 | low | Confirm `bayesian-changepoint-detection==0.2.dev1` is the only PyPI artifact with the Adams-MacKay impl; pin a stable release if one exists, else document why a `.dev` pin + add a hash-pinned lock. | dependency contract documented | no | doc/dep |
| F-053 | low | Make `build_rule_inputs_for_date` delegate to the vectorized `_rolling_*_series` reducers (single source of truth) **or delete it** (zero production callers); fix the stale `axis_builders/network_fragility.py:60` docstring. | parity test exact, not approx | no | **sub** |
| Amb #3 | — | **DECISION:** document the 63/126-session CPI offset as an approximation of the calendar-month offset, OR compute against the CPI obs exactly 3/6 calendar months prior. Pin with a synthetic-CPI test. | CPI-offset semantics pinned | no | doc/code |
| Amb #4 | — | **DECISION:** document the followthrough breakout_level 20d-preferred tie-break (or switch to strict max(m20,m50)). (Pairs with F-055.) | tie-break pinned + tested | no | doc/code |

### M7 — Replay-gate soundness (milestone `/code-review` findings, 2026-06-01)

> Surfaced by the M0-boundary `/code-review` (xhigh, 48-agent multi-angle) over the
> F-001 walk-forward replay producer + §6 gate. The gate is **sound on the happy path**
> (default config, no explicit PIT, non-empty batch, same host —
> `test_walkforward_replay_check` proves it) but unsound off it. CR-001…CR-007 are
> capital-protection-gate correctness; CR-008…CR-011 are robustness. Per `AGENTS.md`
> "never mark an issue acceptable", the docstring "out of scope" note (CR-004) is **not**
> a resolution. These are F-001 follow-ons (M2 surface) recorded as their own milestone.

| ID | Sev | Ideal fix (distilled) | Test | RF? | Reuse/Sub |
|---|---|---|---|---|---|
| CR-001 | high | The walk-forward *original* classifies from in-memory inputs while replay classifies from the reloaded archive — **not like-for-like**, so any archive round-trip lossiness reads as a regime regression. Mirror `run_shadow_regime.py:278-291`: reload the archived inputs and classify the ORIGINAL from the archive, making the §6 gate immune to round-trip noise. | replay of a macro-dependent date matches only when the original is archive-fed; a seeded round-trip delta is caught | no | reuse |
| CR-002 | high | Macro archive is **non-idempotent**: `_macro_series_frame` stores the post-`/100` `implied_vol_30d` WITH the `logical_name` column, so `load_archived_macro_series`→`loaders.load_macro_series` re-applies `/100` (100× too small). Make the round-trip idempotent (omit `logical_name` on archive, or archive raw points, or a transform-free archived reader). | `implied_vol_30d` archive→reload round-trips to identity | no | reuse |
| CR-003 | high | Replay drives the engine from `--config-path` (default→default config), **ignoring the per-run `config_version`** in the DB (a coarse Literal, not a content hash). Archive the resolved config (path + content hash); drive the replay engine from it; `_replay_gate_reasons` binds the config hash. | non-default-config batch replays faithfully; config-hash mismatch fails the gate | no | reuse |
| CR-004 | high | Explicit `--pit-constituent-intervals` is **never archived**; replay hardcodes `pit_intervals=None` (different membership), so explicit-PIT batches can never pass. Archive the PIT frame in `write_archived_inputs` and load it in replay. | explicit-PIT batch replays byte-identical | no | reuse |
| CR-005 | medium | Empty batch (zero success runs) → `all_passed = bool([]) and …` is False → gate emits `replay_mismatch_detected`, conflating "nothing to replay" with a real mismatch. Emit a distinct `no_successful_runs_to_replay` reason. | empty batch → distinct reason, not `replay_mismatch_detected` | no | reuse |
| CR-006 | medium | Replay reads DB-stored **absolute** `input_archive_path`/`output_path`; a copied archive (CI) → `FileNotFoundError`. Re-anchor to `output_root` (mirror `build_walkforward_report._nan_leakage:231-235`). | relocated/copied archive replays | no | reuse |
| CR-007 | medium | `_replay_one` dropped the shadow replay's explicit `output_path is not None` guard → opaque `TypeError` aborts the whole batch. Restore a clean per-run `ValueError` naming the date. | null `output_path` → ValueError, batch continues | no | reuse |
| CR-008 | low | F-006 warmup counts **raw AAII rows** via `searchsorted`, not distinct weekly publication dates — a duplicated publication row warms `sentiment_score` early. Count distinct publication dates before the `>=4` threshold. | a duplicated publication row does not warm before 4 distinct weeks | no | reuse |
| CR-009 | low | `build_v2_classify_kwargs` guards `v2_slice is None`, not **emptiness** → an as-of before the first v2_daily row builds full V2 kwargs and raises (status=failure) instead of a V1 fallback. Guard emptiness → V1 path. (Pre-existing; shadow runner shares it.) | early as-of → V1 fallback, not failure | no | reuse |
| CR-010 | low | `success_dates` is recomputed from `runs_df` at report-build time vs the producer's snapshot; a DB status flip between steps fires `replay_dates_mismatch`. Pin the "re-run the producer before building the report" ordering (or stamp+compare the producer's snapshot id). Two-sided: fail-closed may be intended. | ordering pinned + tested, or documented | no | doc/code |
| CR-011 | low | `_replay_one` reads `input_archive_path` with no null guard (symmetric to CR-007; schema is `NOT NULL`, so lower severity). Add the same per-run guard. | null `input_archive_path` → ValueError, batch continues | no | reuse |

---

## 7. Decisions deferred to execution (per-change sign-off)

**Decided this session (see the milestone rows):** Q1 F-002 (surface, 2016+), Q2 F-008 (replace
data-incomplete golden dates), Q3 F-015 (official 10th cohort), Q4 F-004/F-012 (keep + new ADR),
Q5 F-011 (fix code → `min_periods=50`).

Still deferred to their PR with a recommended option: **F-013** (leverage_allowed), **F-051**
(funding-spread gate), **F-043/F-047/F-048/F-052** (spec pins), **Amb #2/#3/#4**
(predicate/offset/tie-break). M0 spec edits (F-035/F-036/F-044, Amb #1/#5) also each get sign-off.

## 8. Definition of done

- **Per finding:** red→green test exists; root-cause fix (no patch); independent validator PASS;
  re-freeze (if any) justified + signed off; cited in the milestone PR.
- **Per milestone:** `/code-review` clean; GitHub Actions green (URL + summary pasted); spec docs
  updated where the resolution was a reconciliation; no finding left partial; **every finding-class
  backed by an enumerative gate (§9 G2) and every fix passed a codebase-wide pattern sweep (§9 G1).**
- **Program:** all 55 findings + 5 ambiguities closed; `spec_review.md` regenerated (or a delta
  appended) showing 0 confirmed open; no new regressions introduced (the F-007→F-011 lesson: every
  fix re-checks the spec *constant*, not just "add a guard").

## 9. Convergence & anti-recurrence (the anti-whack-a-mole contract)

**Why this section exists.** The first review found 48 issues; "all were resolved"; this review found
55 more. Forensics (see `spec_review.md` Appendix A + the round-1↔round-2 diff): of the 55, **1 is a
regression from a prior fix (F-007→F-011), ~7 are incompletely-closed prior findings, and ~47 are
pre-existing issues the first run's *sample* never drew** — not damage from the fixes. The lesson is
structural: **LLM review samples the defect space; it never enumerates it.** Re-review converges to
zero only if each fix removes a *class* of defects and installs a *deterministic gate* that makes the
class un-reintroducible. A fix that closes one finding without a gate is a re-discovery waiting to
happen — proven by F-011 (a fix that added `min_periods` but with the wrong constant, then
self-certified as resolved).

Three hard exit gates are therefore added to every milestone, on top of §8:

**G1 — Pattern sweep, not point fix.** Before a finding's fix is "done", grep the whole codebase for
the *pattern* and fix every instance in the same PR. The PR description states "swept N sites, fixed M";
a silent single-site fix is rejected at gate 3. Required sweeps include:
- F-010 (leak-unsafe `reindex` without `method='ffill'`) → every series→session alignment in `src/`, not just NFCI/CPI.
- F-004 / F-012 (undocumented cold-start fallback branch) → every axis rule path; each fallback gets a spec/ADR anchor or goes default-off.
- F-011 (wrong rolling-window constant) → every `rolling(...)` / `min_periods` / window constant checked against its spec line.
- F-040 (bare-`str` axis label) → every axis output model; any not pinned to a closed `Literal` is fixed.
- F-002 (per-session model-evidence handling) → sweep every call site of `compose_transition_score_for_session` / `compute_transition_score`; all must share the resolved missing-evidence policy.
- F-005 (specialist-before-fail-closed) → sweep every `AgentRouting` construction / routing path for unknown-axis handling ahead of specialist matching.

**G2 — An enumerative gate per finding-class.** A milestone does NOT close until each of its finding
classes is backed by a test that **fails on re-introduction** (enumeration), not one that merely
passes today (sampling):
- Behavioral correctness (M1/M5 axis fixes) → the **value-asserting golden suite** (F-008) covers the changed dates; a regressed label fails CI.
- Closed-set membership → **property tests** asserted over the full enumeration: every emitted axis label ∈ its `Literal` (F-040), every cohort ∈ §5.1 (F-015), every strategy modifier ∈ §10.4 (F-043), every gate label ∈ {valid} ∪ {unknown, insufficient_history} (F-020).
- Spec constants → **constant-pinning assertions** against the spec line (min_periods=50, §4.3 weights, the 252/504/63/126 windows) so editing the constant fails CI (the F-011 lesson).
- Gate enforcement (M3) → a **failure-path test per gate condition** (F-009/F-016/F-017/F-020/F-027/F-028/F-029/F-030): seed the failing condition, assert the gate returns fail. A gate with no red-path test is treated as unenforced.
- Reproducibility (M2) → a **replay round-trip test** (F-001/F-003): recompute a sampled date from archived inputs and assert byte-identity; a missing archived input fails it. The F-003 test MUST use a **macro-dependent** date and assert `macro_series.parquet` is present in the archive dir, else a missing-macro regression passes silently.
- Mandatory model-evidence (F-002) → a **parameterized policy test**: with the HMM/clustering seam ENABLED, the *signed-off* missing-evidence behavior holds (seam-absent ⇒ raise, already tested; per-session-missing ⇒ the Q1 decision), and the test fails if the policy is reverted. Exact assertion pinned by decision Q1.
- Fail-closed routing (F-005) → the cohort-routing regression test asserts the **blocked-modes invariant** (partial outage ⇒ leverage/short-vol blocked, crisis still pre-empts), NOT a cohort *string* (F-015 may rename it).

**G3 — No self-certification.** The agent that wrote the fix never marks it done. The independent
validator (gate 3) must confirm (a) the spec *constant/clause* is met — not just "a guard exists",
(b) the G1 sweep covered all sites, and (c) **the G2 gate actually goes red when the fix is reverted**
(the validator reverts the fix in a scratch worktree and watches the new test fail). A test that does
not fail without the fix is worthless (`AGENTS.md` G) — this check is the teeth. For a multi-file PR,
"revert the fix" = `git checkout HEAD~1 -- <the PR's changed *source* files>` (source only; keep the
new test), run the new test, observe red, restore; the validator records the red output as evidence.

**Convergence metric (the program-level exit signal).** After M0–M6, re-run the regression workers +
a fresh sample audit and record:
- prior-fix regression rate (round-1→2 baseline: 1/48 ≈ 2%) → target **0**;
- net-new findings *in already-fixed areas* → target **0** (new findings allowed only in spec surface not yet gated);
- % of confirmed findings backed by an enumerative gate → target **100**.

The program is **converged** when a fresh review finds nothing in any area that has an enumerative gate
— the only way to surface a new issue is in un-gated spec surface, which the matrix coverage bounds.
At that point CI is the safety net and re-review is only a backstop. Because the spec corpus is finite
and round-2's 643 matrix rows are near-saturation, net-new spec surface is small; once each
finding-class has a CI gate, a regression cannot survive to a review, so expected net-new in gated
areas → ~0 and any residual is confined to the shrinking un-gated remainder.

## 10. Execution order

M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7, one PR each, stop-and-confirm at every boundary. M0 first
because it makes the oracle real and retires stale spec text, so every later milestone is verified
against a trustworthy suite. M7 (replay-gate soundness) was opened by the M0-boundary `/code-review`
and folds the F-001 follow-ons in after M2's contracts land. No milestone closes until its §9 G1–G3
gates are satisfied.

**Status (2026-06-02):** M0 ✅ + M1 ✅ + M2 ✅ + M7 ✅ complete. M2 closed F-042
(config-type boundary guard) + F-019 (per-artifact provenance) on the F-001/F-003 base.
M7 closed all of CR-001…CR-011 (classify-from-archive like-for-like; idempotent macro
round-trip; frozen-config + PIT archival; replay-producer robustness; distinct-week
warmup; empty-slice V1 fallback) — CR-001 gate-3 validated. PR #71 also addressed the 7
cubic review comments (incl. GNXfc pit-survivorship). Order taken: M0 → M1 → M2 → M7 →
**M3** (gate enforcement) next, then M4–M6, working autonomously.
