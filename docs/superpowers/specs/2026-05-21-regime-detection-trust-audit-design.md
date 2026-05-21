# Regime-Detection Trust Audit — Design

**Status**: draft pending Owner sign-off on open items (α)/(γ).
**Source**: consensus reached in Envoy room `room_1779305260690cfc3lp`, cursors [19]–[30].
**Reviewers**: Claude (proposer), Codex (reviewer).

## Premise

The regime-detection codebase has accumulated trust-deficit signals: drift between docs and code, data fetched not matching what docs claim, dead code, stubs labeled as implemented, logic that is wired but never consumed. The audit method here treats the **runner as the executable truth**. Docs are not an oracle until proven against the live path. Every finding is recorded in a **structured ledger** with file:line-level evidence — never prose.

## Audit method — final order

Steps run sequentially. Step −1 is a fast preflight; the binding validity pilot moves to immediately after Step 1 with fresh findings.

### Step −1 — Enum smoke test (30 minutes)

Sanity-check the ledger enum (see schema below) by classifying 3–5 obviously-distinct seed findings drawn from `docs/spec_code_data_audit_2026_05_15.md:104-155` and ADR `docs/decisions/0010-per-label-hysteresis-and-audit-hardening.md:8-20`. Goal: detect obvious enum collapse before any real step runs. If the smoke test exposes a missing class or a one-to-one synonym, fix the enum first.

### Step 1 — Manifest+runner spine audit (first reviewer gate)

**Mode: runtime, non-semantic.** Step 1 is *not* static reachability; it is a manifest spine *trace*. The audit runs Step 1 across the **required runner set** (defined below). For each runner:

1. Load the manifest.
2. Materialize (or verify) the artifacts required by the runner.
3. Resolve runner input paths via `src/regime_data_fetch/manifest_inputs.py:243-319` when the runner uses the manifest resolver; otherwise record the bypass class (see Step 2a).
4. Record actual artifact keys, sha256 hashes, destination paths, `resolved_from_manifest`, and `cli_overrides`.

Step 1 does **not** assert expected classifications. It establishes which inputs the runner actually consumes, and which provenance bypass markers fire. Static reading alone is insufficient because resolver behavior depends on manifest contents and CLI override sets. Full golden replay is the wrong level here because it mixes input-routing bugs with classifier-behavior bugs; Step 1 isolates the first class.

**Required runner set** (Step 1 audit must cover all of these — a single-runner pass is incomplete):

| Runner | Path | Manifest behavior |
|---|---|---|
| Profile engine | `scripts/profile_engine.py` | `materialize_manifest_from_args` + resolver |
| Layer-2 30-day audit | `scripts/audit_layer2_30d.py` | `materialize_manifest_from_args` + resolver |
| V2 calibration | `scripts/run_v2_calibration.py` | `materialize_manifest_from_args` + resolver |
| V2 walk-forward gate | `scripts/run_v2_walkforward_gate.py` | `materialize_manifest_from_args` + resolver |
| Materialize-only | `scripts/materialize_regime_data.py` | `materialize_manifest` direct |
| V1 shadow regime | `scripts/run_shadow_regime.py` | runner-level bypass (no materialize) |
| V1 historical walkforward | `scripts/run_historical_walkforward.py` | materializes but does not bind via resolver |
| Walkforward report builder | `scripts/build_walkforward_report.py` | runner-level bypass |
| Group B candidate approval | `scripts/approve_group_b_candidate.py` | runner-level bypass |

**Implementability constraint** (UNINSTRUMENTED today): `scripts/_v2_calibration_helpers.py:160-172` calls `materialize_if_requested(...)` but **discards** the returned `MaterializedArtifact` list. The helper at `:145-157` records only `args.manifest_resolved_inputs` and `args.manifest_cli_overrides` on the namespace — no sha256, no destination path. Step 1 therefore requires a small audit-harness emission task before it can produce a complete trace: capture the discarded `MaterializedArtifact` list and the bypass markers into a structured provenance JSON per run. This harness is scoped, non-semantic, and is a Step 1 precondition — not classifier work.

**Cross-worktree scope.** Step 1 records cross-worktree provenance because a pass in one local worktree does not prove that a sibling checkout's runner consumes the same code or data path. The current local inventory is:

- `/Users/avinashvankadaru/conductor/repos/regime-detection` (`git rev-parse HEAD`: `a85ad6701d773615a509d7b5056ee12e4936c3cb`)
- `/Users/avinashvankadaru/conductor/workspaces/regime-detection/manila-v2` (`git rev-parse HEAD`: `b42e4cfbcee6998f138143d20e4009cbc99fa9c2`)

The requested shell inventory command, `ls -d /Users/avinashvankadaru/conductor/workspaces/regime-detection-*`, produced `zsh:1: no matches found: /Users/avinashvankadaru/conductor/workspaces/regime-detection-*`; `git worktree list --porcelain` is therefore the authoritative local worktree inventory for this run. If additional conductor worktrees matching `/Users/avinashvankadaru/conductor/workspaces/regime-detection-*` exist in a future environment, they are automatically in scope and must be enumerated in the same provenance bundle.

Detection method: for each in-scope worktree, Step 1 records `git rev-parse HEAD` plus file-content `sha256` for every tracked source file under `src/regime_detection/` and `src/regime_data_fetch/`. Differences across worktrees in the same logical file are recorded as `BROKEN_WIRING × MISMATCH` findings with both source paths cited. Example drift already present locally: `/Users/avinashvankadaru/conductor/repos/regime-detection/src/regime_detection/config.py:13-170` defines V2 config classes inline, while `/Users/avinashvankadaru/conductor/workspaces/regime-detection/manila-v2/src/regime_detection/config.py:12-56` imports split config modules and `/Users/avinashvankadaru/conductor/workspaces/regime-detection/manila-v2/src/regime_detection/_config_layer2.py:205-233` contains the Inflation/Growth config that has no same-path counterpart in the main checkout. Default policy: any divergence in `src/regime_detection/*` is `BLOCKING` pending Owner classification of intentional experiment versus accidental drift.

### Step 2a — Manifest/materialization provenance bundle

**Non-semantic provenance only.** This step does *not* emit new classifier explanations or predicate IDs. It emits fields that exist outside the classifier path today.

Required fields per run:

| Field | Source | Bypass marker? |
|---|---|---|
| Manifest artifact `name` | `src/regime_data_fetch/artifact_manifest.py:37-48` | — |
| Manifest artifact `uri` | same | — |
| Manifest artifact `local_path` | same | — |
| Manifest artifact `sha256` | same; validated at `:102-115` | — |
| Manifest artifact `schema_version` | same | — |
| Manifest artifact `rows` | same | — |
| Manifest artifact `min_date` / `max_date` | same | — |
| Manifest artifact `required_for` (field) | `src/regime_data_fetch/artifact_manifest.py:37-48` | — |
| Manifest artifact `required_for` (filter method) | `src/regime_data_fetch/artifact_manifest.py:177-180` | — |
| Resolver use of `required_for(runner_name)` | `src/regime_data_fetch/manifest_inputs.py:253-254` | — |
| Manifest-level `artifact_set` / `created_at_utc` / `storage_root` | `src/regime_data_fetch/artifact_manifest.py:118-123` | — |
| `resolved_from_manifest` (per field) | `src/regime_data_fetch/manifest_inputs.py:317` | — |
| `cli_overrides` (per field) | `src/regime_data_fetch/manifest_inputs.py:318` | **per-field bypass marker** |
| `manifest_path_provided` (bool) | runner entry; see `src/regime_data_fetch/materialization.py:150-151` | **whole-manifest bypass marker** |
| `materialize_called_by_runner` (bool) | runner-level instrumentation (new, non-semantic) | **runner-level bypass marker** |
| Materialization sha verification result | `src/regime_data_fetch/materialization.py:61-68` (via `expected_sha256=artifact.sha256` on `store.get_file`) | — |
| Store-level sha mismatch failure | `src/regime_data_fetch/artifact_store.py:43-48`, `:151-156` | — |

**Bypass paths confirmed in the codebase (see Step 1 evidence):**

1. **Whole-manifest bypass.** `materialize_if_requested(manifest_path=None)` at `src/regime_data_fetch/materialization.py:150-151` returns `[]` and skips sha verification entirely.
2. **Per-field CLI override.** `resolve_runner_input_paths` at `src/regime_data_fetch/manifest_inputs.py:269-289` accepts CLI values that bypass manifest resolution per field.
3. **Runner-level bypass.** Three runners read source artifacts without calling `materialize_manifest_from_args`: `scripts/run_shadow_regime.py`, `scripts/build_walkforward_report.py`, `scripts/approve_group_b_candidate.py`. These runners have **no provenance** today.
4. **Materialize-but-don't-bind bypass.** `scripts/run_historical_walkforward.py:386-401` calls `materialize_if_requested(...)` (so sha verifies if `--manifest` is provided) but then passes the **original CLI paths** (`args.market_data`, `args.event_calendar`) into `run_walkforward(...)` rather than the manifest-resolved paths. The artifact set is verified; the runner doesn't bind to it. This is distinct from class 3 because materialization still runs.
5. **Cross-worktree divergence.** The same logical feature has been shipped to multiple conductor worktrees with no visible merge point. The local drift example above shows config schema code split in `manila-v2` while the main checkout retains the older inline layout; Step 1 in one worktree cannot prove correctness for runs originating in any other worktree.

The runner-level bypass marker (`materialize_called_by_runner`) is the only new instrumentation introduced by Step 2a. It is non-semantic: a single boolean recorded at runner entry.

**Classifier self-reports are explicitly excluded.** Fields like `classification_status`, `classification_reason`, `data_quality`, `reporting_label`, `source_used`, and `rule_evidence` (summarized at `scripts/audit_layer2_30d.py:204-252`) are **semantic breadcrumbs**, not provenance — they are produced by the system we are auditing and may be silently wrong.

### Step 2b — Golden-run differential replay

Run the engine against pinned dates/fixtures/manifests. Assert against the provenance bundle from 2a *plus* final-label equivalence. Distinct from Step 1: 2b catches classifier-behavior drift on the verified spine. Step 2b assumes Step 1 has already established that the spine used the intended inputs.

### Step 3 — Doc-anchored claim audit

Walk every design doc. Extract each factual claim (formula, data source, entry condition, manifest path). Verify against the **verified live path** from Step 1 — not against the code directly. This distinguishes three failure classes: (i) doc and live path agree; (ii) doc claims path A, code implements B, but B is live (code violates spec); (iii) doc claims path A, code implements both A and B, and the runner uses B (doc describes a path the runner never uses — orphan spec).

### Step 4 — Reachability + coverage sweep

**Hypothesis-only.** Step 4 produces *candidate* orphans, dead code, and test-only-reachable code via static call-graph + coverage tools. It has **no deletion authority**. Every candidate must be confirmed by:

- Grep for dynamic dispatch patterns (registry decorators, `importlib.import_module`, `__init_subclass__`, entry_points).
- Manual check that the symbol is not referenced indirectly via `MANIFEST_INPUT_SPECS` (`src/regime_data_fetch/manifest_inputs.py:95-107`), `_FEATURE_STORE_BUILDERS` (`src/regime_detection/feature_store.py:679-698`), or `register_manifest_input_args` (`scripts/profile_engine.py:397-401`, `scripts/audit_layer2_30d.py:379-383`).

Static reachability cannot be the live-set filter for Step 1, because registry-dispatched code is statically unreachable. Step 4 runs *after* the live set is known from Step 1.

## Ledger schema

Every audit finding is one row. The schema is **two-axis** — failure domain × mechanism — because the single-axis 11-value enum is not mutually exclusive against Owner's stated trust failure classes (one Owner class can map to multiple mechanism values; see review B4).

| Field | Type | Notes |
|---|---|---|
| `path` | string | `file:line`, runner artifact path, manifest key, or command-output excerpt. **Never prose.** |
| `claim_or_invariant` | string | the spec claim or invariant being checked |
| `observed_evidence` | string | what the audit actually observed |
| `failure_domain` | enum (below) | Owner-facing trust class — exactly one |
| `mechanism` | enum (below) | how it manifests in code — exactly one |
| `severity` | enum (below) | `BLOCKING` / `TRUST_GAP` / `INFO` |
| `owner_decision_needed` | string \| null | what Owner must decide before resolution |

**Failure-domain enum (4 values — maps to Owner's stated classes):**

```
DEAD_CODE_DOMAIN         — code that exists but contributes nothing live
WRONG_CODE               — code that runs but disagrees with intended behavior
UNIMPLEMENTED_LOGIC      — behavior claimed but not implemented
BROKEN_WIRING            — components present but coordination is wrong (no caller, wrong call, lost output)
```

**Mechanism enum (11 values — how the failure manifests):**

```
MATCHES_SPEC          — code agrees with doc/spec (used for negative findings during pilot)
MISMATCH              — code and doc disagree on a specific behavior
MISSING_FROM_CODE     — doc describes behavior not implemented in code
MISSING_FROM_SPEC     — code implements behavior not documented
STALE_DATA            — artifact present but older than expected window
SILENT_FALLBACK       — runtime took a non-preferred path without surfacing it
UNREACHABLE           — symbol exists but no path from a runner entry
DEAD_CODE             — symbol unreachable AND has no test
UNINSTRUMENTED        — invariant cannot be checked without semantic refactor
UNCONSUMED_OUTPUT     — code computes a value that nothing downstream reads
TEST_ONLY_REACHABLE   — code reachable only via tests/fixtures
```

**Indicative domain × mechanism combinations** (not exhaustive, not constraining):

| Domain | Common mechanisms |
|---|---|
| `DEAD_CODE_DOMAIN` | `UNREACHABLE`, `DEAD_CODE`, `TEST_ONLY_REACHABLE` |
| `WRONG_CODE` | `MISMATCH`, `SILENT_FALLBACK`, `STALE_DATA` |
| `UNIMPLEMENTED_LOGIC` | `MISSING_FROM_CODE`, `UNINSTRUMENTED` |
| `BROKEN_WIRING` | `UNCONSUMED_OUTPUT`, `UNREACHABLE` (callee missing), `SILENT_FALLBACK` (resolver bypass), `MISSING_FROM_SPEC` (undocumented path) |

The reviewer picks one mechanism that best fits the evidence; the domain is then derivable but recorded explicitly so Owner can filter the ledger by their mental model.

**Severity enum (3 values):**

```
BLOCKING      — must resolve before next run/release
TRUST_GAP     — known gap; needs decision but not blocking
INFO          — informational; no action required
```

**Evidence rule:** every row's `path` and `observed_evidence` must be one of `file:line`, runner artifact path, manifest key, or command-output excerpt with reproducible command. No prose-only evidence.

## Validity gate — adversarial discriminability test

The previously-proposed "<20% reviewer disagreement" gate is **discarded** because the only available reviewers are AI agents (Claude + Codex) with shared priors — agreement between us measures similarity, not enum soundness.

Replacement test, run immediately after Step 1 produces fresh findings:

1. Curate 8–12 findings: at least 3 from Step 1 fresh output, at least 3 **synthetic/pathological cases where the intended class is known by construction**, and the remainder from existing artifacts (`docs/spec_code_data_audit_2026_05_15.md:104-155`, ADR 0010, `.context/profile_engine_2016_2026_no_rule_reason_split_final.json:65-88`, `tests/fixtures/verification/golden_dates_report.yaml:19-44`). **Note**: the latter two artifacts contain classifier self-reports (`classification_status`, `source_used`, `rule_evidence`-style outputs). They are valid *enum-discriminability seeds* — material for testing whether the enum separates failure modes — but they are **not** Step 2a independent provenance and must not be cited as such.
2. **Blind** both reviewers to the intended class on synthetic cases.
3. Each reviewer produces exactly one primary classification, one severity, and evidence in the allowed forms.
4. **Negative test:** a second pass attempts to reclassify each finding under its nearest competing enum value. The enum **passes** only when the evidence rule and definitions force a stable primary class, OR when the competing class is clearly the same concept and the two should be merged.
5. Failure mode = merge or rename; do not add new classes without an Owner-supplied pathological case requiring one.

**Constraint on synthetic case construction:** the §Open items (α) section now provides three Owner-anchored mechanisms (stale-comments, intentionally-None-but-unwired, three-way divergence) plus five named sites. At least one synthetic case in the validity gate must be drawn verbatim from those mechanisms or sites — not reconstructed by AI reviewers.

## Open items — Owner-blocking

These must be answered before Step 1 begins.

### (α) Canonical trust-failure cases

Drawn from cross-session pattern analysis (sessions `3e3cf8a3`, `68022971`, `b683bcf0`) and the four parallel worktrees (`manila-v2`, `provo`, `san-diego`, `nicosia`, `vaduz`, `zurich-v1`). Three recurring mechanisms, each evidenced in multiple sessions, plus five named feature sites currently labeled shipped but with unfinished work.

**Mechanism 1 — Stale comments labeling shipped features as "deferred".**
Comments became the de-facto source of truth and never got updated as code shipped.
- Example: `src/regime_detection/_config_layer2.py:20-34` lists `broad_usd`, `yield_change_zscore`, and label set as deferred. All three are fully implemented in `src/regime_detection/monetary_pressure.py:164-242`.
- Domain × mechanism: `WRONG_CODE × MISMATCH`.

**Mechanism 2 — Config stubs marked "intentionally None" but never wired by consumer code.**
The config field exists; the YAML omits it deliberately ("intentionally omitted"); the consumer code never reads `config.<field>` and uses a hardcoded default in production.
- Example: `TrendCharacterV2Config` defined with per-label hysteresis fields; YAML intentionally omits the section; consumer code never read `config.trend_character_v2` until session `3e3cf8a3` fixed the wiring. Pre-fix, production silently used hardcoded defaults.
- Domain × mechanism: `UNIMPLEMENTED_LOGIC × UNCONSUMED_OUTPUT`.

**Mechanism 3 — Three-way doc / code / config divergence on implementation status.**
The same feature reads as "shipped" in one source, "stub" in the second, and "disabled" in the third. No single source of truth.
- Example: §2D Event Calendar V2. Spec describes full implementation; config YAML says "stub, not yet implemented"; code has `EventCalendarV2Config(enabled=False)`.
- Domain × mechanism: `BROKEN_WIRING × MISMATCH`.

**Five named sites currently labeled shipped but with unfinished work** (each is a synthetic seed for the §Validity gate test):

1. **§2D Event Calendar V2** — Group A done; Group B approval-gated; YAML omits; consumer logic partially wired.
2. **`trend_character_v2` per-label hysteresis** — config class existed, YAML omitted, consumer never read it; fixed in session `3e3cf8a3` on `2026-05-19`. Pre-fix behavior is the trust-failure case.
3. **`vol_crush` rule in §1C** — spec defines; code marks deferred pending implied-vol ingestion; nothing fires; status not surfaced in user-facing run reports.
4. **GMM cluster label mapping** — `docs/verification/cluster_label_map.candidate.yaml` ships with operator-review placeholders that were never filled.
5. **HMM state label mapping** — `docs/verification/hmm_state_label_map.candidate.yaml` same pattern; manual review per spec §6.2 never completed.

These materials make the §Validity gate's synthetic-case construction Owner-anchored rather than AI-reconstructed: at least one synthetic case in the validity gate must be drawn verbatim from this list.

### (β) Bypass-path proof  — **resolved during spec authorship**

**Question:** does any code path proceed without manifest sha verification today?

**Answer:** yes, three classes of bypass exist (documented in Step 2a). The provenance bundle now includes three explicit bypass markers (`manifest_path_provided`, `materialize_called_by_runner`, per-field `cli_overrides`). No further Owner decision required for this item.

### (γ) Historical-manifest scope

**Question:** should Step 1 also run against pinned historical manifests (e.g., snapshots in `.context/profile_engine_*.json`) to catch silent regressions in loader/resolver behavior, or audit today's manifest only?

`<TBD — Owner>`

Today-only is faster and answers "is the current state correct." Historical replay also catches "did the loader behavior change silently between runs," which directly maps to the trust-crisis framing of "data fetched not matching what docs claim."

## Out of scope (deferred)

- **Predicate-ID registry.** Predicates today are named functions (`evaluate_goldilocks` / `evaluate_inflation_shock` at `src/regime_detection/inflation_growth_rules.py:30-78`, precedence walker at `:253-275`, `VolumeLiquidity.RULE_PRECEDENCE` at `src/regime_detection/volume_liquidity_rules.py:62-66`, `NetworkFragility` precedence at `src/regime_detection/network_fragility_rules.py:69-76`) but have no stable cross-axis ID registry. Adding one is a refactor, not instrumentation; it changes the surface that Step 1 audits and is therefore deferred to a separate scoped task if Owner wants per-predicate provenance later.
- **Cursor-loss bug in the autonomous Envoy watcher.** Parked thread at room cursor [17]. Unrelated to the audit method.

## Open work — task seeding

This spec was produced under Envoy Task A (`msg_cf1ee8770310dff4412e9c2cdc18c9c4`). Review proceeds under Task B (Codex). Revision under Task C (Claude).
