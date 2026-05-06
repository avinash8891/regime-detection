# AGENTS.md

Operating discipline for coding agents in this repository. Project-specific context (schemas, architecture, tasks, acceptance criteria) lives in the project's spec file, not here. Read this file at session start. It wins on conflict unless resolved explicitly.

## Non-negotiable rules

1. **Announce intent** before each unit of work: what, files touched, tests, expected user-visible output.
2. **No secrets, auth headers, or raw request/response payloads** in code, logs, errors, or output. Read from env or secret manager. Never print, never commit.
3. **Search before writing.** Grep the codebase for the concept before any new function >15 lines. Report findings.
4. **Stop at each task boundary.** Commit with clear tag, push, wait for "continue." No auto-chaining.
5. **Validate at every layer boundary**, including raw external inputs — one source-specific model per source before normalization.
6. **TDD for all V1 work.** For every vertical slice in the V1 spec: write the failing tests first (golden date(s), cold-start/NaN behavior, NYSE calendar enforcement, hysteresis edge cases), then implement until they pass. No untested implementation.
7. **Tests run and pass.** Paste actual test-runner output into the completion message.
8. **No scope creep.** The spec is authoritative. "While I was here" additions banned. "In blast radius" is not a license.
9. **Single source of truth for persistent state.** No shadow copies, no cross-task intermediate stores. Raw inputs and final outputs OK.

## Failure modes + counter-mechanisms

- **A. Wire-first.** First file touched = the entry point, with the signature that will call the new code. Task not done until a real invocation exercises it end-to-end. Before commit, grep for imports of the new module — must return at least one caller outside its own package, or the module is unwired.
- **B. One home per concept.** Each external service client, each shared utility, each data-model file — one location, declared up front. Nobody re-instantiates locally. Data models live in one file; adding or changing a field is announced, not silent. Payload normalizers must be public and invoked at *every* validation call site — if a second caller appears, promote the helper out of `_private` rather than duplicating the normalization logic. Skipping normalization at one site means valid payloads will raise validation errors and abort whatever flow that site drives.
- **C. No patch-on-patch.** Bug → failing test first, then fix. Read the whole function before touching it — don't parachute a guard around the failing call. Before removing or renaming anything (function, table, field, env var), grep every caller in one pass — "unused in this file" ≠ "unused globally." Commit subject identifies the class: `fix` (root cause), `patch` (symptom workaround — opens follow-up issue), `refactor` (no behavior change), `feat` (new capability). Commit message names the wrong assumption ("code assumed X, but X is false because Y"). Delete-before-add bias: fix by removing lines when possible. If a patch adds >5 lines in one file or >15 total, stop and reconsider — probably a symptom not a root cause. Three-strikes rule: if a function accumulates three branches for specific edge cases, STOP and refactor — the data model is wrong, not the branches. Trust the framework: don't re-validate what pydantic/the DB/the type system already validates.
- **D. Read-before-accept.** Plain-language diff summary in domain language before every commit. Flag trust-point lines (table names, fields, endpoints, regex, filter conditions) for user spot-check.
- **E. One commit, one deliverable.** Stated in one sentence up front. No related cleanup, no uncalled-for refactors. If a file outside the stated deliverable needs to change, ask first.
- **F. No confident-wrong.** Every external API or library claim verified via docs lookup, signature pasted in announcement. Uncertainty stated explicitly. User corrections override training — don't argue, update.
- **G. Real tests only.** Never mock internal code; mock external services using captured real fixtures (redacted), not hand-written shapes. One integration test per task against real data. Assertions exercise behavior, not structure — exact values or counts beat `is not None`. If the test would pass with the function body replaced by a plausible constant, the test is worthless. When an assertion fails, the code or fixture is wrong — not the assertion. Don't relax `==` to `>=`, widen ranges, or loosen types to turn red tests green.
- **H. Log before feature.** Structured stdlib logging, UTC timestamps. Every ERROR line answers "what do I do about this?"
- **I. Quarantine bad external data.** Source-specific validation row-by-row. Malformed rows → quarantine file + log, continue. >1% failure rate = STOP and show user. When a new edge case surfaces in production data, add it to the test fixture first, then fix the code.
- **J. UTC in persistent state.** All stored timestamps UTC with timezone. Conversion to local time happens at display edges only. Naive datetimes banned from storage and comparisons.
- **K. Evidence with claims.** "It works" requires pasted output, row counts for mutations, or relevant command excerpts. "Tests pass" alone is not evidence. Ambiguous success reported as ambiguous ("47/48 passing" is not "tests pass"). User can demand verification at any point — respond with artifacts, not restated claims.
- **L. Agent findings are hypotheses.** Before acting on a review agent's claim ("dead code," "unused import," "missing reference," "broken ref"), grep or read to verify. Fixing imaginary bugs introduces real ones.

## Error policy

- Deterministic errors (schema, validation, logic, SQL): propagate loud. Never swallow.
- External flakiness (5xx, rate-limit, timeout): catch, log, degrade. Never kill the whole run.
- Status integrity: a run or step is "ok" only if every required sub-step succeeded. Partial success with a silent skip = failed run, reported as failed.
- **Terminal-state bookkeeping precedes deterministic-error validation.** If a handler decides work reached a terminal state (`halted`, `manual_review`, `finished`), the state mutation and operator notification must run *before* any code that can raise. A validation error propagating before terminal fields are written leaves state inconsistent and is indistinguishable from a crash. Pattern: write terminal fields first, then validate or compile best-effort enrichment with its own error handling that logs and continues.

## Hygiene

- `get_logger` must NOT set `propagate = False` — pytest caplog captures via the root logger; blocking propagation makes all `caplog` assertions return empty strings. No duplicate output risk in production (no root handler attached outside tests).
- `PYTEST_CURRENT_TEST` is NOT set during pytest module import/collection — only during test execution. Module-level code cannot use it as an import guard. Use it only inside functions.
- No new dependency without justification in commit message. Stdlib → existing deps → new dep, in that order.
- Hardcoded tunable numbers (thresholds, limits, batch sizes, weights) = config smell. Put them in config, validated on load.
- **Env-var-backed tunables use lazy accessor functions, not module-level constants.** `MAX_X = int(os.environ.get("PROJECT_MAX_X", "10"))` at import time means pytest's `monkeypatch.setenv` after the module is imported has no effect — import order silently determines whether overrides stick. Wrap the read in `def max_x() -> int:` and call it at the use site. Validation (int parse, range check) lives inside the accessor and raises with a named env-var on bad input.
- Size-down tactics: data over logic (lookup tables beat if-chains), stdlib first, no wrapper-only-renames, no premature abstraction, no scaffolding comments, delete dead code on contact (including commented-out blocks — git preserves what was there).

## Violations

Cite by number or letter (`violates rule 4` or `violates C`), self-correct before proceeding, state what changed. If a rule seems wrong for a specific case, flag the conflict and propose a resolution — never silently ignore.
