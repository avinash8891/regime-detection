# CLAUDE.md — regime-detection

## What this repo is (read first)

This is **one** regime-detection engine, built in **two phases**. "V1" and "V2" are
**phases of a single system, not two separate systems**. Some features shipped in
phase 1 and others in phase 2; together they are one engine. The package is
currently at phase 2: `src/regime_detection/__init__.py` sets `__version__ = "2.0.0"`.

- `docs/regime_engine_v1_final_spec.md` — phase-1 features and the phase-1 output contract.
- `docs/regime_engine_v2_spec.md` — phase-2 features that **extend the same engine**.
- `docs/historical_walkforward_spec.md`, `docs/shadow_runner_spec.md` — qualification gates.

Where a phase-1 spec statement was later superseded by phase 2 (e.g. inflation/growth
and credit/funding are "not implemented in V1" in §7.1 but **are** implemented as
real V2 axes), the **shipped phase-2 behavior is current**; the phase-1 text is
historical. Treat such cross-phase disagreements as internal spec inconsistency to
reconcile toward the shipped behavior — verify the code first, then fix whichever
(doc or code) is actually wrong. Do not default to "the code violates V1."

## Consequences for reviews and agents

- **Do not flag the engine for "being V2."** `engine_version` and the default config
  derive from the package `__version__` (currently 2.0.0), so the runtime emits
  `regime-engine-v2.0.0` and defaults to `configs/core3-v2.0.0.yaml`. The **binding**
  rule in V1 §2.4 is the *package ↔ engine_version coupling test* (line 185), **not**
  the literal `v1.0.0` shown in the example JSON (line 178). The `v1.0.0` /
  `core3-v1.0.0.yaml` literals in §2.4 / §2.4.1 are stale phase-1 examples.
- **Phase-1 ("V1") wire output remains a frozen, byte-identical replay contract.**
  Extending the engine in phase 2 must not mutate phase-1 frozen outputs. Changing a
  V1 wire field requires an updated `tests/test_v1_frozen_replay.py` fixture **and**
  the `tests/_v1_frozen_models.py` shim, P1-reviewed (see `AGENTS.md` → Review
  guidelines).
- A correctness fix that legitimately changes a phase-1 output is allowed only with an
  explicit, justified fixture re-freeze — never a silent relaxation of a golden/fixture
  assertion to make a test pass.

## Where things live

- **Operating discipline for agents:** `AGENTS.md` (wins on conflict; this file is
  framing/context, AGENTS.md is the rules of engagement).
- **Phase specs:** the four `docs/*_spec.md` files above.
- **Latest spec-compliance review:** `spec_review.md` (root).

## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default five-label triage vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repo. See `docs/agents/domain.md`.
