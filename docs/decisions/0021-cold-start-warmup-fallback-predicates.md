# ADR 0021 — Cold-start warmup fallback predicates (correlation_to_one, liquidity_gap_behavior)

**Status:** Accepted
**Date:** 2026-06-01
**Resolves:** spec-compliance findings F-004 (`cold_start_corr_to_one`) and F-012 (`cold_start_liquidity_gap`)
**Owner decision:** keep the fallbacks enabled and authorize them in-spec (Q4 = B).

## Context

Two V2 rule paths carry a **cold-start fallback predicate** that fires while the
normal percentile rule cannot yet be evaluated — i.e. before enough history exists
to compute the rolling percentile the spec rule references:

- **`correlation_to_one`** (V2 §3.5 network fragility) — `cold_start_corr_to_one`
  path in `src/regime_detection/network_fragility_rules.py`, enabled by default via
  `cold_start_corr_to_one_enabled=True` (`src/regime_detection/_config_core.py`,
  `configs/core3-v2.0.0.yaml`). Its 504-session percentile (`avg_pairwise_corr_percentile_504d`)
  is unavailable during warmup.
- **`liquidity_gap_behavior`** (V2 §1E volume/liquidity) — `cold_start_liquidity_gap`
  path in `src/regime_detection/volume_liquidity_rules.py`, enabled by default via
  `cold_start_liquidity_gap_enabled=True` (`src/regime_detection/_config_layer1.py`).
  Its 252-session percentile is unavailable during warmup.

The literal V2 §3.5 / §1E rule text describes only the percentile branch and does not
mention these fallbacks. The original review flagged them as undocumented extensions.

## Decision

**Keep both fallbacks enabled by default; authorize them here as deliberate
warmup-only fail-safes.** They are cited from the spec rule sections (§3.5, §1E) by
cross-reference to this ADR.

Rationale, grounded in the engine's operating window:

- **The emitted regime-detection window starts 2016**; all data from 2009 onward is
  warmup that calibrates the models and percentiles but is never emitted
  (`README.md` "Main Profile Runner"; `CLAUDE.md`).
- By 2016 the 504-/252-session percentiles for these axes are **fully populated** (the
  underlying price/correlation inputs exist from 2009), so the **normal percentile
  branch governs every emitted session**. The cold-start fallback predicates therefore
  **never fire in emitted output** — they only affect the pre-2016 warmup ramp, which
  is not shown.
- Because they cannot affect emitted classification, disabling them would be a no-op
  for production; keeping them preserves a conservative early-fire fail-safe for any
  future configuration whose emitted window begins before a percentile is warm (e.g. a
  shorter warmup load), without widening any emitted high-risk label today.

## Consequences

- No code or config change: `cold_start_corr_to_one_enabled` and
  `cold_start_liquidity_gap_enabled` remain `True` by default.
- V2 §3.5 (`correlation_to_one`) and §1E (`liquidity_gap_behavior`) carry a
  cross-reference to this ADR so code and spec agree.
- An enumerative guard (test) asserts the fallback predicates are inert in the emitted
  window: with a fully-warmed percentile history, the fallback branch is not taken.
- If a future profile shrinks the warmup load such that an emitted session falls inside
  a percentile warmup, the fallback becomes observable — that is the intended fail-safe,
  and is the only condition under which this ADR has emitted-output effect.
