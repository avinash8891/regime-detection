# ADR 0015 — Data-Quality Completeness Gate (Two-Tier)

**Status:** Accepted (R1, R2)
**Date:** 2026-05-23
**Context:** A comment-audit of
`src/regime_detection/data_quality.py` surfaced that the
`INSUFFICIENT_COMPLETENESS_FLOOR = 0.70` constant was annotated as
"spec §2.8" — but the spec has no §2.8 heading and the 0.70 value
appears nowhere in `docs/regime_engine_v2_spec.md` or any ADR. The
floor is a v1-inherited engine convention that shipped into v2 without
governance. This ADR ratifies the two-tier completeness gate (hard
floor + soft tunable) and the status precedence in
`assess_series_input_quality`, closing the governance gap.

## Problem

`assess_series_input_quality` produces five statuses
(`insufficient_history`, `stale_data`, `insufficient_data`, `degraded`,
`ok`) based on three numeric inputs:

- `required_trading_days` — window length (caller-supplied).
- `max_freshness_days` — staleness budget (caller-supplied).
- `min_completeness` — caller's soft completeness floor.

The function additionally compares completeness against a hardcoded
`INSUFFICIENT_COMPLETENESS_FLOOR = 0.70` to decide between
`insufficient_data` (forces classifier `unknown`) and `degraded` (allows
classifier to proceed with a flag).

Spec status:

- Implementation Ambiguity Log #10 (spec line 542) authorizes the
  helper's existence and the `skip_raw_label_short_circuit` flag but is
  silent on numeric completeness thresholds.
- Spec line 452 (Ambiguity Log #2, §3.2) explicitly notes the spec did
  NOT specify per-window completeness floors, and resolves §3.2 with
  `min_window_completeness = 0.9` — a different concept (per-window
  dispersion-completeness, exposed in v2 config) and a different number.
- No ADR (0001-0014) covers the 0.70 floor.

The pre-edit comment claimed "spec §2.8" authority for the value. That
claim is false; the comment misled readers and audits.

## Decision

### R1 — Ratify the 0.70 hard floor

`INSUFFICIENT_COMPLETENESS_FLOOR = 0.70` is ratified as the v2 engine
convention for the hard completeness floor. Rationale:

- Inherited from v1 unchanged across the v2 migration; backtest fixtures
  rely on this value.
- Distinct from the per-axis `min_completeness` knob in
  `DataQualityConfig` (caller-supplied, typically 0.85-0.95) — the
  hard floor is the universal "below this we can't compute anything
  meaningful" line.
- Below the floor, the classifier output should be `unknown`. Between
  `[0.70, min_completeness)`, the output can be the classified label
  but with `status="degraded"` so downstream consumers can decay
  conviction.

The value remains a module constant rather than a config knob because
(a) it is not currently tunable per-axis in production, and (b) lifting
it to config would require a coordinated config-schema migration. A
future ADR may demote it to `DataQualityConfig.hard_completeness_floor`
if calibration evidence warrants.

### R2 — Pin status precedence in `assess_series_input_quality`

The five statuses are produced in the following order; the first match
wins:

1. **`insufficient_history`** — any required window is shorter than
   `required_trading_days`. Forces classifier `unknown`.
2. **`stale_data`** — measured `freshness_days > max_freshness_days`.
   Forces classifier `unknown`.
3. **`insufficient_data`** — `completeness < INSUFFICIENT_COMPLETENESS_FLOOR`
   (i.e. `< 0.70`). Forces classifier `unknown`.
4. **`raw_label == "unknown"` short-circuit** — V1 callers fold into
   `insufficient_history` here. V2 callers opt out via
   `skip_raw_label_short_circuit=True` (Ambiguity Log #10).
5. **`degraded`** — completeness is in `[0.70, min_completeness)`.
   Classifier proceeds, status flag warns downstream.
6. **`ok`** — completeness `>= min_completeness`, freshness within
   budget, raw_label not forced. Classifier proceeds clean.

The precedence is fixed: stale-but-complete-enough still produces
`stale_data` (not `degraded`); a window in the hard-floor zone still
produces `insufficient_data` even if its single observation is recent.

`quality_forces_unknown(dq)` returns True iff `dq.status` is in
`{"insufficient_data", "insufficient_history", "stale_data"}`.
`degraded` and `ok` pass through.

## What changed under this ADR

| File | Change |
|---|---|
| `src/regime_detection/data_quality.py` (module header) | Added module docstring naming the authoritative anchor (Ambiguity Log #10) and pinning the R2 precedence. |
| `src/regime_detection/data_quality.py:12` | Annotation on `INSUFFICIENT_COMPLETENESS_FLOOR` cites this ADR as ratification source; removed the false "spec §2.8" claim. |
| `src/regime_detection/data_quality.py:114` | Replaced bare `10**9` magic with named `_NO_VALID_OBSERVATION_FRESHNESS_DAYS` constant. |

## Open questions

**O1 — Per-axis tunability of the hard floor.** Should
`INSUFFICIENT_COMPLETENESS_FLOOR` be promoted to
`DataQualityConfig.hard_completeness_floor` so individual axes can pick
a different floor? Today the value is module-global. Calibration §9.1
may identify axes (e.g. central-bank-text with sparse releases) where
a softer floor produces more usable signal. Defer until calibration
evidence warrants.

**O2 — Status vocabulary unification.** `DataQuality.status` uses
`insufficient_data`, `degraded`, `ok`; spec line 77-84 defines a
*classifier-output* status table with `data_unavailable`, `classified`,
`not_wired`, etc. These are different namespaces (input-quality vs
classifier-output) but the boundary is undocumented. A future
docstring or ADR could pin the namespace separation explicitly.

## Consequences

- **No behavior change.** R1 and R2 ratify shipped behavior. The code
  changes under this ADR are annotation + constant naming only.
- **Future audits will not re-flag the 0.70 floor.** The previous
  audit was misled by the false "spec §2.8" comment; the new
  annotation routes auditors to this ADR.
- **If O1 is resolved positively** (promote to config), the migration
  is straightforward: add `hard_completeness_floor: float` to
  `DataQualityConfig` with default 0.70, update
  `assess_series_input_quality` to read from caller-supplied config,
  re-run the regime-detection profile to confirm zero drift.
