# Comment Audit — Open Questions Log

**Started:** 2026-05-23 (overnight autonomous session)
**Format:** One section per file. Each entry has a short description, the lines
involved, what's doubtful, and the recommended next step. The orchestrator
deferred these because resolving them requires a design judgment call I'm
not equipped to make autonomously.

Resolve in priority order: items marked **[HIGH]** affect classifier behavior;
**[MED]** affect documentation accuracy; **[LOW]** are style/hygiene.

---

## Resolution status

All previously logged items have been resolved as of 2026-05-23 (resolution
pass). The audit closed with the following dispositions:

### Resolved with code/comment changes

- **File 21 [HIGH]** `_RISK_RANK` semantics → ADR 0016 drafted
  (`docs/decisions/0016-trend-direction-risk-rank-vs-precedence.md`); comment
  at `trend_direction.py:27-32` rewritten to point at the ADR and explain
  risk-rank semantics.
- **File 21 [MED]** Vectorized vs per-day v2 evidence asymmetry → identified
  as a **behavioral** asymmetry, not just evidence-emission. Fix applied to
  `raw_label_for_day` so v2 rule eval runs even when v1 inputs are NaN (per
  spec §1A L239 precedence: unknown is the tail and may be overridden by
  any v2 rule). Vectorized path was already correct; per-day brought into
  alignment. v1 byte-identity preserved for v1-only callers (no v2 args).
- **File 31 [MED]** Docstring hysteresis `unknown: 2` → `unknown: 0` (spec
  L1449 + yaml).
- **File 34** "six features" → "five base + four derived percentiles"
  (matches the actual bullet list).
- **File 37** "V2 §10 ABSOLUTE RULE" → "V2 §10 no-auto-label rule" (the
  spec's literal `ABSOLUTE RULE` heading is about a different rule).
- **File 39** Three "documented implementation decision" → concrete
  Ambiguity Log #66 (4-table system) and Log #64 (5-session rolling max)
  pointers.
- **File 42** Dropped stale "§2A monetary_pressure classifier is not
  shipped yet" example (the classifier IS shipped).
- **File 47 [LOW]** timeline.py multi-claim review — all 4 verified
  case-by-case: "every V2 axis builder is shipped" TRUE (9 builders in
  `axis_builders/`); "byte-identity for callers that omit all three
  configs" softened to the more precise "preserves V1's slicing window";
  "+21/+63" warmup arithmetic TRUE against feature inputs; "V1 wire
  contract" statements TRUE against `RegimeConfig.market` /
  `RegimeOutput.market` semantics.
- **File 53** L19 misleading warm-up comment rewritten.
- **File 58** L49 "(documented implementation decision)" → "(Ambiguity Log
  #46)".
- **Files 2/4/5** Spec-citation drift sweep — 60+ stale citations
  re-anchored via dedicated A+B pass.

### Resolved via verification (no change needed)

- **File 21 [LOW]** Magic numbers — already applied autonomously during the
  audit (`_SIDEWAYS_ABS_RETURN_63D`, `_WITHIN_PCT_SMA200`).
- **File 22 [LOW]** "V2 §10 do not invent a sentiment proxy" — already
  reworded to "V2 §10 no-hallucination rule".
- **File 22 [LOW]** SF Fed second sentiment voice — already relabeled
  "Engine-local extension (NOT in V2 §1A)".
- **File 60** `I1:` prefix — verified as internal designation referenced in
  matching test contracts (`test_network_fragility_classifier.py:488,512`),
  not fabricated authority. Kept as-is.

### Open follow-up (not blocking)

- **File 60 [LOW]** Multiple single-agent flags (NaN-handling vs §3.3 prose,
  "per the spec" overstatement, "v2/v2 calendar drift" phrasing, "V1 axes"
  wording when V2 axes also wired). Each is documentation precision; apply
  piecemeal if a future session revisits `axis_builders/network_fragility.py`.

---
