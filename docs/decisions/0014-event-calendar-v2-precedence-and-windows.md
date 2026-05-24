# ADR 0014 — Event Calendar V2 Precedence and Per-Type Windows

**Status:** Accepted (R1, R2, R3)
**Date:** 2026-05-22
**Context:** A comment-audit of
`src/regime_detection/event_calendar.py` surfaced two governance gaps where
V2 §2D added four new event labels but didn't pin their precedence or
per-type trading-day windows. The code shipped defensible defaults, but the
V2 §10 absolute rule (spec line 1118: "when the spec is ambiguous or silent,
stop and ask; do not invent") was breached without an ADR. This ADR closes
that gap.

## Problem

V1 spec §7.2 + ADR 0002 fully pin the V1 event-calendar contract:
- Precedence: `fed_week > cpi_week > nfp_week > expiry_week > earnings_season > normal_calendar > unknown` (ADR 0002 §63-64).
- Per-type windows: `fed_week [-2,+2]`, `cpi_week [-1,+1]`, `nfp_week [-1,+1]` (V1 spec §7.2 lines 757-759).

V2 §2D (spec lines 3362-3398) adds four new labels:
- `election_window` — with a spec-pinned `[-5, +10]` default (line 3366).
- `geopolitical_event` — approval-gated overlay-promoted (line 3367); no window pinned.
- `budget_week` — deterministic fiscal-deadline row (line 3365); no window pinned.
- `global_rate_decision` — ECB/BOE/BOJ scheduled meetings (line 3368); no window pinned.

Neither §2D nor Ambiguity Log #50 (lines 1739-1773) extends V1's precedence
chain to cover the new labels. The code at
`src/regime_detection/event_calendar.py:34-46` invented an 11-element
ordering, and the code at lines 61-69 invented `(0, 0)` windows for three
of the new labels. Both choices ship in production via
`core3-v2.0.0.yaml`.

## Decision

### R1 — Event-calendar precedence (full V1 + V2 chain)

Ratify the following precedence ordering, with V1 sub-sequence preserved
verbatim from ADR 0002:

```text
geopolitical_event > election_window > fed_week > global_rate_decision
                   > budget_week > cpi_week > nfp_week > expiry_week
                   > earnings_season > normal_calendar > unknown
```

Rationale per slot:

- **`geopolitical_event` first** — approval-gated and overlay-promoted
  (V2 §2D line 3367 + Ambiguity Log #50 line 1750). When the overlay
  promotes a row, the event is by definition a real shock; spurious
  candidate rows never fire. Outranking everything (including FOMC)
  reflects that the operator has affirmed a market-moving event.
- **`election_window` second** — widest window (`[-5, +10]`, the rarest
  cadence of all V1+V2 events). When it fires it dominates the macro
  narrative for ~3 weeks of trading. Slotting it above `fed_week` reflects
  that election uncertainty supersedes scheduled CB cadence.
- **`fed_week` third** — preserved V1 position relative to `cpi_week`/
  `nfp_week`. Domestic CB meeting outranks foreign CB meetings.
- **`global_rate_decision` fourth** — slotted between `fed_week` and
  `cpi_week`. ECB/BOE/BOJ are cross-axis macro events that pre-empt
  scheduled US data releases for cross-asset traders. Below `fed_week`
  because the Fed dominates US session structure.
- **`budget_week` fifth** — fiscal deadlines are calendar events that
  matter as policy backdrop but rarely move markets on the day; below
  `global_rate_decision` for that reason.
- **`cpi_week` / `nfp_week` / `expiry_week` / `earnings_season` /
  `normal_calendar` / `unknown`** — V1 sub-sequence preserved.

### R2 — Per-type windows for V2 labels

Ratify the following per-type trading-day windows:

| Label | Window | Source |
|---|---|---|
| `election_window` | `[-5, +10]` | V2 §2D line 3366 (already spec-pinned) |
| `geopolitical_event` | `(0, 0)` | This ADR — manual no-window fallback only; generated approved rows can override with row-level `window_days` |
| `budget_week` | `(0, 0)` | This ADR — fires on the deadline day only; budget runup behavior is out of scope |
| `global_rate_decision` | `(0, 0)` | This ADR — **known asymmetry vs `fed_week (-2, +2)`** (see "Open question" below) |

The known asymmetry between `fed_week` and `global_rate_decision` is
recorded as Open Question O1 below. The ADR ratifies `(0, 0)` as the
**provisional** default pending calibration work.

For `geopolitical_event`, this table is not the GPR persistence policy. It is
only the fallback for approved rows that do not carry `window_days`. GPR-derived
candidates can suggest `(0, 0)`, `(-1, 3)`, or `(-2, 5)`; once approved and
rendered, the row-level value takes precedence over this default.

### R3 — Extend ADR 0002 §52 90-day publication-date default to V2 scheduled types

ADR 0002 §52 authorizes "if `publication_date` is absent, default it to
`date - 90 calendar days`" for FOMC, CPI, and NFP rows. The V2 set of
scheduled event types added by §2D — `ECB_decision`, `BOE_decision`,
`BOJ_decision`, `global_rate_decision`, `election`, `budget` — share the
same property as the V1 scheduled types: official meeting / release
schedules are publicly known months in advance. Applying the 90-day
fallback to V2 types is therefore consistent with the spec contract and
preserves V1's stateless-replay invariant ("only consult events whose
`publication_date` is on or before the firing session" per V1 §2.2,
verified at spec line 222).

R3 explicitly extends ADR 0002 §52 to all members of
`_SCHEDULED_TYPES` in
`src/regime_detection/loaders.py:19-29` — i.e. the union of
{FOMC, CPI, NFP} (V1) and {ECB_decision, BOE_decision, BOJ_decision,
global_rate_decision, election, budget} (V2).

`_V2_MANUAL_TYPES` (`geopolitical_event`, `ad_hoc` approval-overlay rows)
intentionally fall through to `publication_date = row["date"]` because
they are operator-supplied and the spec contract is that they only fire
when the overlay promotes them — there is no "publicly known months
ahead" property to anchor a 90-day default against.

## Open questions

**O1 — `global_rate_decision` window asymmetry vs `fed_week`.** The Fed's
FOMC meetings fire across `[-2, +2]` trading days; foreign CB meetings
(ECB/BOE/BOJ) currently fire only on the event day. The rationale for
asymmetry is that foreign-CB events don't dominate US session structure
the way the Fed does, so the surrounding sessions don't carry the same
information content. However, this is an untested calibration claim. The
calibration §9.1 study should examine whether US-equity / cross-asset
behavior actually quiets down within `[-2, +2]` of a major ECB meeting,
and revise to `(-1, +1)` or `(-2, +2)` if so. The knob's location in
`_WINDOWS` makes this a one-line code change.

**O2 — `budget_week` runup behavior.** Some fiscal deadlines drive
multi-day positioning ahead of the date (debt ceiling, government
shutdown). `(0, 0)` may understate these. Future ADR can expand to
`(-N, 0)` if a measurable runup effect is identified.

## Implementation

R1 and R2 are reflected in code as of this ADR:

| File | Change |
|---|---|
| `src/regime_detection/event_calendar.py:34-46` | Existing `_PRECEDENCE` ratified. Added an explanatory comment above the tuple citing this ADR. |
| `src/regime_detection/event_calendar.py:61-69` | Existing `_WINDOWS` ratified. Added an explanatory comment above the dict citing this ADR for the V2 entries. |
| `src/regime_detection/event_calendar.py` (constants block) | Added `_FORWARD_EVENT_WARNING_DAYS = 90` (ADR 0002 §"Optional operator guard") and `_SESSION_PADDING_DAYS = 40` (derived from max `_WINDOWS` end_offset + safety margin). Replaced the magic numbers at `classify_event_calendar` L86 and at `compute_event_window_just_passed` L348-349. |
| `src/regime_detection/event_calendar.py` `compute_event_window_just_passed` docstring | Added explicit sentence that only V1 window types drive the rule per ADR 0005 Q3. |
| `tests/` | Follow-up: add precedence boundary tests covering V1↔V2 transitions and a V1-only filter regression test for `event_window_just_passed`. Logged as test-backfill TODO below. |

## Test backfill (follow-up)

The following tests should be added to `tests/test_event_calendar.py` and
`tests/test_volatility_state_v2_vol_crush.py`:

1. **Precedence boundaries** — pairwise tests covering each V1↔V2 transition:
   - `fed_week` vs `election_window` (expect `election_window`).
   - `fed_week` vs `global_rate_decision` (expect `fed_week`).
   - `global_rate_decision` vs `cpi_week` (expect `global_rate_decision`).
   - `budget_week` vs `cpi_week` (expect `budget_week`).
   - `geopolitical_event` overlay-promoted vs everything (expect `geopolitical_event`).

2. **V1-only filter for `event_window_just_passed`** — a regression test
   that an `ECB_decision` event with `publication_date <= as_of_date` and
   the event date 1-3 NYSE sessions before `as_of_date` does NOT cause
   `event_window_just_passed` to fire (preserves V1-byte-identity).

3. **Forward-event warning** — a `caplog` test that
   `_FORWARD_EVENT_WARNING_DAYS = 90` fires the logger.warning per ADR
   0002 §57.

## Consequences

- **No behavior change.** R1 and R2 ratify existing shipped behavior. The
  code change is annotation only (comments cite this ADR; magic numbers
  extracted to named constants).
- **Future asymmetry revision** (O1) on `global_rate_decision` window may
  shift `vol_crush` and cross-axis predicate firings on the ~80 ECB/BOE/BOJ
  meeting days per year. The knob is config-shaped so a future ADR can
  retune without a spec amendment.
- **Spec hygiene follow-up:** consider amending V2 §2D to inline this
  precedence chain + windows table so future readers see the contract
  alongside the label definitions.
