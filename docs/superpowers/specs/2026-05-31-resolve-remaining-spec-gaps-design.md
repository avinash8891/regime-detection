# Design — Resolve Remaining Spec Gaps (A / B / C)

**Date:** 2026-05-31
**Author:** autonomous session (user stepped out, pre-authorized "resolve all the issues")
**Scope:** the verified remaining gaps after the spec-compliance fix campaign.

This design is approved-by-delegation: the user explicitly instructed autonomous
implementation with verification-first. Decisions and assumptions are documented
here in lieu of an interactive approval gate.

## A. PIT survivorship-bias rejection gate + `allow_survivorship_biased_breadth`

**Gap (V2PRE-010 / V2 §10):** V2 PIT breadth must not silently fall back to a
survivorship-biased current-constituent universe. Today, when the V2 breadth
feature builder is invoked without point-in-time inputs
(`pit_constituent_intervals` / `constituent_ohlcv`), it silently returns the
sector-only path (PIT features all `None`, a `bias_warning` attached) — a
warning, not a rejection. No opt-in flag exists.

**Design (final — ingestion layer):** The gate lives at the PIT data-ingestion
boundary `regime_data_fetch.pit_constituents.read_pit_intervals`, with a public
detector `is_survivorship_biased_universe(intervals)`:
- A point-in-time universe INCLUDES removed/delisted members — at least one
  closed membership interval (non-null `end_date`). A current-only snapshot
  (every interval open) is survivorship-biased.
- `read_pit_intervals(path, *, allow_survivorship_biased_breadth=False)` raises
  `ValueError` after applying delisting corrections if the loaded universe is
  biased and research mode is not approved. Real PIT feeds (with delistings)
  pass; a current-only snapshot is rejected.

**Why ingestion, not the runtime `_build_breadth_state_v2`:** an initial attempt
gated at the feature-store build (rejecting when PIT inputs were absent). That
was wrong on two counts. (1) Spec §1D line 329 EXPLICITLY allows V1 ETF-proxy
breadth as unbiased fallback when no PIT universe is present — so "PIT absent"
must NOT be rejected. (2) The real requirement (line 326-327) is row-level
validation + "explicit rejection of survivorship-biased universes", which is a
property of the loaded DATA, validated once at ingestion — not on every
`classify`. The runtime gate also broke ~20 tests that legitimately pass
placeholder universes directly. The ingestion gate validates real loaded data,
leaves the etf_proxy fallback intact, and has zero blast radius on the runtime
tests (which never call `read_pit_intervals`).

**Toggle placement:** "biased research mode" is an operational, per-run decision,
not a property of the frozen shipped config — so the toggle is the
`read_pit_intervals` keyword `allow_survivorship_biased_breadth` (default
fail-closed), not a `core3-v2.0.0.yaml` field. The three production loaders
(`profile_engine`, `run_v2_calibration`, `audit_layer2_30d`) load real PIT data
with delistings and pass under the default deny.

**Invariants preserved:** V1 never loads a PIT universe. The production V2 path
loads the real fja05680 universe (delistings present → passes). The existing
`read_pit_intervals` tests already use universes with closed intervals → pass.

## B. Data / fixture gaps

**B2 (doable now):** `2020-08-15` in `golden_dates_v2.yaml` is a Saturday. Re-anchor
to `2020-08-14` (Friday, NYSE session, within V2 fixture coverage), verify it
classifies live, re-derive its `expected_v2_fields`, remove it from
`_V2_LIVE_FIXTURE_UNSUPPORTED_GOLDEN_DATES`, and update `_V2_SPEC_GOLDEN_DATES`.

**B1 / B3 (data-blocked — cannot do autonomously):** pre-2019 V2 OHLCV+VIX history
back to ~2010 (for the 4 pre-2019 §9.4 dates) and the true PIT vendor feed both
require external market-data acquisition. Fabricating data is forbidden (golden
tests require real data). These are tracked as an acquisition spec with the exact
data required; the existing `_V2_LIVE_FIXTURE_UNSUPPORTED_GOLDEN_DATES` guard keeps
them honest (explicitly unsupported, not silently skipped) until the data lands.

## C. Spec-deferred items

**`classify_series(...)` V1.1 helper (V1GEC-017):** implement as a thin wrapper
over `classify_window` that flattens the timeline to a `pd.DataFrame` (one row per
`as_of_date`, columns = per-axis active labels + transition_risk state). It is a
V1.1 helper in the spec (not V2 scaffolding), low-risk, in-repo. TDD.

**§5.3 vol-crush exposure response (V2AMB2-043):** NOT implemented here — per ADR
0020 it is a downstream strategy/position-management contract (50% long-vol cut
over a 5-day cooldown), not this engine's responsibility. The engine correctly
emits the `vol_crush` label. Resolution = confirm ADR 0020 documents the boundary
(it does). No code in this repo.

## D. By-design non-gaps (no work)

`upvol_downvol_ratio` and `sector_breadth` are computed but evidence-only — no §1D
label consumes them and the spec defines none. The five named V2 breadth labels
all ship and emit. No action.

## Testing

- A: feature-store-level tests — raises without PIT inputs + flag False; allowed
  with flag True; passes with PIT inputs present.
- B2: the existing §9.4 live-classification test covers it once re-anchored.
- C: unit tests for `classify_series` shape/content + window-independence.
- Full suite + ruff/black green before completion.
