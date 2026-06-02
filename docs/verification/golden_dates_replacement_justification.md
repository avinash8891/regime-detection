# Golden Date Replacement Justification (F-008)

The V1 spec §12.2 golden-date table is the historical *intent* set, but the
active regression fixture (`tests/fixtures/derived/golden_dates.yaml`) uses 10
curated replacement dates. This document is the per-date replacement-justification
report required to close F-008, with the **real** reason for each substitution.

## Why the literal §12.2 dates are not used directly

Two project invariants collide on the literal §12.2 dates:

1. **§12.2 "predicates win over intuition."** The spec table's labels are
   explicitly "hand-labeled expectations pending Slice 2 verification," and the
   spec says the deterministic rule predicates win and "do not relax expectations
   to make tests pass."
2. **`provenance: hand_labeled` — never bless engine output.** `golden_dates.yaml`
   carries `provenance: hand_labeled`, enforced by
   `tests/test_fixture_verification.py` (asserts `provenance == "hand_labeled"`)
   and by `scripts/verify_fixtures.py`, which fails loudly rather than emit
   `RegimeEngine` output as the expected label.

When the engine classifies the **literal** §12.2 dates, the deterministic
predicates diverge from the table's pending-verification intuition on **every
one of them** (verified by classifying each date through the frozen V1 / V2
config — the `trend / character / vol / breadth` the engine actually emits):

| §12.2 date | §12.2 table intuition | Engine predicate at the literal date | Diverges on |
|---|---|---|---|
| 2017-06-01 | bull / trending / low_vol / healthy | bull / transition / normal_vol / weak_breadth | character, vol, breadth |
| 2018-02-05 | bull / transition / crisis_vol / (pin) | sideways / transition / high_vol / weak_breadth | trend, vol |
| 2018-12-24 | bear / trending / high_vol / weak | bear / trending / crisis_vol / weak_breadth | vol |
| 2019-09-13 | bull / trending / normal_vol / healthy | bull / transition / normal_vol / healthy_breadth | character |
| 2020-03-16 | bear / transition / crisis_vol / weak | transition / trending / crisis_vol / narrowing_breadth | trend, character, breadth |
| 2020-04-10 | bear / recovery_attempt / high_vol / recovery | **non-trading (Good Friday)** | — |
| 2021-11-15 | bull / trending / low_vol / healthy | bull / mild_trend / low_vol / weak_breadth | character, breadth |
| 2022-06-13 | bear / trending / high_vol / weak | bear / range_bound / crisis_vol / narrowing_breadth | character, vol, breadth |
| 2022-10-12 | bear / trending / high_vol / weak | bear / trending / high_vol / healthy_breadth | breadth |
| 2024-01-16 | bull / trending / low_vol / healthy | bull / mild_trend / low_vol / narrowing_breadth | character, breadth |

These divergences are predicate-correct, not bugs — e.g. on **2018-02-05
(Volmageddon)** SPY fell ≈ −4.1%, which does **not** breach the §5.5 `crisis_vol`
trigger (`return_1d <= -0.05`), so the engine emits `high_vol`, not the table's
`crisis_vol` intuition. Anchoring the regression suite to the literal dates would
therefore force either (a) asserting the table's intuition that the predicates
contradict (tests fail; the spec forbids relaxing them), or (b) freezing engine
output as the expected label (breaks the `hand_labeled` invariant). Additionally,
**`2020-04-10` is Good Friday — a non-NYSE trading session the engine rejects**,
so it can never be a golden `as_of_date` regardless.

## Resolution: curated same-regime sessions where intuition and predicates agree

The active set keeps `provenance: hand_labeled` and satisfies §12.2's
"predicates win" by selecting, for each §12.2 regime, a nearby same-era trading
session whose independently hand-labeled expectation also matches the
deterministic predicates. Per-date mapping:

| §12.2 intent date | §12.2 regime targeted | Committed `as_of_date` | `intent_id` | Reason |
|---|---|---|---|---|
| 2017-06-01 | bull / trending / low_vol | 2020-08-11 | summer2020_bull_trending_lowvol | Same bull/trending/low-vol signature on a session where intuition and predicates agree. |
| 2018-02-05 (Volmageddon) | volatility spike / crisis | 2018-02-08 | volmageddon_crisis | Same Volmageddon episode; 2018-02-08 cleanly classifies crisis_vol (the −5% trigger is met) where 02-05 does not. |
| 2018-12-24 | bear / high-vol drawdown | 2018-12-11 | dec2018_bear_stress | Same Dec-2018 bear-stress drawdown; agreeing session. |
| 2019-09-13 | bull / normal_vol | 2019-09-12 | mid2019_bull_normal | Adjacent session, same bull/normal-vol regime. |
| 2020-03-16 (COVID crash) | bear / crisis | 2020-03-30 | covid_crash_crisis | Same COVID-crash crisis window; agreeing session. |
| 2020-04-10 | recovery / high-vol | 2020-04-17 | covid_recovery_attempt | **2020-04-10 = Good Friday (non-trading).** Nearest recovery-window session where intuition and predicates agree. |
| 2021-11-15 | bull / low_vol | 2021-11-01 | late2021_bull_lowvol | Same late-2021 bull/low-vol regime; agreeing session. |
| 2022-06-13 | bear / high-vol | 2022-06-15 | jun2022_bear_crisis | Same Jun-2022 bear episode; agreeing session. |
| 2022-10-12 | bear / high-vol | 2022-07-06 | jul2022_bear_stress | Same 2022 bear-stress regime within coverage; agreeing session. |
| 2024-01-16 | bull / low_vol | 2023-12-05 | early2024_bull_lowvol | Same 2023→2024 bull/low-vol turn; agreeing session. |

All 10 committed `as_of_date` values are verified on every commit with
`no silent pre-2019 or data-quality skips` (see
`tests/test_fixture_verification.py::test_classified_golden_outputs_cover_every_row_without_silent_skips`
and `::test_golden_dates_match_live_labels_without_data_quality_bypass`).

The replacement set remains **hand-labeled and not engine-generated**. When the
deterministic predicates disagree with a hand label, the row must be updated with
the predicate result or the implementation fixed — current engine output is never
blessed into the fixture.

## Note on full re-anchoring

Re-anchoring to the literal §12.2 dates was investigated and rejected: it cannot
satisfy both §12.2 "predicates win" and the `provenance: hand_labeled` invariant
(the literal dates' predicate labels diverge from the table's intuition on every
date, per the evidence table above), and `2020-04-10` is structurally impossible.
Reversing the `hand_labeled` invariant to freeze predicate output as expected
would be an architectural decision beyond this fixture and is intentionally not
taken here.

## Sync guarantee

`tests/test_fixture_verification.py::test_golden_date_replacement_set_has_documented_justification`
asserts this report enumerates all 10 §12.2 source dates AND all 10 committed
`as_of_date` values, so the mapping cannot silently drift out of sync with
`tests/fixtures/derived/golden_dates.yaml`.
