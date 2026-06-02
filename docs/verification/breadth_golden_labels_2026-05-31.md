# Breadth Golden Labels - 2026-05-31

F-011 exposed that the hand-labeled golden fixture had one ambiguous
`breadth_state` expectation, while the engine contract emits both
`raw_label` and `active_label`. Raw labels are the daily rule predicate
result. Active labels are the post-hysteresis output consumed downstream.

The fixture now pins `breadth_state_raw` and `breadth_state_active` so tests
can verify both contracts without treating hysteresis as a rule mismatch.

| as_of_date | intent_id | previous breadth_state | breadth_state_raw | breadth_state_active | reason |
| --- | --- | --- | --- | --- | --- |
| 2020-08-11 | summer2020_bull_trending_lowvol | neutral_breadth | healthy_breadth | divergent_fragile | Raw ETF-proxy breadth is healthy; active label remains divergent_fragile from prior-state hysteresis. |
| 2018-02-08 | volmageddon_crisis | weak_breadth | weak_breadth | weak_breadth | Pre-2019 row is outside the current classified V2 golden fixture; keep the existing manual stress label explicit for both contracts. |
| 2018-12-11 | dec2018_bear_stress | weak_breadth | weak_breadth | weak_breadth | Pre-2019 row is outside the current classified V2 golden fixture; keep the existing manual stress label explicit for both contracts. |
| 2019-09-12 | mid2019_bull_normal | healthy_breadth | healthy_breadth | healthy_breadth | Raw and active labels agree. |
| 2020-03-30 | covid_crash_crisis | weak_breadth | weak_breadth | weak_breadth | Raw and active labels agree. |
| 2020-04-17 | covid_recovery_attempt | weak_breadth | weak_breadth | weak_breadth | Raw and active labels agree. |
| 2021-11-01 | late2021_bull_lowvol | weak_breadth | weak_breadth | weak_breadth | Raw and active labels agree. |
| 2022-06-15 | jun2022_bear_crisis | weak_breadth | narrowing_breadth | narrowing_breadth | Current breadth rules classify this as narrowing rather than weak. |
| 2022-07-06 | jul2022_bear_stress | weak_breadth | weak_breadth | weak_breadth | Raw and active labels agree. |
| 2023-12-05 | early2024_bull_lowvol | healthy_breadth | healthy_breadth | weak_breadth | Raw ETF-proxy breadth is healthy; active label remains weak_breadth from prior-state hysteresis. |

The old `breadth_state` key remains for historical context, but F-011 tests no
longer consume it. New breadth golden checks must assert the explicit raw and
active keys.
