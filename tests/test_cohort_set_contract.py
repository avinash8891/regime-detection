"""Enumerative gate for resolution finding F-015 (§9 G2 anti-recurrence gate).

The set of cohorts the router can ever emit must equal the spec §5.1 *closed*
set of 10 cohorts: 8 axis-predicate specialists + ``data_outage_specialist``
(fail-closed) + ``default_neutral`` (fallback). This test fails if code adds an
11th cohort or drops one, so spec §5.1 and the router cannot silently drift.

The expected set is the spec contract (docs/regime_engine_v2_spec.md §5.1),
intentionally pinned here rather than derived from ``COHORTS`` — deriving it from
the code would make the gate circular and unable to catch a wrong cohort.
"""

from __future__ import annotations

from regime_detection.cohort_routing import COHORTS, _DATA_OUTAGE

# docs/regime_engine_v2_spec.md §5.1 (cohort list + precedence + Ambiguity Log pin).
_SPEC_5_1_COHORTS = frozenset(
    {
        "crisis_specialist",
        "euphoria_specialist",
        "bear_stress_specialist",
        "tightening_specialist",
        "easing_specialist",
        "recovery_specialist",
        "chop_mean_reversion_specialist",
        "bull_low_vol_specialist",
        "data_outage_specialist",
        "default_neutral",
    }
)


def test_router_emittable_cohort_set_equals_spec_5_1_closed_set() -> None:
    emittable = frozenset(COHORTS) | {_DATA_OUTAGE}
    assert emittable == _SPEC_5_1_COHORTS
    assert len(_SPEC_5_1_COHORTS) == 10


def test_data_outage_specialist_is_a_distinct_fail_closed_cohort() -> None:
    # F-015: data_outage_specialist is now a formal member of the closed §5.1 cohort set
    # (the 10th cohort, in COHORTS at the precedence position after crisis), but it
    # remains DISTINCT — it carries NO routing_rule and is emitted by the dedicated
    # fail-closed branch (any core risk axis "unknown"), not an axis-predicate match,
    # and it is not the permissive fallback.
    from regime_detection.config import load_default_regime_config

    assert _DATA_OUTAGE == "data_outage_specialist"
    assert _DATA_OUTAGE in COHORTS  # formalized 10th cohort
    assert COHORTS.index(_DATA_OUTAGE) == 1  # precedence: right after crisis_specialist
    assert _DATA_OUTAGE != "default_neutral"
    # Distinctness: no axis-predicate routing rule — it is fail-closed, not rule-matched.
    routing_rules = load_default_regime_config().cohort_routing.routing_rules
    assert _DATA_OUTAGE not in routing_rules
