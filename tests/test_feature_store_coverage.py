from __future__ import annotations

from regime_detection.feature_store import (
    _FEATURE_SPECS,
    _FEATURE_STORE_BUILDERS,
    FeatureStore,
)


_FEATURE_STORE_NON_FEATURE_FIELDS = frozenset({"spy_index", "availability"})


def test_every_feature_store_field_has_a_builder_or_spec() -> None:
    declared = set(FeatureStore.model_fields.keys()) - _FEATURE_STORE_NON_FEATURE_FIELDS
    registered = {s.name for s in _FEATURE_SPECS} | {
        b.name for b in _FEATURE_STORE_BUILDERS
    }
    missing = declared - registered
    assert not missing, (
        f"FeatureStore fields with no spec/builder: {sorted(missing)}"
    )
    # Note: registered may contain intermediate state names (e.g. sentiment_score,
    # realized_vol_21d) that are not FeatureStore fields. That is expected — those
    # features populate _FeatureStoreBuildState only and feed downstream consumers
    # like HMM and clustering. We only assert that every FeatureStore field is covered.


def test_no_feature_appears_in_both_specs_and_legacy_builders() -> None:
    spec_names = {s.name for s in _FEATURE_SPECS}
    builder_names = {b.name for b in _FEATURE_STORE_BUILDERS}
    overlap = spec_names & builder_names
    assert not overlap, (
        f"feature {sorted(overlap)} defined twice (spec + builder); "
        "Task 1.9 should have removed it from _FEATURE_STORE_BUILDERS"
    )


def test_spec_required_inputs_are_unique_within_each_spec() -> None:
    for spec in _FEATURE_SPECS:
        assert len(spec.required_inputs) == len(set(spec.required_inputs)), (
            f"spec {spec.name!r} has duplicate required_inputs: {spec.required_inputs}"
        )


def test_spec_required_inputs_is_a_tuple_not_a_set() -> None:
    for spec in _FEATURE_SPECS:
        assert isinstance(spec.required_inputs, tuple), (
            f"spec {spec.name!r} required_inputs must be tuple "
            f"(deterministic order), got {type(spec.required_inputs).__name__}"
        )
