from __future__ import annotations

from pathlib import Path

from regime_detection.feature_store import _FEATURE_SPECS, FeatureStore

_FEATURE_STORE_NON_FEATURE_FIELDS = frozenset({"spy_index", "availability"})
_REPO_ROOT = Path(__file__).resolve().parents[1]
_V1_CONFIG_PATH = (
    _REPO_ROOT / "src" / "regime_detection" / "configs" / "core3-v1.0.0.yaml"
)


def test_every_feature_store_field_has_a_spec() -> None:
    """After PR 2 cleanup, every FeatureStore field must be registered as a
    spec. The legacy _FEATURE_STORE_BUILDERS registry has been deleted."""
    declared = set(FeatureStore.model_fields.keys()) - _FEATURE_STORE_NON_FEATURE_FIELDS
    registered = {s.name for s in _FEATURE_SPECS}
    missing = declared - registered
    # registered may contain intermediate state names (sentiment_score,
    # news_sentiment_score, realized_vol_21d, drawdown_63d) that are not
    # FeatureStore fields — those specs have report=False. We only assert that
    # every FeatureStore field is covered by some spec; we do NOT assert the
    # converse.
    assert not missing, f"FeatureStore fields with no spec: {sorted(missing)}"


def test_spec_required_inputs_are_unique_within_each_spec() -> None:
    for spec in _FEATURE_SPECS:
        assert len(spec.required_inputs) == len(
            set(spec.required_inputs)
        ), f"spec {spec.name!r} has duplicate required_inputs: {spec.required_inputs}"


def test_spec_required_inputs_is_a_tuple_not_a_set() -> None:
    for spec in _FEATURE_SPECS:
        assert isinstance(spec.required_inputs, tuple), (
            f"spec {spec.name!r} required_inputs must be tuple "
            f"(deterministic order), got {type(spec.required_inputs).__name__}"
        )


_ALLOWED_REASONS = frozenset({"populated", "not_configured", "missing_required_inputs"})


def test_availability_report_uses_only_allowed_reason_strings(
    market_df_for_asof,
) -> None:
    """End-to-end check that every emitted reason string is in the allowed
    vocabulary. Catches accidental wire-format drift."""
    from datetime import date

    from regime_detection.config import load_regime_config
    from regime_detection.feature_store import build_feature_store
    from regime_detection.market_context import build_market_context

    as_of = date(2023, 12, 14)
    context = build_market_context(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        config=load_regime_config(_V1_CONFIG_PATH),
    )
    store = build_feature_store(context)

    for name, availability in store.availability.items():
        assert availability.reason in _ALLOWED_REASONS, (
            f"feature {name!r} emitted reason {availability.reason!r} "
            f"not in allowed vocabulary {sorted(_ALLOWED_REASONS)}"
        )
