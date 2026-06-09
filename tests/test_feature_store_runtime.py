from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from regime_detection.feature_store_runtime import (
    FeatureAvailability,
    FeatureSpec,
    _Unavailable,
    _run_feature_specs,
)


@dataclass
class _ToyState:
    """Minimal mutable state for orchestrator tests."""

    inputs: dict[str, int] = field(default_factory=dict)
    outputs: dict[str, int] = field(default_factory=dict)
    context: object | None = None


def _v2_state(**kwargs) -> _ToyState:
    return _ToyState(
        **kwargs,
        context=SimpleNamespace(config=SimpleNamespace(config_version="core3-v2.0.0")),
    )


def _make_spec(
    name: str,
    *,
    required: tuple[str, ...] = (),
    missing: tuple[str, ...] | None = None,
    raises: Exception | None = None,
) -> FeatureSpec[int, _ToyState]:
    def resolve(state: _ToyState):
        if missing is not None:
            return _Unavailable(missing_inputs=missing)
        return {"x": state.inputs.get("x", 0)}

    def build(x: int) -> int:
        if raises is not None:
            raise raises
        return x * 2

    def store(state: _ToyState, value: int) -> None:
        state.outputs[name] = value

    return FeatureSpec(
        name=name,
        policy="raise",
        required_inputs=required,
        resolve=resolve,
        build=build,
        store=store,
    )


def test_resolve_returns_kwargs_then_build_runs_and_store_writes() -> None:
    state = _ToyState(inputs={"x": 3})
    specs = (_make_spec("alpha", required=("x",)),)

    report = _run_feature_specs(specs, state)

    assert state.outputs == {"alpha": 6}
    assert report == {
        "alpha": FeatureAvailability(
            feature="alpha",
            available=True,
            policy="raise",
            reason="populated",
            required_inputs=("x",),
        )
    }


def test_unavailable_with_missing_inputs_raises_missing_required_inputs_reason() -> (
    None
):
    state = _v2_state()
    specs = (_make_spec("beta", required=("x", "y"), missing=("y",)),)

    with pytest.raises(
        RuntimeError,
        match=(
            r"feature spec 'beta' unavailable: missing_required_inputs; "
            r"missing_inputs=\('y',\)"
        ),
    ):
        _run_feature_specs(specs, state)
    assert state.outputs == {}


def test_unavailable_with_empty_missing_raises_not_configured_reason() -> None:
    state = _v2_state()
    specs = (_make_spec("gamma", required=("config",), missing=()),)

    with pytest.raises(
        RuntimeError,
        match=(
            r"feature spec 'gamma' unavailable: not_configured; " r"missing_inputs=\(\)"
        ),
    ):
        _run_feature_specs(specs, state)
    assert state.outputs == {}


def test_build_exception_propagates() -> None:
    state = _ToyState(inputs={"x": 1})
    specs = (_make_spec("delta", required=("x",), raises=RuntimeError("boom")),)

    with pytest.raises(RuntimeError, match="boom"):
        _run_feature_specs(specs, state)


def test_spec_ordering_preserved_in_returned_dict() -> None:
    state = _ToyState(inputs={"x": 1})
    specs = (
        _make_spec("first", required=("x",)),
        _make_spec("second", required=("x",)),
        _make_spec("third", required=("x",)),
    )

    report = _run_feature_specs(specs, state)

    assert list(report.keys()) == ["first", "second", "third"]


def test_spec_with_report_false_runs_but_omits_availability_entry() -> None:
    state = _ToyState(inputs={"x": 5})
    specs = (
        _make_spec("public", required=("x",)),
        FeatureSpec(
            name="internal",
            policy="raise",
            required_inputs=("x",),
            resolve=lambda s: {"x": s.inputs.get("x", 0)},
            build=lambda x: x * 3,
            store=lambda s, v: s.outputs.__setitem__("internal", v),
            report=False,
        ),
    )

    report = _run_feature_specs(specs, state)

    # Build/store side-effects happen for both specs.
    assert state.outputs == {"public": 10, "internal": 15}
    # But only the public spec produces an availability entry.
    assert set(report.keys()) == {"public"}
    assert "internal" not in report


def test_spec_with_report_false_still_raises_when_unavailable() -> None:
    """Internal specs cannot hide missing required inputs under fail-loud mode."""
    state = _v2_state()
    specs = (
        FeatureSpec(
            name="internal_missing",
            policy="none",
            required_inputs=("x",),
            resolve=lambda s: _Unavailable(missing_inputs=("x",)),
            build=lambda x: x * 2,
            store=lambda s, v: s.outputs.__setitem__("internal_missing", v),
            report=False,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"feature spec 'internal_missing' unavailable: "
            r"missing_required_inputs; missing_inputs=\('x',\)"
        ),
    ):
        _run_feature_specs(specs, state)
    assert state.outputs == {}  # build did not run


def test_spec_with_build_returning_none_raises() -> None:
    """When inputs resolve but build returns None, the orchestrator fails loud."""
    state = _v2_state(inputs={"x": 5})
    none_returning_build_spec: FeatureSpec[int | None, _ToyState] = FeatureSpec(
        name="sometimes_none",
        policy="none",
        required_inputs=("x",),
        resolve=lambda s: {"x": s.inputs.get("x", 0)},
        # Build returns None despite valid inputs — simulates a compute_*_features
        # function that can fail to produce a value when intermediate data is
        # insufficient.
        build=lambda x: None,
        store=lambda s, v: (
            s.outputs.__setitem__("sometimes_none", v) if v is not None else None
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"feature spec 'sometimes_none' returned None after inputs resolved; "
            r"insufficient history or failed model fit must fail loudly"
        ),
    ):
        _run_feature_specs((none_returning_build_spec,), state)
    assert state.outputs == {}


def test_spec_with_build_returning_none_and_report_false_raises() -> None:
    """report=False affects observability, not fail-loud validation."""
    state = _v2_state(inputs={"x": 5})
    internal_none_spec: FeatureSpec[int | None, _ToyState] = FeatureSpec(
        name="internal_none",
        policy="none",
        required_inputs=("x",),
        resolve=lambda s: {"x": s.inputs.get("x", 0)},
        build=lambda x: None,
        store=lambda s, v: None,
        report=False,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"feature spec 'internal_none' returned None after inputs resolved; "
            r"insufficient history or failed model fit must fail loudly"
        ),
    ):
        _run_feature_specs((internal_none_spec,), state)
    assert state.outputs == {}


def test_unknown_config_state_reports_unavailable_instead_of_raising() -> None:
    state = _ToyState()
    specs = (_make_spec("legacy_optional", required=("x",), missing=("x",)),)

    report = _run_feature_specs(specs, state)

    assert report["legacy_optional"] == FeatureAvailability(
        feature="legacy_optional",
        available=False,
        policy="raise",
        reason="missing_required_inputs",
        required_inputs=("x",),
        missing_inputs=("x",),
    )
