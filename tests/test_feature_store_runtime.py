from __future__ import annotations

from dataclasses import dataclass, field

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


def test_unavailable_with_missing_inputs_emits_missing_required_inputs_reason() -> None:
    state = _ToyState()
    specs = (_make_spec("beta", required=("x", "y"), missing=("y",)),)

    report = _run_feature_specs(specs, state)

    assert state.outputs == {}
    assert report["beta"].available is False
    assert report["beta"].reason == "missing_required_inputs"
    assert report["beta"].missing_inputs == ("y",)
    assert report["beta"].required_inputs == ("x", "y")


def test_unavailable_with_empty_missing_emits_not_configured_reason() -> None:
    state = _ToyState()
    specs = (_make_spec("gamma", required=("config",), missing=()),)

    report = _run_feature_specs(specs, state)

    assert state.outputs == {}
    assert report["gamma"].reason == "not_configured"
    assert report["gamma"].missing_inputs == ()


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
