from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

__all__ = [
    "FeatureAvailability",
    "FeatureAvailabilityPolicy",
    "FeatureSpec",
    "_Unavailable",
    "_run_feature_specs",
]

FeatureAvailabilityPolicy = Literal["raise", "none", "unknown", "degraded"]

T = TypeVar("T")
StateT = TypeVar("StateT")
FeatureInputs = dict[str, Any]


class FeatureAvailability(BaseModel):
    """Declared availability result for one feature seam."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    available: bool
    policy: FeatureAvailabilityPolicy
    reason: str
    required_inputs: tuple[str, ...] = ()
    missing_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Unavailable:
    """Sentinel returned by `FeatureSpec.resolve` when required inputs are absent."""

    missing_inputs: tuple[str, ...]


@dataclass(frozen=True)
class FeatureSpec(Generic[T, StateT]):
    """One feature's complete contract: how to gate, build, and store its value.

    `resolve` returns either a kwargs dict to splat into `build`, or `_Unavailable`
    listing the absent required inputs. `build` is total over its typed parameters
    — it has no internal None-guards. `store` writes the built value back into state.

    `report` controls whether the orchestrator emits a `FeatureAvailability` entry
    for this spec. Set to `False` for intermediate state features that are not
    user-observable (e.g. derived series consumed only by other specs). Default
    True — public features that should appear in `FeatureStore.availability`.

    If `build` returns None despite a successful `resolve`, the orchestrator emits
    `available=False, reason="not_configured"` — matching legacy `_availability`
    semantics for `compute_*_features` functions whose preconditions can't always
    be expressed as resolve gates.
    """

    name: str
    policy: FeatureAvailabilityPolicy
    required_inputs: tuple[str, ...]
    resolve: Callable[[StateT], FeatureInputs | _Unavailable]
    build: Callable[..., T]
    store: Callable[[StateT, T], None]
    report: bool = True


def _run_feature_specs(
    specs: tuple[FeatureSpec[Any, StateT], ...],
    state: StateT,
) -> dict[str, FeatureAvailability]:
    report: dict[str, FeatureAvailability] = {}
    for spec in specs:
        resolved = spec.resolve(state)
        if isinstance(resolved, _Unavailable):
            if spec.report:
                reason = (
                    "not_configured"
                    if not resolved.missing_inputs
                    else "missing_required_inputs"
                )
                report[spec.name] = FeatureAvailability(
                    feature=spec.name,
                    available=False,
                    policy=spec.policy,
                    reason=reason,
                    required_inputs=spec.required_inputs,
                    missing_inputs=resolved.missing_inputs,
                )
            continue
        value = spec.build(**resolved)
        spec.store(state, value)
        if spec.report:
            if value is None:
                # Build returned None despite valid inputs — match legacy
                # _availability helper's value-is-None semantics. Some
                # compute_*_features functions can fail to produce a value
                # when intermediate-data preconditions (e.g. training-window
                # length) are not met; those preconditions can't always be
                # gated in resolve.
                report[spec.name] = FeatureAvailability(
                    feature=spec.name,
                    available=False,
                    policy=spec.policy,
                    reason="not_configured",
                    required_inputs=spec.required_inputs,
                )
            else:
                report[spec.name] = FeatureAvailability(
                    feature=spec.name,
                    available=True,
                    policy=spec.policy,
                    reason="populated",
                    required_inputs=spec.required_inputs,
                )
    return report
