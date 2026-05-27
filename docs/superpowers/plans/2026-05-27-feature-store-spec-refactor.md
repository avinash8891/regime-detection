# Feature Store Spec Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the parallel availability registry in `feature_store.py` into per-feature `FeatureSpec` objects so the predicate that decides "buildable" is the same code path that produces the availability report. Eliminate ~228 lines of `_build_feature_availability_report` and ~15 implicit `if config is None: state.x = None; return` guards.

**Architecture:** New `feature_store_runtime.py` holds `FeatureSpec`, `_Unavailable` sentinel, and `_run_feature_specs` orchestrator. Each feature becomes a `FeatureSpec(name, policy, required_inputs, resolve, build, store)`. `resolve` returns typed kwargs or `_Unavailable`; the orchestrator calls `build` only on success and derives availability from the same decision. Two PRs: PR 1 = foundation + 5 always-on V1 features (orchestrator runs alongside old `_FEATURE_STORE_BUILDERS` for the unmigrated 15). PR 2 = remaining 15 features + cleanup.

**Tech Stack:** Python 3.12, Pydantic v2, pandas, pytest. Type checker: pyright strict. Linter: ruff.

**Spec:** `docs/superpowers/specs/2026-05-27-feature-store-spec-refactor-design.md`

---

## File Structure

**Created:**
- `src/regime_detection/feature_store_runtime.py` — `FeatureSpec`, `_Unavailable`, `_run_feature_specs`, `FeatureAvailability` (moved from feature_store.py).
- `tests/test_feature_store_runtime.py` — orchestrator unit tests against toy specs.
- `tests/test_feature_store_coverage.py` — CI guard: every `FeatureStore` field has a spec; `required_inputs` deterministic and unique.
- `tests/test_feature_availability_golden.py` — pins `availability` dict for two `MarketContext` fixtures across the refactor.

**Modified:**
- `src/regime_detection/feature_store.py` — re-exports `FeatureAvailability` (for backward compat), adds `_FEATURE_SPECS` registry, migrates 5 always-on builders to specs in PR 1 and 15 more in PR 2, deletes `_FeatureStoreBuilder` / `_FEATURE_STORE_BUILDERS` / `_build_feature_availability_report` / `_availability` in PR 2.
- `tests/test_feature_store_refactors.py` — rewritten in PR 2 to test the `_FEATURE_SPECS` registry instead of `_FEATURE_STORE_BUILDERS`.
- `pyproject.toml` — add `feature_store_runtime.py` to pyright strict include set in PR 1.

**Untouched (do not modify):**
- All `compute_*_features` modules (`trend_direction.py`, `hmm_state.py`, etc.). Specs wrap them; they keep their signatures.
- `axis_series.py`, `models.py`, `boundary_policies.py`, `engine.py`. Out of scope per spec.

---

## Universal TDD Ritual

Every spec-migration task follows this five-step ritual. Steps embed the ritual inline per task, but reference this section for the rationale.

1. Write the failing test (resolution + build wiring).
2. Run the test to verify it fails with a sensible error.
3. Write the spec (`resolve` + `build` + `store`).
4. Run the test to verify it passes.
5. Run the full feature-store test subset to verify no regression, then commit.

---

# PR 1 — Foundation + 5 Always-On Features

## Task 1.1: Create `feature_store_runtime.py` skeleton

**Files:**
- Create: `src/regime_detection/feature_store_runtime.py`
- Modify: `pyproject.toml` (add to pyright include)

- [ ] **Step 1: Create the module with `FeatureSpec`, `_Unavailable`, `FeatureAvailability`**

```python
# src/regime_detection/feature_store_runtime.py
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
    """

    name: str
    policy: FeatureAvailabilityPolicy
    required_inputs: tuple[str, ...]
    resolve: Callable[[StateT], FeatureInputs | _Unavailable]
    build: Callable[..., T]
    store: Callable[[StateT, T], None]


def _run_feature_specs(
    specs: tuple[FeatureSpec[Any, StateT], ...],
    state: StateT,
) -> dict[str, FeatureAvailability]:
    report: dict[str, FeatureAvailability] = {}
    for spec in specs:
        resolved = spec.resolve(state)
        if isinstance(resolved, _Unavailable):
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
        report[spec.name] = FeatureAvailability(
            feature=spec.name,
            available=True,
            policy=spec.policy,
            reason="populated",
            required_inputs=spec.required_inputs,
        )
    return report
```

- [ ] **Step 2: Add to pyright strict include set**

Modify `pyproject.toml`:

```toml
[tool.pyright]
typeCheckingMode = "strict"
include = [
  "src/regime_detection/engine.py",
  "src/regime_detection/models.py",
  "src/regime_detection/axis_series.py",
  "src/regime_detection/feature_store.py",
  "src/regime_detection/feature_store_runtime.py",  # NEW
  "src/regime_detection/timeline.py",
]
```

- [ ] **Step 3: Run pyright to confirm strict mode passes on the new file**

Run: `python -m pyright src/regime_detection/feature_store_runtime.py`
Expected: 0 errors, 0 warnings.

- [ ] **Step 4: Commit**

```bash
git add src/regime_detection/feature_store_runtime.py pyproject.toml
git commit -m "feat(feature-store): add FeatureSpec orchestrator skeleton

Introduces feature_store_runtime.py with FeatureSpec, _Unavailable sentinel,
and _run_feature_specs orchestrator. Foundation for the refactor that collapses
the parallel availability registry. No callers yet."
```

---

## Task 1.2: Orchestrator unit tests

**Files:**
- Create: `tests/test_feature_store_runtime.py`

- [ ] **Step 1: Write failing tests for the four orchestrator branches**

```python
# tests/test_feature_store_runtime.py
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
```

- [ ] **Step 2: Run tests to confirm they fail (functions exist but tests not yet exercised)**

Run: `python -m pytest tests/test_feature_store_runtime.py -v 2>&1 | tail -30 ; echo "EXIT:$?"`
Expected: All 5 tests PASS (the orchestrator was implemented in Task 1.1; these tests pin its contract). EXIT:0.

If any fail: the orchestrator implementation in Task 1.1 has a bug — fix it and re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_feature_store_runtime.py
git commit -m "test(feature-store): pin orchestrator contract for FeatureSpec

Covers the four orchestrator branches (kwargs→build→store; _Unavailable with
missing inputs; _Unavailable with empty missing; build raises) plus spec
ordering preservation. Toy state isolates the orchestrator from MarketContext."
```

---

## Task 1.3: Move `FeatureAvailability` import in `feature_store.py`

The model is now defined in `feature_store_runtime.py`. Re-export from `feature_store.py` to preserve the public import path (`from regime_detection.feature_store import FeatureAvailability`).

**Files:**
- Modify: `src/regime_detection/feature_store.py:207-224` (delete local class), `feature_store.py:8` (update Pydantic imports), `feature_store.py:114` (keep export)

- [ ] **Step 1: Replace the local `FeatureAvailability` definition with an import**

In `src/regime_detection/feature_store.py`, find the block at line 207-224 starting with `class FeatureAvailability(BaseModel):` through the end of the class. Replace with: nothing (delete the block).

At the top of the file, add to the imports (next to other `regime_detection.*` imports):

```python
from regime_detection.feature_store_runtime import (
    FeatureAvailability,
    FeatureAvailabilityPolicy,
    FeatureSpec,
    _Unavailable,
    _run_feature_specs,
)
```

Delete the existing local definition at line 134:
```python
FeatureAvailabilityPolicy = Literal["raise", "none", "unknown", "degraded"]
```
(Now imported from runtime.)

The `__all__` block at line 114 already includes `"FeatureAvailability"` — leave it. This makes `from regime_detection.feature_store import FeatureAvailability` continue to work.

- [ ] **Step 2: Run pyright and ruff to confirm no new errors**

Run: `python -m pyright src/regime_detection/feature_store.py src/regime_detection/feature_store_runtime.py ; echo "EXIT:$?"`
Expected: EXIT:0.

Run: `python -m ruff check src/regime_detection/feature_store.py ; echo "EXIT:$?"`
Expected: EXIT:0.

- [ ] **Step 3: Run existing feature store tests to confirm no regression**

Run: `python -m pytest tests/test_v2_feature_store_and_axis_seams.py tests/test_feature_store_refactors.py -v 2>&1 | tail -40 ; echo "EXIT:$?"`
Expected: All tests PASS. EXIT:0.

- [ ] **Step 4: Commit**

```bash
git add src/regime_detection/feature_store.py
git commit -m "refactor(feature-store): move FeatureAvailability to runtime module

Re-exports from feature_store.py preserve the public import path. No behavior
change. Sets up the orchestrator wiring in subsequent tasks."
```

---

## Task 1.4: Migrate `trend_direction` to a spec

**Pattern (applies to all 5 always-on specs):**
- `policy="raise"` (matches today's `_build_feature_availability_report`)
- `resolve` returns kwargs unconditionally — never returns `_Unavailable` (per locked-in decision #2: preserve raise-on-missing-SPY)
- `build` is a thin wrapper around the existing `compute_*` function

**Files:**
- Modify: `src/regime_detection/feature_store.py` (add spec, leave old builder in place for now)
- Modify: `tests/test_feature_store_refactors.py` (add resolution test) OR create `tests/test_feature_store_specs.py` for per-spec tests

> **Decision:** Per-spec resolution tests go in a new file `tests/test_feature_store_specs.py` to keep them grouped. `test_feature_store_refactors.py` is rewritten in PR 2.

- [ ] **Step 1: Write the failing resolution test**

Create `tests/test_feature_store_specs.py` with:

```python
# tests/test_feature_store_specs.py
from __future__ import annotations

import pandas as pd
import pytest

from regime_detection.feature_store import _FEATURE_SPECS, build_feature_store


def _spec_by_name(name: str):
    matches = [s for s in _FEATURE_SPECS if s.name == name]
    if not matches:
        raise AssertionError(f"no spec named {name!r} in _FEATURE_SPECS")
    if len(matches) > 1:
        raise AssertionError(f"duplicate specs named {name!r}: {matches}")
    return matches[0]


def test_trend_direction_resolve_returns_spy_close_kwargs() -> None:
    from regime_detection.feature_store import _FeatureStoreBuildState
    from tests.helpers.market_context import make_minimal_market_context

    ctx = make_minimal_market_context()
    state = _FeatureStoreBuildState(
        context=ctx, spy_ohlcv=ctx.spy_ohlcv, spy_close=ctx.spy_ohlcv["close"]
    )

    spec = _spec_by_name("trend_direction")
    resolved = spec.resolve(state)

    assert isinstance(resolved, dict)
    assert set(resolved.keys()) == {"spy_close"}
    assert resolved["spy_close"] is state.spy_close
```

> **Note on `tests/helpers/market_context.py`:** if a `make_minimal_market_context` helper already exists in the test suite, import it. If not, add one in Step 2.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_feature_store_specs.py::test_trend_direction_resolve_returns_spy_close_kwargs -v 2>&1 | tail -20 ; echo "EXIT:$?"`
Expected: FAIL with `ImportError: cannot import name '_FEATURE_SPECS'` (the registry doesn't exist yet) OR FAIL with "no spec named 'trend_direction'" once the registry is empty.

- [ ] **Step 3: Add the `trend_direction` spec and create `_FEATURE_SPECS` registry**

In `src/regime_detection/feature_store.py`, before the existing `_FEATURE_STORE_BUILDERS` tuple at line 738, add:

```python
# --- New spec-based builders (PR 1) ------------------------------------------

def _build_trend_direction(spy_close: pd.Series) -> TrendDirectionFeatures:
    return compute_trend_direction_features(spy_close)


def _resolve_trend_direction(state: _FeatureStoreBuildState):
    return {"spy_close": state.spy_close}


_FEATURE_SPECS: tuple[FeatureSpec[object, _FeatureStoreBuildState], ...] = (
    FeatureSpec(
        name="trend_direction",
        policy="raise",
        required_inputs=("spy_ohlcv.close",),
        resolve=_resolve_trend_direction,
        build=_build_trend_direction,
        store=lambda s, v: setattr(s, "trend_direction", v),
    ),
)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_feature_store_specs.py::test_trend_direction_resolve_returns_spy_close_kwargs -v 2>&1 | tail -10 ; echo "EXIT:$?"`
Expected: PASS. EXIT:0.

- [ ] **Step 5: Commit (no wiring change yet — old builder still runs)**

```bash
git add src/regime_detection/feature_store.py tests/test_feature_store_specs.py
git commit -m "feat(feature-store): add trend_direction spec

First always-on V1 feature migrated to FeatureSpec. Old _FeatureStoreBuilder
for trend_direction stays in place until Task 1.9 wires the orchestrator into
build_feature_store. Both define trend_direction; the spec is the canonical
form once wired."
```

---

## Task 1.5: Migrate `trend_character` to a spec

**Files:**
- Modify: `src/regime_detection/feature_store.py` (add spec to `_FEATURE_SPECS`)
- Modify: `tests/test_feature_store_specs.py` (add resolution test)

`trend_character` reads `state.spy_ohlcv`, `state.spy_close`, and the optional `state.context.config.trend_character_v2`. The `_build_trend_character_feature` at line 422-448 has two branches (with/without v2 config). The spec preserves both branches inside `build`.

- [ ] **Step 1: Write the failing resolution test**

Append to `tests/test_feature_store_specs.py`:

```python
def test_trend_character_resolve_returns_ohlcv_kwargs_v1_path() -> None:
    from regime_detection.feature_store import _FeatureStoreBuildState
    from tests.helpers.market_context import make_minimal_market_context

    ctx = make_minimal_market_context()
    state = _FeatureStoreBuildState(
        context=ctx, spy_ohlcv=ctx.spy_ohlcv, spy_close=ctx.spy_ohlcv["close"]
    )

    spec = _spec_by_name("trend_character")
    resolved = spec.resolve(state)

    assert isinstance(resolved, dict)
    assert {"close", "high", "low", "volume", "tc_v2_config"}.issubset(resolved.keys())
    assert resolved["close"] is state.spy_close
    assert resolved["tc_v2_config"] is None  # v1-only context
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_feature_store_specs.py::test_trend_character_resolve_returns_ohlcv_kwargs_v1_path -v 2>&1 | tail -10 ; echo "EXIT:$?"`
Expected: FAIL ("no spec named 'trend_character'").

- [ ] **Step 3: Add the spec**

In `src/regime_detection/feature_store.py`, add:

```python
def _build_trend_character(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series | None,
    tc_v2_config,  # TrendCharacterV2Config | None — type from regime_detection.config
) -> TrendCharacterFeatures:
    if tc_v2_config is not None:
        return compute_trend_character_features(
            close=close, high=high, low=low, volume=volume,
            bb_width_period=tc_v2_config.bb_width_period,
            bb_width_multiplier=tc_v2_config.bb_width_multiplier,
            bb_width_expanding_lookback=tc_v2_config.bb_width_expanding_lookback,
            followthrough_lookback_sessions=tc_v2_config.followthrough_lookback_sessions,
            followthrough_window_count=tc_v2_config.followthrough_window_count,
            followthrough_hold_sessions=tc_v2_config.followthrough_hold_sessions,
        )
    return compute_trend_character_features(close=close, high=high, low=low, volume=volume)


def _resolve_trend_character(state: _FeatureStoreBuildState):
    volume = (
        _series_column(state.spy_ohlcv, "volume")
        if "volume" in state.spy_ohlcv.columns
        else None
    )
    return {
        "close": state.spy_close,
        "high": _series_column(state.spy_ohlcv, "high"),
        "low": _series_column(state.spy_ohlcv, "low"),
        "volume": volume,
        "tc_v2_config": state.context.config.trend_character_v2,
    }
```

Append to the `_FEATURE_SPECS` tuple:

```python
    FeatureSpec(
        name="trend_character",
        policy="raise",
        required_inputs=("spy_ohlcv.close", "spy_ohlcv.high", "spy_ohlcv.low"),
        resolve=_resolve_trend_character,
        build=_build_trend_character,
        store=lambda s, v: setattr(s, "trend_character", v),
    ),
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_feature_store_specs.py::test_trend_character_resolve_returns_ohlcv_kwargs_v1_path -v 2>&1 | tail -10 ; echo "EXIT:$?"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/regime_detection/feature_store.py tests/test_feature_store_specs.py
git commit -m "feat(feature-store): add trend_character spec

Preserves both v1 and v2 branches of compute_trend_character_features inside
the spec's build. Old _FeatureStoreBuilder still runs until Task 1.9."
```

---

## Task 1.6: Migrate `volatility` to a spec

**Files:**
- Modify: `src/regime_detection/feature_store.py`
- Modify: `tests/test_feature_store_specs.py`

- [ ] **Step 1: Write the failing test**

```python
def test_volatility_resolve_returns_close_and_vix_proxy() -> None:
    from regime_detection.feature_store import _FeatureStoreBuildState
    from tests.helpers.market_context import make_minimal_market_context

    ctx = make_minimal_market_context()
    state = _FeatureStoreBuildState(
        context=ctx, spy_ohlcv=ctx.spy_ohlcv, spy_close=ctx.spy_ohlcv["close"]
    )

    spec = _spec_by_name("volatility")
    resolved = spec.resolve(state)

    assert isinstance(resolved, dict)
    assert set(resolved.keys()) == {"close", "vix_proxy_close"}
    assert resolved["close"] is state.spy_close
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_feature_store_specs.py::test_volatility_resolve_returns_close_and_vix_proxy -v 2>&1 | tail -10`
Expected: FAIL.

- [ ] **Step 3: Add the spec**

```python
def _build_volatility(close: pd.Series, vix_proxy_close: pd.Series | None) -> VolatilityFeatures:
    return compute_volatility_features(close=close, vix_proxy_close=vix_proxy_close)


def _resolve_volatility(state: _FeatureStoreBuildState):
    return {"close": state.spy_close, "vix_proxy_close": state.context.vix_proxy_close}
```

Append to `_FEATURE_SPECS`:
```python
    FeatureSpec(
        name="volatility",
        policy="raise",
        required_inputs=("spy_ohlcv.close",),
        resolve=_resolve_volatility,
        build=_build_volatility,
        store=lambda s, v: setattr(s, "volatility", v),
    ),
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_feature_store_specs.py::test_volatility_resolve_returns_close_and_vix_proxy -v 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/regime_detection/feature_store.py tests/test_feature_store_specs.py
git commit -m "feat(feature-store): add volatility spec"
```

---

## Task 1.7: Migrate `breadth` to a spec

**Files:**
- Modify: `src/regime_detection/feature_store.py`
- Modify: `tests/test_feature_store_specs.py`

`breadth` reads `state.spy_close`, `state.context.rsp_close`, and reindexes against `state.spy_ohlcv.index`. The spec resolve produces three kwargs.

- [ ] **Step 1: Write the failing test**

```python
def test_breadth_resolve_returns_spy_close_and_aligned_rsp() -> None:
    from regime_detection.feature_store import _FeatureStoreBuildState
    from tests.helpers.market_context import make_minimal_market_context

    ctx = make_minimal_market_context()
    state = _FeatureStoreBuildState(
        context=ctx, spy_ohlcv=ctx.spy_ohlcv, spy_close=ctx.spy_ohlcv["close"]
    )

    spec = _spec_by_name("breadth")
    resolved = spec.resolve(state)

    assert isinstance(resolved, dict)
    assert set(resolved.keys()) == {"spy_close", "rsp_close"}
    # rsp_close should already be reindexed against spy_ohlcv.index
    assert list(resolved["rsp_close"].index) == list(state.spy_ohlcv.index)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_feature_store_specs.py::test_breadth_resolve_returns_spy_close_and_aligned_rsp -v 2>&1 | tail -10`
Expected: FAIL.

- [ ] **Step 3: Add the spec**

```python
def _build_breadth(spy_close: pd.Series, rsp_close: pd.Series) -> BreadthFeatures:
    return compute_breadth_features(spy_close=spy_close, rsp_close=rsp_close)


def _resolve_breadth(state: _FeatureStoreBuildState):
    return {
        "spy_close": state.spy_close,
        "rsp_close": state.context.rsp_close.reindex(state.spy_ohlcv.index),
    }
```

Append to `_FEATURE_SPECS`:
```python
    FeatureSpec(
        name="breadth",
        policy="raise",
        required_inputs=("spy_ohlcv.close", "rsp_close"),
        resolve=_resolve_breadth,
        build=_build_breadth,
        store=lambda s, v: setattr(s, "breadth", v),
    ),
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_feature_store_specs.py::test_breadth_resolve_returns_spy_close_and_aligned_rsp -v 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/regime_detection/feature_store.py tests/test_feature_store_specs.py
git commit -m "feat(feature-store): add breadth spec"
```

---

## Task 1.8: Migrate `sma_50` to a spec

**Files:**
- Modify: `src/regime_detection/feature_store.py`
- Modify: `tests/test_feature_store_specs.py`

- [ ] **Step 1: Write the failing test**

```python
def test_sma_50_resolve_returns_spy_close() -> None:
    from regime_detection.feature_store import _FeatureStoreBuildState
    from tests.helpers.market_context import make_minimal_market_context

    ctx = make_minimal_market_context()
    state = _FeatureStoreBuildState(
        context=ctx, spy_ohlcv=ctx.spy_ohlcv, spy_close=ctx.spy_ohlcv["close"]
    )

    spec = _spec_by_name("sma_50")
    resolved = spec.resolve(state)

    assert isinstance(resolved, dict)
    assert set(resolved.keys()) == {"spy_close"}
    assert resolved["spy_close"] is state.spy_close
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_feature_store_specs.py::test_sma_50_resolve_returns_spy_close -v 2>&1 | tail -10`
Expected: FAIL.

- [ ] **Step 3: Add the spec**

```python
def _build_sma_50(spy_close: pd.Series) -> pd.Series:
    return simple_moving_average(spy_close, window=50)


def _resolve_sma_50(state: _FeatureStoreBuildState):
    return {"spy_close": state.spy_close}
```

Append to `_FEATURE_SPECS`:
```python
    FeatureSpec(
        name="sma_50",
        policy="raise",
        required_inputs=("spy_ohlcv.close",),
        resolve=_resolve_sma_50,
        build=_build_sma_50,
        store=lambda s, v: setattr(s, "sma_50", v),
    ),
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_feature_store_specs.py::test_sma_50_resolve_returns_spy_close -v 2>&1 | tail -10`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/regime_detection/feature_store.py tests/test_feature_store_specs.py
git commit -m "feat(feature-store): add sma_50 spec — completes 5 always-on PR 1 specs"
```

---

## Task 1.9: Wire `_FEATURE_SPECS` into `build_feature_store`

Now the 5 always-on features have specs AND old builders. Replace the old builders for these 5 with the orchestrator; leave the remaining 15 old builders intact. The orchestrator runs first, then the remaining old builders. Availability report = `{orchestrator output} | {hand-written entries for the unmigrated 15}`.

**Files:**
- Modify: `src/regime_detection/feature_store.py:738-758` (remove 5 migrated builders from `_FEATURE_STORE_BUILDERS`)
- Modify: `src/regime_detection/feature_store.py:1043+` (in `build_feature_store`, call orchestrator then old builders)
- Modify: `src/regime_detection/feature_store.py:814-1041` (in `_build_feature_availability_report`, remove the 5 hardcoded entries for migrated features)

- [ ] **Step 1: Update `_FEATURE_STORE_BUILDERS` to exclude the 5 migrated features**

Remove these 5 lines from `_FEATURE_STORE_BUILDERS`:
```python
_FeatureStoreBuilder("trend_direction", _build_trend_direction_feature),
_FeatureStoreBuilder("trend_character", _build_trend_character_feature),
_FeatureStoreBuilder("volatility", _build_volatility_feature),
_FeatureStoreBuilder("breadth", _build_breadth_feature),
_FeatureStoreBuilder("sma_50", _build_sma_50_feature),
```

Delete the now-unused `_build_trend_direction_feature`, `_build_trend_character_feature`, `_build_volatility_feature`, `_build_breadth_feature`, `_build_sma_50_feature` functions (lines 391-466 minus the others). Keep all other `_build_*` functions.

- [ ] **Step 2: Remove the 5 hardcoded report entries**

In `_build_feature_availability_report` (line 814), delete the 5 entries for `"trend_direction"`, `"trend_character"`, `"volatility"`, `"breadth"`, `"sma_50"` at lines 877-911.

- [ ] **Step 3: Wire the orchestrator into `build_feature_store`**

In `build_feature_store` (line 1043), find the call to `_run_feature_store_builders(_FEATURE_STORE_BUILDERS, build_state)` and the call to `_build_feature_availability_report(build_state)`. Replace with:

```python
    spec_report = _run_feature_specs(_FEATURE_SPECS, build_state)
    _run_feature_store_builders(_FEATURE_STORE_BUILDERS, build_state)
    legacy_report = _build_feature_availability_report(build_state)
    combined_availability = {**spec_report, **legacy_report}
    ...
    availability=combined_availability,
```

(Adjust to match the actual existing call sites — read lines 1043-1085 first to see exact construction.)

- [ ] **Step 4: Run the full feature-store test subset**

Run: `python -m pytest tests/test_v2_feature_store_and_axis_seams.py tests/test_feature_store_refactors.py tests/test_feature_store_specs.py -v 2>&1 | tail -40 ; echo "EXIT:$?"`
Expected: All PASS. EXIT:0.

If any test fails: read the diff carefully — the most likely cause is the orchestrator output overriding a legacy entry that has different `reason` or `required_inputs`. Check the merge order and the per-feature differences.

- [ ] **Step 5: Commit**

```bash
git add src/regime_detection/feature_store.py
git commit -m "refactor(feature-store): wire 5 always-on specs through orchestrator

Removes _FeatureStoreBuilder entries and report dict entries for the 5 always-on
V1 features. build_feature_store now runs _FEATURE_SPECS via _run_feature_specs
first, then the remaining 15 legacy builders. Availability dict is merged from
both sources. PR 2 collapses everything onto the spec path."
```

---

## Task 1.10: Coverage CI test

**Files:**
- Create: `tests/test_feature_store_coverage.py`

Asserts: every `FeatureStore` model field (minus `spy_index`, `availability`) is registered either as a `FeatureSpec.name` or in `_FEATURE_STORE_BUILDERS`. Asserts `required_inputs` uniqueness and deterministic order per locked-in decision #4.

- [ ] **Step 1: Write the test**

```python
# tests/test_feature_store_coverage.py
from __future__ import annotations

from regime_detection.feature_store import (
    _FEATURE_SPECS,
    _FEATURE_STORE_BUILDERS,
    FeatureStore,
)


_FEATURE_STORE_NON_FEATURE_FIELDS = {"spy_index", "availability"}


def test_every_feature_store_field_has_a_builder_or_spec() -> None:
    declared = set(FeatureStore.model_fields.keys()) - _FEATURE_STORE_NON_FEATURE_FIELDS
    registered = {s.name for s in _FEATURE_SPECS} | {
        b.name for b in _FEATURE_STORE_BUILDERS
    }
    missing = declared - registered
    extra = registered - declared
    assert not missing, f"FeatureStore fields with no spec/builder: {sorted(missing)}"
    assert not extra, f"Registered specs/builders with no FeatureStore field: {sorted(extra)}"


def test_no_feature_appears_in_both_specs_and_legacy_builders() -> None:
    spec_names = {s.name for s in _FEATURE_SPECS}
    builder_names = {b.name for b in _FEATURE_STORE_BUILDERS}
    overlap = spec_names & builder_names
    assert not overlap, f"feature {sorted(overlap)} defined twice (spec + builder)"


def test_spec_required_inputs_are_unique_within_each_spec() -> None:
    for spec in _FEATURE_SPECS:
        assert len(spec.required_inputs) == len(set(spec.required_inputs)), (
            f"spec {spec.name!r} has duplicate required_inputs: {spec.required_inputs}"
        )


def test_spec_required_inputs_is_a_tuple_not_a_set() -> None:
    for spec in _FEATURE_SPECS:
        assert isinstance(spec.required_inputs, tuple), (
            f"spec {spec.name!r} required_inputs must be tuple (deterministic order), "
            f"got {type(spec.required_inputs).__name__}"
        )
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest tests/test_feature_store_coverage.py -v 2>&1 | tail -20 ; echo "EXIT:$?"`
Expected: All PASS. EXIT:0.

If `test_every_feature_store_field_has_a_builder_or_spec` fails: a feature was deleted from one registry without being added to the other. Look at the `missing` / `extra` set printed in the failure.

- [ ] **Step 3: Commit**

```bash
git add tests/test_feature_store_coverage.py
git commit -m "test(feature-store): coverage guard against orphan features

Asserts every FeatureStore field is registered either as a spec or legacy
builder; no feature appears in both; required_inputs is unique and tuple-typed.
PR 2 tightens the first check to require spec-only registration."
```

---

## Task 1.11: Audit `reason` vocabulary and pin in test

Per locked-in decision #1: the orchestrator must produce the exact same `reason` strings the current report does. Audit, then pin.

**Files:**
- Modify: `tests/test_feature_store_coverage.py` (add reason vocab test)

- [ ] **Step 1: Audit current reason strings**

Run: `grep -n 'reason=' src/regime_detection/feature_store.py`
Expected: shows every `reason=` literal. Collect the distinct values.

From inspection of `_build_feature_availability_report` and `_availability`, the vocabulary is:
- `"populated"` (always-on V1 hardcoded + `_availability` when `value is not None`)
- `"not_configured"` (`_availability` when `missing_inputs` is empty but value is None)
- `"missing_required_inputs"` (`_availability` when `missing_inputs` is non-empty)

The orchestrator's branch `"not_configured" if not missing else "missing_required_inputs"` + `"populated"` covers all three. No change needed.

- [ ] **Step 2: Pin the vocabulary in a test**

Append to `tests/test_feature_store_coverage.py`:

```python
_ALLOWED_REASONS = frozenset({"populated", "not_configured", "missing_required_inputs"})


def test_availability_report_uses_only_allowed_reason_strings() -> None:
    from tests.helpers.market_context import make_minimal_market_context
    from regime_detection.feature_store import build_feature_store

    ctx = make_minimal_market_context()
    store = build_feature_store(ctx)

    for name, availability in store.availability.items():
        assert availability.reason in _ALLOWED_REASONS, (
            f"feature {name!r} emitted reason {availability.reason!r} "
            f"not in allowed vocabulary {sorted(_ALLOWED_REASONS)}"
        )
```

- [ ] **Step 3: Run the test**

Run: `python -m pytest tests/test_feature_store_coverage.py::test_availability_report_uses_only_allowed_reason_strings -v 2>&1 | tail -10 ; echo "EXIT:$?"`
Expected: PASS.

If FAIL: a fourth `reason` string is being emitted somewhere. Add it to `_ALLOWED_REASONS` AND add a branch in the orchestrator if the orchestrator should emit it.

- [ ] **Step 4: Commit**

```bash
git add tests/test_feature_store_coverage.py
git commit -m "test(feature-store): pin availability reason vocabulary

Asserts every emitted reason string is in {populated, not_configured,
missing_required_inputs}. Prevents accidental wire-format drift during PR 2."
```

---

## Task 1.12: Golden snapshot test for `availability` dict

Pins the full `availability` dict for at least two representative `MarketContext` fixtures so PR 2's per-feature migrations cannot silently change behavior.

**Files:**
- Create: `tests/test_feature_availability_golden.py`

- [ ] **Step 1: Write the test using existing fixtures**

```python
# tests/test_feature_availability_golden.py
from __future__ import annotations

from regime_detection.feature_store import build_feature_store


def test_availability_dict_pure_v1_context_snapshot() -> None:
    from tests.helpers.market_context import make_minimal_market_context

    ctx = make_minimal_market_context()
    store = build_feature_store(ctx)
    actual = {
        name: avail.model_dump() for name, avail in store.availability.items()
    }

    expected = {
        "trend_direction": {
            "feature": "trend_direction", "available": True, "policy": "raise",
            "reason": "populated", "required_inputs": ("spy_ohlcv.close",),
            "missing_inputs": (),
        },
        "trend_character": {
            "feature": "trend_character", "available": True, "policy": "raise",
            "reason": "populated",
            "required_inputs": ("spy_ohlcv.close", "spy_ohlcv.high", "spy_ohlcv.low"),
            "missing_inputs": (),
        },
        "volatility": {
            "feature": "volatility", "available": True, "policy": "raise",
            "reason": "populated", "required_inputs": ("spy_ohlcv.close",),
            "missing_inputs": (),
        },
        "breadth": {
            "feature": "breadth", "available": True, "policy": "raise",
            "reason": "populated", "required_inputs": ("spy_ohlcv.close", "rsp_close"),
            "missing_inputs": (),
        },
        "sma_50": {
            "feature": "sma_50", "available": True, "policy": "raise",
            "reason": "populated", "required_inputs": ("spy_ohlcv.close",),
            "missing_inputs": (),
        },
        # ... legacy entries from _build_feature_availability_report for the
        # unmigrated 15 features. Fill in by capturing current output:
        # run the test once, copy the printed dict, paste below.
    }

    assert actual == expected, (
        f"availability snapshot drifted.\nactual: {actual}\nexpected: {expected}"
    )
```

> **Bootstrap note:** the expected dict for the 15 unmigrated features is best captured by running the test once with `expected = {}`, printing `actual`, and pasting it in. This is acceptable for golden tests — the goal is to detect any future drift, not to manually compute the right answer.

- [ ] **Step 2: Capture the actual current snapshot**

Run with `expected = {}`:
```bash
python -m pytest tests/test_feature_availability_golden.py -v 2>&1 | tail -100
```

Copy the printed `actual` dict and paste it as `expected`. Re-run; should PASS.

- [ ] **Step 3: Add a second fixture for full-V2 coverage**

If a full-V2 `MarketContext` fixture helper exists (`make_full_v2_market_context` or similar), add a second test using it. If not, skip this for PR 1 and add in PR 2 when more spec coverage exists.

- [ ] **Step 4: Commit**

```bash
git add tests/test_feature_availability_golden.py
git commit -m "test(feature-store): golden snapshot for availability dict

Pins the exact availability dict structure for a minimal V1 context. Any
PR 2 migration that changes wire format must update this snapshot explicitly."
```

---

## Task 1.13: Full test suite + open PR 1

- [ ] **Step 1: Run linters and type checker**

Run: `python -m ruff format --check src/ tests/ ; echo "EXIT:$?"`
Expected: EXIT:0. If non-zero: `python -m ruff format src/ tests/` to fix.

Run: `python -m ruff check src/ tests/ ; echo "EXIT:$?"`
Expected: EXIT:0.

Run: `python -m pyright ; echo "EXIT:$?"`
Expected: EXIT:0.

- [ ] **Step 2: Run targeted feature-store tests**

Run:
```bash
python -m pytest \
  tests/test_v2_feature_store_and_axis_seams.py \
  tests/test_feature_store_refactors.py \
  tests/test_feature_store_specs.py \
  tests/test_feature_store_runtime.py \
  tests/test_feature_store_coverage.py \
  tests/test_feature_availability_golden.py \
  -v 2>&1 | tail -50 ; echo "EXIT:$?"
```
Expected: all PASS, EXIT:0.

- [ ] **Step 3: Open PR 1**

Use the `superpowers:new-branch-and-pr` skill or open directly:

```bash
gh pr create --base main --title "refactor(feature-store): introduce FeatureSpec orchestrator + migrate 5 always-on features" --body "$(cat <<'EOF'
## Summary
- Adds `feature_store_runtime.py` with `FeatureSpec`, `_Unavailable`, `_run_feature_specs` orchestrator.
- Migrates 5 always-on V1 features (`trend_direction`, `trend_character`, `volatility`, `breadth`, `sma_50`) to specs.
- Leaves the remaining 15 features on the legacy `_FEATURE_STORE_BUILDERS` path; orchestrator output is merged with legacy report.
- Adds coverage CI test, reason vocabulary test, and golden availability snapshot.

## Why
The current `_build_feature_availability_report` (228 lines) re-derives the same gate predicates the builders use, joined only by a name string. This PR introduces the seam that lets the report be **derived** from the build decision. PR 2 migrates the remaining 15 features and deletes the legacy path.

Spec: `docs/superpowers/specs/2026-05-27-feature-store-spec-refactor-design.md`

## Test plan
- [x] `pytest tests/test_feature_store_runtime.py` — orchestrator unit tests
- [x] `pytest tests/test_feature_store_specs.py` — 5 per-spec resolution tests
- [x] `pytest tests/test_feature_store_coverage.py` — coverage + vocabulary guards
- [x] `pytest tests/test_feature_availability_golden.py` — snapshot of full availability dict
- [x] `pytest tests/test_v2_feature_store_and_axis_seams.py` — end-to-end regression check
- [x] `ruff check` + `pyright` clean

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# PR 2 — Remaining 15 Features + Cleanup

PR 2 ships after PR 1 merges to main. Same `subagent-driven-development` flow; 15 of the tasks (one per feature) are independent and can be dispatched in parallel after Task 2.0 completes.

## Task 2.0: Branch from main + verify PR 1 state

- [ ] **Step 1: Pull main, create PR 2 branch**

```bash
git checkout main && git pull
git checkout -b avinash8891/feature-store-spec-refactor-pr2
```

- [ ] **Step 2: Verify PR 1 state by running its tests**

Run: `python -m pytest tests/test_feature_store_runtime.py tests/test_feature_store_specs.py tests/test_feature_store_coverage.py tests/test_feature_availability_golden.py -v 2>&1 | tail -20 ; echo "EXIT:$?"`
Expected: PASS, EXIT:0. If FAIL, abort — PR 1 didn't merge cleanly and PR 2 should not start.

---

## Per-Feature Migration Template

Each of Tasks 2.1–2.15 follows this template. Adapt the resolve / build / store / test bodies to the per-feature specifics in the task body.

**The 5 steps per feature:**

1. **Write the failing resolution test(s)** in `tests/test_feature_store_specs.py`. At minimum one happy-path test (resolve returns kwargs) and one missing-input test per distinct absence shape (resolve returns `_Unavailable(missing=(...))`).
2. **Run to verify failure** — `python -m pytest tests/test_feature_store_specs.py::<test_name> -v ; echo "EXIT:$?"`. Expected: FAIL.
3. **Add the spec** to `feature_store.py`: a pure `_build_*` function, a `_resolve_*` function, and a `FeatureSpec(...)` entry appended to `_FEATURE_SPECS`. Remove the corresponding entry from `_FEATURE_STORE_BUILDERS`. Remove the corresponding entry from `_build_feature_availability_report`. Delete the now-orphan `_build_*_feature(state)` function and any helper that only it called.
4. **Run to verify pass** — `python -m pytest tests/test_feature_store_specs.py tests/test_feature_availability_golden.py tests/test_v2_feature_store_and_axis_seams.py -v ; echo "EXIT:$?"`. Expected: PASS.
5. **Commit** — `git commit -m "feat(feature-store): migrate <name> to spec"`.

---

## Task 2.1: Migrate `sentiment_score`

**Today's builder:** `_build_sentiment_score_feature` at line 395. Reads `state.context.aaii_sentiment`, returns from `_build_sentiment_score_series` which returns None when any input is missing.

**Spec shape:**
- `name="sentiment_score"`, `policy="none"`
- `required_inputs=("aaii_sentiment",)`
- `resolve`: returns `_Unavailable(missing=("aaii_sentiment",))` if `state.context.aaii_sentiment` is None or empty or missing required column. Else returns `{"aaii_sentiment": state.context.aaii_sentiment, "session_index": _as_datetime_index(state.spy_close.index)}`.
- `build(aaii_sentiment, session_index) -> pd.Series | None`: existing `_build_sentiment_score_series` logic, but the None-returning branches inside it move into `resolve`.
- `store=lambda s, v: setattr(s, "sentiment_score", v)`

**Tests:**
- `test_sentiment_score_resolve_missing_aaii_returns_unavailable`
- `test_sentiment_score_resolve_present_aaii_returns_kwargs`

Follow the 5-step template above.

---

## Task 2.2: Migrate `news_sentiment_score`

**Today's builder:** `_build_news_sentiment_score_feature` at line 402. Gated by `state.news_sentiment_config` AND `state.context.news_sentiment`.

**Spec shape:**
- `name="news_sentiment_score"`, `policy="none"`
- `required_inputs=("news_sentiment_config", "news_sentiment")`
- `resolve`: returns `_Unavailable(missing=tuple(...))` listing whichever of `news_sentiment_config` / `news_sentiment` is absent or empty. Else `{"news_sentiment": ..., "session_index": ..., "config": ...}`.
- `build(news_sentiment, session_index, config) -> pd.Series`: wraps existing `_build_news_sentiment_score_series` minus its None-guards.
- `store=lambda s, v: setattr(s, "news_sentiment_score", v)`

**Tests:** missing config → unavailable; missing news_sentiment → unavailable; both present → kwargs.

Follow the 5-step template.

---

## Task 2.3: Migrate `trend_direction_v2`

**Today's builder:** `_build_trend_direction_v2_feature` at line 410. Gated by `state.trend_direction_v2_config`.

**Spec shape:**
- `name="trend_direction_v2"`, `policy="none"`
- `required_inputs=("trend_direction_v2_config", "spy_ohlcv.close")`
- `resolve`: returns `_Unavailable(missing=("trend_direction_v2_config",))` if `state.trend_direction_v2_config is None`. Else `{"spy_close": ..., "config": ..., "sentiment_score": state.sentiment_score, "news_sentiment_score": state.news_sentiment_score}`.
- `build(spy_close, config, sentiment_score, news_sentiment_score) -> TrendDirectionV2Features`: wraps `compute_trend_v2_features(...)`.
- `store=lambda s, v: setattr(s, "trend_direction_v2", v)`

**Tests:** missing config → unavailable; config present → kwargs.

Follow the template.

---

## Task 2.4: Migrate `network_fragility`

**Today's builder:** `_build_network_fragility_feature` at line 469. Gated by `state.context.sector_etf_closes`. Optional `state.network_fragility_config` changes which `compute_network_fragility_features` overload runs.

**Spec shape:**
- `name="network_fragility"`, `policy="none"`
- `required_inputs=("sector_etf_closes",)`
- `resolve`: `_Unavailable(("sector_etf_closes",))` if `state.context.sector_etf_closes is None`. Else `{"sector_etf_closes": ..., "cross_asset_closes": state.context.cross_asset_closes or {}, "spy_close": ..., "config": state.network_fragility_config}`.
- `build(sector_etf_closes, cross_asset_closes, spy_close, config) -> NetworkFragilityFeatures`: branches on `config is None` to pick the overload, same as today's builder.
- `store=lambda s, v: setattr(s, "network_fragility", v)`

**Tests:** missing sector_etf_closes → unavailable; present → kwargs (default config); present + config → kwargs (with config).

Follow the template.

---

## Task 2.5: Migrate `volatility_state_v2`

**Today's builder:** `_build_volatility_state_v2_feature` at line 496. Gated by `state.volatility_state_v2_config`. Computes optional `event_window_just_passed` from calendar.

**Spec shape:**
- `name="volatility_state_v2"`, `policy="none"`
- `required_inputs=("volatility_state_v2_config", "spy_ohlcv.ohlc")`
- `resolve`: `_Unavailable(("volatility_state_v2_config",))` if config is None. Else build the kwargs dict including the computed `event_window_just_passed` (move the calendar-derivation logic into resolve since it depends only on state).
- `build(open_, high, low, close, config, rules_config, implied_vol_30d, event_window_just_passed) -> VolatilityV2Features`: wraps `compute_volatility_v2_features`.
- `store=lambda s, v: setattr(s, "volatility_state_v2", v)`

**Tests:** missing config → unavailable; config present → kwargs (verify event_window key is in dict, even if value is None).

Follow the template.

---

## Task 2.6: Migrate `breadth_state_v2`

**Today's builder:** `_build_breadth_state_v2_feature` at line 525. Gated by `state.breadth_state_v2_config` AND `state.context.sector_etf_closes` AND at-least-one-of `SECTOR_ETFS` present.

**Spec shape:**
- `name="breadth_state_v2"`, `policy="none"`
- `required_inputs=("breadth_state_v2_config", "sector_etf_closes")`
- `resolve`: build a missing list — `"breadth_state_v2_config"` if config None, `"sector_etf_closes"` if closes None, `"sector_etf_closes.any_sector_etf"` if closes present but no SECTOR_ETF symbol intersects. If non-empty, return `_Unavailable(missing=tuple(...))`. Else `{"sector_etf_closes": ..., "config": ..., "pit_constituent_intervals": ..., "constituent_ohlcv": ...}`.
- `build(sector_etf_closes, config, pit_constituent_intervals, constituent_ohlcv) -> BreadthV2Features`: wraps `compute_breadth_v2_features`.
- `store=lambda s, v: setattr(s, "breadth_state_v2", v)`

**Tests:** missing config → unavailable; missing closes → unavailable; closes present but no SECTOR_ETF match → unavailable with `"sector_etf_closes.any_sector_etf"` in missing; happy path → kwargs.

Follow the template.

---

## Task 2.7: Migrate `volume_liquidity_v2`

**Today's builder:** `_build_volume_liquidity_v2_feature` at line 541. Gated by config + SPY volume column existing + non-NaN.

**Spec shape:**
- `name="volume_liquidity_v2"`, `policy="none"`
- `required_inputs=("volume_liquidity_v2_config", "spy_ohlcv.volume")`
- `resolve`: build missing list — `"volume_liquidity_v2_config"` if None; `"spy_ohlcv.volume"` if column absent; `"spy_ohlcv.volume.non_nan"` if all-NaN. If non-empty, return `_Unavailable`. Else `{"volume": spy_volume, "config": ...}`.
- `build(volume, config) -> VolumeLiquidityV2Features`: wraps `compute_volume_liquidity_v2_features`.
- `store=lambda s, v: setattr(s, "volume_liquidity_v2", v)`

**Tests:** missing config; missing volume column; all-NaN volume; happy path.

Follow the template.

---

## Task 2.8: Migrate `monetary`

**Today's builder:** `_build_monetary_feature` at line 560. Gated by config + macro_series present + DGS2/DGS10/broad_usd_index keys present. Also optionally computes `cb_text_score_series` from central_bank_text data.

**Spec shape:**
- `name="monetary"`, `policy="raise"`
- `required_inputs=("monetary_pressure_v2_config", "macro_series", "2y_yield", "dgs10", "broad_usd_index")`
- `resolve`: build missing list using `_missing_macro_keys` helper. If config None → `_Unavailable(("monetary_pressure_v2_config",))`. Else check macro keys. Move the central_bank_text → daily score derivation into resolve (depends only on state). Return kwargs dict with `dgs2`, `dgs10`, `broad_usd_index`, `central_bank_text_score` (possibly None), `config`.
- `build(dgs2, dgs10, broad_usd_index, central_bank_text_score, config) -> MonetaryPressureV2Features`: wraps `compute_monetary_pressure_features`.
- `store=lambda s, v: setattr(s, "monetary", v)`

**Tests:** missing config; missing macro_series; missing each of DGS2/DGS10/broad_usd_index; all present with no central_bank_text; all present with central_bank_text.

Follow the template.

---

## Task 2.9: Migrate `realized_vol_21d`

**Today's builder:** `_build_realized_vol_21d_feature` at line 593. Computed only when one of hmm/clustering/change_point config is non-None.

**Spec shape:**
- `name="realized_vol_21d"`, `policy="none"`
- `required_inputs=("hmm_or_clustering_or_change_point_config",)` — single rolled-up name since it's a disjunction
- `resolve`: if none of the three configs are set, return `_Unavailable(missing=("hmm_or_clustering_or_change_point_config",))`. Else `{"spy_close": state.spy_close}`.
- `build(spy_close) -> pd.Series`: returns `realized_vol(spy_close, 21)`.
- `store=lambda s, v: setattr(s, "realized_vol_21d", v)`

**Tests:** none of the three configs set → unavailable; any one set → kwargs.

Follow the template.

---

## Task 2.10: Migrate `drawdown_63d`

**Today's builder:** `_build_drawdown_63d_feature` at line 605. Computed only when hmm or clustering config is non-None.

**Spec shape:**
- `name="drawdown_63d"`, `policy="none"`
- `required_inputs=("hmm_or_clustering_config",)`
- `resolve`: if neither hmm nor clustering config set, `_Unavailable`. Else `{"spy_close": state.spy_close}`.
- `build(spy_close) -> pd.Series`: returns `compute_trailing_drawdown(spy_close, 63)`.
- `store=lambda s, v: setattr(s, "drawdown_63d", v)`

**Tests:** neither config set → unavailable; one set → kwargs.

Follow the template.

---

## Task 2.11: Migrate `hmm`

**Today's builder:** `_build_hmm_feature` at line 619. Gated by hmm config + volume_liquidity_v2 + network_fragility. Reads volatility.return_1d, realized_vol_21d, drawdown_63d (all upstream-required).

**Spec shape:**
- `name="hmm"`, `policy="none"`
- `required_inputs=("hmm_config", "volume_liquidity_v2", "network_fragility")`
- `resolve`: build missing list checking the three gates. If non-empty, `_Unavailable`. Else `{"config": ..., "volume_liquidity": ..., "network_fragility": ..., "return_1d": state.volatility.return_1d, "realized_vol_21d": ..., "drawdown_63d": ...}`. (`state.volatility` is non-None because PR 1's spec guarantees it — `policy="raise"`.)
- `build(config, volume_liquidity, network_fragility, return_1d, realized_vol_21d, drawdown_63d) -> HMMFeatures`: wraps `compute_hmm_features` with kwargs mapping.
- `store=lambda s, v: setattr(s, "hmm", v)`

**Tests:** missing config; missing volume_liquidity_v2; missing network_fragility; all present → kwargs.

Follow the template.

---

## Task 2.12: Migrate `clustering`

**Today's builder:** `_build_clustering_feature` at line 640. Gated by clustering config + breadth_state_v2.pct_above_50dma + network_fragility + trend_direction_v2.

**Spec shape:**
- `name="clustering"`, `policy="none"`
- `required_inputs=("clustering_config", "breadth_state_v2.pct_above_50dma", "network_fragility", "trend_direction_v2")`
- `resolve`: build missing list per the four gates (note the pct_above_50dma sub-field check). If non-empty, `_Unavailable`. Else `{"return_21d": state.trend_character.return_21d, "return_63d": state.trend_direction_v2.return_63d, "realized_vol_21d": ..., "drawdown_63d": ..., "adx_14": state.trend_character.adx_14, "avg_pairwise_corr_63d": ..., "pct_above_50dma": ..., "config": ...}`.
- `build(...) -> ClusteringFeatures`: wraps `compute_clustering_features`.
- `store=lambda s, v: setattr(s, "clustering", v)`

**Tests:** each of the four gates absent → unavailable; all present → kwargs.

Follow the template.

---

## Task 2.13: Migrate `credit_funding`

**Today's builder:** `_build_credit_funding_feature` at line 666. Gated by config + cross_asset_closes + macro_series + all required keys in each.

**Spec shape:**
- `name="credit_funding"`, `policy="none"`
- `required_inputs=("credit_funding_config", "cross_asset_closes", "macro_series")` (key-level missingness shows up in `missing_inputs` per resolve)
- `resolve`: build missing list using `_missing_cross_asset_keys` + `_missing_macro_keys`. If non-empty, `_Unavailable`. Else a large kwargs dict with all the per-key reads.
- `build(...) -> CreditFundingFeatures`: wraps `compute_credit_funding_features`. The `nan_oas` fallback for HY/IG OAS moves into resolve (it depends on `spy_close.index`).
- `store=lambda s, v: setattr(s, "credit_funding", v)`

**Tests:** missing config; missing cross_asset_closes; missing macro_series; specific missing keys; happy path.

Follow the template.

---

## Task 2.14: Migrate `inflation_growth`

**Today's builder:** `_build_inflation_growth_feature` at line 695. Similar shape to credit_funding — config + cross_asset_closes + macro_series + per-key checks.

**Spec shape:** mirror credit_funding's structure with inflation_growth's specific keys (`_IG_CROSS_ASSET_KEYS`, `_IG_MACRO_KEYS`).

**Tests:** missing config; missing each input collection; happy path.

Follow the template.

---

## Task 2.15: Migrate `change_point`

**Today's builder:** `_build_change_point_feature` at line 728. Gated by change_point config only. Depends on `state.realized_vol_21d`.

**Spec shape:**
- `name="change_point"`, `policy="none"`
- `required_inputs=("change_point_config", "realized_vol_21d")`
- `resolve`: if config None → `_Unavailable(("change_point_config",))`. If `state.realized_vol_21d` None → `_Unavailable(("realized_vol_21d",))`. Else `{"realized_vol_21d": ..., "config": ...}`.
- `build(realized_vol_21d, config) -> ChangePointFeatures`: wraps `compute_change_point_features`.
- `store=lambda s, v: setattr(s, "change_point", v)`

**Tests:** missing config; missing realized_vol_21d; both present.

Follow the template.

---

## Task 2.16: Delete legacy code

After Tasks 2.1–2.15 land, `_FEATURE_STORE_BUILDERS` should be empty and `_build_feature_availability_report` should have no entries left.

**Files:**
- Modify: `src/regime_detection/feature_store.py`

- [ ] **Step 1: Verify `_FEATURE_STORE_BUILDERS` is empty**

Run: `grep -n "_FEATURE_STORE_BUILDERS = " src/regime_detection/feature_store.py`
Expected: shows the tuple definition. Verify by reading that it's now `()` or has no entries.

- [ ] **Step 2: Delete the legacy machinery**

Remove from `feature_store.py`:
- `_FeatureStoreBuilder` class definition
- `_run_feature_store_builders` function
- `_FEATURE_STORE_BUILDERS` tuple
- `_availability` function
- `_build_feature_availability_report` function
- `_missing_macro_keys`, `_missing_cross_asset_keys`, `_missing_sector_inputs` helpers if no longer used (likely moved into resolve functions during PR 2 migrations — confirm by grepping for callers)

In `build_feature_store`, remove the calls:
- Remove `_run_feature_store_builders(_FEATURE_STORE_BUILDERS, build_state)`
- Remove `legacy_report = _build_feature_availability_report(build_state)`
- Replace `combined_availability = {**spec_report, **legacy_report}` with `combined_availability = spec_report`

- [ ] **Step 3: Run targeted tests**

Run: `python -m pytest tests/test_v2_feature_store_and_axis_seams.py tests/test_feature_store_specs.py tests/test_feature_store_runtime.py tests/test_feature_store_coverage.py tests/test_feature_availability_golden.py -v 2>&1 | tail -40 ; echo "EXIT:$?"`
Expected: all PASS, EXIT:0.

- [ ] **Step 4: Commit**

```bash
git add src/regime_detection/feature_store.py
git commit -m "refactor(feature-store): delete legacy _FeatureStoreBuilder path

All 20 features now run through the FeatureSpec orchestrator. Removes
_FeatureStoreBuilder, _FEATURE_STORE_BUILDERS, _build_feature_availability_report
(228 lines), _availability, and unused _missing_* helpers. feature_store.py
drops from ~1100 to ~700 lines."
```

---

## Task 2.17: Tighten coverage test

The PR 1 coverage test allowed registration in either `_FEATURE_SPECS` or `_FEATURE_STORE_BUILDERS`. Tighten to require spec-only.

**Files:**
- Modify: `tests/test_feature_store_coverage.py`

- [ ] **Step 1: Update the coverage test**

Replace `test_every_feature_store_field_has_a_builder_or_spec` with:

```python
def test_every_feature_store_field_has_a_spec() -> None:
    declared = set(FeatureStore.model_fields.keys()) - _FEATURE_STORE_NON_FEATURE_FIELDS
    registered = {s.name for s in _FEATURE_SPECS}
    missing = declared - registered
    extra = registered - declared
    assert not missing, f"FeatureStore fields with no spec: {sorted(missing)}"
    assert not extra, f"Specs with no FeatureStore field: {sorted(extra)}"
```

Remove the `from regime_detection.feature_store import _FEATURE_STORE_BUILDERS` import (which no longer exists). Remove `test_no_feature_appears_in_both_specs_and_legacy_builders` (now meaningless).

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_feature_store_coverage.py -v 2>&1 | tail -20 ; echo "EXIT:$?"`
Expected: PASS, EXIT:0.

- [ ] **Step 3: Commit**

```bash
git add tests/test_feature_store_coverage.py
git commit -m "test(feature-store): require spec-only registration

PR 1's coverage test allowed either spec or legacy builder. Now all features
must be specs. Removes the dual-registration test (legacy path deleted)."
```

---

## Task 2.18: Rewrite `test_feature_store_refactors.py`

The old file imports `_FeatureStoreBuilder` and `_FEATURE_STORE_BUILDERS`, both gone. Rewrite to assert the same invariants against `_FEATURE_SPECS`.

**Files:**
- Modify: `tests/test_feature_store_refactors.py`

- [ ] **Step 1: Read the old test file to understand what it tests**

Run: `cat tests/test_feature_store_refactors.py`

Identify the contracts being tested (likely: builder iteration produces the expected names; build state is correctly populated; etc.).

- [ ] **Step 2: Rewrite each test against the spec registry**

Replace each test's imports and core assertion to use `_FEATURE_SPECS` and the orchestrator. Example replacement for "the builder tuple has these N names":

```python
def test_feature_specs_have_expected_names() -> None:
    from regime_detection.feature_store import _FEATURE_SPECS
    actual_names = tuple(s.name for s in _FEATURE_SPECS)
    expected_names = (
        "trend_direction",
        "trend_character",
        "volatility",
        "breadth",
        "sma_50",
        "sentiment_score",
        "news_sentiment_score",
        "trend_direction_v2",
        "network_fragility",
        "volatility_state_v2",
        "breadth_state_v2",
        "volume_liquidity_v2",
        "monetary",
        "realized_vol_21d",
        "drawdown_63d",
        "hmm",
        "clustering",
        "credit_funding",
        "inflation_growth",
        "change_point",
    )
    assert actual_names == expected_names
```

For toy-builder tests like the old `_FeatureStoreBuilder("moving_average", ...)` usage in test_feature_store_refactors.py, rewrite as toy `FeatureSpec` instances (the orchestrator's unit tests in `test_feature_store_runtime.py` are the canonical pattern).

- [ ] **Step 3: Run the rewritten tests**

Run: `python -m pytest tests/test_feature_store_refactors.py -v 2>&1 | tail -30 ; echo "EXIT:$?"`
Expected: PASS, EXIT:0.

- [ ] **Step 4: Commit**

```bash
git add tests/test_feature_store_refactors.py
git commit -m "test(feature-store): rewrite refactor tests against _FEATURE_SPECS

Old tests imported _FeatureStoreBuilder and _FEATURE_STORE_BUILDERS, both
removed in Task 2.16. Rewritten to assert equivalent invariants against
the spec registry and orchestrator."
```

---

## Task 2.19: Extend golden snapshot, run full suite, open PR 2

- [ ] **Step 1: Add a second fixture to the golden snapshot test**

In `tests/test_feature_availability_golden.py`, add a test using a full-V2 `MarketContext` fixture (with all configs set and all macro/cross-asset keys present). Capture the actual output as the expected snapshot, same bootstrap pattern as Task 1.12.

- [ ] **Step 2: Run linters and type checker**

Run: `python -m ruff format --check src/ tests/ ; echo "EXIT:$?"`
Run: `python -m ruff check src/ tests/ ; echo "EXIT:$?"`
Run: `python -m pyright ; echo "EXIT:$?"`
Expected: EXIT:0 for all three.

- [ ] **Step 3: Run targeted feature-store tests**

Run:
```bash
python -m pytest \
  tests/test_v2_feature_store_and_axis_seams.py \
  tests/test_feature_store_refactors.py \
  tests/test_feature_store_specs.py \
  tests/test_feature_store_runtime.py \
  tests/test_feature_store_coverage.py \
  tests/test_feature_availability_golden.py \
  tests/test_breadth_state_v2_features.py \
  tests/test_breadth_state_v2_pit_features.py \
  tests/test_monetary_pressure_features.py \
  tests/test_network_fragility_features.py \
  tests/test_trend_direction_v2_features.py \
  tests/test_volatility_state_v2_features.py \
  tests/test_volume_liquidity_v2_features.py \
  -v 2>&1 | tail -50 ; echo "EXIT:$?"
```
Expected: all PASS, EXIT:0.

- [ ] **Step 4: Verify line count reduction**

Run: `wc -l src/regime_detection/feature_store.py`
Expected: ~700 lines (down from 1102).

- [ ] **Step 5: Open PR 2**

```bash
gh pr create --base main --title "refactor(feature-store): complete FeatureSpec migration, delete legacy registry" --body "$(cat <<'EOF'
## Summary
- Migrates the remaining 15 features to `FeatureSpec` (PR 1 covered the first 5).
- Deletes `_FeatureStoreBuilder`, `_FEATURE_STORE_BUILDERS`, `_build_feature_availability_report` (228 lines), `_availability`, and the `_missing_*` helpers.
- Tightens the coverage CI test to require spec-only registration.
- Rewrites `tests/test_feature_store_refactors.py` against the spec registry.
- Extends the golden snapshot to a full-V2 fixture.

## Result
- `feature_store.py`: 1102 → ~700 lines.
- One source of truth per feature: `resolve` decides both buildability and availability.
- `~15` implicit `if config is None: state.x = None; return` guards removed.

Spec: `docs/superpowers/specs/2026-05-27-feature-store-spec-refactor-design.md`
Follows: PR 1 (already merged).

## Test plan
- [x] `pytest tests/test_feature_store_*` — all PASS
- [x] `pytest tests/test_v2_feature_store_and_axis_seams.py` — regression check
- [x] `pytest tests/test_*_features.py` for the 7 V2 axis modules — regression check
- [x] `ruff check` + `pyright` clean
- [x] Golden snapshot covers V1-minimal AND full-V2 fixtures

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review Notes

**Coverage check vs. spec:**
- §1 Architecture: Tasks 1.1, 1.3 create the runtime module and wire it ✓
- §2 FeatureSpec shape: Task 1.1 implements ✓
- §3 Migration sequencing C: PR 1 (Tasks 1.1–1.13) does foundation + 5 always-on; PR 2 (Tasks 2.0–2.19) does 15 + cleanup ✓
- §4 Behavior preservation: Tasks 1.12 (golden) + 1.11 (vocabulary) + 1.13 (full suite) + 2.19 (full suite, extended golden) ✓
- §5 Testing strategy: orchestrator (1.2), per-spec resolution (1.4–1.8, 2.1–2.15), end-to-end golden (1.12, 2.19) ✓
- §6 Out of scope: respected (no models.py / axis_series.py / boundary_policies.py touches) ✓
- Locked-in decisions: #1 vocabulary pin (1.11), #2 raise-on-missing-SPY (always-on resolves return kwargs unconditionally — Tasks 1.4–1.8), #3 realized_vol/drawdown stay in PR 2 (Tasks 2.9, 2.10), #4 required_inputs uniqueness (Task 1.10) ✓

**Type consistency check:**
- `FeatureSpec[T, StateT]` signature defined in Task 1.1; used as `FeatureSpec[object, _FeatureStoreBuildState]` in feature_store.py (Tasks 1.4+). ✓
- `_run_feature_specs` signature `tuple[FeatureSpec[Any, StateT], ...] -> dict[str, FeatureAvailability]` defined in Task 1.1; called consistently in Task 1.9. ✓
- `FeatureAvailability` model unchanged from current; field names match in golden snapshots. ✓

**No placeholders:**
- No "TBD" / "TODO" / "appropriate" / "etc." in step content. ✓
- Code blocks shown wherever code changes. ✓
- The per-feature task bodies (2.1–2.15) describe shape rather than spelling every line — this is intentional DRY against the template. Each task body still names every required input, every kwarg, every test case, and the exact `policy`/`store` so a subagent can implement without re-deriving. ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-27-feature-store-spec-refactor.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task. Each task is small enough to fit in a single subagent's context. Tasks 2.1–2.15 in PR 2 are independent and can run in parallel batches after PR 1 lands.

**2. Inline Execution** — Execute all tasks in this session via `superpowers:executing-plans` with checkpoints between PR 1 and PR 2.

Which approach?
