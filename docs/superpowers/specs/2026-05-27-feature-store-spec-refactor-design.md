# Feature Store: collapse parallel availability registry into per-feature specs

**Date:** 2026-05-27
**Status:** approved design, plan pending
**Scope:** `src/regime_detection/feature_store.py`, new `src/regime_detection/feature_store_runtime.py`, one test file rewrite

## Problem

`feature_store.py` carries two parallel registries that encode the same per-feature gate logic:

1. `_FEATURE_STORE_BUILDERS` (line 738) — tuple of `_FeatureStoreBuilder(name, build_fn)` where each builder reads `_FeatureStoreBuildState` and decides internally whether to populate its field via `if config is None: state.x = None; return` guards.
2. `_build_feature_availability_report` (line 814, ~228 lines) — hand-written dict that re-derives the same gate predicates (`state.context.config.hmm is not None`, `state.volume_liquidity_v2 is not None`, etc.) to produce a `FeatureAvailability` entry per feature.

The two are joined only by the name string. Any change to a builder's required inputs must be mirrored by hand in the report dict. Drift is silent: the report can say "available=True" when the builder short-circuited, or "unavailable, missing X" when the builder actually populated, because availability is *inferred from `value is not None`* rather than *decided by the same predicate that gates the build*.

After the recent commits ratcheting strict typing and runtime contracts, `feature_store.py` grew from 801 → 1102 lines, crossing the 1k-line threshold. Most of the growth is the second registry.

## Goal

One source of truth per feature. The predicate that decides "buildable" is the same code path that extracts inputs for the build function. The availability report is **derived**, never hand-written.

Behavior must remain byte-identical: same `FeatureStore` output, same `availability` dict contents, same exceptions at the same boundaries with the same messages.

## Non-goals

- Signature reflection (`inspect.signature` → `required_inputs`). Door left open; not in this refactor.
- Touching `models.py` pyright suppressions.
- Touching `axis_series.py` `AxisDependencyContract`.
- Changing `_FeatureStoreBuildState` shape or making it immutable.

## Architecture

One new module: `src/regime_detection/feature_store_runtime.py`. Holds the orchestrator machinery so it can be unit-tested against toy specs without standing up a full `MarketContext`.

- `feature_store_runtime.py` — `FeatureSpec`, `_Unavailable` sentinel, `_run_feature_specs` orchestrator, `FeatureAvailability` model (moved here from `feature_store.py`).
- `feature_store.py` — keeps `FeatureStore`, `_FeatureStoreBuildState`, per-feature `compute_*` thin wrappers, the registry `_FEATURE_SPECS: tuple[FeatureSpec[Any], ...]`, and the public `build_feature_store(context)` entry point.

## The `FeatureSpec` shape

```python
@dataclass(frozen=True)
class _Unavailable:
    missing_inputs: tuple[str, ...]

FeatureInputs = dict[str, Any]  # resolved kwargs to pass to build

@dataclass(frozen=True)
class FeatureSpec[T]:
    name: str
    policy: FeatureAvailabilityPolicy
    required_inputs: tuple[str, ...]
    resolve: Callable[[_FeatureStoreBuildState], FeatureInputs | _Unavailable]
    build: Callable[..., T]
    store: Callable[[_FeatureStoreBuildState, T], None]
```

- `resolve` is the **only** place that decides whether a feature is buildable. It either returns a `dict` of typed kwargs ready to splat into `build`, or `_Unavailable(missing=(...))` listing the absent inputs.
- `build` is total over its typed parameters. No internal None-guards. No reference to `_FeatureStoreBuildState`.
- `store` is a one-line setattr-style write back into state.
- `required_inputs` is the machine-readable list shown in `FeatureAvailability.required_inputs`. It must enumerate the same logical inputs that `resolve` checks; a unit test pins this per spec.

### Orchestrator

```python
def _run_feature_specs(
    specs: tuple[FeatureSpec[Any], ...],
    state: _FeatureStoreBuildState,
) -> dict[str, FeatureAvailability]:
    report: dict[str, FeatureAvailability] = {}
    for spec in specs:
        resolved = spec.resolve(state)
        if isinstance(resolved, _Unavailable):
            report[spec.name] = FeatureAvailability(
                feature=spec.name,
                available=False,
                policy=spec.policy,
                reason="not_configured" if not resolved.missing_inputs
                       else "missing_required_inputs",
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

Key invariant: a feature appears as `available=True` in the report **if and only if** `resolve` returned kwargs and `build` was called. No `value is not None` post-hoc inference.

### Example: HMM spec

```python
def build_hmm(
    config: HMMConfig,
    volume_liquidity: VolumeLiquidityV2Features,
    network_fragility: NetworkFragilityFeatures,
    return_1d: pd.Series,
    realized_vol_21d: pd.Series,
    drawdown_63d: pd.Series,
) -> HMMFeatures:
    return compute_hmm_features(...)

def _resolve_hmm(s: _FeatureStoreBuildState) -> FeatureInputs | _Unavailable:
    missing: list[str] = []
    if s.context.config.hmm is None:
        missing.append("hmm_config")
    if s.volume_liquidity_v2 is None:
        missing.append("volume_liquidity_v2")
    if s.network_fragility is None:
        missing.append("network_fragility")
    if missing:
        return _Unavailable(missing_inputs=tuple(missing))
    return dict(
        config=s.context.config.hmm,
        volume_liquidity=s.volume_liquidity_v2,
        network_fragility=s.network_fragility,
        return_1d=s.volatility.return_1d,
        realized_vol_21d=s.realized_vol_21d,
        drawdown_63d=s.drawdown_63d,
    )

HMM_SPEC = FeatureSpec(
    name="hmm",
    policy="none",
    required_inputs=("hmm_config", "volume_liquidity_v2", "network_fragility"),
    resolve=_resolve_hmm,
    build=build_hmm,
    store=lambda s, v: setattr(s, "hmm", v),
)
```

## Migration sequencing (two PRs, Option C)

### PR 1 — foundation + 5 always-on V1 features

1. Create `feature_store_runtime.py` with `FeatureSpec`, `_Unavailable`, `_run_feature_specs`. Move `FeatureAvailability` over from `feature_store.py`; re-export from old location to keep public imports working.
2. Migrate 5 always-on V1 features to specs: `trend_direction`, `trend_character`, `volatility`, `breadth`, `sma_50`. All `policy="raise"`, `resolve` returns kwargs unconditionally — these never gate.
3. Orchestrator runs the new specs first, then the existing `_FEATURE_STORE_BUILDERS` for the unmigrated 15. Availability report = `{specs report} | {hand-written entries for unmigrated 15}`.
4. Add CI test `tests/test_feature_store_coverage.py` asserting every key in `FeatureStore.model_fields` (minus `spy_index`, `availability`) appears either as a `FeatureSpec.name` or in `_FEATURE_STORE_BUILDERS`. (After PR 2, the OR collapses.)
5. Full pytest suite passes unchanged. Golden snapshot test added — see Testing section.

### PR 2 — remaining 15 features + cleanup

1. Convert the 15 remaining features to specs: `sentiment_score`, `news_sentiment_score`, `trend_direction_v2`, `network_fragility`, `volatility_state_v2`, `breadth_state_v2`, `volume_liquidity_v2`, `monetary`, `realized_vol_21d`, `drawdown_63d`, `hmm`, `clustering`, `credit_funding`, `inflation_growth`, `change_point`. Each spec is one commit inside the PR. Total feature count: 5 (PR 1) + 15 (PR 2) = 20, matching today's `_FEATURE_STORE_BUILDERS` registry.
2. Delete `_FeatureStoreBuilder`, `_FEATURE_STORE_BUILDERS`, `_build_feature_availability_report`, `_availability`, the bare-dict report-builder helpers (`_missing_macro_keys`, `_missing_cross_asset_keys`, `_missing_sector_inputs` move into the resolve functions that need them).
3. Tighten the coverage test from PR 1 to require every feature be a spec.
4. Rewrite `tests/test_feature_store_refactors.py` — drops `_FeatureStoreBuilder` / `_FEATURE_STORE_BUILDERS` imports, asserts the same coverage and ordering invariants against `_FEATURE_SPECS` + orchestrator.

End state: `feature_store.py` ~700 lines (from 1102). `_build_feature_availability_report` (228 lines) gone. ~15 implicit `if config is None: state.x = None; return` guards inside builders gone (one per gated feature).

## Behavior preservation

Non-negotiable. The refactor must produce:

- The same `FeatureStore` for any `MarketContext`.
- The same `availability` dict keys, `available` booleans, `policy` values, `reason` strings, `required_inputs` tuples, `missing_inputs` tuples.
- The same exceptions raised at the same boundaries with the same messages.

Verification:

- Full pytest suite passes before PR 1 and after PR 2 with zero changed assertions (other than the deliberately rewritten `test_feature_store_refactors.py` and new tests).
- Golden-snapshot test pins the availability dict for representative `MarketContext` fixtures across both refactor PRs.

## Testing strategy

Three layers, added incrementally across the two PRs.

### Orchestrator unit tests (PR 1, `tests/test_feature_store_runtime.py`)

Toy `FeatureSpec` instances driving `_run_feature_specs` against a minimal fake state. Cases:

- `resolve` returns kwargs → `build` called with those kwargs → `store` writes value → report entry has `available=True`, `reason="populated"`, `missing_inputs=()`.
- `resolve` returns `_Unavailable(missing=("foo",))` → `build` NOT called → report entry has `available=False`, `reason="missing_required_inputs"`, `missing_inputs=("foo",)`.
- `resolve` returns `_Unavailable(missing=())` → report `reason="not_configured"`.
- `build` raises → orchestrator propagates (no swallowing).
- Spec ordering preserved in the returned dict.

### Per-spec resolution tests (in existing per-feature test files)

For each migrated spec, one test per absence shape, landing in the same PR that migrates the spec. PR 1 covers resolution tests for the 5 always-on features (single happy-path test each, since they cannot return `_Unavailable`). PR 2 covers resolution tests for the 15 gated features. Example for HMM: missing config → `_Unavailable(missing=("hmm_config",))`; missing `volume_liquidity_v2` → `_Unavailable(missing=("volume_liquidity_v2",))`; all present → kwargs dict matches expected shape. This is the test that catches "spec author forgot to check a required input."

### End-to-end behavior preservation

Existing `test_v2_feature_store_and_axis_seams.py` and per-feature tests pass unchanged. New `tests/test_feature_availability_golden.py` pins the `availability` dict for at least two `MarketContext` fixtures (pure-V1 minimal context; full-V2 context with all configs and inputs). Snapshot asserted as a `dict[str, FeatureAvailability]` for byte-equality.

## CI guardrail

The coverage test added in PR 1 (tightened in PR 2) makes "forgot to register a feature" impossible. Cheap, durable, keeps working even if a future refactor reshapes the registry.

## What the type system catches vs. what tests catch

| Drift class | Caught by | Mechanism |
|---|---|---|
| Spec author calls `build` with wrong kwarg names | Runtime via tests | Per-spec resolution test asserts kwargs match `build`'s signature |
| Spec author forgets to check a required input | Per-spec resolution test | Test asserts `_Unavailable(missing=(...))` for each absence shape |
| `FeatureStore` field added without spec | Coverage CI test | Asserts every field has a spec |
| Builder reads an input not in `required_inputs` | Per-spec resolution test | Test passes a state missing that input and asserts `_Unavailable` |
| Availability report drifts from build gate | Impossible by construction | Same `resolve` predicate decides both |

The last row is the whole point of the refactor.

## Locked-in decisions (resolved open items)

1. **`reason` string vocabulary preserved exactly.** The implementation plan's first step in PR 1 grep's `_build_feature_availability_report` for every distinct `reason` value and pins each one in a test. The orchestrator's branch `"not_configured" if not missing else "missing_required_inputs"` plus the always-true `"populated"` case must cover the entire current vocabulary. If any other `reason` string is emitted today, the orchestrator gains a branch for it; this refactor is not the place to change wire output.

2. **Always-on features preserve raise-on-missing-SPY behavior.** The 5 PR-1 specs (`trend_direction`, `trend_character`, `volatility`, `breadth`, `sma_50`) have `resolve` functions that return kwargs unconditionally — no SPY-presence check. If `spy_ohlcv.close` is missing, `build` crashes the same way it does today (from inside `compute_*_features`). Rationale: `policy="raise"` already means "this should never be missing in production"; converting to `_Unavailable` would change failure semantics. `build_feature_store` callers are responsible for SPY presence; the spec does not pretend to defend against its absence.

3. **`realized_vol_21d` and `drawdown_63d` stay in PR 2.** Although both are config-gateless, they depend on the V1 `volatility` spec running first (they read `state.volatility.return_1d` and SPY-derived series). PR 1's job is to validate the orchestrator on the *simplest* features — adding ordering-dependent specs early would muddle the validation. Move both to PR 2 where the full ordering story is exercised.

4. **Coverage test asserts `required_inputs` uniqueness and deterministic ordering.** The PR 1 CI test, in addition to "every `FeatureStore` field has a spec," asserts: (a) within each spec, `required_inputs` contains no duplicates; (b) `required_inputs` is a `tuple` not a `set`-derived structure (deterministic order). Cheap, catches a real class of construction bugs.
