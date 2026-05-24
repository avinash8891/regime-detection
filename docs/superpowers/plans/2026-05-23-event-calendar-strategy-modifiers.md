# Event Calendar Strategy Modifiers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add direct strategy execution modifiers driven by `event_calendar.matching_labels`, without replacing transition-risk macro-event evidence or using `primary_label` for logic.

**Architecture:** Event calendar remains computed once by `compute_event_calendar_outputs` and emitted under `structural_causal_state.event_calendar`. `build_regime_timeline` passes `event_output.matching_labels` and a new V2 `strategy_event_modifiers` config into `build_strategy_response`. Strategy response applies configured event groups as execution overlays using existing response fields and records modifier names in `modifiers_applied`.

**Tech Stack:** Python, Pydantic config models, YAML config, pytest/RTK, ruff.

---

## File Structure

- Modify `src/regime_detection/_config_evidence_strategy.py`
  - Add `StrategyEventModifierRule` and `StrategyEventModifiersConfig`.
- Modify `src/regime_detection/config.py`
  - Import/export the new config types and add `strategy_event_modifiers` to `RegimeConfig`.
- Modify `src/regime_detection/configs/core3-v2.0.0.yaml`
  - Add the default V2 event strategy rules.
- Do not modify `src/regime_detection/configs/core3-v1.0.0.yaml`
  - Keep V1 behavior unchanged unless a later task explicitly refreshes V1 frozen fixtures.
- Modify `src/regime_detection/strategy_response.py`
  - Accept `event_calendar_labels` and optional `event_modifier_config`.
  - Apply matching-label overlays after base/transition modifiers.
- Modify `src/regime_detection/timeline.py`
  - Pass `event_output.matching_labels` and `working_context.config.strategy_event_modifiers`.
- Modify tests:
  - `tests/test_transition_and_strategy.py`
  - `tests/test_v2_config.py`
  - `tests/test_schema_and_timeline.py`
  - optionally `tests/test_v1_frozen_replay.py` only to confirm unchanged V1 replay.
- Modify docs:
  - `docs/regime_engine_v2_spec.md`
  - `docs/regime_engine_v1_final_spec.md`
  - `docs/transition_risk.md`
  - optional ADR if reviewers want a durable decision record.

## Design Rules

- Logic must consume `event_calendar.matching_labels`, never `primary_label`.
- Strategy event modifiers are execution overlays, not regime labels and not transition-risk replacements.
- Event overlays should only tighten risk. Do not increase size, enable leverage, or loosen confirmation.
- Stricter overlapping event rules win through `min(position_size_multiplier, cap)` and boolean tightening.
- Do not add required `StrategyResponse` fields. Use existing fields and `modifiers_applied`.
- Defaults:
  - `macro_event_window`: labels `fed_week`, `cpi_week`, `nfp_week`, `global_rate_decision`; cap size at `0.75`; disable leverage expansion; require new-long confirmation.
  - `policy_or_event_risk_window`: labels `budget_week`, `election_window`, `geopolitical_event`; cap size at `0.50`; disallow leverage; prefer cash/hedges; require new-long confirmation.
  - Do not globally throttle `expiry_week` or `earnings_season` by default.

## Sub-Agent Execution To-Do

Run with `subagent-driven-development` using one fresh worker per task:

- [ ] Worker 1: Config model + YAML defaults.
- [ ] Worker 2: Strategy response logic + direct unit tests.
- [ ] Worker 3: Timeline integration + engine/schema tests.
- [ ] Worker 4: Docs/spec updates + stale-reference grep.
- [ ] Reviewer A: Spec-compliance review of all tasks.
- [ ] Reviewer B: Code-quality/blast-radius review of all tasks.

## Task 1: Add Config Home

**Files:**
- Modify: `src/regime_detection/_config_evidence_strategy.py`
- Modify: `src/regime_detection/config.py`
- Modify: `src/regime_detection/configs/core3-v2.0.0.yaml`
- Test: `tests/test_v2_config.py`

- [ ] **Step 1: Write failing config tests**

Add tests that prove the default V2 config ships two named event modifier rules and rejects out-of-range caps:

```python
def test_v2_default_config_has_strategy_event_modifiers() -> None:
    cfg = load_default_regime_config()
    assert cfg.strategy_event_modifiers is not None
    rules = cfg.strategy_event_modifiers.rules
    assert rules["macro_event_window"].labels == [
        "fed_week",
        "cpi_week",
        "nfp_week",
        "global_rate_decision",
    ]
    assert rules["macro_event_window"].position_size_cap == 0.75
    assert rules["policy_or_event_risk_window"].labels == [
        "budget_week",
        "election_window",
        "geopolitical_event",
    ]
    assert rules["policy_or_event_risk_window"].position_size_cap == 0.50


def test_strategy_event_modifier_config_rejects_invalid_position_cap(tmp_path: Path) -> None:
    pkg_file = importlib.resources.files("regime_detection").joinpath(
        "configs/core3-v2.0.0.yaml"
    )
    data = yaml.safe_load(pkg_file.read_text(encoding="utf-8"))
    data["strategy_event_modifiers"]["rules"]["macro_event_window"][
        "position_size_cap"
    ] = 1.25
    bad_yaml = tmp_path / "bad-strategy-event-modifier.yaml"
    bad_yaml.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_regime_config(bad_yaml)
```

- [ ] **Step 2: Run the config tests and verify RED**

Run:

```bash
rtk pytest tests/test_v2_config.py::test_v2_default_config_has_strategy_event_modifiers tests/test_v2_config.py::test_strategy_event_modifier_config_rejects_invalid_position_cap; echo EXIT:$?
```

Expected: fails because `RegimeConfig` has no `strategy_event_modifiers`.

- [ ] **Step 3: Implement config models**

Add to `src/regime_detection/_config_evidence_strategy.py`:

```python
class StrategyEventModifierRule(StrictBaseModel):
    """Execution overlay applied when any configured event-calendar label matches."""

    labels: list[str]
    position_size_cap: float = Field(ge=0.0, le=1.0)
    leverage_allowed: bool | None = None
    allow_leverage_expansion: bool | None = None
    require_confirmation_for_new_longs: bool | None = None
    prefer_cash_or_hedges: bool | None = None


class StrategyEventModifiersConfig(StrictBaseModel):
    """V2 strategy-response event-window overlays keyed by modifier name."""

    rules: dict[str, StrategyEventModifierRule]
```

Expose in `src/regime_detection/config.py` imports, `__all__`, and `RegimeConfig`:

```python
strategy_event_modifiers: StrategyEventModifiersConfig | None = None
```

Add to `core3-v2.0.0.yaml` near strategy/cohort config:

```yaml
strategy_event_modifiers:
  rules:
    macro_event_window:
      labels: [fed_week, cpi_week, nfp_week, global_rate_decision]
      position_size_cap: 0.75
      allow_leverage_expansion: false
      require_confirmation_for_new_longs: true
    policy_or_event_risk_window:
      labels: [budget_week, election_window, geopolitical_event]
      position_size_cap: 0.50
      leverage_allowed: false
      prefer_cash_or_hedges: true
      require_confirmation_for_new_longs: true
```

- [ ] **Step 4: Run config tests and ruff**

Run:

```bash
rtk pytest tests/test_v2_config.py::test_v2_default_config_has_strategy_event_modifiers tests/test_v2_config.py::test_strategy_event_modifier_config_rejects_invalid_position_cap; echo EXIT:$?
python3 -m ruff check src/regime_detection/_config_evidence_strategy.py src/regime_detection/config.py tests/test_v2_config.py; echo EXIT:$?
```

Expected: `EXIT:0` and `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/regime_detection/_config_evidence_strategy.py src/regime_detection/config.py src/regime_detection/configs/core3-v2.0.0.yaml tests/test_v2_config.py
git commit -m "feat(regime): configure strategy event modifiers"
```

## Task 2: Apply Event Modifiers In Strategy Response

**Files:**
- Modify: `src/regime_detection/strategy_response.py`
- Test: `tests/test_transition_and_strategy.py`

- [ ] **Step 1: Write failing strategy tests**

Add tests:

```python
from regime_detection.config import load_default_regime_config


def test_strategy_response_applies_macro_event_window_modifier() -> None:
    cfg = load_default_regime_config()
    response = build_strategy_response(
        trend_direction_active="bull",
        trend_character_active="trending",
        volatility_state_active="normal_vol",
        breadth_state_active="healthy_breadth",
        transition_risk_state="stable",
        event_calendar_labels=("earnings_season", "fed_week"),
        event_modifier_config=cfg.strategy_event_modifiers,
    )

    assert response.position_size_multiplier == 0.75
    assert response.allow_leverage_expansion is False
    assert response.require_confirmation_for_new_longs is True
    assert response.modifiers_applied == ["bull_healthy_low_vol", "macro_event_window"]


def test_strategy_response_applies_stricter_policy_event_modifier_on_overlap() -> None:
    cfg = load_default_regime_config()
    response = build_strategy_response(
        trend_direction_active="bull",
        trend_character_active="trending",
        volatility_state_active="normal_vol",
        breadth_state_active="healthy_breadth",
        transition_risk_state="stable",
        event_calendar_labels=("fed_week", "election_window"),
        event_modifier_config=cfg.strategy_event_modifiers,
    )

    assert response.position_size_multiplier == 0.50
    assert response.leverage_allowed is False
    assert response.prefer_cash_or_hedges is True
    assert response.modifiers_applied == [
        "bull_healthy_low_vol",
        "macro_event_window",
        "policy_or_event_risk_window",
    ]
```

- [ ] **Step 2: Run strategy tests and verify RED**

Run:

```bash
rtk pytest tests/test_transition_and_strategy.py::test_strategy_response_applies_macro_event_window_modifier tests/test_transition_and_strategy.py::test_strategy_response_applies_stricter_policy_event_modifier_on_overlap; echo EXIT:$?
```

Expected: fails because `build_strategy_response` has no `event_calendar_labels` argument.

- [ ] **Step 3: Implement strategy logic**

Update the signature:

```python
from regime_detection.config import StrategyEventModifiersConfig

def build_strategy_response(
    *,
    trend_direction_active: str,
    trend_character_active: str,
    volatility_state_active: str,
    breadth_state_active: str,
    transition_risk_state: str,
    event_calendar_labels: tuple[str, ...] = ("normal_calendar",),
    event_modifier_config: StrategyEventModifiersConfig | None = None,
) -> StrategyResponse:
```

Add helper:

```python
def _matching_event_modifier_names(
    *, labels: tuple[str, ...], config: StrategyEventModifiersConfig | None
) -> list[str]:
    if config is None:
        return []
    label_set = set(labels)
    return [
        name
        for name, rule in config.rules.items()
        if label_set.intersection(rule.labels)
    ]
```

After existing transition-risk modifier blocks and before `StrategyResponse(...)`, apply tightening overlays:

```python
for modifier_name in _matching_event_modifier_names(
    labels=event_calendar_labels, config=event_modifier_config
):
    rule = event_modifier_config.rules[modifier_name]
    position_size_multiplier = min(position_size_multiplier, rule.position_size_cap)
    if rule.leverage_allowed is False:
        leverage_allowed = False
    if rule.allow_leverage_expansion is False:
        allow_leverage_expansion = False
    if rule.require_confirmation_for_new_longs is True:
        require_confirmation_for_new_longs = True
    if rule.prefer_cash_or_hedges is True:
        prefer_cash_or_hedges = True
    modifiers.append(modifier_name)
```

- [ ] **Step 4: Run strategy tests and existing strategy tests**

Run:

```bash
rtk pytest tests/test_transition_and_strategy.py::test_strategy_response_de_risks_crisis_final_state tests/test_transition_and_strategy.py::test_strategy_response_handles_recovery_attempt_final_state tests/test_transition_and_strategy.py::test_strategy_response_de_risks_high_transition_risk_final_state tests/test_transition_and_strategy.py::test_strategy_response_applies_macro_event_window_modifier tests/test_transition_and_strategy.py::test_strategy_response_applies_stricter_policy_event_modifier_on_overlap; echo EXIT:$?
python3 -m ruff check src/regime_detection/strategy_response.py tests/test_transition_and_strategy.py; echo EXIT:$?
```

Expected: `EXIT:0` and `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/regime_detection/strategy_response.py tests/test_transition_and_strategy.py
git commit -m "feat(regime): apply event calendar strategy overlays"
```

## Task 3: Wire Timeline And Engine Output

**Files:**
- Modify: `src/regime_detection/timeline.py`
- Test: `tests/test_schema_and_timeline.py`
- Test: `tests/test_transition_and_strategy.py`

- [ ] **Step 1: Write failing integration test**

Add a timeline test that patches `build_strategy_response` and asserts matching labels/config are passed:

```python
def test_timeline_passes_event_calendar_matching_labels_to_strategy_response(
    mocker, market_df_for_asof, event_calendar_df
) -> None:
    spy = mocker.spy(timeline_module, "build_strategy_response")
    as_of = date(2024, 1, 19)
    engine = RegimeEngine()
    engine.classify(
        as_of_date=as_of,
        market_data=market_df_for_asof(as_of),
        event_calendar=event_calendar_df,
    )

    kwargs = spy.call_args.kwargs
    assert "event_calendar_labels" in kwargs
    assert isinstance(kwargs["event_calendar_labels"], tuple)
    assert kwargs["event_modifier_config"] is not None
```

Use the module alias already present in `tests/test_schema_and_timeline.py`; if absent, import `regime_detection.timeline as timeline_module`.

- [ ] **Step 2: Run integration test and verify RED**

Run:

```bash
rtk pytest tests/test_schema_and_timeline.py::test_timeline_passes_event_calendar_matching_labels_to_strategy_response; echo EXIT:$?
```

Expected: fails because timeline does not pass those kwargs.

- [ ] **Step 3: Wire timeline**

Modify `src/regime_detection/timeline.py` call:

```python
strategy_response=build_strategy_response(
    trend_direction_active=trend_direction_output.active_label,
    trend_character_active=trend_character_output.active_label,
    volatility_state_active=volatility_output.active_label,
    breadth_state_active=breadth_output.active_label,
    transition_risk_state=transition_output.state,
    event_calendar_labels=event_output.matching_labels,
    event_modifier_config=working_context.config.strategy_event_modifiers,
),
```

- [ ] **Step 4: Run integration and focused schema tests**

Run:

```bash
rtk pytest tests/test_schema_and_timeline.py::test_timeline_passes_event_calendar_matching_labels_to_strategy_response tests/test_schema_and_timeline.py::test_runtime_evidence_fields_use_named_payloads; echo EXIT:$?
python3 -m ruff check src/regime_detection/timeline.py tests/test_schema_and_timeline.py; echo EXIT:$?
```

Expected: `EXIT:0` and `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/regime_detection/timeline.py tests/test_schema_and_timeline.py
git commit -m "feat(regime): wire event labels into strategy response"
```

## Task 4: Documentation And Spec Updates

**Files:**
- Modify: `docs/regime_engine_v2_spec.md`
- Modify: `docs/regime_engine_v1_final_spec.md`
- Modify: `docs/transition_risk.md`

- [ ] **Step 1: Update V2 spec**

Document that:

- `event_calendar.matching_labels` has two downstream consumers:
  - transition score `macro_event`
  - strategy response execution overlays
- `primary_label` remains display-only.
- Strategy overlays tighten execution only and are configured by `strategy_event_modifiers`.

Add an example YAML block matching `core3-v2.0.0.yaml`.

- [ ] **Step 2: Update V1 spec**

Clarify V1 remains unchanged unless `strategy_event_modifiers` exists in config. Do not add V1 frozen fixture changes.

- [ ] **Step 3: Update transition-risk docs**

Replace the statement “Strategy consumes only `transition_risk.state`” with:

```markdown
Strategy consumes `transition_risk.state` for regime-instability overlays and
`event_calendar.matching_labels` for scheduled-event execution overlays. The
event overlay does not change transition-risk state or score.
```

- [ ] **Step 4: Grep stale claims**

Run:

```bash
rg -n "Strategy consumes only `transition_risk.state`|event_calendar_active|event_calendar\\.active_label|event_calendar\\.label" docs src tests scripts --glob '!tests/fixtures/**'; echo EXIT:$?
```

Expected: `EXIT:1` or only intentionally historical ADR references.

- [ ] **Step 5: Commit**

```bash
git add docs/regime_engine_v2_spec.md docs/regime_engine_v1_final_spec.md docs/transition_risk.md
git commit -m "docs(regime): specify event calendar strategy overlays"
```

## Task 5: Final Verification And Review

**Files:**
- No production edits expected.

- [ ] **Step 1: Run focused tests**

```bash
rtk pytest tests/test_v2_config.py tests/test_transition_and_strategy.py tests/test_schema_and_timeline.py tests/test_v2_comparison.py tests/test_v1_frozen_replay.py; echo EXIT:$?
```

Expected: `EXIT:0`.

- [ ] **Step 2: Run lint**

```bash
python3 -m ruff check src/regime_detection/_config_evidence_strategy.py src/regime_detection/config.py src/regime_detection/strategy_response.py src/regime_detection/timeline.py tests/test_v2_config.py tests/test_transition_and_strategy.py tests/test_schema_and_timeline.py; echo EXIT:$?
```

Expected: `All checks passed!` and `EXIT:0`.

- [ ] **Step 3: Run direct pytest for clearer output**

```bash
python3 -m pytest tests/test_v2_config.py tests/test_transition_and_strategy.py tests/test_schema_and_timeline.py tests/test_v2_comparison.py -q; echo EXIT:$?
```

Expected: all selected tests pass and `EXIT:0`.

- [ ] **Step 4: Run diff hygiene**

```bash
git diff --check; echo EXIT:$?
git status -sb
```

Expected: `EXIT:0`; branch clean or only intended commits ahead before push.

- [ ] **Step 5: Final review subagents**

Dispatch:

- Spec reviewer: verify every requirement in this plan is implemented and no V1 frozen wire field changed unexpectedly.
- Code quality reviewer: verify event overlay logic only tightens risk, no `primary_label` use in logic, no silent skips, config validation is strict.

- [ ] **Step 6: Push and report**

```bash
git push
```

Report commits, test outputs, and any full-suite limitations.

## Self-Review

- Spec coverage: Covers config home, runtime strategy logic, timeline wiring, docs/specs, focused tests, V1 frozen guard.
- Placeholder scan: No placeholder markers or vague “add tests” instructions remain.
- Type consistency: Plan consistently uses `event_calendar_labels: tuple[str, ...]`, `event_modifier_config: StrategyEventModifiersConfig | None`, and `strategy_event_modifiers` on `RegimeConfig`.
