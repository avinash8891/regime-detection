# Verified Regime Issues Long-Term Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the verified regime-engine issues with durable architecture changes, explicit contracts, and regression tests.

**Architecture:** Separate data quality from regime state, add one canonical strategy decision resolver, make cold-start behavior explicit, and reconcile evidence-only/model-label contracts with config and docs. Do not patch around symptoms; each task introduces or tightens a single contract and proves it with focused tests.

**Tech Stack:** Python 3.14, pandas, Pydantic v2, existing `rtk pytest`/direct pytest test harness, existing `RegimeOutput` models and config YAML.

---

## Verified Scope

Included:
- `unknown` labels enter hysteresis and can become stable state after sufficiently long data outages.
- `strategy_response`, `agent_routing`, and `strategy_family_constraints` are emitted in parallel with no canonical effective resolver.
- Severe percentile-gated labels have cold-start dead zones: `correlation_to_one`, `systemic_stress`, `deleveraging`, `liquidity_gap_behavior`.
- `systemic_stress` cannot fire when credit/funding is unavailable; this is silent except for label fallback.
- `volume_liquidity` currently emits `normal_volume` when gap percentile fields are NaN but volume/return are normal.
- `disinflation_yield_independent=True` intentionally diverges from the quoted spec behavior and needs source-of-truth reconciliation.
- HMM/GMM label maps default to `None` in production config, so mapped labels are absent unless operator config supplies maps.
- `parse_datetime_index()` returns tz-naive timestamps while `calendar.as_date()` rejects tz-naive timestamps.
- Central-bank text evidence is low-accuracy lexicon evidence and needs an explicit evidence-quality contract.

Excluded:
- “Default neutral permits every strategy” is not verified. `strategy_family_constraints.default_neutral` blocks `breakout` and `short_vol`.
- “Future-dated rows produce negative freshness through the standard quality path” is not verified. `_window_to_asof()` truncates to `<= as_of_date`.

---

## File Responsibility Map

- `src/regime_detection/hysteresis.py`: keep pure label hysteresis, or add a new explicit data-gap-aware wrapper if used globally.
- `src/regime_detection/axis_builders/per_label.py`: centralize DQ-to-hysteresis behavior for V2 per-label axes.
- `src/regime_detection/axis_builders/breadth.py`: migrate the breadth custom path to the same data-gap-aware contract.
- `src/regime_detection/strategy_constraints.py`: new canonical resolver for effective strategy permissions.
- `src/regime_detection/models.py`: add output models for effective strategy constraints and possibly data-gap metadata.
- `src/regime_detection/timeline.py`: wire canonical strategy constraints and model/evidence metadata into `RegimeOutput`.
- `src/regime_detection/network_fragility_rules.py`: cold-start fallback predicates and explicit credit-unavailable evidence.
- `src/regime_detection/credit_funding.py`: cold-start fallback for `deleveraging`.
- `src/regime_detection/volume_liquidity_rules.py`: make NaN gap-percentile behavior explicit and non-quiet.
- `src/regime_detection/_config_layer1.py`, `_config_layer2.py`, `_config_evidence_strategy.py`: add validated config knobs for cold-start policy, strategy resolver policy, label maps, and evidence quality.
- `src/regime_detection/configs/core3-v2.0.0.yaml`: production defaults for the new explicit policies.
- `src/regime_detection/temporal.py`: normalize parsed session indexes to the engine’s intended timezone/date contract.
- `src/regime_detection/central_bank_text.py`: surface evidence-quality metadata and prevent accidental rule consumption.
- `docs/regime_engine_v2_spec.md`, ADR docs under `docs/decisions/`: reconcile spec/ADR/config truth.
- Tests under `tests/`: one regression file per behavior, using real existing fixtures or exact scalar rule inputs.

---

### Task 1: Freeze Hysteresis Across Data-Quality Outages

**Files:**
- Modify: `src/regime_detection/hysteresis.py`
- Modify: `src/regime_detection/axis_builders/per_label.py`
- Modify: `src/regime_detection/axis_builders/breadth.py`
- Test: `tests/test_per_label_hysteresis.py`
- Test: `tests/test_network_fragility_classifier.py`
- Test: `tests/test_breadth_state_v2_labels.py`

- [ ] Write failing tests proving that a data-quality `unknown` run does not replace the last known stable label while inside a configured freeze window.
- [ ] Add an explicit data-gap-aware hysteresis function or wrapper that takes `raw_labels` plus per-day `DataQuality`.
- [ ] Preserve pure `apply_per_label_asymmetric_hysteresis()` for cases where `unknown` is intentionally a real label.
- [ ] Add config for `max_unknown_freeze_days` per affected axis, with validated non-negative integers.
- [ ] Migrate `build_per_label_axis_outputs()` to use the data-gap-aware path.
- [ ] Migrate `build_breadth_axis_series()` off its custom post-hysteresis override and onto the same contract.
- [ ] Verify recovery behavior: after the freeze window expires, the axis must emit explicit stale/unknown state with evidence, not silently pretend the prior regime is current.
- [ ] Run: `python3.14 -m pytest -o addopts='' tests/test_per_label_hysteresis.py tests/test_network_fragility_classifier.py tests/test_breadth_state_v2_labels.py -q; echo "EXIT:$?"`.
- [ ] Run: `rtk pytest tests/test_per_label_hysteresis.py tests/test_network_fragility_classifier.py tests/test_breadth_state_v2_labels.py -q; echo "EXIT:$?"`.
- [ ] Commit: `fix(regime): separate data gaps from hysteresis state`.

### Task 2: Add One Canonical Effective Strategy Resolver

**Files:**
- Create: `src/regime_detection/strategy_constraints.py`
- Modify: `src/regime_detection/models.py`
- Modify: `src/regime_detection/timeline.py`
- Test: `tests/test_strategy_constraints.py`
- Test: `tests/test_schema_and_timeline.py`
- Test: `tests/test_transition_and_strategy.py`

- [ ] Write failing tests for contradictory inputs, e.g. `strategy_response.allow_breakout=True` while family constraints block `breakout`.
- [ ] Define a Pydantic `EffectiveStrategyConstraint` model with fields: `family`, `allowed`, `sources`, `blocking_reasons`, and resolved scalar settings.
- [ ] Implement resolver precedence: most restrictive result wins across `agent_routing.blocked_strategy_modes`, `strategy_family_constraints`, and `strategy_response`.
- [ ] Wire `effective_strategy_constraints` into `RegimeOutput` without removing existing fields.
- [ ] Add timeline tests proving the resolver consumes the actual `agent_routing`, `strategy_family_constraints`, and `strategy_response` emitted for the same day.
- [ ] Add docs explaining that downstream consumers should read `effective_strategy_constraints` as the source of truth.
- [ ] Run: `python3.14 -m pytest -o addopts='' tests/test_strategy_constraints.py tests/test_schema_and_timeline.py tests/test_transition_and_strategy.py -q; echo "EXIT:$?"`.
- [ ] Run: `rtk pytest tests/test_strategy_constraints.py tests/test_schema_and_timeline.py tests/test_transition_and_strategy.py -q; echo "EXIT:$?"`.
- [ ] Commit: `feat(regime): add effective strategy constraint resolver`.

### Task 3: Replace Cold-Start Dead Zones With Explicit Fallback Policy

**Files:**
- Modify: `src/regime_detection/network_fragility_rules.py`
- Modify: `src/regime_detection/credit_funding.py`
- Modify: `src/regime_detection/volume_liquidity_rules.py`
- Modify: `src/regime_detection/_config_layer1.py`
- Modify: `src/regime_detection/_config_layer2.py`
- Modify: `src/regime_detection/configs/core3-v2.0.0.yaml`
- Test: `tests/test_network_fragility_rules.py`
- Test: `tests/test_credit_funding.py`
- Test: `tests/test_volume_liquidity_rules.py`

- [ ] Write failing tests showing current NaN percentile behavior for `correlation_to_one`, `deleveraging`, and `liquidity_gap_behavior`.
- [ ] Add explicit cold-start fallback config for each percentile-gated severe rule, using absolute-level thresholds calibrated from existing rule inputs.
- [ ] For `network_fragility`, allow `correlation_to_one` fallback only when raw correlation/eigen/vol/drawdown evidence is present and severe.
- [ ] For `credit_funding`, allow `deleveraging` fallback only when SPY/TLT/USD/realized-vol conditions are severe and correlation percentile is unavailable due to warmup, not stale data.
- [ ] For `volume_liquidity`, stop returning `normal_volume` when required gap-stress inputs are unavailable; return `unknown` or a named `insufficient_gap_history` evidence reason.
- [ ] Add evidence fields indicating whether a label used full percentile path or cold-start fallback path.
- [ ] Add config validation that fallback thresholds cannot be looser than the percentile path’s intended severity.
- [ ] Run: `python3.14 -m pytest -o addopts='' tests/test_network_fragility_rules.py tests/test_credit_funding.py tests/test_volume_liquidity_rules.py -q; echo "EXIT:$?"`.
- [ ] Run: `rtk pytest tests/test_network_fragility_rules.py tests/test_credit_funding.py tests/test_volume_liquidity_rules.py -q; echo "EXIT:$?"`.
- [ ] Commit: `fix(regime): add explicit cold-start severe-label fallbacks`.

### Task 4: Make Credit-Unavailable Systemic Stress a First-Class State

**Files:**
- Modify: `src/regime_detection/network_fragility_rules.py`
- Modify: `src/regime_detection/axis_builders/network_fragility.py`
- Modify: `src/regime_detection/models.py`
- Test: `tests/test_network_fragility_rules.py`
- Test: `tests/test_network_fragility_classifier.py`

- [ ] Write failing tests where all systemic-stress market conditions are met but `credit_funding_label=None`.
- [ ] Decide and encode the long-term policy: either fire a distinct `systemic_stress_unconfirmed` label or emit `correlation_to_one` with mandatory `blocked_by_missing_credit_funding` evidence.
- [ ] Add the selected label/evidence to model literals and risk rank only if the label path is chosen.
- [ ] Ensure cohort routing treats the selected state as crisis-equivalent for safety if systemic conditions are otherwise met.
- [ ] Add tests proving missing credit cannot silently downgrade crisis conditions without evidence.
- [ ] Run: `python3.14 -m pytest -o addopts='' tests/test_network_fragility_rules.py tests/test_network_fragility_classifier.py tests/test_cohort_routing.py -q; echo "EXIT:$?"`.
- [ ] Run: `rtk pytest tests/test_network_fragility_rules.py tests/test_network_fragility_classifier.py tests/test_cohort_routing.py -q; echo "EXIT:$?"`.
- [ ] Commit: `fix(regime): surface credit-unavailable systemic stress`.

### Task 5: Reconcile Disinflation Spec, ADR, Config, And Tests

**Files:**
- Modify: `docs/regime_engine_v2_spec.md`
- Modify: `docs/decisions/0011-inflation-growth-rule-coverage-fix.md`
- Modify: `src/regime_detection/_config_layer2.py`
- Modify: `src/regime_detection/configs/core3-v2.0.0.yaml`
- Test: `tests/test_inflation_growth.py`
- Test: `tests/test_v2_config.py`

- [ ] Identify the authoritative behavior: yield-independent disinflation by default, or yield-confirmed disinflation by default.
- [ ] Update the spec or ADR so there is one source of truth.
- [ ] Keep `disinflation_yield_independent` explicit in YAML; do not rely on hidden defaults.
- [ ] Add a config test asserting the production default and citing the authoritative doc.
- [ ] Add rule tests for both `True` and `False` config modes.
- [ ] Run: `python3.14 -m pytest -o addopts='' tests/test_inflation_growth.py tests/test_v2_config.py -q; echo "EXIT:$?"`.
- [ ] Run: `rtk pytest tests/test_inflation_growth.py tests/test_v2_config.py -q; echo "EXIT:$?"`.
- [ ] Commit: `fix(regime): reconcile disinflation rule authority`.

### Task 6: Make HMM/GMM Label Mapping Operationally Explicit

**Files:**
- Modify: `src/regime_detection/_config_evidence_strategy.py`
- Modify: `src/regime_detection/configs/core3-v2.0.0.yaml`
- Modify: `src/regime_detection/timeline.py`
- Test: `tests/test_clustering.py`
- Test: `tests/test_hmm_state.py`
- Test: `tests/test_schema_and_timeline.py`

- [ ] Write failing tests proving production config currently emits `mapped_label=None`.
- [ ] Add explicit config policy: `label_map_required_for_output: true|false` for HMM and clustering.
- [ ] If maps remain optional, add `mapping_status` and `mapping_reason` to outputs so `None` is explainable.
- [ ] If maps are required, load committed verification maps and fail config validation when missing/incomplete.
- [ ] Ensure version coupling is strict: map model version must match fitted model version.
- [ ] Run: `python3.14 -m pytest -o addopts='' tests/test_clustering.py tests/test_hmm_state.py tests/test_schema_and_timeline.py -q; echo "EXIT:$?"`.
- [ ] Run: `rtk pytest tests/test_clustering.py tests/test_hmm_state.py tests/test_schema_and_timeline.py -q; echo "EXIT:$?"`.
- [ ] Commit: `fix(regime): make evidence label mapping explicit`.

### Task 7: Normalize Session Date/Timezone Contract

**Files:**
- Modify: `src/regime_detection/temporal.py`
- Modify: `src/regime_detection/calendar.py` only if the public contract changes.
- Test: `tests/test_temporal_normalization.py`
- Test: `tests/test_loaders.py`

- [ ] Write failing tests showing `parse_datetime_index()` output cannot be passed to `as_date()`.
- [ ] Decide the engine contract: plain NYSE session dates or tz-aware NY/Eastern timestamps.
- [ ] Implement the contract in `parse_datetime_index()` and update docstrings.
- [ ] Add tests for naive strings, tz-aware strings, DST boundary dates, and mixed invalid inputs.
- [ ] Ensure persistent model output still stores dates consistently with existing wire contract.
- [ ] Run: `python3.14 -m pytest -o addopts='' tests/test_temporal_normalization.py tests/test_loaders.py -q; echo "EXIT:$?"`.
- [ ] Run: `rtk pytest tests/test_temporal_normalization.py tests/test_loaders.py -q; echo "EXIT:$?"`.
- [ ] Commit: `fix(regime): normalize session date parsing contract`.

### Task 8: Add Evidence-Quality Contract For Central Bank Text

**Files:**
- Modify: `src/regime_detection/central_bank_text.py`
- Modify: `src/regime_detection/monetary_pressure.py`
- Modify: `src/regime_detection/models.py`
- Modify: `docs/verification/lexicon_validation.md`
- Test: `tests/test_central_bank_text.py`
- Test: `tests/test_monetary_pressure_classifier.py`

- [ ] Write tests proving central-bank text score appears only as evidence and cannot affect rule labels.
- [ ] Add evidence metadata: `classifier_type=lexicon`, `sentence_accuracy=0.539`, `conditional_accuracy=0.709`, `rule_consumption=false`.
- [ ] Add a model field or nested evidence object so downstream consumers cannot confuse the score with a validated rule input.
- [ ] Add validation that `central_bank_text_score` is not present in `MonetaryPressureRuleInputs`.
- [ ] Run: `python3.14 -m pytest -o addopts='' tests/test_central_bank_text.py tests/test_monetary_pressure_classifier.py -q; echo "EXIT:$?"`.
- [ ] Run: `rtk pytest tests/test_central_bank_text.py tests/test_monetary_pressure_classifier.py -q; echo "EXIT:$?"`.
- [ ] Commit: `fix(regime): label central bank text as evidence-only`.

### Task 9: Full Integration And Replay Gate

**Files:**
- Modify only files touched by earlier tasks if integration breaks.
- Test: `tests/test_v1_frozen_replay.py`
- Test: `tests/test_schema_and_timeline.py`
- Test: `tests/test_v2_config.py`
- Test: full default suite.

- [ ] Run V1 frozen replay before final integration changes.
- [ ] Run V2 config/schema tests after all new output fields are wired.
- [ ] Run direct default pytest: `python3.14 -m pytest ; echo "EXIT:$?"`.
- [ ] Run RTK failures-only gate: `rtk pytest ; echo "EXIT:$?"`.
- [ ] If V1 frozen output changes, stop and either preserve byte identity or update the V1 shim/fixture only under the explicit AGENTS.md V1 replay rule.
- [ ] Update release notes or ADR index with the final behavior changes.
- [ ] Commit: `test(regime): verify long-term issue fixes end to end`.

---

## Execution Order

1. Task 1 first: data-gap/hysteresis semantics affect many later outputs.
2. Task 2 second: downstream strategy consumers need a single resolver before safety behavior changes.
3. Tasks 3 and 4 together: both address severe-label under-detection.
4. Task 5 independently: resolves spec/config truth.
5. Task 6 independently: evidence mapping governance.
6. Task 7 independently: date/time contract.
7. Task 8 independently: evidence quality contract.
8. Task 9 last: integration and replay validation.

## Acceptance Criteria

- No verified issue remains as undocumented implicit behavior.
- Data outages do not silently mutate regime hysteresis state inside the configured freeze window.
- Downstream strategy consumers have one canonical effective constraint field.
- Cold-start severe-label behavior is explicit, tested, and evidenced.
- Missing credit/funding cannot silently hide systemic-stress conditions.
- Disinflation behavior has one authoritative source across spec, ADR, config, and tests.
- HMM/GMM mapped-label absence is either impossible by config or explicitly represented in output.
- Session date parsing has one timezone/date contract.
- Central-bank text evidence is visibly evidence-only with quality metadata.
- Direct pytest and RTK gates pass with actual `EXIT:0` output.
