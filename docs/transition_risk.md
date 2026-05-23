# Transition Risk

Transition risk measures whether the current regime is becoming unstable enough
to reduce risk before the main regime labels fully flip.

It is calculated in five stages.

## 1. Build Required Inputs

Transition uses the current active labels from the main axes:

- `trend_direction.active_label`
- `trend_character.active_label`
- `volatility_state.active_label`
- `breadth_state.active_label`
- `event_calendar.matching_labels`

It also uses market and history inputs:

- SPY close
- SPY 50-day moving average
- whether stable axis labels changed today (`stable_changed_today`)
- how many axes switched today (`axis_switch_count`)
- rolling 5-session sum of axis switches (`recent_axis_switch_count`)
- sessions since the most recent axis switch within the last 60 NYSE
  sessions (`days_since_axis_switch`; `None` if no switch in that window)
- whether `trend_direction.stable_label` was bear at any point in the prior
  60 NYSE sessions, **excluding today**. The implementation shifts the bear
  series by one session before the rolling lookback so the
  `recovery_attempt` rule cannot fire while today's stable label is still
  bear (transition-window hysteresis lag).

And V2 score evidence:

- trend drawdown / moving-average break
- short-vs-long realized volatility
- percent above 50dma
- network fragility
- credit funding, when available
- volume liquidity, when available
- event calendar
- HMM/change-point/cluster instability, when available

Missing required score infrastructure raises a runtime error. Missing optional
evidence is omitted, and weights are renormalized only if enough configured
weight coverage remains.

Required score infrastructure:

- `feature_store.volatility_state_v2`
- `feature_store.breadth_state_v2.pct_above_50dma`
- `feature_store.network_fragility`
- `feature_store.trend_direction_v2`
- `context.config.transition_score`

Optional evidence:

- credit funding
- volume liquidity
- HMM
- change point
- cluster

## 2. Compute Component Scores

Each warning area becomes a normalized `0.0` to `1.0` score:

- `trend_break`
- `volatility_acceleration`
- `breadth_deterioration`
- `correlation_fragility`
- `credit_stress`
- `liquidity_stress`
- `macro_event`
- `model_instability`

Examples:

- `volatility_acceleration` rises when short-term realized volatility is high
  versus long-term realized volatility.
- `breadth_deterioration` rises when fewer stocks are above their 50-day moving
  average.
- `trend_break` rises when SPY is below its 50-day moving average or in a
  252-day drawdown.
- `correlation_fragility` rises when assets move together, eigenvalue
  concentration rises, effective rank falls, or top-3 absorption is high.
- `model_instability` rises when HMM probability shifts, change-point score
  rises, or cluster ID changes.

## 3. Compute Weighted Transition Score

The component scores are combined using configured weights:

| Component | Weight |
| --- | ---: |
| `trend_break` | `0.18` |
| `volatility_acceleration` | `0.16` |
| `breadth_deterioration` | `0.16` |
| `correlation_fragility` | `0.14` |
| `credit_stress` | `0.12` |
| `liquidity_stress` | `0.10` |
| `macro_event` | `0.06` |
| `model_instability` | `0.08` |

The output is `transition_risk.score`, from `0.0` to `1.0`.

Score bands:

| Score | Base State |
| --- | --- |
| `< 0.35` | `stable` |
| `0.35` to `< 0.55` | `weakening` |
| `0.55` to `< 0.75` | `transition_warning` |
| `>= 0.75` | `high_transition_risk` |

## 4. Apply Hard Overrides

The weighted score gives the normal transition-pressure level, but some market
patterns override it. Override precedence is evaluated **before** the
`insufficient_data` check, so an emergency rule like `crisis` fires even when
another axis is `unknown`. The score catches gradual deterioration; hard
overrides preserve emergency and well-known market patterns that should not be
diluted by calm components.

Numeric thresholds below are the defaults in
`TransitionScoreConfig.overrides`; deployments can tune them in the YAML
config without code changes.

| # | Rule | Final state | Trigger condition |
|---|---|---|---|
| 1 | `crisis` | `crisis` | `volatility_state.active_label == "crisis_vol"`. Single-axis override — fires unconditionally. |
| 2 | `bear_stress` | `bear_stress` | `trend_direction.active_label == "bear"` AND `volatility_state.active_label ∈ {high_vol, crisis_vol}` AND (`breadth_state.active_label ∈ {weak_breadth, divergent_fragile, unknown}` OR `credit_stress ≥ overrides.credit_stress` (default `0.70`)). |
| 3 | `fragile_bull` | `fragile_bull` | `trend_direction.active_label == "bull"` AND (`breadth_state.active_label == "divergent_fragile"` OR `correlation_fragility ≥ overrides.correlation_fragility` (default `0.70`) OR `credit_stress ≥ overrides.credit_stress`). |
| 4 | `recovery_attempt` | `recovery_attempt` | `trend_character.active_label == "recovery_attempt"` **OR** (`prior_bear` AND `close > sma_50` AND `breadth_state.active_label ∈ {recovery_breadth, healthy_breadth}`). OR-composed: either signal alone fires the rule. |
| 5 | `sideways_stress` | `watch` | `trend_direction.active_label == "sideways"` AND `volatility_state.active_label == "high_vol"` AND `breadth_state.active_label ∈ {weak_breadth, divergent_fragile}`. |
| 6 | `event_transition_watch` | `watch` | `macro_event ≥ overrides.macro_event_min` (default `1.0`) AND `score ≥ overrides.score_elevated_min` (default `0.35`) AND `macro_event` is the dominant component (≥ every other component). |
| 7 | `post_switch_cooldown` | `watch` | Within `cooldown_window_days` of an axis switch (default 5 NYSE sessions) AND not in `crisis_vol`. Applied **only when the score-derived state is `stable`**. |
| 8 | `insufficient_data` | `insufficient_data` | Any of `trend_direction`, `trend_character`, `volatility_state`, `breadth_state` active label is `unknown`. |
| 9 | _none_ | score-derived | Score-band-to-state mapping from §3. |

## 5. Debounce Final State

After the raw state is selected, final `transition_risk.state` is debounced.

Default confirmation windows:

| State | Required Consecutive Raw Prints |
| --- | ---: |
| `stable` | `1` |
| `watch` | `1` |
| `weakening` | `2` |
| `transition_warning` | `2` |
| `high_transition_risk` | `2` |
| `fragile_bull` | `2` |
| `recovery_attempt` | `2` |
| `bear_stress` | `1` |
| `crisis` | `1` |
| `insufficient_data` | `1` |

If a new state has not persisted long enough, the public state stays at the
prior active state and `state_confirmation_pending` is added to
`triggered_rules`.

The confirmation windows are **deliberately asymmetric**: `stable` and the
emergency rules (`crisis`, `bear_stress`) confirm in a single print so the
engine can de-risk immediately, while non-stable promotions (`weakening`,
`transition_warning`, `high_transition_risk`, `fragile_bull`,
`recovery_attempt`) require two consecutive prints so a single noisy session
cannot drive a regime change. This favours fast de-escalation over fast
escalation and is the intended trade-off.

## Final Output

Transition produces:

- `transition_risk.state`
- `transition_risk.score`
- `transition_risk.score_components`
- `transition_risk.primary_drivers`
- `transition_risk.triggered_rules`
- `transition_risk.evidence`
- `transition_risk.data_quality`

Strategy consumes only `transition_risk.state`.

Audit and reporting should show the score, components, drivers, rules, data
quality, and switch-count evidence. Downstream code should not rebuild
transition-risk decisions from component scores.

`primary_drivers` lists the components with values at or above
`overrides.primary_driver_min` (default `0.35`), ranked by component value,
capped at 3 entries.
