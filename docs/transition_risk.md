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
- whether stable axis labels changed recently
- how many axes switched today
- how many axes switched recently
- whether `trend_direction.stable_label` was bear in the prior 60 sessions

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
patterns override it.

Current override precedence:

1. `crisis`
2. `bear_stress`
3. `fragile_bull`
4. `recovery_attempt`
5. `sideways_stress` -> public state `watch`
6. `event_transition_watch` -> `watch`
7. `post_switch_cooldown` -> `watch` only if the score state is otherwise
   stable
8. `insufficient_data`
9. score-derived state

The score catches gradual deterioration. Hard overrides preserve emergency and
well-known market patterns that should not be diluted by calm components.

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
