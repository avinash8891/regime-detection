# ADR 0016: trend_direction `_RISK_RANK` vs spec precedence — two different orderings

**Date:** 2026-05-23
**Status:** Accepted

## Context

The comment audit of `src/regime_detection/trend_direction.py` flagged an apparent
contradiction between the `_RISK_RANK` dict (lines 31-39) and the V2 §1A
precedence (`docs/regime_engine_v2_spec.md` line 239):

```python
# In trend_direction.py
_RISK_RANK: dict[TrendDirectionLabel, int] = {
    "bull": 0,
    "sideways": 1,
    "recovery": 1,
    "transition": 2,
    "unknown": 2,
    "bear": 3,
    "euphoria": 4,
}
```

```text
# In spec §1A line 239
euphoria > bull > recovery > bear > sideways > transition > unknown
```

The orderings are structurally different:

- Spec puts `euphoria` first (winning override); code puts `euphoria` at 4
  (highest risk rank).
- Spec puts `bull` second; code puts `bull` at 0 (lowest risk rank).
- Spec puts `bear` fourth; code puts `bear` at 3.

An auditor reading both side-by-side could reasonably suspect a bug.

## Decision

**`_RISK_RANK` encodes downside risk (for hysteresis decay), not override
precedence (for which label wins when multiple fire).** The two are
semantically independent and SHOULD use different orderings.

### Risk-rank semantics

`_RISK_RANK` is consumed by `axis_builders/trend_direction.py` and passed to the
hysteresis layer (`hysteresis.py`) via `AxisOutput.evidence["risk_rank"]`. The
hysteresis layer uses risk-rank to decide:

1. **Asymmetric escalation/de-escalation cadence.** A label with higher
   risk-rank should be EASIER to escalate to (faster lock-in on regime
   deterioration) and HARDER to de-escalate from (slower release on
   spurious calm).
2. **Crisis override.** Labels with risk-rank ≥ N trigger downstream
   crisis-style modifiers in `transition_risk.py` and
   `strategy_response.py`.

Under these semantics:

- `euphoria = 4` is correct: an irrationally frothy market is the most
  dangerous state for risk management — speculative excess collapses
  fastest.
- `bear = 3` is correct: an established downtrend is high-risk.
- `bull = 0` is correct: a healthy uptrend is low-risk.
- `unknown = 2` (mid-rank) is correct: a NaN-driven unknown should
  neither fast-track escalation past `bear` nor strand the engine in a
  low-risk label across cold-start gaps.

### Precedence semantics (separate concern, not encoded in `_RISK_RANK`)

The spec precedence at §1A line 239 says: when multiple labels could
fire at session `t`, the LEFTMOST label wins. Precedence is encoded in
`trend_direction_v2._V2_TREND_PRECEDENCE`:

```python
_V2_TREND_PRECEDENCE: tuple[str, ...] = (
    "euphoria", "bull", "recovery", "bear",
    "sideways", "transition", "unknown",
)
```

This tuple is consumed by `evaluate_v2_trend_label` to resolve override
authority. `_RISK_RANK` plays no role in precedence resolution.

## Consequences

- **No code change required.** The current `_RISK_RANK` values are correct
  under risk-rank semantics.
- **Comment clarification at `trend_direction.py:27-30` already
  documents the distinction** ("Risk-rank intuition: recovery is mid-rally
  off a deep drawdown ..."), but the existing comment could be tightened.
  Future PR may rewrite the docstring to make the precedence-vs-risk
  separation more explicit.
- **Future label additions** must populate `_RISK_RANK` based on
  hysteresis-relevant downside risk, NOT on spec precedence position.
- **Reviewers reading the file side-by-side with the spec must NOT
  treat the ordering mismatch as a bug** — it is intentional and
  semantically correct.

## Related

- `regime_detection/trend_direction.py:31-39` (`_RISK_RANK`)
- `regime_detection/trend_direction_v2.py:_V2_TREND_PRECEDENCE`
- `regime_detection/hysteresis.py` (consumer of risk-rank)
- ADR 0010 (per-label hysteresis foundation)
- V2 spec §1A line 239 (precedence definition)

## Other axes

`_RISK_RANK` dicts exist on every L1 axis (`trend_character.py`,
`volatility_state.py`, `breadth_state.py`, etc.) and every L2 axis
(`credit_funding.py`, `inflation_growth.py`, etc.). Each encodes downside
risk for hysteresis, NOT spec precedence. This ADR applies symmetrically
to all of them.
