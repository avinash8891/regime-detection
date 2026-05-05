from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from regime_detection.hysteresis import apply_asymmetric_hysteresis
from regime_detection.models import TransitionRiskOutput
from regime_detection.trend_direction import compute_features as td_compute_features
from regime_detection.trend_direction import raw_label_for_day as td_raw_label_for_day
from regime_detection.trend_direction import _RISK_RANK as TD_RISK_RANK  # type: ignore[attr-defined]
from regime_detection.trend_character import compute_features as tc_compute_features
from regime_detection.trend_character import raw_label_for_day as tc_raw_label_for_day
from regime_detection.trend_character import _RISK_RANK as TC_RISK_RANK  # type: ignore[attr-defined]
from regime_detection.volatility_state import compute_features as vol_compute_features
from regime_detection.volatility_state import raw_label_for_day as vol_raw_label_for_day
from regime_detection.volatility_state import _RISK_RANK as VOL_RISK_RANK  # type: ignore[attr-defined]
from regime_detection.breadth_state import compute_features as br_compute_features
from regime_detection.breadth_state import raw_label_for_day as br_raw_label_for_day
from regime_detection.breadth_state import _RISK_RANK as BR_RISK_RANK  # type: ignore[attr-defined]


_PRECEDENCE = [
    "crisis_override",
    "bear_stress_warning",
    "bull_fragile_warning",
    "recovery_attempt",
    "post_switch_cooldown",
    "stable",
    "unknown",
]


@dataclass(frozen=True)
class TransitionRiskInputs:
    close: pd.Series
    high: pd.Series
    low: pd.Series
    vix_proxy_close: pd.Series
    rsp_close: pd.Series
    as_of_date: date
    trend_direction_deescalation_days: int
    trend_character_deescalation_days: int
    volatility_deescalation_days: int
    breadth_deescalation_days: int


def classify_transition_risk(
    *,
    inp: TransitionRiskInputs,
    trend_direction_active: str,
    trend_character_active: str,
    volatility_active: str,
    breadth_active: str,
) -> TransitionRiskOutput:
    dt = pd.Timestamp(inp.as_of_date)

    # Compute stable series for cooldown and the "stable was bear in last 60 days" rule.
    td_stable = _trend_direction_stable_series(
        close=inp.close, as_of_date=inp.as_of_date, deescalation_days=inp.trend_direction_deescalation_days
    )
    tc_stable = _trend_character_stable_series(
        close=inp.close,
        high=inp.high,
        low=inp.low,
        as_of_date=inp.as_of_date,
        deescalation_days=inp.trend_character_deescalation_days,
    )
    vol_stable = _volatility_stable_series(
        close=inp.close,
        vix_proxy_close=inp.vix_proxy_close,
        as_of_date=inp.as_of_date,
        deescalation_days=inp.volatility_deescalation_days,
    )
    br_stable = _breadth_stable_series(
        spy_close=inp.close,
        rsp_close=inp.rsp_close,
        as_of_date=inp.as_of_date,
        deescalation_days=inp.breadth_deescalation_days,
    )

    crisis_override = volatility_active == "crisis_vol"
    bear_stress_warning = (
        (trend_direction_active == "bear")
        and (volatility_active in ["high_vol", "crisis_vol"])
        and (breadth_active in ["weak_breadth", "divergent_fragile", "unknown"])
    )
    bull_fragile_warning = (trend_direction_active == "bull") and (breadth_active == "divergent_fragile")

    # Recovery attempt: either explicit trend_character label, or "post-bear rebound" clause.
    recovery_attempt_a = trend_character_active == "recovery_attempt"
    recovery_attempt_b = _recovery_attempt_post_bear_clause(td_stable=td_stable, close=inp.close, dt=dt, breadth_active=breadth_active)
    recovery_attempt = recovery_attempt_a or recovery_attempt_b

    # Cooldown: any axis had a stable_label switch in the last 5 NYSE trading days.
    cooldown_days = min(
        _days_since_last_switch(td_stable),
        _days_since_last_switch(tc_stable),
        _days_since_last_switch(vol_stable),
        _days_since_last_switch(br_stable),
    )
    post_switch_cooldown = (cooldown_days is not None) and (cooldown_days <= 5)

    # Emergency override breaks cooldown.
    if crisis_override:
        post_switch_cooldown = False

    matched: list[str] = []
    if crisis_override:
        matched.append("crisis_override")
    if bear_stress_warning:
        matched.append("bear_stress_warning")
    if bull_fragile_warning:
        matched.append("bull_fragile_warning")
    if recovery_attempt:
        matched.append("recovery_attempt")
    if post_switch_cooldown:
        matched.append("post_switch_cooldown")
    if not matched:
        matched.append("stable")

    label = _pick_by_precedence(matched)
    evidence: dict[str, Any] = {
        "matched": [l for l in _PRECEDENCE if l in set(matched)],
        "selected_via_precedence": label,
        "recovery_attempt_branch_a": recovery_attempt_a,
        "recovery_attempt_branch_b": recovery_attempt_b,
        "cooldown_days_since_last_switch": cooldown_days,
    }
    return TransitionRiskOutput(label=label, evidence=evidence)


def _pick_by_precedence(labels: list[str]) -> str:
    s = set(labels)
    for lab in _PRECEDENCE:
        if lab in s:
            return lab
    return "unknown"


def _trim_series(s: pd.Series, *, as_of_date: date) -> pd.Series:
    out = s.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    dt = pd.Timestamp(as_of_date)
    if dt not in out.index:
        raise ValueError(f"as_of_date missing from series: {as_of_date.isoformat()}")
    return out.loc[:dt]


def _stable_from_raw(raw_labels: list[str], *, risk_rank: dict[str, int], deescalation_days: int) -> list[str]:
    stable, _active = apply_asymmetric_hysteresis(
        raw_labels=raw_labels,
        risk_rank=risk_rank,
        deescalation_days=deescalation_days,
    )
    return stable


def _trend_direction_stable_series(*, close: pd.Series, as_of_date: date, deescalation_days: int) -> list[str]:
    s = _trim_series(close, as_of_date=as_of_date)
    f = td_compute_features(s)
    raw: list[str] = []
    for day in s.index:
        lbl, _ev = td_raw_label_for_day(f, day)
        raw.append(lbl)
    return _stable_from_raw(raw, risk_rank=TD_RISK_RANK, deescalation_days=deescalation_days)


def _trend_character_stable_series(
    *, close: pd.Series, high: pd.Series, low: pd.Series, as_of_date: date, deescalation_days: int
) -> list[str]:
    c = _trim_series(close, as_of_date=as_of_date)
    h = _trim_series(high, as_of_date=as_of_date)
    l = _trim_series(low, as_of_date=as_of_date)
    f = tc_compute_features(close=c, high=h, low=l)
    raw: list[str] = []
    for day in c.index:
        lbl, _ev = tc_raw_label_for_day(f, day)
        raw.append(lbl)
    return _stable_from_raw(raw, risk_rank=TC_RISK_RANK, deescalation_days=deescalation_days)


def _volatility_stable_series(
    *, close: pd.Series, vix_proxy_close: pd.Series, as_of_date: date, deescalation_days: int
) -> list[str]:
    c = _trim_series(close, as_of_date=as_of_date)
    v = _trim_series(vix_proxy_close, as_of_date=as_of_date)
    f = vol_compute_features(close=c, vix_proxy_close=v)
    raw: list[str] = []
    for day in c.index:
        lbl, _ev = vol_raw_label_for_day(f, day)
        raw.append(lbl)
    return _stable_from_raw(raw, risk_rank=VOL_RISK_RANK, deescalation_days=deescalation_days)


def _breadth_stable_series(
    *, spy_close: pd.Series, rsp_close: pd.Series, as_of_date: date, deescalation_days: int
) -> list[str]:
    spy = _trim_series(spy_close, as_of_date=as_of_date)
    rsp = _trim_series(rsp_close, as_of_date=as_of_date)
    f = br_compute_features(spy_close=spy, rsp_close=rsp)
    raw: list[str] = []
    for day in spy.index:
        lbl, _ev = br_raw_label_for_day(f, day)
        raw.append(lbl)
    return _stable_from_raw(raw, risk_rank=BR_RISK_RANK, deescalation_days=deescalation_days)


def _days_since_last_switch(stable_labels: list[str]) -> int | None:
    if len(stable_labels) < 2:
        return None
    last_switch_idx: int | None = None
    for i in range(1, len(stable_labels)):
        if stable_labels[i] != stable_labels[i - 1]:
            last_switch_idx = i
    if last_switch_idx is None:
        return None
    return (len(stable_labels) - 1) - last_switch_idx


def _recovery_attempt_post_bear_clause(
    *,
    td_stable: list[str],
    close: pd.Series,
    dt: pd.Timestamp,
    breadth_active: str,
) -> bool:
    # Clause:
    # - trend_direction.stable_label was bear at any point in last 60 trading days
    # - close > SMA_50
    # - breadth_state.active_label in [recovery_breadth, healthy_breadth]
    # Note: recovery_breadth is PIT-only in V1; it can never be emitted by ETF-proxy breadth.
    if breadth_active not in ["recovery_breadth", "healthy_breadth"]:
        return False
    c = close.copy()
    c.index = pd.to_datetime(c.index)
    c = c.sort_index()
    if dt not in c.index:
        return False
    sma50 = c.rolling(50).mean()
    if pd.isna(sma50.loc[dt]) or pd.isna(c.loc[dt]):
        return False
    if not bool(c.loc[dt] > sma50.loc[dt]):
        return False

    # Check last 60 trading days of stable direction, inclusive of as_of_date.
    window = td_stable[-60:] if len(td_stable) >= 60 else td_stable
    return any(lbl == "bear" for lbl in window)

