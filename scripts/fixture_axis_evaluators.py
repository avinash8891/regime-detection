from __future__ import annotations

from typing import Any

import pandas as pd


def _is_nan(x: Any) -> bool:
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _eval_trend_direction(
    feat: dict[str, pd.Series], dt: pd.Timestamp
) -> tuple[str, dict[str, bool]]:
    close = feat["close"].loc[dt]
    sma50 = feat["SMA_50"].loc[dt]
    sma200 = feat["SMA_200"].loc[dt]
    ret63 = feat["return_63d"].loc[dt]

    required_nan = any(_is_nan(x) for x in [close, sma50, sma200, ret63])
    if required_nan:
        return "unknown", {"unknown_required_nan": True}

    within_5pct_sma200 = (close >= sma200 * 0.95) and (close <= sma200 * 1.05)

    bull = (close > sma50) and (close > sma200) and (sma50 > sma200)
    bear = (close < sma50) and (close < sma200) and (sma50 < sma200)
    sideways = (abs(ret63) < 0.05) and within_5pct_sma200
    transition = not (bull or bear or sideways)

    label = "transition"
    if bull:
        label = "bull"
    elif bear:
        label = "bear"
    elif sideways:
        label = "sideways"

    return label, {
        "bull": bull,
        "bear": bear,
        "sideways": sideways,
        "transition": transition,
    }


_TD_RISK_RANK: dict[str, int] = {
    "bull": 0,
    "sideways": 1,
    "transition": 2,
    "bear": 3,
    "unknown": 2,
}


def _apply_trend_direction_hysteresis(
    *, raw_labels: list[str], deescalation_days: int
) -> tuple[list[str], list[str]]:
    stable: list[str] = []
    active: list[str] = []

    stable_label = raw_labels[0]
    pending: str | None = None
    cnt = 0

    for raw in raw_labels:
        rr = _TD_RISK_RANK[raw]
        sr = _TD_RISK_RANK[stable_label]

        if rr > sr:
            stable_label = raw
            pending = None
            cnt = 0
        elif rr < sr:
            if deescalation_days == 0:
                stable_label = raw
                pending = None
                cnt = 0
            else:
                if pending != raw:
                    pending = raw
                    cnt = 1
                else:
                    cnt += 1
                if cnt >= deescalation_days:
                    stable_label = raw
                    pending = None
                    cnt = 0
        else:
            if raw != stable_label:
                if deescalation_days == 0:
                    stable_label = raw
                    pending = None
                    cnt = 0
                else:
                    if pending != raw:
                        pending = raw
                        cnt = 1
                    else:
                        cnt += 1
                    if cnt >= deescalation_days:
                        stable_label = raw
                        pending = None
                        cnt = 0
            else:
                pending = None
                cnt = 0

        stable.append(stable_label)
        active.append(raw if rr > _TD_RISK_RANK[stable_label] else stable_label)

    return stable, active


def _apply_asymmetric_hysteresis(
    *, raw_labels: list[str], risk_rank: dict[str, int], deescalation_days: int
) -> tuple[list[str], list[str]]:
    stable: list[str] = []
    active: list[str] = []

    stable_label = raw_labels[0]
    pending: str | None = None
    cnt = 0

    for raw in raw_labels:
        rr = risk_rank[raw]
        sr = risk_rank[stable_label]

        if rr > sr:
            stable_label = raw
            pending = None
            cnt = 0
        elif rr < sr or raw != stable_label:
            if deescalation_days == 0:
                stable_label = raw
                pending = None
                cnt = 0
            else:
                if pending != raw:
                    pending = raw
                    cnt = 1
                else:
                    cnt += 1
                if cnt >= deescalation_days:
                    stable_label = raw
                    pending = None
                    cnt = 0
        else:
            pending = None
            cnt = 0

        stable.append(stable_label)
        active.append(raw if rr > risk_rank[stable_label] else stable_label)

    return stable, active


def _eval_trend_character(
    feat: dict[str, pd.Series], dt: pd.Timestamp
) -> tuple[str, dict[str, bool]]:
    adx = feat["ADX_14"].loc[dt]
    ret10 = feat["return_10d"].loc[dt]
    ret21 = feat["return_21d"].loc[dt]
    ret63 = feat["return_63d"].loc[dt]
    prior_dd = feat["prior_63d_drawdown"].loc[dt]
    close = feat["close"].loc[dt]
    sma50 = feat["SMA_50"].loc[dt]

    required_nan = any(
        _is_nan(x) for x in [adx, ret10, ret21, ret63, prior_dd, close, sma50]
    )
    if required_nan:
        return "unknown", {"unknown_required_nan": True}

    close_window = feat["close"].loc[:dt].tail(20)
    if len(close_window) == 20 and not close_window.isna().any():
        cw_max = float(close_window.max())
        cw_min = float(close_window.min())
        midpoint = (cw_max + cw_min) / 2.0
        midpoint_excursion = (
            float(((close_window - midpoint).abs() / midpoint).max())
            if midpoint > 0
            else float("nan")
        )
    else:
        midpoint_excursion = float("nan")

    recovery_attempt = (prior_dd <= -0.10) and (close > sma50) and (ret10 >= 0.05)
    trending = (adx >= 20) and (abs(ret21) >= 0.05)
    range_bound = (
        (not _is_nan(midpoint_excursion))
        and (abs(ret63) < 0.05)
        and (midpoint_excursion <= 0.05)
        and (adx < 20)
    )
    chop = (adx < 20) and (abs(ret10) < 0.03) and (abs(ret21) < 0.05)
    transition = not (recovery_attempt or trending or range_bound or chop)

    if recovery_attempt:
        label = "recovery_attempt"
    elif trending:
        label = "trending"
    elif range_bound:
        label = "range_bound"
    elif chop:
        label = "chop"
    else:
        label = "transition"

    return label, {
        "recovery_attempt": recovery_attempt,
        "trending": trending,
        "range_bound": range_bound,
        "chop": chop,
        "transition": transition,
    }


_TC_RISK_RANK: dict[str, int] = {
    "trending": 0,
    "breakout_expansion": 0,
    "recovery_attempt": 1,
    "range_bound": 1,
    "chop": 1,
    "transition": 2,
    "unknown": 2,
}


def _eval_volatility_state(
    feat: dict[str, pd.Series], dt: pd.Timestamp
) -> tuple[str, dict[str, bool]]:
    ret1 = feat["return_1d"].loc[dt]
    ret5 = feat["return_5d"].loc[dt]
    ret21 = feat["return_21d"].loc[dt]
    vol_pct = feat["realized_vol_percentile_252d"].loc[dt]
    vix_pct = feat["vix_percentile_252d"].loc[dt]

    required_nan = any(_is_nan(x) for x in [ret1, ret5, ret21, vol_pct])
    if required_nan:
        return "unknown", {"unknown_required_nan": True}

    vix_crisis = (not _is_nan(vix_pct)) and (vix_pct >= 0.95)
    vix_high = (not _is_nan(vix_pct)) and (vix_pct >= 0.80)

    crisis = (
        (ret1 <= -0.05)
        or (ret5 <= -0.08)
        or ((vol_pct >= 0.90) and (ret21 <= -0.05))
        or vix_crisis
    )
    high_vol = (vol_pct >= 0.80) or vix_high
    low_vol = vol_pct <= 0.30
    normal_vol = not (crisis or high_vol or low_vol)

    if crisis:
        label = "crisis_vol"
    elif high_vol:
        label = "high_vol"
    elif low_vol:
        label = "low_vol"
    else:
        label = "normal_vol"

    return label, {
        "crisis_vol": crisis,
        "high_vol": high_vol,
        "low_vol": low_vol,
        "normal_vol": normal_vol,
        "vix_percentile_present": not _is_nan(vix_pct),
    }


_VS_RISK_RANK: dict[str, int] = {
    "low_vol": 0,
    "normal_vol": 1,
    "high_vol": 2,
    "crisis_vol": 3,
    "unknown": 2,
}


def _eval_breadth_state_etf_proxy(
    feat: dict[str, pd.Series], dt: pd.Timestamp
) -> tuple[str, dict[str, bool]]:
    ratio = feat["relative_breadth_ratio"].loc[dt]
    ratio_sma50 = feat["relative_breadth_sma50"].loc[dt]
    ratio_ret20 = feat["relative_breadth_return_20d"].loc[dt]
    idx_dist = feat["index_distance_from_63d_high"].loc[dt]

    required_nan = any(_is_nan(x) for x in [ratio, ratio_sma50, ratio_ret20, idx_dist])
    if required_nan:
        return "unknown", {"unknown_required_nan": True}

    divergent_fragile = (
        (idx_dist >= -0.05) and (ratio < ratio_sma50) and (ratio_ret20 <= -0.03)
    )
    weak_breadth = (ratio < ratio_sma50) and (ratio_ret20 < 0)
    healthy_breadth = (ratio > ratio_sma50) and (ratio_ret20 >= 0)
    neutral_breadth = not (divergent_fragile or weak_breadth or healthy_breadth)

    if divergent_fragile:
        label = "divergent_fragile"
    elif weak_breadth:
        label = "weak_breadth"
    elif healthy_breadth:
        label = "healthy_breadth"
    else:
        label = "neutral_breadth"

    return label, {
        "divergent_fragile": divergent_fragile,
        "weak_breadth": weak_breadth,
        "healthy_breadth": healthy_breadth,
        "neutral_breadth": neutral_breadth,
    }


_BS_RISK_RANK: dict[str, int] = {
    "healthy_breadth": 0,
    "neutral_breadth": 1,
    "weak_breadth": 2,
    "divergent_fragile": 3,
    "unknown": 2,
}


def _eval_transition_risk(
    trend_direction: str,
    trend_character: str,
    volatility_state: str,
    breadth_state: str,
    *,
    trend_direction_stable_history: list[str],
    stable_changed_today: bool,
    days_since_axis_switch: int | None,
    close: float,
    sma_50: float,
) -> tuple[str, dict[str, Any]]:
    any_unknown = any(
        lab == "unknown"
        for lab in [trend_direction, trend_character, volatility_state, breadth_state]
    )
    crisis_override = volatility_state == "crisis_vol"
    bear_stress_warning = (
        (trend_direction == "bear")
        and (volatility_state in ["high_vol", "crisis_vol"])
        and (breadth_state in ["weak_breadth", "divergent_fragile", "unknown"])
    )
    bull_fragile_warning = (
        (trend_direction == "bull") and (breadth_state == "divergent_fragile")
    )
    prior_bear = any(label == "bear" for label in trend_direction_stable_history[-60:])
    recovery_attempt = trend_character == "recovery_attempt" or (
        prior_bear
        and (not _is_nan(close))
        and (not _is_nan(sma_50))
        and (close > sma_50)
        and (breadth_state in ["recovery_breadth", "healthy_breadth"])
    )
    post_switch_cooldown = bool(
        stable_changed_today
        and days_since_axis_switch is not None
        and days_since_axis_switch <= 5
    )

    if crisis_override:
        label = "crisis_override"
    elif bear_stress_warning:
        label = "bear_stress_warning"
    elif bull_fragile_warning:
        label = "bull_fragile_warning"
    elif recovery_attempt:
        label = "recovery_attempt"
    elif post_switch_cooldown and not crisis_override:
        label = "post_switch_cooldown"
    elif any_unknown:
        label = "unknown"
    else:
        label = "stable"

    return label, {
        "crisis_override": crisis_override,
        "bear_stress_warning": bear_stress_warning,
        "bull_fragile_warning": bull_fragile_warning,
        "recovery_attempt": recovery_attempt,
        "post_switch_cooldown": post_switch_cooldown,
        "stable_changed_today": stable_changed_today,
        "days_since_axis_switch": days_since_axis_switch,
    }


INTENTS: list[dict[str, Any]] = [
    {
        "intent_id": "summer2020_bull_trending_lowvol",
        "intent_date": "2020-08-11",
        "intent": {
            "trend_direction": "bull",
            "trend_character": "trending",
            "volatility_state": "low_vol",
            "breadth_state": "neutral_breadth",
        },
        "search_window_trading_days": 120,
        "notes": "Summer 2020 bull with low vol and neutral breadth",
    },
    {
        "intent_id": "volmageddon_crisis",
        "intent_date": "2018-02-09",
        "intent": {
            "trend_character": "transition",
            "volatility_state": "crisis_vol",
            "transition_risk": "crisis_override",
        },
        "search_window_trading_days": 10,
        "notes": "Volmageddon episode; crisis_vol day",
    },
    {
        "intent_id": "dec2018_bear_stress",
        "intent_date": "2018-12-20",
        "intent": {
            "trend_direction": "bear",
            "trend_character": "trending",
            "volatility_state": "high_vol",
            "breadth_state": "weak_breadth",
            "transition_risk": "bear_stress_warning",
        },
        "search_window_trading_days": 10,
        "notes": "Late-2018 selloff; stress warning",
    },
    {
        "intent_id": "mid2019_bull_normal",
        "intent_date": "2019-06-28",
        "intent": {
            "trend_direction": "bull",
            "trend_character": "range_bound",
            "volatility_state": "normal_vol",
            "breadth_state": "healthy_breadth",
        },
        "search_window_trading_days": 60,
        "notes": "Bull market normal conditions; §1B range_bound catches the tight oscillation",
    },
    {
        "intent_id": "covid_crash_crisis",
        "intent_date": "2020-03-30",
        "intent": {
            "trend_direction": "bear",
            "volatility_state": "crisis_vol",
            "breadth_state": "weak_breadth",
            "transition_risk": "crisis_override",
        },
        "search_window_trading_days": 10,
        "notes": "COVID crash episode; pick nearest crisis_vol day with bear direction",
    },
    {
        "intent_id": "covid_recovery_attempt",
        "intent_date": "2020-04-17",
        "intent": {
            "trend_character": "recovery_attempt",
            "volatility_state": "high_vol",
        },
        "search_window_trading_days": 15,
        "notes": "Post-crash recovery attempt",
    },
    {
        "intent_id": "late2021_bull_lowvol",
        "intent_date": "2021-11-15",
        "intent": {
            "trend_direction": "bull",
            "trend_character": "trending",
            "volatility_state": "low_vol",
            "breadth_state": "weak_breadth",
        },
        "search_window_trading_days": 20,
        "notes": "Late-2021 bull / low vol with weak breadth (ETF-proxy rules)",
    },
    {
        "intent_id": "jun2022_bear_crisis",
        "intent_date": "2022-06-29",
        "intent": {
            "trend_direction": "bear",
            "trend_character": "trending",
            "volatility_state": "crisis_vol",
            "breadth_state": "weak_breadth",
            "transition_risk": "crisis_override",
        },
        "search_window_trading_days": 10,
        "notes": "2022 drawdown; crisis-vol episode",
    },
    {
        "intent_id": "jul2022_bear_stress",
        "intent_date": "2022-07-12",
        "intent": {
            "trend_direction": "bear",
            "trend_character": "trending",
            "volatility_state": "high_vol",
            "breadth_state": "weak_breadth",
            "transition_risk": "bear_stress_warning",
        },
        "search_window_trading_days": 10,
        "notes": "2022 bear market; stress warning (mid-2022 episode)",
    },
    {
        "intent_id": "early2024_bull_lowvol",
        "intent_date": "2023-12-19",
        "intent": {
            "trend_direction": "bull",
            "trend_character": "trending",
            "volatility_state": "low_vol",
            "breadth_state": "healthy_breadth",
        },
        "search_window_trading_days": 10,
        "notes": "Early 2024 bull / low vol / healthy breadth",
    },
]


def evaluate_all(
    feat: dict[str, pd.Series], *, hysteresis_days: dict[str, int]
) -> pd.DataFrame:
    idx = feat["close"].index
    out = pd.DataFrame(index=idx)

    td = []
    tc = []
    vs = []
    bs = []
    td_ev = []
    tc_ev = []
    vs_ev = []
    bs_ev = []
    for dt in idx:
        td_label, td_e = _eval_trend_direction(feat, dt)
        tc_label, tc_e = _eval_trend_character(feat, dt)
        vs_label, vs_e = _eval_volatility_state(feat, dt)
        bs_label, bs_e = _eval_breadth_state_etf_proxy(feat, dt)
        td.append(td_label)
        tc.append(tc_label)
        vs.append(vs_label)
        bs.append(bs_label)
        td_ev.append(td_e)
        tc_ev.append(tc_e)
        vs_ev.append(vs_e)
        bs_ev.append(bs_e)

    td_stable, td_active = _apply_trend_direction_hysteresis(
        raw_labels=td, deescalation_days=hysteresis_days["trend_direction"]
    )
    out["trend_direction"] = td_active
    out["_trend_direction_raw"] = td
    out["_trend_direction_stable"] = td_stable
    tc_stable, tc_active = _apply_asymmetric_hysteresis(
        raw_labels=tc,
        risk_rank=_TC_RISK_RANK,
        deescalation_days=hysteresis_days["trend_character"],
    )
    out["trend_character"] = tc_active
    out["_trend_character_raw"] = tc
    out["_trend_character_stable"] = tc_stable
    vs_stable, vs_active = _apply_asymmetric_hysteresis(
        raw_labels=vs,
        risk_rank=_VS_RISK_RANK,
        deescalation_days=hysteresis_days["volatility_state"],
    )
    out["volatility_state"] = vs_active
    out["_volatility_state_raw"] = vs
    out["_volatility_state_stable"] = vs_stable
    bs_stable, bs_active = _apply_asymmetric_hysteresis(
        raw_labels=bs,
        risk_rank=_BS_RISK_RANK,
        deescalation_days=hysteresis_days["breadth_state"],
    )
    out["breadth_state"] = bs_active
    out["_breadth_state_raw"] = bs
    out["_breadth_state_stable"] = bs_stable
    tr: list[str] = []
    tr_ev: list[dict[str, Any]] = []
    stable_keys_per_row = list(
        zip(td_stable, tc_stable, vs_stable, bs_stable, strict=True)
    )
    switch_days_ago: list[int | None] = []
    for i in range(len(stable_keys_per_row)):
        last_switch: int | None = None
        for j in range(i, 0, -1):
            if stable_keys_per_row[j] != stable_keys_per_row[j - 1]:
                last_switch = i - j
                break
        switch_days_ago.append(last_switch)

    for i, (dt, td_a, tc_a, vs_a, bs_a) in enumerate(
        zip(idx, td_active, tc_active, vs_active, bs_active, strict=True)
    ):
        stable_changed_today = (
            i > 0 and stable_keys_per_row[i] != stable_keys_per_row[i - 1]
        )
        close = feat["close"].loc[dt]
        sma_50 = feat["SMA_50"].loc[dt]
        tr_label, tr_e = _eval_transition_risk(
            trend_direction=td_a,
            trend_character=tc_a,
            volatility_state=vs_a,
            breadth_state=bs_a,
            trend_direction_stable_history=td_stable[: i + 1],
            stable_changed_today=stable_changed_today,
            days_since_axis_switch=switch_days_ago[i],
            close=float(close) if not _is_nan(close) else float("nan"),
            sma_50=float(sma_50) if not _is_nan(sma_50) else float("nan"),
        )
        tr.append(tr_label)
        tr_ev.append(tr_e)

    out["transition_risk"] = tr
    out["_td_evidence"] = [
        {
            "raw_label": raw,
            "stable_label": st,
            "active_label": act,
            "rule_evidence": ev,
            "deescalation_days": hysteresis_days["trend_direction"],
            "risk_rank": _TD_RISK_RANK,
        }
        for raw, st, act, ev in zip(td, td_stable, td_active, td_ev, strict=True)
    ]
    out["_tc_evidence"] = [
        {
            "raw_label": raw,
            "stable_label": st,
            "active_label": act,
            "rule_evidence": ev,
            "deescalation_days": hysteresis_days["trend_character"],
            "risk_rank": _TC_RISK_RANK,
        }
        for raw, st, act, ev in zip(tc, tc_stable, tc_active, tc_ev, strict=True)
    ]
    out["_vs_evidence"] = [
        {
            "raw_label": raw,
            "stable_label": st,
            "active_label": act,
            "rule_evidence": ev,
            "deescalation_days": hysteresis_days["volatility_state"],
            "risk_rank": _VS_RISK_RANK,
        }
        for raw, st, act, ev in zip(vs, vs_stable, vs_active, vs_ev, strict=True)
    ]
    out["_bs_evidence"] = [
        {
            "raw_label": raw,
            "stable_label": st,
            "active_label": act,
            "rule_evidence": ev,
            "deescalation_days": hysteresis_days["breadth_state"],
            "risk_rank": _BS_RISK_RANK,
        }
        for raw, st, act, ev in zip(bs, bs_stable, bs_active, bs_ev, strict=True)
    ]
    out["_tr_evidence"] = tr_ev
    return out
