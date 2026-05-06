#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "tests" / "fixtures" / "raw"
DERIVED_PATH = REPO_ROOT / "tests" / "fixtures" / "derived" / "golden_dates.yaml"
REPORT_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "verification" / "golden_dates_report.yaml"
)
CONFIG_PATHS = [
    REPO_ROOT / "src" / "regime_detection" / "configs" / "core3-v1.0.0.yaml",
    REPO_ROOT / "configs" / "core3-v1.0.0.yaml",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_hysteresis_days() -> dict[str, int]:
    cfg_path = next((p for p in CONFIG_PATHS if p.exists()), None)
    if cfg_path is None:
        raise SystemExit("Could not find core config yaml (core3-v1.0.0.yaml) for fixture verification.")
    data = yaml.safe_load(cfg_path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("hysteresis"), dict):
        raise SystemExit(f"Invalid config structure in {cfg_path}")
    h = data["hysteresis"]
    return {
        "trend_direction": int(h["trend_direction_deescalation_days"]),
        "trend_character": int(h["trend_character_deescalation_days"]),
        "volatility_state": int(h["volatility_deescalation_days"]),
        "breadth_state": int(h["breadth_deescalation_days"]),
        "composite": int(h["composite_deescalation_days"]),
        "event_calendar": int(h["event_calendar_days"]),
    }


def _pct_rank_last(arr: np.ndarray) -> float:
    # Percentile rank of last element within the window, inclusive.
    x = arr[-1]
    if np.isnan(x):
        return float("nan")
    arr2 = arr[~np.isnan(arr)]
    if arr2.size == 0:
        return float("nan")
    return float(np.mean(arr2 <= x))


def _wilder_ewm(series: pd.Series, n: int, min_periods: int | None = None) -> pd.Series:
    return series.ewm(alpha=1 / n, adjust=False, min_periods=min_periods or n).mean()


def _compute_adx_14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=close.index)

    n = 14
    atr = _wilder_ewm(tr, n)
    # Guard against zero ATR and zero denominator in DX; flat/missing windows should not
    # create inf/NaN that leaks into label selection.
    atr_safe = atr.replace(0.0, np.nan)
    plus_di = 100 * _wilder_ewm(plus_dm, n) / atr_safe
    minus_di = 100 * _wilder_ewm(minus_dm, n) / atr_safe
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom) * 100
    dx = dx.fillna(0.0)
    return _wilder_ewm(dx, n)


@dataclass(frozen=True)
class Inputs:
    spy: pd.DataFrame
    rsp: pd.DataFrame
    vixy: pd.DataFrame


def _load_raw() -> Inputs:
    def load(sym: str) -> pd.DataFrame:
        path = RAW_DIR / f"{sym}.csv"
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values("date").set_index("date")
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise SystemExit(f"{path} missing required columns: {missing}")
        if df.index.has_duplicates:
            raise SystemExit(f"{path} has duplicate dates in index")
        return df

    spy = load("SPY")
    rsp = load("RSP")
    vixy = load("VIXY")

    # Ensure the expected time alignment for deterministic feature computation.
    common = spy.index.intersection(rsp.index).intersection(vixy.index)
    if len(common) == 0:
        raise SystemExit("Raw fixtures have no overlapping dates across SPY/RSP/VIXY")
    if not spy.index.equals(common) or not rsp.index.equals(common) or not vixy.index.equals(common):
        # Do not silently align via pandas index union; that's a common source of NaN-induced
        # 'unknown' outputs and unintentional fixture drift.
        raise SystemExit(
            "Raw fixtures must have identical trading-day indices across SPY/RSP/VIXY. "
            f"Counts: SPY={len(spy.index)} RSP={len(rsp.index)} VIXY={len(vixy.index)} common={len(common)}"
        )

    return Inputs(spy=spy, rsp=rsp, vixy=vixy)


def _compute_features(inp: Inputs) -> dict[str, pd.Series]:
    spy = inp.spy
    rsp = inp.rsp
    vixy = inp.vixy

    close = spy["close"]
    high = spy["high"]
    low = spy["low"]

    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    return_1d = close / close.shift(1) - 1
    return_5d = close / close.shift(5) - 1
    return_10d = close / close.shift(10) - 1
    return_21d = close / close.shift(21) - 1
    return_63d = close / close.shift(63) - 1

    prior_63d_drawdown = close / close.rolling(63).max() - 1

    adx_14 = _compute_adx_14(high=high, low=low, close=close)

    daily_returns = close.pct_change()
    realized_vol_21d = daily_returns.rolling(21).std() * np.sqrt(252)
    realized_vol_percentile_252d = realized_vol_21d.rolling(252, min_periods=252).apply(
        _pct_rank_last, raw=True
    )

    vix_close = vixy["close"]
    vix_percentile_252d = vix_close.rolling(252, min_periods=252).apply(_pct_rank_last, raw=True)

    # ETF proxy breadth
    ratio = rsp["close"] / spy["close"]
    ratio_sma_50 = ratio.rolling(50).mean()
    ratio_return_20d = ratio / ratio.shift(20) - 1
    index_distance_from_63d_high = spy["close"] / spy["close"].rolling(63).max() - 1

    return {
        "close": close,
        "SMA_50": sma_50,
        "SMA_200": sma_200,
        "return_1d": return_1d,
        "return_5d": return_5d,
        "return_10d": return_10d,
        "return_21d": return_21d,
        "return_63d": return_63d,
        "prior_63d_drawdown": prior_63d_drawdown,
        "ADX_14": adx_14,
        "realized_vol_21d": realized_vol_21d,
        "realized_vol_percentile_252d": realized_vol_percentile_252d,
        "vix_percentile_252d": vix_percentile_252d,
        "relative_breadth_ratio": ratio,
        "relative_breadth_sma50": ratio_sma_50,
        "relative_breadth_return_20d": ratio_return_20d,
        "index_distance_from_63d_high": index_distance_from_63d_high,
    }


def _is_nan(x: Any) -> bool:
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _eval_trend_direction(feat: dict[str, pd.Series], dt: pd.Timestamp) -> tuple[str, dict[str, bool]]:
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
    """
    Match the engine's deterministic asymmetric hysteresis for trend_direction:
    - Escalation (higher risk_rank) updates stable immediately.
    - De-escalation requires N consecutive days.
    - active_label fast-path: if raw riskier than stable => active=raw else active=stable.
    """
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
    """
    Generic asymmetric hysteresis used by multiple axes.
    - Escalation updates stable immediately.
    - De-escalation requires N consecutive days on the candidate label.
    - Equal-rank label changes are treated as de-escalation candidates.
    - active fast-path: if raw riskier than stable => active=raw else active=stable.
    """
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


def _eval_trend_character(feat: dict[str, pd.Series], dt: pd.Timestamp) -> tuple[str, dict[str, bool]]:
    adx = feat["ADX_14"].loc[dt]
    ret10 = feat["return_10d"].loc[dt]
    ret21 = feat["return_21d"].loc[dt]
    prior_dd = feat["prior_63d_drawdown"].loc[dt]
    close = feat["close"].loc[dt]
    sma50 = feat["SMA_50"].loc[dt]

    required_nan = any(_is_nan(x) for x in [adx, ret10, ret21, prior_dd, close, sma50])
    if required_nan:
        return "unknown", {"unknown_required_nan": True}

    recovery_attempt = (prior_dd <= -0.10) and (close > sma50) and (ret10 >= 0.05)
    trending = (adx >= 20) and (abs(ret21) >= 0.05)
    chop = (adx < 20) and (abs(ret10) < 0.03) and (abs(ret21) < 0.05)
    transition = not (recovery_attempt or trending or chop)

    # precedence: recovery_attempt > trending > chop > transition
    if recovery_attempt:
        label = "recovery_attempt"
    elif trending:
        label = "trending"
    elif chop:
        label = "chop"
    else:
        label = "transition"

    return label, {
        "recovery_attempt": recovery_attempt,
        "trending": trending,
        "chop": chop,
        "transition": transition,
    }


_TC_RISK_RANK: dict[str, int] = {
    "trending": 0,
    "chop": 1,
    "recovery_attempt": 1,
    "transition": 2,
    "unknown": 2,
}


def _eval_volatility_state(feat: dict[str, pd.Series], dt: pd.Timestamp) -> tuple[str, dict[str, bool]]:
    ret1 = feat["return_1d"].loc[dt]
    ret5 = feat["return_5d"].loc[dt]
    ret21 = feat["return_21d"].loc[dt]
    vol_pct = feat["realized_vol_percentile_252d"].loc[dt]
    vix_pct = feat["vix_percentile_252d"].loc[dt]

    required_nan = any(_is_nan(x) for x in [ret1, ret5, ret21, vol_pct, vix_pct])
    if required_nan:
        return "unknown", {"unknown_required_nan": True}

    crisis = (
        (ret1 <= -0.05)
        or (ret5 <= -0.08)
        or ((vol_pct >= 0.90) and (ret21 <= -0.05))
        or (vix_pct >= 0.95)
    )
    high_vol = (vol_pct >= 0.80) or (vix_pct >= 0.80)
    low_vol = vol_pct <= 0.30
    normal_vol = not (crisis or high_vol or low_vol)

    # precedence: crisis > high > low > normal
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

    divergent_fragile = (idx_dist >= -0.05) and (ratio < ratio_sma50) and (ratio_ret20 <= -0.03)
    weak_breadth = (ratio < ratio_sma50) and (ratio_ret20 < 0)
    healthy_breadth = (ratio > ratio_sma50) and (ratio_ret20 >= 0)
    neutral_breadth = not (divergent_fragile or weak_breadth or healthy_breadth)

    # precedence: divergent_fragile > weak > healthy > neutral
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


def _eval_transition_risk(
    dt: pd.Timestamp,
    trend_direction: str,
    trend_character: str,
    volatility_state: str,
    breadth_state: str,
) -> tuple[str, dict[str, bool]]:
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
    bull_fragile_warning = (trend_direction == "bull") and (breadth_state == "divergent_fragile")
    recovery_attempt = trend_character == "recovery_attempt"

    # V1 precedence:
    # crisis_override > bear_stress_warning > bull_fragile_warning > recovery_attempt > stable > unknown
    if crisis_override:
        label = "crisis_override"
    elif bear_stress_warning:
        label = "bear_stress_warning"
    elif bull_fragile_warning:
        label = "bull_fragile_warning"
    elif recovery_attempt:
        label = "recovery_attempt"
    elif any_unknown:
        label = "unknown"
    else:
        label = "stable"

    return label, {
        "crisis_override": crisis_override,
        "bear_stress_warning": bear_stress_warning,
        "bull_fragile_warning": bull_fragile_warning,
        "recovery_attempt": recovery_attempt,
    }


GoldenIntent = dict[str, Any]


INTENTS: list[dict[str, Any]] = [
    {
        "intent_id": "bull_trending_lowvol_healthy",
        "intent_date": "2020-08-11",
        "intent": {
            "trend_direction": "bull",
            "trend_character": "trending",
            "volatility_state": "low_vol",
            "breadth_state": "healthy_breadth",
            "transition_risk": "stable",
        },
        "search_window_trading_days": 120,
        "notes": "Steady bull, trending, low vol, healthy breadth",
    },
    {
        "intent_id": "volmageddon_crisis",
        "intent_date": "2018-02-09",
        "intent": {
            # do not constrain trend_direction: spec labels depend on SMA cross state
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
            "trend_character": "transition",
            "volatility_state": "normal_vol",
            "breadth_state": "healthy_breadth",
            "transition_risk": "stable",
        },
        "search_window_trading_days": 10,
        "notes": "Bull market normal conditions (trend_character may remain transition under hysteresis)",
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
            "transition_risk": "recovery_attempt",
        },
        "search_window_trading_days": 10,
        "notes": "Post-crash recovery attempt; breadth pinned by rules in ETF-proxy mode",
    },
    {
        "intent_id": "late2021_bull_lowvol",
        "intent_date": "2021-11-15",
        "intent": {
            "trend_direction": "bull",
            "trend_character": "trending",
            "volatility_state": "low_vol",
            "breadth_state": "weak_breadth",
            "transition_risk": "stable",
        },
        "search_window_trading_days": 20,
        "notes": "Late-2021 bull / low vol with weak breadth (ETF-proxy rules)",
    },
    {
        "intent_id": "jun2022_bear_stress",
        "intent_date": "2022-06-29",
        "intent": {
            "trend_direction": "bear",
            "trend_character": "trending",
            "volatility_state": "high_vol",
            "breadth_state": "weak_breadth",
            "transition_risk": "bear_stress_warning",
        },
        "search_window_trading_days": 10,
        "notes": "2022 drawdown; stress warning",
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
            "transition_risk": "stable",
        },
        "search_window_trading_days": 10,
        "notes": "Early 2024 bull / low vol / healthy breadth",
    },
]


def _evaluate_all(feat: dict[str, pd.Series]) -> pd.DataFrame:
    idx = feat["close"].index
    out = pd.DataFrame(index=idx)

    td = []
    tc = []
    vs = []
    bs = []
    tr = []
    td_ev = []
    tc_ev = []
    vs_ev = []
    bs_ev = []
    tr_ev = []
    for dt in idx:
        td_label, td_e = _eval_trend_direction(feat, dt)
        tc_label, tc_e = _eval_trend_character(feat, dt)
        vs_label, vs_e = _eval_volatility_state(feat, dt)
        bs_label, bs_e = _eval_breadth_state_etf_proxy(feat, dt)
        tr_label, tr_e = _eval_transition_risk(
            dt=dt,
            trend_direction=td_label,
            trend_character=tc_label,
            volatility_state=vs_label,
            breadth_state=bs_label,
        )
        td.append(td_label)
        tc.append(tc_label)
        vs.append(vs_label)
        bs.append(bs_label)
        tr.append(tr_label)
        td_ev.append(td_e)
        tc_ev.append(tc_e)
        vs_ev.append(vs_e)
        bs_ev.append(bs_e)
        tr_ev.append(tr_e)

    hyst = _load_hysteresis_days()

    # Apply V1 trend_direction hysteresis so fixtures match engine outputs (active_label).
    td_stable, td_active = _apply_trend_direction_hysteresis(
        raw_labels=td, deescalation_days=hyst["trend_direction"]
    )
    out["trend_direction"] = td_active
    out["_trend_direction_raw"] = td
    out["_trend_direction_stable"] = td_stable
    # Apply V1 trend_character hysteresis so fixtures match engine outputs (active_label).
    tc_stable, tc_active = _apply_asymmetric_hysteresis(
        raw_labels=tc, risk_rank=_TC_RISK_RANK, deescalation_days=hyst["trend_character"]
    )
    out["trend_character"] = tc_active
    out["_trend_character_raw"] = tc
    out["_trend_character_stable"] = tc_stable
    out["volatility_state"] = vs
    out["breadth_state"] = bs
    out["transition_risk"] = tr
    out["_td_evidence"] = [
        {
            "raw_label": raw,
            "stable_label": st,
            "active_label": act,
            "rule_evidence": ev,
            "deescalation_days": hyst["trend_direction"],
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
            "deescalation_days": hyst["trend_character"],
            "risk_rank": _TC_RISK_RANK,
        }
        for raw, st, act, ev in zip(tc, tc_stable, tc_active, tc_ev, strict=True)
    ]
    out["_vs_evidence"] = vs_ev
    out["_bs_evidence"] = bs_ev
    out["_tr_evidence"] = tr_ev
    return out


def _nearest_trading_days(index: pd.DatetimeIndex, base: pd.Timestamp, k: int) -> pd.DatetimeIndex:
    # Assumes index is sorted, tz-naive.
    pos = int(np.searchsorted(index.values, base.to_datetime64()))
    lo = max(0, pos - k)
    hi = min(len(index), pos + k + 1)
    return index[lo:hi]


def _pick_fixture_date(
    labels: pd.DataFrame,
    intent_date: str,
    intent: dict[str, str],
    search_window_trading_days: int,
) -> pd.Timestamp:
    base = pd.Timestamp(intent_date)
    if base not in labels.index:
        raise SystemExit(
            f"intent_date must be an NYSE trading day present in labels index. "
            f"Got intent_date={intent_date}."
        )
    # Only search within the explicitly provided trading-day window.
    # If the intent can't be satisfied within this episode window, the intent
    # must be rewritten (date and/or expected labels) rather than widening
    # the search and accidentally pinning a different episode.
    window = _nearest_trading_days(labels.index, base, int(search_window_trading_days))
    df = labels.loc[labels.index.intersection(window)]
    mask = pd.Series(True, index=df.index)
    for k, v in intent.items():
        mask &= df[k].eq(v)
    candidates = df[mask]
    if len(candidates) > 0:
        if base in candidates.index:
            return base

        delta_days = np.abs(((candidates.index - base) / np.timedelta64(1, "D")).astype(int))
        best = int(delta_days.min())
        subset = candidates.iloc[np.where(delta_days == best)[0]]
        return pd.Timestamp(subset.index.min())

    actual = labels.loc[base] if base in labels.index else None
    raise SystemExit(
        f"No fixture candidate found for intent={intent} starting from intent_date={intent_date}. "
        f"Base row={None if actual is None else actual.to_dict()}"
    )


def _serialize_scalar(x: Any) -> Any:
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        if np.isnan(x):
            return None
        # Normalize float noise so fixture generation is stable across
        # minor numpy/pandas/BLAS differences.
        return float(round(float(x), 12))
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return str(pd.Timestamp(x).date())
    return x


def _serialize_obj(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _serialize_obj(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_serialize_obj(v) for v in x]
    return _serialize_scalar(x)


def main() -> None:
    derived_doc, report_doc = generate_docs()
    DERIVED_PATH.write_text(yaml.safe_dump(derived_doc, sort_keys=False))
    REPORT_PATH.write_text(yaml.safe_dump(report_doc, sort_keys=False))
    print(json.dumps({"derived": str(DERIVED_PATH), "report": str(REPORT_PATH)}, indent=2))


def generate_docs(
    *, generated_at_utc: str | None = None, generated_by_commit: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    inp = _load_raw()
    feat = _compute_features(inp)
    labels = _evaluate_all(feat)

    generated_at = generated_at_utc or _utc_iso_now()
    generated_by = generated_by_commit or _git_head_sha()

    raw_hashes = {
        "SPY.csv": _sha256_file(RAW_DIR / "SPY.csv"),
        "RSP.csv": _sha256_file(RAW_DIR / "RSP.csv"),
        "VIXY.csv": _sha256_file(RAW_DIR / "VIXY.csv"),
    }

    derived_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []

    for item in INTENTS:
        intent_date = item["intent_date"]
        intent = item["intent"]
        pick = _pick_fixture_date(
            labels=labels,
            intent_date=intent_date,
            intent=intent,
            search_window_trading_days=int(item["search_window_trading_days"]),
        )
        row = labels.loc[pick]
        as_of = str(pick.date())

        expected = {
            "trend_direction": row["trend_direction"],
            "trend_character": row["trend_character"],
            "volatility_state": row["volatility_state"],
            "breadth_state": row["breadth_state"],
            "transition_risk": row["transition_risk"],
        }

        derived_rows.append(
            {
                "intent_id": item["intent_id"],
                "intent_date": intent_date,
                "as_of_date": as_of,
                "expected": expected,
            }
        )

        feature_keys = [
            "close",
            "SMA_50",
            "SMA_200",
            "return_1d",
            "return_5d",
            "return_10d",
            "return_21d",
            "return_63d",
            "prior_63d_drawdown",
            "ADX_14",
            "realized_vol_21d",
            "realized_vol_percentile_252d",
            "vix_percentile_252d",
            "relative_breadth_ratio",
            "relative_breadth_sma50",
            "relative_breadth_return_20d",
            "index_distance_from_63d_high",
        ]
        features_at = {k: _serialize_scalar(feat[k].loc[pick]) for k in feature_keys}

        base = pd.Timestamp(intent_date)
        delta_calendar_days = int(abs((pick - base) / np.timedelta64(1, "D")))

        report_rows.append(
            {
                "intent_id": item["intent_id"],
                "intent_date": intent_date,
                "as_of_date": as_of,
                "delta_calendar_days": delta_calendar_days,
                "notes": item.get("notes", ""),
                "expected": expected,
                "features": features_at,
                "predicate_evaluations": {
                    "trend_direction": _serialize_obj(row["_td_evidence"]),
                    "trend_character": _serialize_obj(row["_tc_evidence"]),
                    "volatility_state": _serialize_obj(row["_vs_evidence"]),
                    "breadth_state": _serialize_obj(row["_bs_evidence"]),
                    "transition_risk": _serialize_obj(row["_tr_evidence"]),
                },
            }
        )

    derived_doc = {
        "generated_at_utc": generated_at,
        "generated_by_commit": generated_by,
        "raw_file_sha256": raw_hashes,
        "rows": derived_rows,
    }
    report_doc = {
        "generated_at_utc": generated_at,
        "generated_by_commit": generated_by,
        "raw_file_sha256": raw_hashes,
        "rows": report_rows,
    }
    return derived_doc, report_doc


def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if out:
            return out
    except Exception:
        pass
    return "unknown"


if __name__ == "__main__":
    main()
