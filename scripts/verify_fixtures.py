#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    plus_di = 100 * _wilder_ewm(plus_dm, n) / atr
    minus_di = 100 * _wilder_ewm(minus_dm, n) / atr
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = ((plus_di - minus_di).abs() / denom) * 100
    dx = dx.replace([np.inf, -np.inf], np.nan).fillna(0.0)
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
        return df

    return Inputs(spy=load("SPY"), rsp=load("RSP"), vixy=load("VIXY"))


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


_TD_RISK_RANK = {"bull": 0, "sideways": 1, "transition": 2, "bear": 3, "unknown": 2}


def _apply_hysteresis_active(
    *, raw_labels: list[str], deescalation_days: int
) -> tuple[list[str], list[str]]:
    # Matches docs/regime_engine_v1_final_spec.md §2.10 semantics.
    stable: list[str] = []
    active: list[str] = []

    stable_label = raw_labels[0]
    pending_label: str | None = None
    pending_count = 0

    for raw in raw_labels:
        raw_rank = _TD_RISK_RANK[raw]
        stable_rank = _TD_RISK_RANK[stable_label]

        if raw_rank > stable_rank:
            stable_label = raw
            pending_label = None
            pending_count = 0
        elif raw_rank == stable_rank and raw != stable_label:
            # Same-rank label changes are immediate (no debounce) to match engine hysteresis.
            stable_label = raw
            pending_label = None
            pending_count = 0
        elif raw_rank < stable_rank:
            if deescalation_days == 0:
                stable_label = raw
                pending_label = None
                pending_count = 0
            else:
                if pending_label != raw:
                    pending_label = raw
                    pending_count = 1
                else:
                    pending_count += 1
                if pending_count >= deescalation_days:
                    stable_label = raw
                    pending_label = None
                    pending_count = 0
        else:
            pending_label = None
            pending_count = 0

        stable.append(stable_label)
        if _TD_RISK_RANK[raw] > _TD_RISK_RANK[stable_label]:
            active.append(raw)
        else:
            active.append(stable_label)

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
            "trend_character": "transition",
            "volatility_state": "low_vol",
            "breadth_state": "neutral_breadth",
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
            "breadth_state": "weak_breadth",
            "transition_risk": "stable",
        },
        "search_window_trading_days": 10,
        "notes": "Bull market normal conditions",
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
            "trend_direction": "bear",
            "trend_character": "recovery_attempt",
            "volatility_state": "high_vol",
            "breadth_state": "weak_breadth",
            "transition_risk": "recovery_attempt",
        },
        "search_window_trading_days": 10,
        "notes": "Post-crash recovery attempt; breadth pinned by rules in ETF-proxy mode",
    },
    {
        "intent_id": "late2021_bull_lowvol",
        "intent_date": "2020-12-08",
        "intent": {
            "trend_direction": "bull",
            "trend_character": "trending",
            "volatility_state": "low_vol",
            "breadth_state": "healthy_breadth",
            "transition_risk": "stable",
        },
        "search_window_trading_days": 10,
        "notes": "Late-2021 bull; breadth may be narrower than expected, verify by rules",
    },
    {
        "intent_id": "jun2022_bear_stress",
        "intent_date": "2022-06-29",
        "intent": {
            "trend_direction": "bear",
            "trend_character": "transition",
            "volatility_state": "crisis_vol",
            "breadth_state": "weak_breadth",
            "transition_risk": "bear_stress_warning",
        },
        "search_window_trading_days": 10,
        "notes": "2022 drawdown; stress warning",
    },
    {
        "intent_id": "oct2022_bear_stress",
        "intent_date": "2022-07-12",
        "intent": {
            "trend_direction": "bear",
            "trend_character": "trending",
            "volatility_state": "high_vol",
            "breadth_state": "weak_breadth",
            "transition_risk": "bear_stress_warning",
        },
        "search_window_trading_days": 10,
        "notes": "2022 bear market; stress warning near Oct lows",
    },
    {
        "intent_id": "early2024_bull_lowvol",
        "intent_date": "2023-12-19",
        "intent": {
            "trend_direction": "bull",
            "trend_character": "transition",
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

    out["trend_direction"] = td
    out["trend_character"] = tc
    out["volatility_state"] = vs
    out["breadth_state"] = bs
    out["transition_risk"] = tr
    out["_td_evidence"] = td_ev
    out["_tc_evidence"] = tc_ev
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
    # Start near the intended date, then widen if needed. This matches the spec's
    # "rules win; replace fixture date, not predicates" principle, while still
    # biasing towards the same historical episode.
    windows = [
        search_window_trading_days,
        2 * search_window_trading_days,
        5 * search_window_trading_days,
    ]

    last_df: pd.DataFrame | None = None
    for w in windows:
        window = _nearest_trading_days(labels.index, base, int(w))
        df = labels.loc[labels.index.intersection(window)]
        last_df = df
        mask = pd.Series(True, index=df.index)
        for k, v in intent.items():
            col = k
            if k == "trend_direction" and "trend_direction_active" in df.columns:
                col = "trend_direction_active"
            if k == "trend_character" and "trend_character_active" in df.columns:
                col = "trend_character_active"
            if k == "volatility_state" and "volatility_state_active" in df.columns:
                col = "volatility_state_active"
            if k == "breadth_state" and "breadth_state_active" in df.columns:
                col = "breadth_state_active"
            mask &= df[col].eq(v)
        candidates = df[mask]
        if len(candidates) == 0:
            continue

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
        return float(x)
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


def generate_docs(*, generated_at_utc: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    inp = _load_raw()
    feat = _compute_features(inp)
    labels = _evaluate_all(feat)
    # Apply hysteresis for trend_direction to align pinned fixtures with engine outputs.
    td_stable, td_active = _apply_hysteresis_active(
        raw_labels=labels["trend_direction"].tolist(),
        deescalation_days=3,
    )
    labels["trend_direction_stable"] = td_stable
    labels["trend_direction_active"] = td_active

    # Trend character hysteresis uses the same asymmetric mechanics with its own ranks.
    tc_risk_rank = {
        "trending": 0,
        "chop": 1,
        "recovery_attempt": 1,
        "transition": 2,
        "unknown": 2,
    }

    def apply_generic(
        raw_labels: list[str], risk_rank: dict[str, int], deescalation_days: int
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
            elif rr == sr and raw != stable_label:
                # Same-rank label changes are immediate (no debounce) to match engine hysteresis.
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
                pending = None
                cnt = 0
            stable.append(stable_label)
            active.append(raw if risk_rank[raw] > risk_rank[stable_label] else stable_label)
        return stable, active

    _, tc_active = apply_generic(labels["trend_character"].tolist(), tc_risk_rank, 3)
    labels["trend_character_active"] = tc_active

    # Volatility hysteresis (de-escalation 2 days).
    vol_risk_rank = {
        "low_vol": 0,
        "normal_vol": 1,
        "high_vol": 2,
        "crisis_vol": 3,
        "unknown": 2,
    }
    _, vol_active = apply_generic(labels["volatility_state"].tolist(), vol_risk_rank, 2)
    labels["volatility_state_active"] = vol_active

    # Breadth hysteresis (de-escalation 2 days).
    breadth_risk_rank = {
        "healthy_breadth": 0,
        "neutral_breadth": 1,
        "weak_breadth": 2,
        "divergent_fragile": 3,
        "unknown": 2,
    }
    _, breadth_active = apply_generic(labels["breadth_state"].tolist(), breadth_risk_rank, 2)
    labels["breadth_state_active"] = breadth_active

    generated_at = generated_at_utc or _utc_iso_now()

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
            "trend_direction": row["trend_direction_active"],
            "trend_character": row["trend_character_active"],
            "volatility_state": row["volatility_state_active"],
            "breadth_state": row["breadth_state_active"],
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
        "raw_file_sha256": raw_hashes,
        "rows": derived_rows,
    }
    report_doc = {
        "generated_at_utc": generated_at,
        "raw_file_sha256": raw_hashes,
        "rows": report_rows,
    }
    return derived_doc, report_doc


if __name__ == "__main__":
    main()
