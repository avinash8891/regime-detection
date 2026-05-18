#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fixture_axis_evaluators import INTENTS  # noqa: E402
from fixture_axis_evaluators import evaluate_all  # noqa: E402


REPO_ROOT = SCRIPT_DIR.parents[0]
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
    labels = evaluate_all(feat, hysteresis_days=_load_hysteresis_days())

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
