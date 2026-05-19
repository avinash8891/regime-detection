#!/usr/bin/env python3
"""Verify golden-date fixtures against the actual engine.

Runs RegimeEngine.classify_window on the raw fixture CSVs and compares
the engine's output labels against the INTENTS. No shadow reimplementation
of classification logic — the engine IS the source of truth.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fixture_axis_evaluators import INTENTS  # noqa: E402
from regime_detection.engine import RegimeEngine  # noqa: E402

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


def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if out:
            return out
    except Exception:
        pass
    return "unknown"


def _load_market_data() -> pd.DataFrame:
    spy = pd.read_csv(RAW_DIR / "SPY.csv", parse_dates=["date"])
    rsp = pd.read_csv(RAW_DIR / "RSP.csv", parse_dates=["date"])
    vixy = pd.read_csv(RAW_DIR / "VIXY.csv", parse_dates=["date"])
    spy["symbol"] = "SPY"
    rsp["symbol"] = "RSP"
    vixy["symbol"] = "VIXY"
    return pd.concat([spy, rsp, vixy], ignore_index=True)


def _serialize_scalar(x: Any) -> Any:
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        if np.isnan(x):
            return None
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


def _classify_all_intents(
    market_data: pd.DataFrame,
) -> dict[date, Any]:
    engine = RegimeEngine()
    intent_dates = sorted(
        date.fromisoformat(item["intent_date"]) for item in INTENTS
    )
    end = max(intent_dates)
    earliest = min(intent_dates)
    span_days = (end - earliest).days
    lookback_sessions = max(1, int(span_days / 365.25 * 252) + 30)
    timeline = engine.classify_window(
        end_date=end,
        market_data=market_data,
        lookback_days=lookback_sessions,
    )
    return {out.as_of_date: out for out in timeline.outputs}


def _pick_fixture_date(
    by_date: dict[date, Any],
    intent_date: str,
    intent: dict[str, str],
    search_window_trading_days: int,
) -> date:
    base = date.fromisoformat(intent_date)
    available = sorted(by_date.keys())
    base_idx = next((i for i, d in enumerate(available) if d >= base), len(available))
    lo = max(0, base_idx - search_window_trading_days)
    hi = min(len(available), base_idx + search_window_trading_days + 1)
    window = available[lo:hi]

    for d in window:
        out = by_date[d]
        match = True
        for axis, expected_label in intent.items():
            actual = _get_axis_label(out, axis)
            if actual != expected_label:
                match = False
                break
        if match:
            return d

    actual_labels = {
        axis: _get_axis_label(by_date[base], axis) for axis in intent
    } if base in by_date else None
    raise SystemExit(
        f"No fixture candidate for intent={intent} near {intent_date}. "
        f"Actual at base={actual_labels}"
    )


def _get_axis_label(out: Any, axis: str) -> str:
    if axis == "transition_risk":
        return out.transition_risk.label
    attr = getattr(out, axis, None)
    if attr is not None and hasattr(attr, "active_label"):
        return attr.active_label
    return "unknown"


def main() -> None:
    derived_doc, report_doc = generate_docs()
    DERIVED_PATH.write_text(yaml.safe_dump(derived_doc, sort_keys=False))
    REPORT_PATH.write_text(yaml.safe_dump(report_doc, sort_keys=False))
    print(json.dumps({"derived": str(DERIVED_PATH), "report": str(REPORT_PATH)}, indent=2))


def generate_docs(
    *, generated_at_utc: str | None = None, generated_by_commit: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    market_data = _load_market_data()
    by_date = _classify_all_intents(market_data)

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
            by_date=by_date,
            intent_date=intent_date,
            intent=intent,
            search_window_trading_days=int(item["search_window_trading_days"]),
        )
        out = by_date[pick]
        as_of = str(pick)

        expected = {
            "trend_direction": _get_axis_label(out, "trend_direction"),
            "trend_character": _get_axis_label(out, "trend_character"),
            "volatility_state": _get_axis_label(out, "volatility_state"),
            "breadth_state": _get_axis_label(out, "breadth_state"),
            "transition_risk": _get_axis_label(out, "transition_risk"),
        }

        derived_rows.append(
            {
                "intent_id": item["intent_id"],
                "intent_date": intent_date,
                "as_of_date": as_of,
                "expected": expected,
            }
        )

        base = date.fromisoformat(intent_date)
        delta_calendar_days = abs((pick - base).days)

        evidence = {}
        for axis in ["trend_direction", "trend_character", "volatility_state", "breadth_state"]:
            attr = getattr(out, axis, None)
            if attr is not None and hasattr(attr, "evidence"):
                evidence[axis] = _serialize_obj(dict(attr.evidence))
        evidence["transition_risk"] = _serialize_obj(
            dict(out.transition_risk.evidence)
        )

        report_rows.append(
            {
                "intent_id": item["intent_id"],
                "intent_date": intent_date,
                "as_of_date": as_of,
                "delta_calendar_days": delta_calendar_days,
                "notes": item.get("notes", ""),
                "expected": expected,
                "predicate_evaluations": evidence,
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


if __name__ == "__main__":
    main()
