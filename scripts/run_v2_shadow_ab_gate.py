#!/usr/bin/env python3
"""V2 §9.3 60-session shadow A/B gate runner.

Simulates the §9.3 live shadow A/B over the most recent 60 NYSE sessions:
classifies each session twice (v1-mode vs full-v2-mode) and emits a
markdown disagreement report at
``docs/verification/v2_shadow_ab_60session.md``.

Per ``docs/v2_slice_gate_checklist.md`` item 7: zero unexpected wire
diffs in v1 fields; v2 enrichments match expectations. v1-field
disagreements (trend_direction / trend_character / volatility_state /
breadth_state / transition_risk.label) should be ZERO. v2-NEW field
activations (transition_risk.score, agent_routing, change_point,
credit_funding_state, inflation_growth_state, cluster) are EXPECTED to
differ — those are wins, not regressions. The markdown surfaces both
tables separately so reviewers can see the distinction at a glance.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from regime_detection.calendar import nyse_calendar  # noqa: E402
from regime_detection.config import load_default_regime_config  # noqa: E402
from regime_detection.engine import RegimeEngine  # noqa: E402
from regime_detection.fragility_universe import SECTOR_ETFS  # noqa: E402
from regime_detection.market_context import build_market_context  # noqa: E402
from regime_detection.versioning import engine_version as resolved_engine_version  # noqa: E402

from _v2_calibration_helpers import (  # noqa: E402
    CROSS_ASSET_SYMBOLS,
    load_close_dict,
    load_macro_series,
    load_market_data,
)


logger = logging.getLogger("v2_shadow_ab_gate")

# v1 fields whose active_label MUST stay identical when V2 enrichments
# are added. Anything different here is a regression.
V1_FIELDS: list[str] = [
    "trend_direction",
    "trend_character",
    "volatility_state",
    "breadth_state",
    "transition_risk_label",
]

# v2 fields whose activation is EXPECTED to differ under v2-mode.
V2_FIELDS: list[str] = [
    "transition_risk_score",
    "agent_routing",
    "change_point",
    "credit_funding_state",
    "inflation_growth_state",
    "cluster",
    "monetary_pressure_state",
    "volume_liquidity_state",
    "network_fragility",
]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def _extract_v1_fields(output: Any) -> dict[str, Any]:
    return {
        "trend_direction": output.trend_direction.active_label,
        "trend_character": output.trend_character.active_label,
        "volatility_state": output.volatility_state.active_label,
        "breadth_state": output.breadth_state.active_label,
        "transition_risk_label": output.transition_risk.label,
    }


def _extract_v2_fields(output: Any) -> dict[str, Any]:
    return {
        "transition_risk_score": output.transition_risk.score,
        "agent_routing": (
            output.agent_routing.active_cohort if output.agent_routing is not None else None
        ),
        "change_point": (
            output.change_point.score if output.change_point is not None else None
        ),
        "credit_funding_state": (
            output.credit_funding_state.active_label
            if output.credit_funding_state is not None
            else None
        ),
        "inflation_growth_state": (
            output.inflation_growth_state.active_label
            if output.inflation_growth_state is not None
            else None
        ),
        "cluster": (output.cluster.cluster_id if output.cluster is not None else None),
        "monetary_pressure_state": (
            output.monetary_pressure_state.active_label
            if output.monetary_pressure_state is not None
            else None
        ),
        "volume_liquidity_state": (
            output.volume_liquidity_state.active_label
            if output.volume_liquidity_state is not None
            else None
        ),
        "network_fragility": (
            output.network_fragility.active_label
            if output.network_fragility is not None
            else None
        ),
    }


def _classify_per_session(
    *,
    engine: RegimeEngine,
    sessions: list[dt.date],
    market_data: pd.DataFrame,
    v2_kwargs: dict[str, Any] | None,
    mode_label: str,
) -> tuple[dict[dt.date, dict[str, Any]], dict[dt.date, dict[str, Any]], int]:
    v1_field_records: dict[dt.date, dict[str, Any]] = {}
    v2_field_records: dict[dt.date, dict[str, Any]] = {}
    errors = 0
    total = len(sessions)
    for idx, as_of_date in enumerate(sessions, start=1):
        market_slice = market_data[market_data["date"] <= as_of_date].copy().reset_index(drop=True)
        kwargs: dict[str, Any] = {
            "as_of_date": as_of_date,
            "market_data": market_slice,
        }
        if v2_kwargs:
            kwargs.update(v2_kwargs)
        try:
            output = engine.classify(**kwargs)
        except Exception as exc:
            errors += 1
            logger.warning("[%s] %s classify failed: %s", mode_label, as_of_date.isoformat(), exc)
            continue
        v1_field_records[as_of_date] = _extract_v1_fields(output)
        v2_field_records[as_of_date] = _extract_v2_fields(output)
        if idx % 10 == 0 or idx == total:
            logger.info("[%s] %d/%d sessions classified", mode_label, idx, total)
    return v1_field_records, v2_field_records, errors


def _compute_v1_disagreements(
    v1_mode_records: dict[dt.date, dict[str, Any]],
    v2_mode_records: dict[dt.date, dict[str, Any]],
) -> tuple[dict[str, int], dict[str, list[tuple[dt.date, Any, Any]]]]:
    counts = {f: 0 for f in V1_FIELDS}
    examples: dict[str, list[tuple[dt.date, Any, Any]]] = {f: [] for f in V1_FIELDS}
    common_dates = sorted(set(v1_mode_records.keys()) & set(v2_mode_records.keys()))
    for d in common_dates:
        a = v1_mode_records[d]
        b = v2_mode_records[d]
        for field in V1_FIELDS:
            if a.get(field) != b.get(field):
                counts[field] += 1
                examples[field].append((d, a.get(field), b.get(field)))
    return counts, examples


def _compute_v2_activations(
    v1_mode_records: dict[dt.date, dict[str, Any]],
    v2_mode_records: dict[dt.date, dict[str, Any]],
) -> tuple[dict[str, int], dict[str, list[tuple[dt.date, Any, Any]]]]:
    """For each v2-NEW field: count sessions where v1-mode==None/null and
    v2-mode is populated (the expected activation). Also count sessions
    where both are populated but values differ.
    """
    activations = {f: 0 for f in V2_FIELDS}
    examples: dict[str, list[tuple[dt.date, Any, Any]]] = {f: [] for f in V2_FIELDS}
    common_dates = sorted(set(v1_mode_records.keys()) & set(v2_mode_records.keys()))
    for d in common_dates:
        a = v1_mode_records[d]
        b = v2_mode_records[d]
        for field in V2_FIELDS:
            av = a.get(field)
            bv = b.get(field)
            if av != bv:
                activations[field] += 1
                examples[field].append((d, av, bv))
    return activations, examples


def _format_examples_table(rows: list[tuple[dt.date, Any, Any]], n: int = 5) -> str:
    if not rows:
        return "_(none)_"
    recent = rows[-n:]
    lines = ["| session | v1-mode | v2-mode |", "|---|---|---|"]
    for d, av, bv in recent:
        lines.append(f"| {d.isoformat()} | `{av!r}` | `{bv!r}` |")
    return "\n".join(lines)


def _build_markdown(
    *,
    start_date: dt.date,
    end_date: dt.date,
    sessions: list[dt.date],
    v1_disagreements: dict[str, int],
    v1_examples: dict[str, list[tuple[dt.date, Any, Any]]],
    v2_activations: dict[str, int],
    v2_examples: dict[str, list[tuple[dt.date, Any, Any]]],
    v1_errors: int,
    v2_errors: int,
    engine_version: str,
) -> str:
    v1_table = ["| v1 field | disagreement count |", "|---|---|"]
    for f in V1_FIELDS:
        v1_table.append(f"| {f} | {v1_disagreements[f]} |")

    v2_table = ["| v2 field | activation/diff count |", "|---|---|"]
    for f in V2_FIELDS:
        v2_table.append(f"| {f} | {v2_activations[f]} |")

    md = [
        "# V2 60-Session Shadow A/B (§9.3)",
        "",
        f"- Window: {start_date.isoformat()} → {end_date.isoformat()} ({len(sessions)} NYSE sessions)",
        f"- Engine version: {engine_version}",
        f"- v1-mode errors (sessions): {v1_errors}",
        f"- v2-mode errors (sessions): {v2_errors}",
        f"- Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "## Gate intent (docs/v2_slice_gate_checklist.md item 7)",
        "",
        "Zero unexpected wire diffs in v1 fields; v2 enrichments match",
        "expectations. The two tables below separate the regression-class",
        "(v1 fields — MUST be zero) from the activation-class (v2 fields —",
        "EXPECTED to be non-zero when V2 inputs are wired in).",
        "",
        "## v1-field disagreements (v1-mode vs v2-mode)",
        "",
        "These fields belong to the V1 wire contract and MUST remain",
        "identical when V2 kwargs are added. Any non-zero count here is a",
        "regression.",
        "",
        "\n".join(v1_table),
        "",
    ]
    for field in V1_FIELDS:
        md.append(f"### {field} — most recent disagreement examples")
        md.append("")
        md.append(_format_examples_table(v1_examples[field]))
        md.append("")

    md.extend(
        [
            "## v2-field activations (expected non-zero deltas)",
            "",
            "These fields are NEW in v2 — under v1-mode they are typically",
            "``None``/omitted and under v2-mode they populate when the",
            "corresponding seam is lit. Non-zero counts here are the v2",
            "wins, not regressions.",
            "",
            "\n".join(v2_table),
            "",
        ]
    )
    for field in V2_FIELDS:
        md.append(f"### {field} — most recent activation examples")
        md.append("")
        md.append(_format_examples_table(v2_examples[field]))
        md.append("")
    return "\n".join(md) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 §9.3 60-session shadow A/B gate runner.")
    parser.add_argument("--n-sessions", type=int, default=60)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "verification" / "v2_shadow_ab_60session.md",
    )
    return parser.parse_args()


def main() -> int:
    _setup_logging()
    args = _parse_args()

    data_root = REPO_ROOT / "data" / "raw"
    daily_dir = data_root / "daily_ohlcv"
    macro_parquet = data_root / "macro" / "fred_macro_series.parquet"
    pmi_path = REPO_ROOT / "data" / "manual_inputs" / "pmi" / "ism_manufacturing_pmi.tsv"

    if not daily_dir.exists():
        raise SystemExit(f"daily_ohlcv directory not found at {daily_dir}")
    if not macro_parquet.exists():
        raise SystemExit(f"macro parquet not found at {macro_parquet}")

    logger.info("Loading market data...")
    market_data = load_market_data(daily_dir)
    end_date = market_data["date"].max()

    # Walk back through the NYSE calendar to get the most-recent N sessions.
    look_back_start = end_date - dt.timedelta(days=max(args.n_sessions * 2, 180))
    all_sessions = list(
        nyse_calendar()
        .schedule(start_date=look_back_start, end_date=end_date)
        .index.date
    )
    sessions = all_sessions[-args.n_sessions:]
    if len(sessions) < args.n_sessions:
        raise SystemExit(
            f"Only {len(sessions)} sessions available between {look_back_start} and {end_date}; "
            f"need {args.n_sessions}."
        )
    start_date = sessions[0]

    logger.info("Window: %s → %s (%d sessions)", start_date.isoformat(), end_date.isoformat(), len(sessions))

    # Build bootstrap context to derive SPY session index.
    config = load_default_regime_config()
    bootstrap_context = build_market_context(
        end_date=end_date,
        market_data=market_data,
        config=config,
    )
    spy_index = bootstrap_context.spy_ohlcv.index

    logger.info("Loading sector/cross-asset closes + macro series...")
    sector_etf_closes = load_close_dict(daily_dir, list(SECTOR_ETFS), spy_index)
    cross_asset_closes = load_close_dict(daily_dir, CROSS_ASSET_SYMBOLS, spy_index)
    macro_series = load_macro_series(macro_parquet, pmi_path)

    v2_kwargs = {
        "sector_etf_closes": sector_etf_closes,
        "cross_asset_closes": cross_asset_closes,
        "macro_series": macro_series,
    }

    engine = RegimeEngine()
    engine_version = resolved_engine_version()

    logger.info("Running v1-mode (no V2 kwargs)...")
    v1_mode_v1, v1_mode_v2, v1_errors = _classify_per_session(
        engine=engine,
        sessions=sessions,
        market_data=market_data,
        v2_kwargs=None,
        mode_label="v1",
    )

    logger.info("Running v2-mode (full V2 inputs)...")
    v2_mode_v1, v2_mode_v2, v2_errors = _classify_per_session(
        engine=engine,
        sessions=sessions,
        market_data=market_data,
        v2_kwargs=v2_kwargs,
        mode_label="v2",
    )

    v1_disagreements, v1_examples = _compute_v1_disagreements(v1_mode_v1, v2_mode_v1)
    v2_activations, v2_examples = _compute_v2_activations(v1_mode_v2, v2_mode_v2)

    markdown = _build_markdown(
        start_date=start_date,
        end_date=end_date,
        sessions=sessions,
        v1_disagreements=v1_disagreements,
        v1_examples=v1_examples,
        v2_activations=v2_activations,
        v2_examples=v2_examples,
        v1_errors=v1_errors,
        v2_errors=v2_errors,
        engine_version=engine_version,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
