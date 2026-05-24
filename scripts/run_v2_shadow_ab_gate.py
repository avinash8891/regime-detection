#!/usr/bin/env python3
"""V2 §9.3 60-session shadow A/B gate runner.

Simulates the §9.3 live shadow A/B over the most recent 60 NYSE sessions:
classifies each session twice (v1-mode vs full-v2-mode) and emits a
markdown disagreement report at
``docs/verification/v2_shadow_ab_60session.md``.

Per ``docs/v2_slice_gate_checklist.md`` item 7: zero unexpected wire
diffs in v1 fields; v2 enrichments match expectations. v1-field
disagreements (trend_direction / trend_character / volatility_state /
breadth_state / transition_risk.state) should be ZERO. v2-NEW field
activations (transition_risk.score, transition_risk score-component presence,
transition_risk primary drivers, transition_risk triggered rules,
transition_risk data quality, agent_routing, change_point,
credit_funding_state, inflation_growth_state, cluster) are EXPECTED to differ —
those are wins, not regressions. The markdown surfaces both tables separately so
reviewers can see the distinction at a glance.
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
from regime_detection.config import load_default_regime_config, load_regime_config  # noqa: E402
from regime_detection.engine import RegimeEngine  # noqa: E402
from regime_detection.fragility_universe import SECTOR_ETFS  # noqa: E402
from regime_detection.loaders import (  # noqa: E402
    load_central_bank_text_score,
    load_cpi_vintages_first_release,
    load_event_calendar,
)
from regime_detection.market_context import build_market_context  # noqa: E402
from regime_detection.versioning import engine_version as resolved_engine_version  # noqa: E402
from regime_data_fetch.universe import FIXED_UNIVERSE_TREE_NAME  # noqa: E402

from _v2_calibration_helpers import (  # noqa: E402
    CROSS_ASSET_SYMBOLS,
    add_manifest_args,
    apply_manifest_input_defaults,
    apply_manifest_input_paths,
    axis_reporting_label as _reporting_label,
    constituent_ohlcv_from_sector_closes,
    load_close_dict,
    load_macro_series,
    load_market_data,
    manifest_input_overrides,
    materialize_manifest_from_args,
    register_manifest_input_args,
    synthetic_pit_intervals_from_sector_closes,
)


logger = logging.getLogger("v2_shadow_ab_gate")

# v1 fields whose active_label MUST stay identical when V2 enrichments
# are added. Anything different here is a regression.
V1_FIELDS: list[str] = [
    "trend_direction",
    "trend_character",
    "volatility_state",
    "breadth_state",
    "transition_risk_state",
]

# v2 fields whose activation is EXPECTED to differ under v2-mode.
V2_FIELDS: list[str] = [
    "transition_risk_score",
    "transition_risk_score_components_present",
    "transition_risk_primary_drivers",
    "transition_risk_triggered_rules",
    "transition_risk_data_quality",
    "agent_routing",
    "change_point",
    "credit_funding_state",
    "credit_funding_effective_state",
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


def _session_error_exit_code(
    *,
    v1_errors: int,
    v2_errors: int,
    allow_session_errors: bool,
) -> int:
    if allow_session_errors:
        return 0
    if v1_errors or v2_errors:
        return 1
    return 0


def _extract_v1_fields(output: Any) -> dict[str, Any]:
    return {
        "trend_direction": output.trend_direction.active_label,
        "trend_character": output.trend_character.active_label,
        "volatility_state": output.volatility_state.active_label,
        "breadth_state": output.breadth_state.active_label,
        "transition_risk_state": output.transition_risk.state,
    }


def _extract_v2_fields(output: Any) -> dict[str, Any]:
    transition_risk = output.transition_risk
    data_quality = getattr(transition_risk, "data_quality", None)
    if isinstance(data_quality, dict):
        transition_data_quality = data_quality.get("status")
    else:
        transition_data_quality = getattr(data_quality, "status", None)
    return {
        "transition_risk_score": transition_risk.score,
        "transition_risk_score_components_present": bool(
            getattr(transition_risk, "score_components", None)
        ),
        "transition_risk_primary_drivers": list(
            getattr(transition_risk, "primary_drivers", []) or []
        ),
        "transition_risk_triggered_rules": list(
            getattr(transition_risk, "triggered_rules", []) or []
        ),
        "transition_risk_data_quality": transition_data_quality,
        "agent_routing": (
            output.agent_routing.active_cohort
            if output.agent_routing is not None
            else None
        ),
        "change_point": (
            output.change_point.score if output.change_point is not None else None
        ),
        "credit_funding_state": (
            _reporting_label(output.credit_funding_state)
            if output.credit_funding_state is not None
            else None
        ),
        "credit_funding_effective_state": (
            _reporting_label(output.credit_funding_effective_state)
            if output.credit_funding_effective_state is not None
            else None
        ),
        "inflation_growth_state": (
            _reporting_label(output.inflation_growth_state)
            if output.inflation_growth_state is not None
            else None
        ),
        "cluster": (output.cluster.cluster_id if output.cluster is not None else None),
        "monetary_pressure_state": (
            _reporting_label(output.monetary_pressure_state)
            if output.monetary_pressure_state is not None
            else None
        ),
        "volume_liquidity_state": (
            _reporting_label(output.volume_liquidity_state)
            if output.volume_liquidity_state is not None
            else None
        ),
        "network_fragility": (
            _reporting_label(output.network_fragility)
            if output.network_fragility is not None
            else None
        ),
    }


def _classify_per_session(
    *,
    engine: RegimeEngine,
    sessions: list[dt.date],
    market_data: pd.DataFrame,
    event_calendar: pd.DataFrame,
    v2_kwargs: dict[str, Any] | None,
    mode_label: str,
) -> tuple[dict[dt.date, dict[str, Any]], dict[dt.date, dict[str, Any]], int]:
    v1_field_records: dict[dt.date, dict[str, Any]] = {}
    v2_field_records: dict[dt.date, dict[str, Any]] = {}
    errors = 0
    total = len(sessions)
    for idx, as_of_date in enumerate(sessions, start=1):
        as_of_timestamp = pd.Timestamp(as_of_date)
        market_dates = pd.to_datetime(market_data["date"])
        market_slice = (
            market_data[market_dates <= as_of_timestamp]
            .copy()
            .reset_index(drop=True)
        )
        kwargs: dict[str, Any] = {
            "as_of_date": as_of_date,
            "market_data": market_slice,
            "event_calendar": event_calendar,
        }
        if v2_kwargs:
            kwargs.update(v2_kwargs)
        try:
            output = engine.classify(**kwargs)
        except (ValueError, RuntimeError) as exc:
            errors += 1
            logger.warning(
                "[%s] %s classify failed: %s", mode_label, as_of_date.isoformat(), exc
            )
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
    parser = argparse.ArgumentParser(
        description="V2 §9.3 60-session shadow A/B gate runner."
    )
    parser.add_argument("--daily-dir", type=Path, default=None)
    parser.add_argument("--macro-parquet", type=Path, default=None)
    parser.add_argument("--event-calendar", dest="event_calendar", type=Path, default=None)
    parser.add_argument("--config-path", type=Path, default=None)
    # Optional manifest-routed inputs come from MANIFEST_INPUT_SPECS.
    register_manifest_input_args(parser, include_required_paths=False)
    parser.add_argument("--n-sessions", type=int, default=60)
    parser.add_argument(
        "--output",
        "--out",
        type=Path,
        default=REPO_ROOT / "docs" / "verification" / "v2_shadow_ab_60session.md",
    )
    add_manifest_args(parser, data_root_default=REPO_ROOT / "data" / "raw", action="running")
    parser.add_argument(
        "--allow-session-errors",
        action="store_true",
        help="Allow exploratory report output even when per-session classify errors occur.",
    )
    args = parser.parse_args()
    args.manifest_input_overrides = manifest_input_overrides(sys.argv[1:])
    if args.daily_dir is None:
        args.daily_dir = args.data_root / FIXED_UNIVERSE_TREE_NAME
    apply_manifest_input_defaults(args, args.data_root)
    materialize_manifest_from_args(
        args,
        repo_root=REPO_ROOT,
        required_for="v2_calibration",
    )
    apply_manifest_input_paths(
        args,
        runner_name="v2_calibration",
        repo_root=REPO_ROOT,
        required_fields=frozenset({"daily_dir", "macro_parquet", "event_calendar"}),
    )
    return args


def main() -> int:
    _setup_logging()
    args = _parse_args()

    daily_dir = args.daily_dir
    macro_parquet = args.macro_parquet
    pmi_path = args.pmi_path

    if not daily_dir.exists():
        raise SystemExit(f"daily_ohlcv directory not found at {daily_dir}")
    if not macro_parquet.exists():
        raise SystemExit(f"macro parquet not found at {macro_parquet}")
    if not args.event_calendar.exists():
        raise SystemExit(f"event_calendar file not found at {args.event_calendar}")

    logger.info("Loading market data...")
    market_data = load_market_data(daily_dir)
    event_calendar = load_event_calendar(args.event_calendar)
    end_date = market_data["date"].max()

    # Walk back through the NYSE calendar to get the most-recent N sessions.
    look_back_start = end_date - dt.timedelta(days=max(args.n_sessions * 2, 180))
    all_sessions = list(
        nyse_calendar()
        .schedule(start_date=look_back_start, end_date=end_date)
        .index.date
    )
    sessions = all_sessions[-args.n_sessions :]
    if len(sessions) < args.n_sessions:
        raise SystemExit(
            f"Only {len(sessions)} sessions available between {look_back_start} and {end_date}; "
            f"need {args.n_sessions}."
        )
    start_date = sessions[0]

    logger.info(
        "Window: %s → %s (%d sessions)",
        start_date.isoformat(),
        end_date.isoformat(),
        len(sessions),
    )

    # Build bootstrap context to derive SPY session index.
    config = load_regime_config(args.config_path) if args.config_path else load_default_regime_config()
    bootstrap_context = build_market_context(
        end_date=end_date,
        market_data=market_data,
        config=config,
    )
    spy_index = bootstrap_context.spy_ohlcv.index

    logger.info("Loading sector/cross-asset closes + macro series...")
    sector_etf_closes = load_close_dict(daily_dir, list(SECTOR_ETFS), spy_index)
    cross_asset_closes = load_close_dict(daily_dir, CROSS_ASSET_SYMBOLS, spy_index)
    macro_series = load_macro_series(
        macro_parquet,
        pmi_path,
        cpi_nowcast_parquet=args.cpi_nowcast_parquet,
        eps_weekly_history_parquet=args.aggregate_forward_eps_weekly_history_parquet,
    )

    # v2 §2A central-bank-text + first-release CPI seams (audit M1 / M2).
    fomc_parquet = args.fomc_minutes_parquet
    powell_parquet = args.powell_speeches_parquet
    cpi_vintages_parquet = args.cpi_vintages_parquet
    central_bank_text_releases = load_central_bank_text_score(
        fomc_minutes_source=(
            fomc_parquet if fomc_parquet is not None and fomc_parquet.exists() else None
        ),
        powell_speeches_source=(
            powell_parquet
            if powell_parquet is not None and powell_parquet.exists()
            else None
        ),
    )
    cpi_first_release = (
        load_cpi_vintages_first_release(cpi_vintages_parquet)
        if cpi_vintages_parquet is not None and cpi_vintages_parquet.exists()
        else None
    )

    v2_kwargs = {
        "sector_etf_closes": sector_etf_closes,
        "cross_asset_closes": cross_asset_closes,
        "macro_series": macro_series,
        "pit_constituent_intervals": synthetic_pit_intervals_from_sector_closes(sector_etf_closes),
        "constituent_ohlcv": constituent_ohlcv_from_sector_closes(sector_etf_closes),
        "central_bank_text_releases": (
            central_bank_text_releases if not central_bank_text_releases.empty else None
        ),
        "cpi_first_release": cpi_first_release,
    }

    engine = RegimeEngine(config_path=args.config_path)
    engine_version = resolved_engine_version()

    logger.info("Running v1-mode (no V2 kwargs)...")
    v1_mode_v1, v1_mode_v2, v1_errors = _classify_per_session(
        engine=engine,
        sessions=sessions,
        market_data=market_data,
        event_calendar=event_calendar,
        v2_kwargs=None,
        mode_label="v1",
    )

    logger.info("Running v2-mode (full V2 inputs)...")
    v2_mode_v1, v2_mode_v2, v2_errors = _classify_per_session(
        engine=engine,
        sessions=sessions,
        market_data=market_data,
        event_calendar=event_calendar,
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
    exit_code = _session_error_exit_code(
        v1_errors=v1_errors,
        v2_errors=v2_errors,
        allow_session_errors=args.allow_session_errors,
    )
    if exit_code:
        logger.error(
            "Session classify errors occurred: v1_errors=%d v2_errors=%d. "
            "Re-run with --allow-session-errors only for exploratory output.",
            v1_errors,
            v2_errors,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
