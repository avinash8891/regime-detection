#!/usr/bin/env python3
"""V2 §9.1 walk-forward performance gate runner.

Iterates ``engine.classify(...)`` per NYSE session over a ≥1y out-of-sample
window, once in v1-mode (no V2 kwargs) and once in v2-mode (full
sector/cross-asset/macro kwargs threaded), tallies per-session
wire-level metrics, and emits a markdown comparison artifact at
``docs/verification/v2_walkforward_perf_gate.md``.

Per the spec §9.1 gate (and ``docs/v2_slice_gate_checklist.md`` item 6),
the strategy-PnL metrics (drawdown / sharpe / false-switch) are computed
downstream when v2 outputs route into a backtester. This runner ships
the wire-level lit-vs-unlit precondition required before the strategy
gate can run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any, get_args

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from regime_detection.calendar import nyse_calendar  # noqa: E402
from regime_detection.comparison import V2_GATE_METRIC_NAMES  # noqa: E402
from regime_detection.config import load_default_regime_config  # noqa: E402
from regime_detection.engine import RegimeEngine  # noqa: E402
from regime_detection.fragility_universe import SECTOR_ETFS  # noqa: E402
from regime_detection.loaders import (  # noqa: E402
    load_central_bank_text_score,
    load_cpi_vintages_first_release,
    load_event_calendar,
)
from regime_detection.market_context import build_market_context  # noqa: E402
from regime_detection.models import ClassificationStatus  # noqa: E402
from regime_detection.versioning import engine_version as resolved_engine_version  # noqa: E402
from regime_data_fetch.universe import FIXED_UNIVERSE_TREE_NAME  # noqa: E402

from _v2_calibration_helpers import (  # noqa: E402
    CROSS_ASSET_SYMBOLS,
    add_manifest_args,
    apply_manifest_input_defaults,
    apply_manifest_input_paths,
    axis_reporting_label as _reporting_label,
    load_close_dict,
    load_macro_series,
    load_market_data,
    manifest_input_overrides,
    materialize_manifest_from_args,
    register_manifest_input_args,
)


logger = logging.getLogger("v2_walkforward_gate")

V1_AXIS_DEFAULT_AGENT = "default"
NON_CLASSIFIED_REPORTING_LABELS = set(get_args(ClassificationStatus)) - {"classified"}


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


def _resolve_default_window(daily_dir: Path) -> tuple[dt.date, dt.date]:
    """Return (start, end). end = max date in daily parquet; start = end-1y-90d."""
    df = pd.read_parquet(daily_dir, columns=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    end_date = df["date"].max()
    # 1y window + 90d trailing-lookback warm-up
    start_date = end_date - dt.timedelta(days=365 + 90)
    return start_date, end_date


def _session_metrics_empty() -> dict[str, int]:
    return {
        "sessions_classified": 0,
        "crisis_fired": 0,
        "bear_stress_fired": 0,
        "fragile_bull_fired": 0,
        "recovery_attempt_fired": 0,
        "watch_fired": 0,
        "weakening_fired": 0,
        "transition_warning_fired": 0,
        "high_transition_risk_fired": 0,
        "insufficient_data_fired": 0,
        "state_confirmation_pending": 0,
        "score_components_dict": 0,
        "agent_routing_field": 0,
        "change_point_score": 0,
        "model_instability_evidence_on_score": 0,
        "credit_funding_state": 0,
        "credit_funding_effective_state": 0,
        "inflation_growth_state": 0,
        "cluster_output": 0,
    }


def _tally_output(metrics: dict[str, int], output: Any) -> None:
    """Tally wire-level signals from a single ``RegimeOutput`` into ``metrics``."""
    metrics["sessions_classified"] += 1
    tr = output.transition_risk
    label = (tr.state or "").lower()
    if label == "crisis":
        metrics["crisis_fired"] += 1
    if label == "bear_stress":
        metrics["bear_stress_fired"] += 1
    if label == "fragile_bull":
        metrics["fragile_bull_fired"] += 1
    if label == "recovery_attempt":
        metrics["recovery_attempt_fired"] += 1
    if label == "watch":
        metrics["watch_fired"] += 1
    if label == "weakening":
        metrics["weakening_fired"] += 1
    if label == "transition_warning":
        metrics["transition_warning_fired"] += 1
    if label == "high_transition_risk":
        metrics["high_transition_risk_fired"] += 1
    if label == "insufficient_data":
        metrics["insufficient_data_fired"] += 1
    if "state_confirmation_pending" in (getattr(tr, "triggered_rules", None) or []):
        metrics["state_confirmation_pending"] += 1
    if tr.score_components:
        metrics["score_components_dict"] += 1
        if "model_instability" in tr.score_components:
            metrics["model_instability_evidence_on_score"] += 1
    if output.agent_routing is not None:
        metrics["agent_routing_field"] += 1
    if output.change_point is not None and output.change_point.score is not None:
        metrics["change_point_score"] += 1
    if output.credit_funding_state is not None:
        metrics["credit_funding_state"] += 1
    if output.credit_funding_effective_state is not None:
        metrics["credit_funding_effective_state"] += 1
    if output.inflation_growth_state is not None:
        metrics["inflation_growth_state"] += 1
    if output.cluster is not None:
        metrics["cluster_output"] += 1


def _axis_activation_empty() -> dict[str, int]:
    return {
        "network_fragility": 0,
        "credit_funding": 0,
        "inflation_growth": 0,
        "monetary_pressure_v2": 0,
        "volume_liquidity_state": 0,
        "agent_routing_non_default": 0,
        "change_point_ge_0_5": 0,
    }


def _is_classified_axis_output(output: Any) -> bool:
    label = (_reporting_label(output) or "").lower()
    return bool(label) and label not in NON_CLASSIFIED_REPORTING_LABELS


def _tally_axis_activation(axes: dict[str, int], output: Any) -> None:
    if _is_classified_axis_output(output.network_fragility):
        axes["network_fragility"] += 1
    if _is_classified_axis_output(output.credit_funding_effective_state):
        axes["credit_funding"] += 1
    if _is_classified_axis_output(output.inflation_growth_state):
        axes["inflation_growth"] += 1
    if _is_classified_axis_output(output.monetary_pressure_state):
        axes["monetary_pressure_v2"] += 1
    if _is_classified_axis_output(output.volume_liquidity_state):
        axes["volume_liquidity_state"] += 1
    if output.agent_routing is not None:
        if (output.agent_routing.active_cohort or "").lower() != V1_AXIS_DEFAULT_AGENT:
            axes["agent_routing_non_default"] += 1
    if output.change_point is not None and output.change_point.score is not None:
        if output.change_point.score >= 0.5:
            axes["change_point_ge_0_5"] += 1


def _classify_window(
    *,
    engine: RegimeEngine,
    sessions: list[dt.date],
    market_data: pd.DataFrame,
    event_calendar: pd.DataFrame,
    v2_kwargs: dict[str, Any] | None,
    mode_label: str,
) -> tuple[dict[str, int], dict[str, int], int]:
    metrics = _session_metrics_empty()
    axes = _axis_activation_empty()
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
        except (
            Exception
        ) as exc:  # fail-open per spec — single bad session doesn't kill the run
            errors += 1
            logger.warning(
                "[%s] %s classify failed: %s", mode_label, as_of_date.isoformat(), exc
            )
            continue
        _tally_output(metrics, output)
        _tally_axis_activation(axes, output)
        if idx % 50 == 0 or idx == total:
            logger.info("[%s] %d/%d sessions classified", mode_label, idx, total)
    return metrics, axes, errors


def _format_table(rows: list[tuple[str, Any, Any, Any]]) -> str:
    header = "| metric | v1 | v2 | delta |\n|---|---|---|---|"
    body = "\n".join(f"| {n} | {v1} | {v2} | {d} |" for n, v1, v2, d in rows)
    return header + "\n" + body


def _build_markdown(
    *,
    start_date: dt.date,
    end_date: dt.date,
    sessions: list[dt.date],
    v1_metrics: dict[str, int],
    v2_metrics: dict[str, int],
    v2_axes: dict[str, int],
    v1_errors: int,
    v2_errors: int,
    engine_version: str,
) -> str:
    n_sessions = len(sessions)

    def _row(label: str, key: str) -> tuple[str, int, int, int]:
        v1 = v1_metrics.get(key, 0)
        v2 = v2_metrics.get(key, 0)
        return (label, v1, v2, v2 - v1)

    metric_rows = [
        _row("sessions classified", "sessions_classified"),
        _row("sessions with crisis fired", "crisis_fired"),
        _row("sessions with bear_stress fired", "bear_stress_fired"),
        _row("sessions with fragile_bull fired", "fragile_bull_fired"),
        _row("sessions with recovery_attempt", "recovery_attempt_fired"),
        _row("sessions with watch", "watch_fired"),
        _row("sessions with weakening", "weakening_fired"),
        _row("sessions with transition_warning", "transition_warning_fired"),
        _row("sessions with high_transition_risk", "high_transition_risk_fired"),
        _row("sessions with insufficient_data", "insufficient_data_fired"),
        _row("sessions with state_confirmation_pending", "state_confirmation_pending"),
        _row("sessions with score_components dict", "score_components_dict"),
        _row("sessions with agent_routing field", "agent_routing_field"),
        _row("sessions with change_point.score", "change_point_score"),
        _row(
            "sessions with model_instability evidence on score",
            "model_instability_evidence_on_score",
        ),
        _row("sessions with credit_funding_state", "credit_funding_state"),
        _row(
            "sessions with credit_funding_effective_state",
            "credit_funding_effective_state",
        ),
        _row("sessions with inflation_growth_state", "inflation_growth_state"),
        _row("sessions with cluster output", "cluster_output"),
    ]

    classified = max(1, v2_metrics["sessions_classified"])
    axis_rows = [
        ("network_fragility (classified)", v2_axes["network_fragility"]),
        ("credit_funding (classified)", v2_axes["credit_funding"]),
        ("inflation_growth (classified)", v2_axes["inflation_growth"]),
        ("monetary_pressure_v2 (classified)", v2_axes["monetary_pressure_v2"]),
        ("volume_liquidity_state (classified)", v2_axes["volume_liquidity_state"]),
        ("agent_routing != default", v2_axes["agent_routing_non_default"]),
        ("change_point >= 0.5", v2_axes["change_point_ge_0_5"]),
    ]

    axis_table_lines = [
        "| axis | sessions lit | activation rate |",
        "|---|---|---|",
    ]
    for label, lit in axis_rows:
        rate = (lit / classified) * 100.0
        axis_table_lines.append(f"| {label} | {lit} | {rate:.1f}% |")

    md = [
        "# V2 Walk-forward Performance Gate (§9.1)",
        "",
        f"- Window: {start_date.isoformat()} → {end_date.isoformat()} ({n_sessions} NYSE sessions)",
        f"- Engine versions: v1={engine_version}, v2={engine_version}",
        f"- v1-mode errors (sessions): {v1_errors}",
        f"- v2-mode errors (sessions): {v2_errors}",
        f"- Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "## Wire-level metrics",
        "",
        _format_table(metric_rows),
        "",
        "## §9.1 Gate Conditions (v2 §9.1 + docs/v2_slice_gate_checklist.md item 6)",
        "",
        "Per the spec, AT LEAST ONE of:",
        "",
        *(f"- {name}" for name in V2_GATE_METRIC_NAMES),
        "",
        "must show v2 improvement. Note: this script ships the per-session wire",
        "comparison; the strategy-PnL metrics (drawdown/sharpe/false-switch) are",
        "operator concerns when v2 outputs route into a backtester (e.g.",
        "vectorbt). The gate currently asserts the wire-level lit-vs-unlit",
        "deltas as a precondition to the strategy gate.",
        "",
        "## Per-axis activation rate (v2 mode only)",
        "",
        "\n".join(axis_table_lines),
        "",
    ]
    return "\n".join(md) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V2 §9.1 walk-forward performance gate runner."
    )
    parser.add_argument("--daily-dir", type=Path, default=None)
    parser.add_argument("--macro-parquet", type=Path, default=None)
    # Optional manifest-routed inputs come from MANIFEST_INPUT_SPECS.
    register_manifest_input_args(parser, include_required_paths=False)
    parser.add_argument("--start-date", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=dt.date.fromisoformat, default=None)
    parser.add_argument(
        "--output",
        "--out",
        type=Path,
        default=REPO_ROOT / "docs" / "verification" / "v2_walkforward_perf_gate.md",
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
    if args.start_date is None or args.end_date is None:
        default_start, default_end = _resolve_default_window(args.daily_dir)
        if args.start_date is None:
            args.start_date = default_start
        if args.end_date is None:
            args.end_date = default_end
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

    # Build a v1 bootstrap context to derive the SPY session index.
    config = load_default_regime_config()
    bootstrap_context = build_market_context(
        end_date=market_data["date"].max(),
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
    logger.info(
        "central_bank_text_releases: %d rows; cpi_first_release: %s",
        len(central_bank_text_releases),
        "wired" if cpi_first_release is not None else "absent (revised CPI fallback)",
    )

    v2_kwargs = {
        "sector_etf_closes": sector_etf_closes,
        "cross_asset_closes": cross_asset_closes,
        "macro_series": macro_series,
        "central_bank_text_releases": (
            central_bank_text_releases if not central_bank_text_releases.empty else None
        ),
        "cpi_first_release": cpi_first_release,
    }

    sessions = list(
        nyse_calendar()
        .schedule(start_date=args.start_date, end_date=args.end_date)
        .index.date
    )
    if not sessions:
        raise SystemExit(
            f"No NYSE sessions in window {args.start_date.isoformat()} → {args.end_date.isoformat()}"
        )
    logger.info(
        "Window: %s → %s (%d NYSE sessions)",
        args.start_date.isoformat(),
        args.end_date.isoformat(),
        len(sessions),
    )

    engine = RegimeEngine()
    engine_version = resolved_engine_version()

    logger.info("Running v1-mode walk-forward (no V2 kwargs)...")
    v1_metrics, _v1_axes, v1_errors = _classify_window(
        engine=engine,
        sessions=sessions,
        market_data=market_data,
        event_calendar=event_calendar,
        v2_kwargs=None,
        mode_label="v1",
    )

    logger.info("Running v2-mode walk-forward (full V2 inputs)...")
    v2_metrics, v2_axes, v2_errors = _classify_window(
        engine=engine,
        sessions=sessions,
        market_data=market_data,
        event_calendar=event_calendar,
        v2_kwargs=v2_kwargs,
        mode_label="v2",
    )

    markdown = _build_markdown(
        start_date=args.start_date,
        end_date=args.end_date,
        sessions=sessions,
        v1_metrics=v1_metrics,
        v2_metrics=v2_metrics,
        v2_axes=v2_axes,
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
