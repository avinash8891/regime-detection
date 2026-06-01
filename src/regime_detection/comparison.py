"""V1-vs-V2 comparison + V2 §9.1 performance gate evaluator.

Pure functions over two `RegimeTimeline` objects (or pre-computed
strategy metrics). Foundation scaffolding for the v2 §9.1 gate:

> Every V2 component must demonstrate, in walk-forward backtest, at
> least one of:
>   - lower max drawdown than V1
>   - higher Sharpe than V1
>   - earlier crisis detection (lower lag in days from event to
>     transition_risk.state = crisis)
>   - lower false-switch rate than V1

``evaluate_v2_gate`` is an OFFLINE promotion gate, NOT a per-session
walk-forward-runner step (F-022, ADR 0023). It compares two whole-backtest
``StrategyMetrics`` summaries — drawdown, Sharpe, detection lag, false-switch
rate — which the per-session regime engine never computes; they come from the
downstream strategy-eval layer (the F-014 ledger producer). The walk-forward /
shadow runners classify regimes only and therefore do not, and cannot, call this
gate. ``passed=False`` means "do not promote the V2 candidate". ``compute_v1_v2_diff``
is the per-session A/B label-diff used by the §9.3 shadow review.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from regime_detection.models import RegimeTimeline

# ---------------------------------------------------------------------------
# Performance gate (v2 §9.1)
# ---------------------------------------------------------------------------


class GateMetric(str, Enum):
    """The four v2 §9.1 metrics. Each is "v2 better than v1" on this axis."""

    LOWER_DRAWDOWN = "lower_drawdown"
    HIGHER_SHARPE = "higher_sharpe"
    EARLIER_CRISIS_DETECTION = "earlier_crisis_detection"
    LOWER_FALSE_SWITCH_RATE = "lower_false_switch_rate"


V2_GATE_METRIC_NAMES: tuple[str, ...] = tuple(metric.name for metric in GateMetric)


@dataclass(frozen=True)
class StrategyMetrics:
    """Downstream strategy metrics fed into the v2 §9.1 gate.

    `max_drawdown` is the most-negative cumulative return over the
    backtest window (e.g. -0.18 for a 18% peak-to-trough drawdown).
    `sharpe` is annualized. `mean_crisis_detection_lag_days` is the
    average NYSE-trading-day lag between a crisis event date and the
    engine raising `transition_risk.state = crisis` (lower is better). `false_switch_rate`
    is in [0, 1] (lower is better).

    The engine itself never computes these — they come from the strategy
    eval layer downstream of classification. The gate evaluator is
    purely a comparison.
    """

    max_drawdown: float
    sharpe: float
    mean_crisis_detection_lag_days: float
    false_switch_rate: float


@dataclass(frozen=True)
class GateResult:
    """v2 §9.1 evaluation result. V2 passes iff at least one metric beats V1."""

    passed: bool
    winning_metrics: tuple[GateMetric, ...]
    v1_metrics: StrategyMetrics
    v2_metrics: StrategyMetrics


def evaluate_v2_gate(
    *,
    v1_metrics: StrategyMetrics,
    v2_metrics: StrategyMetrics,
) -> GateResult:
    """v2 §9.1 gate: V2 passes iff it beats V1 on at least one of the four metrics.

    Offline promotion gate (F-022 / ADR 0023): callers treat ``passed=False`` as
    "block promotion of the V2 candidate". It is invoked by the offline strategy-eval
    / promotion harness, not the per-session walk-forward runner — the runner has no
    StrategyMetrics to supply.

    Args:
        v1_metrics: Strategy metrics for the V1-gated baseline.
        v2_metrics: Strategy metrics for the V2-gated candidate.

    Returns:
        GateResult with `passed=True` if any metric wins; `winning_metrics`
        lists every winning metric (not just the first).
    """
    winners: list[GateMetric] = []

    # Drawdown: more negative = worse → v2 wins if v2.drawdown > v1.drawdown
    # (closer to zero = less peak-to-trough loss).
    if v2_metrics.max_drawdown > v1_metrics.max_drawdown:
        winners.append(GateMetric.LOWER_DRAWDOWN)
    # Sharpe: higher = better.
    if v2_metrics.sharpe > v1_metrics.sharpe:
        winners.append(GateMetric.HIGHER_SHARPE)
    # Detection lag: lower = better.
    if (
        v2_metrics.mean_crisis_detection_lag_days
        < v1_metrics.mean_crisis_detection_lag_days
    ):
        winners.append(GateMetric.EARLIER_CRISIS_DETECTION)
    # False-switch rate: lower = better.
    if v2_metrics.false_switch_rate < v1_metrics.false_switch_rate:
        winners.append(GateMetric.LOWER_FALSE_SWITCH_RATE)

    return GateResult(
        passed=bool(winners),
        winning_metrics=tuple(winners),
        v1_metrics=v1_metrics,
        v2_metrics=v2_metrics,
    )


# ---------------------------------------------------------------------------
# Per-day label diff (v2 §9.3 A/B shadow review)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisLabelDiff:
    """Per-axis active_label disagreement between v1 and v2 on a single session."""

    axis: str
    as_of_date: date
    v1_active_label: str
    v2_active_label: str


@dataclass(frozen=True)
class RegimeDiff:
    """Aggregated v1 vs v2 differences across two parallel RegimeTimelines."""

    label_diffs: tuple[AxisLabelDiff, ...]
    v1_only_top_level_fields: tuple[str, ...]
    v2_only_top_level_fields: tuple[str, ...]


_V1_AXES_TO_COMPARE: tuple[str, ...] = (
    "trend_direction",
    "trend_character",
    "volatility_state",
    "breadth_state",
)
_V2_OPTIONAL_TOP_LEVEL_FIELDS: tuple[str, ...] = (
    "inflation_growth_state",
    "credit_funding_state",
    "credit_funding_effective_state",
    "volume_liquidity_state",
    "change_point",
)


def axis_reporting_label(
    output: object | None, *, default: str | None = None
) -> str | None:
    if output is None:
        return default
    reporting = getattr(output, "reporting_label", None)
    if reporting is not None:
        return str(reporting)
    classification_status = getattr(output, "classification_status", "classified")
    if classification_status != "classified":
        return str(classification_status)
    active_label = getattr(output, "active_label", None)
    if active_label is not None:
        return str(active_label)
    state = getattr(output, "state", None)
    if state is not None:
        return str(state)
    label = getattr(output, "label", default)
    return None if label is None else str(label)


def compute_v1_v2_diff(
    v1_timeline: RegimeTimeline,
    v2_timeline: RegimeTimeline,
) -> RegimeDiff:
    """Per-session diff of two timelines' active_labels.

    Compares both timelines element-by-element. Both must cover the
    same set of NYSE sessions and be in the same order. Raises
    ValueError on length mismatch or date misalignment — silently
    truncating would mask qualification-breaking drift.

    Reports per-axis active_label disagreements (incl. network_fragility)
    and which v2 optional top-level fields are present in one timeline
    but not the other (used by the slice-gating workflow to detect
    accidental V2 leakage / regression).
    """
    if len(v1_timeline.outputs) != len(v2_timeline.outputs):
        raise ValueError(
            "RegimeTimeline length mismatch: "
            f"v1={len(v1_timeline.outputs)} v2={len(v2_timeline.outputs)}"
        )

    diffs: list[AxisLabelDiff] = []
    for v1_out, v2_out in zip(v1_timeline.outputs, v2_timeline.outputs, strict=True):
        if v1_out.as_of_date != v2_out.as_of_date:
            raise ValueError(
                "RegimeTimeline date misalignment: "
                f"v1={v1_out.as_of_date.isoformat()} "
                f"v2={v2_out.as_of_date.isoformat()}"
            )
        for axis in _V1_AXES_TO_COMPARE:
            v1_label = axis_reporting_label(getattr(v1_out, axis))
            v2_label = axis_reporting_label(getattr(v2_out, axis))
            if v1_label != v2_label:
                diffs.append(
                    AxisLabelDiff(
                        axis=axis,
                        as_of_date=v1_out.as_of_date,
                        v1_active_label=v1_label,
                        v2_active_label=v2_label,
                    )
                )
        v1_network_label = axis_reporting_label(v1_out.network_fragility)
        v2_network_label = axis_reporting_label(v2_out.network_fragility)
        if v1_network_label != v2_network_label:
            diffs.append(
                AxisLabelDiff(
                    axis="network_fragility",
                    as_of_date=v1_out.as_of_date,
                    v1_active_label=v1_network_label,
                    v2_active_label=v2_network_label,
                )
            )

    v1_first = v1_timeline.outputs[0]
    v2_first = v2_timeline.outputs[0]
    v1_only = tuple(
        f
        for f in _V2_OPTIONAL_TOP_LEVEL_FIELDS
        if getattr(v1_first, f) is not None and getattr(v2_first, f) is None
    )
    v2_only = tuple(
        f
        for f in _V2_OPTIONAL_TOP_LEVEL_FIELDS
        if getattr(v1_first, f) is None and getattr(v2_first, f) is not None
    )

    return RegimeDiff(
        label_diffs=tuple(diffs),
        v1_only_top_level_fields=v1_only,
        v2_only_top_level_fields=v2_only,
    )
