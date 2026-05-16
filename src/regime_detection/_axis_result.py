"""Shared result type and helper for axis-series classifiers.

Split out from axis_series.py so that individual axis modules can return
``AxisSeriesResult`` without creating a circular import back into
``axis_series.py``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from regime_detection.data_quality import assess_series_input_quality, quality_forces_unknown
from regime_detection.models import AxisOutput, BreadthStateOutput


@dataclass(frozen=True)
class AxisSeriesResult:
    outputs_by_date: dict[date, AxisOutput | BreadthStateOutput]

    @property
    def stable_labels_by_date(self) -> dict[date, str]:
        return {d: out.stable_label for d, out in self.outputs_by_date.items()}

    @property
    def active_labels_by_date(self) -> dict[date, str]:
        return {d: out.active_label for d, out in self.outputs_by_date.items()}


def _build_axis_outputs(
    *,
    dates: Sequence[date],
    raw_labels: Sequence[str],
    stable_labels: Sequence[str],
    active_labels: Sequence[str],
    raw_evidence: Sequence[dict[str, object]],
    risk_rank: Mapping[str, int],
    deescalation_days: int,
    required_inputs: list,
    required_trading_days: int,
    max_freshness_days: int,
    min_completeness: float,
) -> AxisSeriesResult:
    outputs_by_date: dict[date, AxisOutput] = {}
    input_by_date = list(required_inputs)
    for day, raw, stable, active, evidence in zip(
        dates, raw_labels, stable_labels, active_labels, raw_evidence, strict=True
    ):
        dq = assess_series_input_quality(
            as_of_date=day,
            required_inputs=input_by_date,
            required_trading_days=required_trading_days,
            raw_label=raw,
            max_freshness_days=max_freshness_days,
            min_completeness=min_completeness,
        )
        if quality_forces_unknown(dq):
            output = AxisOutput(
                raw_label="unknown",
                stable_label="unknown",
                active_label="unknown",
                evidence={"reason": dq.reason},
                data_quality=dq,
            )
        else:
            output = AxisOutput(
                raw_label=raw,
                stable_label=stable,
                active_label=active,
                evidence={
                    "rule_evidence": evidence,
                    "risk_rank": risk_rank,
                    "deescalation_days": deescalation_days,
                },
                data_quality=dq,
            )
        outputs_by_date[day] = output
    return AxisSeriesResult(outputs_by_date=outputs_by_date)
