from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from regime_detection.axis_series import AxisSeriesResult


def axis_outputs_from_core(**kwargs: object) -> AxisSeriesResult:
    from regime_detection.axis_series import _build_axis_outputs

    return cast("AxisSeriesResult", _build_axis_outputs(**cast(Any, kwargs)))


def new_axis_series_result(**kwargs: object) -> AxisSeriesResult:
    from regime_detection.axis_series import AxisSeriesResult

    return cast("AxisSeriesResult", AxisSeriesResult(**cast(Any, kwargs)))
