from __future__ import annotations

import datetime as dt
from collections.abc import Mapping

import pandas as pd


def require_single_session(index: pd.Index, session: object) -> None:
    """Raise if ``session`` does not appear EXACTLY once in ``index``.

    The per-day axis wrappers (``raw_label_for_day`` in trend_character /
    volatility_state / breadth_state) slice features with ``.loc[[dt]]`` and then
    return ``labels[0]``. If ``dt`` resolves to multiple rows (a duplicate-date data
    issue), that silently picks the first match and masks the bug. Fail loud instead so
    a duplicated session surfaces immediately rather than corrupting a single-day label.
    """
    count = int((index == session).sum())
    if count != 1:
        raise ValueError(
            f"expected exactly one session matching {session!r} in the feature index, "
            f"found {count}"
        )


def cow_safe_assign(
    frame: pd.DataFrame,
    replacements: Mapping[str, object],
    *,
    columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return a new DataFrame with selected columns replaced.

    Pandas 2.3's ``mode.copy_on_write="warn"`` emits FutureWarnings for direct
    column assignment on copied frames. Constructing the replacement frame in
    one pass keeps normalization code compatible with pandas 3.0 Copy-on-Write
    semantics without warning suppression.
    """

    selected_columns = list(columns) if columns is not None else list(frame.columns)
    selected_columns.extend(col for col in replacements if col not in selected_columns)
    return pd.DataFrame(
        {
            col: (
                replacements[col]
                if col in replacements
                else frame[col].to_numpy(copy=True)
            )
            for col in selected_columns
        },
        index=frame.index.copy(),
    )


def optional_date(value: object) -> dt.date | None:
    """Normalize date-like values to ``datetime.date`` while preserving nulls."""

    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))
