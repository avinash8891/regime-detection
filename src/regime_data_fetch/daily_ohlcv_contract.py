"""Canonical daily-OHLCV partition contract.

Single source of truth for the invariants the fixed-universe OHLCV tree
must satisfy on disk and inside manifest artifacts:

  * artifact ``name`` convention: ``f"{FIXED_UNIVERSE_TREE_NAME}_{symbol}"``
  * ``symbol`` column is present, non-null, and equals the partition symbol
  * the partition holds rows for exactly one symbol

Two checks are provided so callers can use whichever shape they already
have: :func:`require_symbol_partition_frame` for a pandas frame and
:func:`require_symbol_partition_table` for a pyarrow table read directly
from bytes.

Previously each caller maintained its own near-duplicate of these checks;
keeping them here prevents drift between the publish-time validator and
the calibration loader.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportMissingTypeStubs=false, reportUnknownParameterType=false

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from regime_data_fetch.universe import FIXED_UNIVERSE_TREE_NAME

if TYPE_CHECKING:  # pragma: no cover - imports only used for typing
    import pandas as pd
    import pyarrow as pa


DAILY_OHLCV_ARTIFACT_PREFIX: str = f"{FIXED_UNIVERSE_TREE_NAME}_"


def daily_ohlcv_artifact_name(symbol: str) -> str:
    """Return the canonical manifest ``name`` for ``symbol``."""
    return f"{DAILY_OHLCV_ARTIFACT_PREFIX}{symbol}"


def parse_daily_ohlcv_artifact_name(name: str) -> str | None:
    """Return the symbol embedded in ``name`` or ``None`` if it does not
    follow the daily-OHLCV partition convention.
    """
    if not name.startswith(DAILY_OHLCV_ARTIFACT_PREFIX):
        return None
    symbol = name[len(DAILY_OHLCV_ARTIFACT_PREFIX) :]
    return symbol or None


def require_symbol_partition_frame(
    frame: "pd.DataFrame",
    *,
    expected_symbol: str,
    source: Path | str,
) -> None:
    """Validate ``frame`` matches the daily-OHLCV partition contract."""
    if "symbol" not in frame.columns:
        raise ValueError(
            _contract_msg(source, expected_symbol, "missing symbol column")
        )
    column = frame["symbol"]
    if column.isna().any():
        raise ValueError(
            _contract_msg(source, expected_symbol, "has null symbol row(s)")
        )
    observed = sorted({str(value) for value in column.unique()})
    if observed != [expected_symbol]:
        raise ValueError(
            f"daily OHLCV symbol contract violation: {source} "
            f"expected {expected_symbol}, observed {observed}"
        )


def require_symbol_partition_table(
    table: "pa.Table",
    *,
    expected_symbol: str,
    source: Path | str,
) -> None:
    """Validate a pyarrow ``table`` matches the daily-OHLCV partition contract."""
    if "symbol" not in table.column_names:
        raise ValueError(
            _contract_msg(source, expected_symbol, "missing symbol column")
        )
    column = table.column("symbol").to_pandas()
    if column.isna().any():
        raise ValueError(
            _contract_msg(source, expected_symbol, "has null symbol row(s)")
        )
    observed = sorted({str(value) for value in column.unique()})
    if observed != [expected_symbol]:
        raise ValueError(
            f"daily OHLCV symbol contract violation: {source} "
            f"expected {expected_symbol}, observed {observed}"
        )


def _contract_msg(source: Path | str, expected_symbol: str, detail: str) -> str:
    return (
        "daily OHLCV symbol contract violation: "
        f"{source} {detail}; expected {expected_symbol}"
    )
