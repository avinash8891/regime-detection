"""Internal helpers shared by loaders.py and event_calendar_loader.py.

Keep this module free of intra-package imports to avoid circular dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd

_PANDAS_READ_CSV = cast(Callable[[Path], pd.DataFrame], cast(Any, pd).read_csv)


def read_csv_dataframe(path: Path) -> pd.DataFrame:
    return _PANDAS_READ_CSV(path)


def column_values(frame: pd.DataFrame, column: str) -> list[object]:
    return list(frame[column])


def is_missing(value: object) -> bool:
    return bool(cast(Any, pd).isna(value))
