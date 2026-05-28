from __future__ import annotations

import numpy as np
import pandas as pd


def aligned_float_values(series: pd.Series, index: pd.Index) -> np.ndarray:
    return series.reindex(index).astype(float).to_numpy()


def optional_aligned_float_values(
    series: pd.Series | None, index: pd.Index
) -> np.ndarray:
    if series is None:
        return np.full(len(index), np.nan)
    return aligned_float_values(series, index)
