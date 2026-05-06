from __future__ import annotations

import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from regime_detection.engine import RegimeEngine  # noqa: E402


def pytest_configure() -> None:
    # Ensure src/ layout is importable without requiring an editable install.
    if str(_REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT / "src"))


_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_RAW_DIR = _FIXTURES_DIR / "raw"
_MARKET_PARQUET_PATH = _RAW_DIR / "market_data.parquet"
_GOLDEN_DATES_PATH = _FIXTURES_DIR / "derived" / "golden_dates.yaml"


@lru_cache(maxsize=1)
def _load_market_data() -> pd.DataFrame:
    if _MARKET_PARQUET_PATH.exists():
        df = pd.read_parquet(_MARKET_PARQUET_PATH)
    else:
        parts = [pd.read_csv(_RAW_DIR / f"{symbol}.csv") for symbol in ("SPY", "RSP", "VIXY")]
        df = pd.concat(parts, ignore_index=True)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
    return df[keep].sort_values(["date", "symbol"]).reset_index(drop=True)


@pytest.fixture(scope="session")
def raw_market_data() -> pd.DataFrame:
    return _load_market_data().copy()


@pytest.fixture(scope="session")
def raw_market_frames(raw_market_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in ("SPY", "RSP", "VIXY"):
        frames[symbol] = raw_market_data[raw_market_data["symbol"] == symbol].copy().reset_index(drop=True)
    return frames


@pytest.fixture(scope="session")
def market_df_for_asof(raw_market_data: pd.DataFrame):
    def _build(as_of: date) -> pd.DataFrame:
        return raw_market_data[raw_market_data["date"] <= as_of].copy().reset_index(drop=True)

    return _build


@pytest.fixture(scope="session")
def golden_rows() -> list[dict[str, object]]:
    golden = yaml.safe_load(_GOLDEN_DATES_PATH.read_text())
    return list(golden["rows"])


@pytest.fixture(scope="session")
def classified_golden_outputs(golden_rows: list[dict[str, object]], market_df_for_asof) -> dict[date, object]:
    engine = RegimeEngine()
    outputs: dict[date, object] = {}
    for row in golden_rows:
        as_of = date.fromisoformat(str(row["as_of_date"]))
        outputs[as_of] = engine.classify(as_of_date=as_of, market_data=market_df_for_asof(as_of))
    return outputs
