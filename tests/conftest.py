from __future__ import annotations

import os
import pickle
import sys
import time
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
_V2_DAILY_OHLCV_PATH = _RAW_DIR / "v2" / "daily_ohlcv.csv"
_V2_FRED_MACRO_PATH = _RAW_DIR / "v2" / "fred_macro_series.csv"
_GOLDEN_DATES_PATH = _FIXTURES_DIR / "derived" / "golden_dates.yaml"
_V2_MACRO_KEY_BY_LOGICAL_NAME = {
    "sofr": "SOFR",
    "iorb": "IORB",
    "nfci": "NFCI",
    "broad_usd_index": "broad_usd_index",
    "hy_oas": "hy_oas",
    "ig_bbb_oas": "ig_bbb_oas",
}


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


@lru_cache(maxsize=1)
def _load_v2_daily_ohlcv() -> pd.DataFrame:
    df = pd.read_csv(_V2_DAILY_OHLCV_PATH)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
    return df[keep].sort_values(["date", "symbol"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _load_v2_fred_macro() -> pd.DataFrame:
    df = pd.read_csv(_V2_FRED_MACRO_PATH)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    keep = ["date", "series_id", "logical_name", "value"]
    return df[keep].sort_values(["date", "logical_name"]).reset_index(drop=True)


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
def v2_daily_ohlcv() -> pd.DataFrame:
    return _load_v2_daily_ohlcv().copy()


@pytest.fixture(scope="session")
def v2_market_df_for_asof(v2_daily_ohlcv: pd.DataFrame):
    def _build(as_of: date) -> pd.DataFrame:
        return v2_daily_ohlcv[
            (v2_daily_ohlcv["date"] <= as_of)
            & (v2_daily_ohlcv["symbol"].isin({"SPY", "RSP", "VIXY"}))
        ].copy().reset_index(drop=True)

    return _build


@pytest.fixture(scope="session")
def v2_close_series_by_symbol(v2_daily_ohlcv: pd.DataFrame) -> dict[str, pd.Series]:
    series_by_symbol: dict[str, pd.Series] = {}
    for symbol, frame in v2_daily_ohlcv.groupby("symbol", sort=True):
        idx = pd.to_datetime(frame["date"])
        series_by_symbol[str(symbol)] = pd.Series(
            frame["close"].astype(float).to_numpy(),
            index=idx,
            name=str(symbol),
        )
    return series_by_symbol


@pytest.fixture(scope="session")
def v2_macro_series_by_key() -> dict[str, pd.Series]:
    macro = _load_v2_fred_macro()
    series_by_key: dict[str, pd.Series] = {}
    for logical_name, key in _V2_MACRO_KEY_BY_LOGICAL_NAME.items():
        frame = macro[macro["logical_name"] == logical_name]
        if frame.empty:
            raise RuntimeError(f"V2 FRED macro fixture missing {logical_name!r}")
        series_by_key[key] = pd.Series(
            frame["value"].astype(float).to_numpy(),
            index=frame["date"],
            name=key,
        )
    return series_by_key


@pytest.fixture(scope="session")
def golden_rows() -> list[dict[str, object]]:
    golden = yaml.safe_load(_GOLDEN_DATES_PATH.read_text())
    return list(golden["rows"])


def _classify_all_golden_rows(
    golden_rows: list[dict[str, object]],
    market_df_for_asof,
) -> dict[date, object]:
    """Classify every golden date in ONE classify_window pass.

    Previously this looped over rows and called ``engine.classify`` per
    row, paying the full ``build_market_context`` + ``build_feature_store``
    cost ~10 times. classify_window emits a per-day timeline from a single
    pipeline run; we slice the requested golden dates out of its outputs.

    PIT correctness: classify_window's per-day emission is V1 §2.2
    stateless-replay compliant — each emitted day's classifier state is
    computed using only data on or before that day. The trainable V2
    seams (HMM/GMM/BOCPD) are NOT exercised here because this fixture
    passes no V2 kwargs (no sector/cross-asset/macro/PIT inputs), so the
    PIT-leak masking added in commit 19e395d is moot for this path.
    """
    engine = RegimeEngine()
    golden_dates = sorted(
        date.fromisoformat(str(row["as_of_date"])) for row in golden_rows
    )
    if not golden_dates:
        return {}
    end = golden_dates[-1]
    earliest = golden_dates[0]
    # NYSE has ~252 sessions per calendar year. Upper-bound lookback to
    # comfortably cover earliest..end inclusive plus engine min-history.
    span_days = (end - earliest).days
    lookback_sessions = max(1, int(span_days / 365.25 * 252) + 30)
    timeline = engine.classify_window(
        end_date=end,
        market_data=market_df_for_asof(end),
        lookback_days=lookback_sessions,
    )
    by_date = {out.as_of_date: out for out in timeline.outputs}
    missing = [d for d in golden_dates if d not in by_date]
    if missing:
        raise RuntimeError(
            f"classify_window did not emit outputs for golden dates: {missing!r}. "
            f"Window end={end}, lookback_sessions={lookback_sessions}, "
            f"emitted span={timeline.outputs[0].as_of_date}..{timeline.outputs[-1].as_of_date}"
        )
    return {d: by_date[d] for d in golden_dates}


def _load_module_for_fixture(name: str, rel_path: str):
    import importlib.util

    script_path = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@pytest.fixture(scope="session")
def walkforward_2023_dec_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the 3-session 2023-12-12..14 walkforward ONCE per worker session.

    The three tests in ``test_historical_walkforward.py`` and
    ``test_build_walkforward_report.py`` previously ran the same walkforward
    independently into their own tmp_paths, paying the ~6s/session classify
    cost 3 times. This fixture caches the walkforward output directory tree
    so the consumer tests can ``shutil.copytree`` it into their own
    tmp_path and run the report builder on the copy without rerunning the
    full classify pipeline.
    """
    cache_dir = tmp_path_factory.mktemp("walkforward_2023_dec_template")
    runner = _load_module_for_fixture(
        "run_historical_walkforward", "scripts/run_historical_walkforward.py"
    )
    market_data_path = _REPO_ROOT / "tests" / "fixtures" / "raw" / "market_data.parquet"
    runner.run_walkforward(
        market_data_path=market_data_path,
        output_root=cache_dir,
        start_date=date(2023, 12, 12),
        end_date=date(2023, 12, 14),
    )
    return cache_dir


@pytest.fixture(scope="session")
def classified_golden_outputs(
    tmp_path_factory: pytest.TempPathFactory,
    golden_rows: list[dict[str, object]],
    market_df_for_asof,
    worker_id: str,
) -> dict[date, object]:
    """Session-scoped fixture, shared across pytest-xdist workers via a disk
    pickle cache. The single-process path computes inline; the multi-worker
    path elects one worker to build via an exclusive-create lockfile, while
    other workers poll for the pickle to land. This eliminates the per-worker
    rebuild cost (~81s × N workers) that previously dominated wall-clock."""
    if worker_id == "master":
        return _classify_all_golden_rows(golden_rows, market_df_for_asof)

    shared_dir = tmp_path_factory.getbasetemp().parent
    cache_path = shared_dir / "classified_golden_outputs.pkl"
    lock_path = shared_dir / "classified_golden_outputs.lock"

    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    try:
        # O_CREAT | O_EXCL: atomic single-winner election across workers.
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        outputs = _classify_all_golden_rows(golden_rows, market_df_for_asof)
        tmp_path = cache_path.with_suffix(".pkl.tmp")
        tmp_path.write_bytes(pickle.dumps(outputs))
        tmp_path.replace(cache_path)
        return outputs
    except FileExistsError:
        pass

    # Lost the election: another worker is building. Poll for the result.
    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        if cache_path.exists():
            return pickle.loads(cache_path.read_bytes())
        time.sleep(0.2)
    raise RuntimeError(
        f"classified_golden_outputs build timed out waiting on peer worker; "
        f"cache_path={cache_path}"
    )
