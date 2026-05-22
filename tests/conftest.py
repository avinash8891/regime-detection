# ruff: noqa: E402
# Imports are intentionally split: thread-cap env vars must be set BEFORE
# numpy/pandas/sklearn import to take effect on BLAS thread pools — see the
# block below. Suppress E402 file-wide for this conftest.
from __future__ import annotations

import os

# IMPORTANT: cap numerical-library thread fanout BEFORE numpy/pandas/sklearn
# import. Under pytest-xdist with ``-n auto`` (8 worker processes here),
# leaving BLAS/OpenMP at default would let each worker spawn 8+ threads, for
# 64+ contending threads on 8 cores. This causes massive CPU stall on heavy
# numerical tests (verified via cProfile: 17.86s of time.sleep in joblib loky
# IPC for a single classify_window call). Test outputs are unaffected — same
# computations, just no parallel fanout inside the per-worker process.
for _envkey in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ.setdefault(_envkey, "1")

import importlib.util
import pickle
import sys
import time
from datetime import date
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from regime_detection.engine import RegimeEngine  # noqa: E402


def pytest_configure() -> None:
    # Ensure src/ layout is importable without requiring an editable install.
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    if str(_REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT / "src"))

    # Force joblib.Parallel to run in-process (n_jobs=1, threading backend)
    # for the pytest session. Under pytest-xdist (8 workers here), each
    # worker would otherwise spawn its own loky process pool inside
    # hmm_state.compute_hmm_features — 8 × 8 = 64 contending procs on 8
    # cores. cProfile confirms the cost: 17.86s of time.sleep in loky IPC
    # for one classify_window call. The seed sweep is deterministic, so
    # output is byte-identical regardless of n_jobs (HMM seeds are picked
    # by log-likelihood comparison after independent fits — order is
    # commutative). xdist already provides the right granularity of
    # parallelism (test-level); nested process parallelism inside each
    # test pessimizes.
    import joblib

    _original_parallel_init = joblib.Parallel.__init__

    def _force_inprocess_parallel(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Run the HMM seed sweep (and any other joblib.Parallel call) fully
        # in-process. xdist already provides test-level parallelism across
        # 8 worker procs; nesting a loky subprocess pool inside each worker
        # caused 17.86s of time.sleep in IPC per classify_window call
        # (verified via cProfile). n_jobs=1 + threading eliminates IPC and
        # leaves serial HMM compute, which empirically beats threaded-fanout
        # under the BLAS thread cap above.
        kwargs["n_jobs"] = 1
        kwargs["backend"] = "threading"
        return _original_parallel_init(self, *args, **kwargs)

    joblib.Parallel.__init__ = _force_inprocess_parallel  # type: ignore[method-assign]


_PROFILE_ENGINE_SHA = "0" * 64


def load_profile_engine_module() -> ModuleType:
    path = _REPO_ROOT / "scripts" / "profile_engine.py"
    spec = importlib.util.spec_from_file_location("profile_engine", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def profile_engine_manifest_artifact(name: str, local_path: str) -> dict[str, object]:
    return {
        "name": name,
        "stage": "canonical",
        "uri": f"s3://bucket/{local_path}",
        "local_path": local_path,
        "sha256": _PROFILE_ENGINE_SHA,
        "schema_version": None,
        "rows": 1,
        "min_date": None,
        "max_date": None,
        "required_for": ["profile_engine"],
    }


def write_profile_engine_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "artifact_set": "profile",
                "created_at_utc": "2026-05-17T00:00:00Z",
                "storage_root": "s3://bucket/root",
                "artifacts": [
                    profile_engine_manifest_artifact(
                        "constituent_ohlcv_AAPL",
                        "data/raw/daily_ohlcv_762/symbol=AAPL/ohlcv.parquet",
                    ),
                    profile_engine_manifest_artifact(
                        "fred_macro_series",
                        "data/raw/macro/fred_macro_series.parquet",
                    ),
                    profile_engine_manifest_artifact(
                        "sp500_pit_constituents",
                        "data/raw/pit_constituents/sp500_ticker_intervals.parquet",
                    ),
                    profile_engine_manifest_artifact(
                        "event_calendar_us",
                        "data/raw/event_calendar/us_events.yaml",
                    ),
                    profile_engine_manifest_artifact(
                        "ism_pmi_history",
                        "data/raw/pmi/us_ism_pmi_history.parquet",
                    ),
                    profile_engine_manifest_artifact(
                        "sf_fed_news_sentiment",
                        "data/raw/news_sentiment/sf_fed_news_sentiment.parquet",
                    ),
                ],
            },
            sort_keys=False,
        )
    )
    return path


_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_RAW_DIR = _FIXTURES_DIR / "raw"
_MARKET_PARQUET_PATH = _RAW_DIR / "market_data.parquet"
_V2_DAILY_OHLCV_PATH = _RAW_DIR / "v2" / "daily_ohlcv.csv"
_V2_FRED_MACRO_PATH = _RAW_DIR / "v2" / "fred_macro_series.csv"
_GOLDEN_DATES_PATH = _FIXTURES_DIR / "derived" / "golden_dates.yaml"
_V2_MACRO_LOGICAL_NAMES = (
    "sofr",
    "iorb",
    "nfci",
    "broad_usd_index",
    "hy_oas",
    "ig_bbb_oas",
)


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
    for logical_name in _V2_MACRO_LOGICAL_NAMES:
        frame = macro[macro["logical_name"] == logical_name]
        if frame.empty:
            raise RuntimeError(f"V2 FRED macro fixture missing {logical_name!r}")
        series_by_key[logical_name] = pd.Series(
            frame["value"].astype(float).to_numpy(),
            index=frame["date"],
            name=logical_name,
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


def _build_real_v2_classify_window_2026_05_13(
    v2_market_df_for_asof,
    v2_close_series_by_symbol: dict[str, pd.Series],
):
    """Compute classify_window once for the real V2 fixture at 2026-05-13
    with the canonical sector + cross-asset closes (no macro). The two
    integration tests asserting on this exact engine state (one via
    ``classify_window``, one via ``classify`` which delegates to
    ``classify_window(lookback_days=1).outputs[-1]`` — see
    ``test_classify_delegates_to_classify_window_with_single_day_lookback``)
    can share this result. Now economical because the joblib in-process
    monkeypatch in ``pytest_configure`` cut the build cost from ~90s to
    ~37s — small enough that the cross-worker setup-wait pays off.
    """
    from regime_detection.engine import RegimeEngine
    from regime_detection.fragility_universe import (
        CROSS_ASSET_SYMBOLS,
        SECTOR_ETFS,
    )

    as_of = date(2026, 5, 13)
    return RegimeEngine().classify_window(
        end_date=as_of,
        market_data=v2_market_df_for_asof(as_of),
        lookback_days=1,
        sector_etf_closes={s: v2_close_series_by_symbol[s] for s in SECTOR_ETFS},
        cross_asset_closes={
            s: v2_close_series_by_symbol[s] for s in CROSS_ASSET_SYMBOLS
        },
    )


@pytest.fixture(scope="session")
def real_v2_classify_window_2026_05_13(
    v2_market_df_for_asof,
    v2_close_series_by_symbol,
    tmp_path_factory: pytest.TempPathFactory,
    worker_id: str,
):
    """Session-scoped, cross-worker pickle-cached classify_window result for
    the real V2 fixture at as_of=2026-05-13 with sector + cross-asset
    closes (no macro). See ``_build_real_v2_classify_window_2026_05_13``.
    """
    if worker_id == "master":
        return _build_real_v2_classify_window_2026_05_13(
            v2_market_df_for_asof, v2_close_series_by_symbol
        )

    shared_dir = tmp_path_factory.getbasetemp().parent
    cache_path = shared_dir / "real_v2_classify_window_2026_05_13.pkl"
    lock_path = shared_dir / "real_v2_classify_window_2026_05_13.lock"

    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        result = _build_real_v2_classify_window_2026_05_13(
            v2_market_df_for_asof, v2_close_series_by_symbol
        )
        tmp = cache_path.with_suffix(".pkl.tmp")
        tmp.write_bytes(pickle.dumps(result))
        tmp.replace(cache_path)
        return result
    except FileExistsError:
        pass

    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        if cache_path.exists():
            return pickle.loads(cache_path.read_bytes())
        time.sleep(0.2)
    raise RuntimeError(
        "real_v2_classify_window_2026_05_13 build timed out waiting on "
        f"peer worker; cache_path={cache_path}"
    )


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
