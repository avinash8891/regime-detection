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
from dataclasses import dataclass
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

from regime_shared.pandas_compat import cow_safe_assign  # noqa: E402
from regime_detection.config import load_regime_config  # noqa: E402
from regime_detection.engine import RegimeEngine  # noqa: E402
from regime_detection.fragility_universe import (
    CROSS_ASSET_SYMBOLS,
    SECTOR_ETFS,
)  # noqa: E402
from regime_detection.loaders import load_event_calendar  # noqa: E402


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
DEFAULT_LIVE_DATA_MANIFEST = (
    _REPO_ROOT / "manifests" / "runs" / "regime_engine_2026-05-17.yaml"
)


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


@dataclass(frozen=True)
class MissingLiveDataInput:
    field: str
    path: Path
    reason: str


@dataclass(frozen=True)
class LiveDataInputs:
    manifest_path: Path
    data_root: Path
    daily_dir: Path
    news_sentiment_parquet: Path | None
    fomc_minutes_parquet: Path | None

    def require_materialized(self, field: str) -> MissingLiveDataInput | None:
        path = getattr(self, field)
        if path is None:
            return MissingLiveDataInput(
                field=field,
                path=self.data_root,
                reason="manifest does not route this optional live-data input",
            )
        assert isinstance(path, Path)
        if path.exists():
            return None
        return MissingLiveDataInput(
            field=field,
            path=path,
            reason="manifest-resolved path is not materialized locally",
        )

    def pytest_skip_unless_materialized(self, *fields: str) -> None:
        missing = [
            item for field in fields if (item := self.require_materialized(field))
        ]
        if not missing:
            return
        details = "; ".join(
            f"{item.field} -> {item.path} ({item.reason})" for item in missing
        )
        pytest.skip(
            "Live integration data is not materialized from the reviewed manifest: "
            f"{details}. Materialize {self.manifest_path} into {self.data_root} first."
        )


def resolve_live_data_inputs(
    *,
    manifest_path: Path | None = None,
    data_root: Path | None = None,
    runner_name: str = "profile_engine",
) -> LiveDataInputs:
    from regime_data_fetch.manifest_inputs import resolve_runner_input_paths

    manifest_path = manifest_path or DEFAULT_LIVE_DATA_MANIFEST
    data_root = data_root or (_REPO_ROOT / "data" / "raw")
    resolved = resolve_runner_input_paths(
        manifest_path=manifest_path,
        data_root=data_root,
        runner_name=runner_name,
        cli_values={},
        cli_overrides=set(),
        repo_root=_REPO_ROOT,
    )
    return LiveDataInputs(
        manifest_path=manifest_path,
        data_root=data_root,
        daily_dir=resolved.daily_dir,
        news_sentiment_parquet=resolved.news_sentiment_parquet,
        fomc_minutes_parquet=resolved.fomc_minutes_parquet,
    )


def load_spy_session_index_from_daily_tree(daily_dir: Path) -> pd.DatetimeIndex:
    spy_path = daily_dir / "symbol=SPY" / "ohlcv.parquet"
    if not spy_path.exists():
        raise FileNotFoundError(spy_path)
    spy = pd.read_parquet(spy_path, columns=["date"])
    dates = pd.to_datetime(spy["date"]).sort_values().unique()
    return pd.DatetimeIndex(dates)


_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_RAW_DIR = _FIXTURES_DIR / "raw"
_MARKET_PARQUET_PATH = _RAW_DIR / "market_data.parquet"
_V2_DAILY_OHLCV_PATH = _RAW_DIR / "v2" / "daily_ohlcv.csv"
_V2_FRED_MACRO_PATH = _RAW_DIR / "v2" / "fred_macro_series.csv"
_EVENT_CALENDAR_PATH = _FIXTURES_DIR / "events" / "us_events.yaml"
_GOLDEN_DATES_PATH = _FIXTURES_DIR / "derived" / "golden_dates.yaml"
_V2_MACRO_LOGICAL_NAMES = (
    "sofr",
    "iorb",
    "nfci",
    "broad_usd_index",
    "hy_oas",
    "ig_bbb_oas",
)


def _fast_v2_test_config():
    engine = RegimeEngine()
    assert engine.config.hmm is not None
    assert engine.config.clustering is not None
    assert engine.config.change_point is not None
    return engine.config.model_copy(
        update={
            "hmm": engine.config.hmm.model_copy(
                update={
                    "n_states": 2,
                    "training_window_days": 100,
                    "random_seeds": (42, 7, 13),
                }
            ),
            "clustering": engine.config.clustering.model_copy(
                update={"training_window_days": 100}
            ),
            "change_point": engine.config.change_point.model_copy(
                update={"training_window_days": 100}
            ),
        }
    )


@lru_cache(maxsize=1)
def _load_market_data() -> pd.DataFrame:
    if not _MARKET_PARQUET_PATH.exists():
        raise RuntimeError(
            "market data fixture must include real VIX rows in market_data.parquet"
        )
    df = pd.read_parquet(_MARKET_PARQUET_PATH)
    df = df.copy()
    if "VIX" not in set(df["symbol"]):
        raise RuntimeError("market data fixture must include real VIX rows")
    keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
    df = cow_safe_assign(df, {"date": pd.to_datetime(df["date"]).dt.date}, columns=keep)
    return df[keep].sort_values(["date", "symbol"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _load_v2_daily_ohlcv() -> pd.DataFrame:
    df = pd.read_csv(_V2_DAILY_OHLCV_PATH)
    df = df.copy()
    keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
    df = cow_safe_assign(df, {"date": pd.to_datetime(df["date"]).dt.date}, columns=keep)
    return df[keep].sort_values(["date", "symbol"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _load_v2_fred_macro() -> pd.DataFrame:
    df = pd.read_csv(_V2_FRED_MACRO_PATH)
    df = df.copy().assign(date=pd.to_datetime(df["date"]))
    keep = ["date", "series_id", "logical_name", "value"]
    return df[keep].sort_values(["date", "logical_name"]).reset_index(drop=True)


def _constituent_ohlcv_from_v2_daily(
    v2_daily_ohlcv: pd.DataFrame, symbols: object
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = v2_daily_ohlcv[v2_daily_ohlcv["symbol"] == symbol].copy()
        if frame.empty:
            raise RuntimeError(f"V2 daily OHLCV fixture missing {symbol} rows")
        idx = pd.to_datetime(frame["date"])
        frames[str(symbol)] = pd.DataFrame(
            {
                "open": frame["open"].astype(float).to_numpy(),
                "high": frame["high"].astype(float).to_numpy(),
                "low": frame["low"].astype(float).to_numpy(),
                "close": frame["close"].astype(float).to_numpy(),
                "volume": frame["volume"].astype("int64").to_numpy(),
                "adjusted_close": frame["close"].astype(float).to_numpy(),
            },
            index=idx,
        )
    return frames


def _close_series_by_symbol_from_v2_daily(
    v2_daily_ohlcv: pd.DataFrame, symbols: object, *, as_of: date
) -> dict[str, pd.Series]:
    series_by_symbol: dict[str, pd.Series] = {}
    for symbol in symbols:
        frame = v2_daily_ohlcv[
            (v2_daily_ohlcv["symbol"] == symbol) & (v2_daily_ohlcv["date"] <= as_of)
        ].copy()
        if frame.empty:
            raise RuntimeError(
                f"V2 daily OHLCV fixture missing {symbol} rows through {as_of}"
            )
        frame = frame.sort_values("date")
        series_by_symbol[str(symbol)] = pd.Series(
            frame["close"].astype(float).to_numpy(),
            index=pd.to_datetime(frame["date"]),
            name=str(symbol),
        )
    return series_by_symbol


def _macro_series_from_v2_fixture(as_of: date) -> dict[str, pd.Series]:
    macro = _load_v2_fred_macro()
    macro = macro[macro["date"].dt.date <= as_of]
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
    broad_usd = series_by_key["broad_usd_index"]
    trend = pd.Series(
        range(len(broad_usd.index)), index=broad_usd.index, dtype="float64"
    )
    series_by_key["2y_yield"] = (4.00 + trend * 0.0002).rename("2y_yield")
    series_by_key["10y_yield"] = (4.25 + trend * 0.0001).rename("10y_yield")
    series_by_key["cpi_all_items"] = (300.0 + trend * 0.01).rename("cpi_all_items")
    series_by_key["pmi_manufacturing"] = (50.0 + trend * 0.0001).rename(
        "pmi_manufacturing"
    )
    return series_by_key


def _v2_macro_fixture_covers(as_of: date) -> bool:
    macro = _load_v2_fred_macro()
    macro = macro[(macro["date"].dt.date <= as_of) & macro["value"].notna()]
    present = set(macro["logical_name"].dropna().astype(str))
    return set(_V2_MACRO_LOGICAL_NAMES).issubset(present)


def _write_v2_macro_fixture_parquet(path: Path) -> Path:
    source = _load_v2_fred_macro().dropna(subset=["value"]).copy()
    daily = _load_v2_daily_ohlcv()
    dates = pd.to_datetime(sorted(daily["date"].unique()))
    trend = pd.Series(range(len(dates)), index=dates, dtype="float64")
    synthetic_series = {
        "2y_yield": 4.00 + trend * 0.0002,
        "10y_yield": 4.25 + trend * 0.0001,
        "cpi_all_items": 300.0 + trend * 0.01,
        "pmi_manufacturing": 50.0 + trend * 0.0001,
    }
    synthetic = pd.DataFrame(
        [
            {
                "date": observed_date,
                "series_id": logical_name.upper(),
                "logical_name": logical_name,
                "value": value,
            }
            for logical_name, series in synthetic_series.items()
            for observed_date, value in series.items()
        ]
    )
    pd.concat([source, synthetic], ignore_index=True).to_parquet(path, index=False)
    return path


@pytest.fixture(scope="session")
def raw_market_data() -> pd.DataFrame:
    return _load_market_data().copy()


@pytest.fixture(scope="session")
def raw_market_frames(raw_market_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in ("SPY", "RSP", "VIX", "VIXY"):
        frames[symbol] = (
            raw_market_data[raw_market_data["symbol"] == symbol]
            .copy()
            .reset_index(drop=True)
        )
    return frames


@pytest.fixture(scope="session")
def market_df_for_asof(raw_market_data: pd.DataFrame):
    def _build(as_of: date) -> pd.DataFrame:
        return (
            raw_market_data[raw_market_data["date"] <= as_of]
            .copy()
            .reset_index(drop=True)
        )

    return _build


@pytest.fixture(scope="session")
def v2_daily_ohlcv() -> pd.DataFrame:
    return _load_v2_daily_ohlcv().copy()


@pytest.fixture(scope="session")
def v2_market_df_for_asof(v2_daily_ohlcv: pd.DataFrame):
    def _build(as_of: date) -> pd.DataFrame:
        out = (
            v2_daily_ohlcv[
                (v2_daily_ohlcv["date"] <= as_of)
                & (v2_daily_ohlcv["symbol"].isin({"SPY", "RSP", "VIX", "VIXY"}))
            ]
            .copy()
            .reset_index(drop=True)
        )
        if "VIX" not in set(out["symbol"]):
            raise RuntimeError("V2 daily OHLCV fixture must include real VIX rows")
        return out.sort_values(["date", "symbol"]).reset_index(drop=True)

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
def v2_sector_etf_closes(
    v2_close_series_by_symbol: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    missing = sorted(set(SECTOR_ETFS).difference(v2_close_series_by_symbol))
    if missing:
        raise RuntimeError(f"V2 daily OHLCV fixture missing sector ETFs: {missing}")
    return {symbol: v2_close_series_by_symbol[symbol] for symbol in SECTOR_ETFS}


@pytest.fixture(scope="session")
def v2_cross_asset_closes(
    v2_close_series_by_symbol: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    required_symbols = set(CROSS_ASSET_SYMBOLS) | {"KRE", "XLY", "XLI", "XLP", "XLU"}
    missing = sorted(required_symbols.difference(v2_close_series_by_symbol))
    if missing:
        raise RuntimeError(
            f"V2 daily OHLCV fixture missing cross-asset symbols: {missing}"
        )
    return {symbol: v2_close_series_by_symbol[symbol] for symbol in required_symbols}


@pytest.fixture(scope="session")
def v2_pit_constituent_intervals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": list(SECTOR_ETFS),
            "start_date": [
                {"XLRE": date(2015, 10, 8), "XLC": date(2018, 6, 19)}.get(
                    t, date(2009, 1, 2)
                )
                for t in SECTOR_ETFS
            ],
            "end_date": [None] * len(SECTOR_ETFS),
        }
    )


@pytest.fixture(scope="session")
def v2_constituent_ohlcv_by_symbol(
    v2_daily_ohlcv: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return _constituent_ohlcv_from_v2_daily(v2_daily_ohlcv, SECTOR_ETFS)


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
    broad_usd = series_by_key["broad_usd_index"]
    trend = pd.Series(
        range(len(broad_usd.index)), index=broad_usd.index, dtype="float64"
    )
    series_by_key["2y_yield"] = (4.00 + trend * 0.0002).rename("2y_yield")
    series_by_key["10y_yield"] = (4.25 + trend * 0.0001).rename("10y_yield")
    series_by_key["cpi_all_items"] = (300.0 + trend * 0.01).rename("cpi_all_items")
    series_by_key["pmi_manufacturing"] = (50.0 + trend * 0.0001).rename(
        "pmi_manufacturing"
    )
    return series_by_key


@pytest.fixture(scope="session")
def v2_macro_parquet_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_v2_macro_fixture_parquet(
        tmp_path_factory.mktemp("v2_macro_fixture") / "fred_macro_series.parquet"
    )


@pytest.fixture(scope="session")
def v2_classify_kwargs_for_asof(
    v2_market_df_for_asof,
    event_calendar_df: pd.DataFrame,
    v2_sector_etf_closes: dict[str, pd.Series],
    v2_cross_asset_closes: dict[str, pd.Series],
    v2_macro_series_by_key: dict[str, pd.Series],
    v2_pit_constituent_intervals: pd.DataFrame,
    v2_constituent_ohlcv_by_symbol: dict[str, pd.DataFrame],
):
    def _build(as_of: date) -> dict[str, object]:
        return {
            "config": _fast_v2_test_config(),
            "market_data": v2_market_df_for_asof(as_of),
            "event_calendar": event_calendar_df,
            "sector_etf_closes": v2_sector_etf_closes,
            "cross_asset_closes": v2_cross_asset_closes,
            "macro_series": v2_macro_series_by_key,
            "pit_constituent_intervals": v2_pit_constituent_intervals,
            "constituent_ohlcv": v2_constituent_ohlcv_by_symbol,
        }

    return _build


@pytest.fixture(scope="session")
def synthetic_v2_kwargs_for_market_data(event_calendar_df: pd.DataFrame):
    def _build(market_data: pd.DataFrame) -> dict[str, object]:
        config = _fast_v2_test_config()
        assert config.network_fragility is not None
        config = config.model_copy(
            update={
                "network_fragility": config.network_fragility.model_copy(
                    update={
                        "percentile_lookback_days": 100,
                        "dispersion_percentile_lookback_days": 100,
                    }
                )
            }
        )
        as_of = max(market_data["date"])
        start = min(market_data["date"])
        v2_daily = _load_v2_daily_ohlcv()
        v2_start = min(v2_daily["date"])
        required_cross_assets = sorted(
            set(CROSS_ASSET_SYMBOLS) | {"KRE", "XLY", "XLI", "XLP", "XLU"}
        )
        if as_of < v2_start or not _v2_macro_fixture_covers(as_of):
            raise RuntimeError(
                "real V2 fixture rows do not cover synthetic_v2_kwargs "
                f"window start={start} as_of={as_of}"
            )
        sector_closes = _close_series_by_symbol_from_v2_daily(
            v2_daily, SECTOR_ETFS, as_of=as_of
        )
        cross_asset_closes = _close_series_by_symbol_from_v2_daily(
            v2_daily, required_cross_assets, as_of=as_of
        )
        macro_series = _macro_series_from_v2_fixture(as_of)
        pit_intervals = pd.DataFrame(
            {
                "ticker": list(SECTOR_ETFS),
                "start_date": [v2_start] * len(SECTOR_ETFS),
                "end_date": [None] * len(SECTOR_ETFS),
            }
        )
        constituent_ohlcv = _constituent_ohlcv_from_v2_daily(
            v2_daily[v2_daily["date"] <= as_of], SECTOR_ETFS
        )
        return {
            "config": config,
            "event_calendar": event_calendar_df,
            "sector_etf_closes": sector_closes,
            "cross_asset_closes": cross_asset_closes,
            "macro_series": macro_series,
            "pit_constituent_intervals": pit_intervals,
            "constituent_ohlcv": constituent_ohlcv,
        }

    return _build


@pytest.fixture(scope="session")
def event_calendar_df() -> pd.DataFrame:
    return load_event_calendar(_EVENT_CALENDAR_PATH).copy()


@pytest.fixture(scope="session")
def golden_rows() -> list[dict[str, object]]:
    golden = yaml.safe_load(_GOLDEN_DATES_PATH.read_text())
    return list(golden["rows"])


def _classify_all_golden_rows(
    golden_rows: list[dict[str, object]],
    v2_market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
    raw_market_data: pd.DataFrame,
    event_calendar_df: pd.DataFrame,
) -> dict[date, object]:
    """Classify every golden date in ONE classify_window pass.

    Previously this looped over rows and called ``engine.classify`` per
    row, paying the full ``build_market_context`` + ``build_feature_store``
    cost ~10 times. classify_window emits a per-day timeline from a single
    pipeline run; we slice the requested golden dates out of its outputs.

    PIT correctness: classify_window's per-day emission is stateless-replay
    compliant — each emitted day's classifier state is computed using only
    data on or before that day. Golden dates before the V2 fixture start are
    V1 regression rows and are classified through the frozen V1 config instead
    of being silently filtered out.
    """
    engine = RegimeEngine()
    all_golden_dates = sorted(
        date.fromisoformat(str(row["as_of_date"])) for row in golden_rows
    )
    if not all_golden_dates:
        return {}
    end = all_golden_dates[-1]
    v2_market_data = v2_market_df_for_asof(end)
    v2_replay_start = date(2020, 1, 1)
    v1_dates = [d for d in all_golden_dates if d < v2_replay_start]
    v2_dates = [d for d in all_golden_dates if d >= v2_replay_start]
    outputs: dict[date, object] = {}

    if v1_dates:
        v1_end = max(v1_dates)
        v1_market_data = (
            raw_market_data[raw_market_data["date"] <= v1_end]
            .copy()
            .reset_index(drop=True)
        )
        v1_lookback_sessions = v1_market_data.loc[
            v1_market_data["symbol"] == "SPY", "date"
        ].nunique()
        v1_timeline = engine.classify_window(
            end_date=v1_end,
            market_data=v1_market_data,
            lookback_days=v1_lookback_sessions,
            config=load_regime_config(
                _REPO_ROOT
                / "src"
                / "regime_detection"
                / "configs"
                / "core3-v1.0.0.yaml"
            ),
            event_calendar=event_calendar_df,
        )
        v1_by_date = {out.as_of_date: out for out in v1_timeline.outputs}
        missing_v1 = [d for d in v1_dates if d not in v1_by_date]
        if missing_v1:
            raise RuntimeError(
                f"classify_window did not emit V1 golden dates: {missing_v1!r}"
            )
        outputs.update({d: v1_by_date[d] for d in v1_dates})

    if v2_dates:
        market_data = v2_market_data
        earliest = v2_dates[0]
        span_days = (end - earliest).days
        lookback_sessions = max(1, int(span_days / 365.25 * 252) + 220)
        available_sessions = market_data.loc[
            market_data["symbol"] == "SPY",
            "date",
        ].nunique()
        lookback_sessions = min(lookback_sessions, available_sessions)
        kwargs = synthetic_v2_kwargs_for_market_data(market_data)
        timeline = engine.classify_window(
            end_date=end,
            market_data=market_data,
            lookback_days=lookback_sessions,
            config=kwargs["config"],
            event_calendar=kwargs["event_calendar"],
            sector_etf_closes=kwargs["sector_etf_closes"],
            cross_asset_closes=kwargs["cross_asset_closes"],
            macro_series=kwargs["macro_series"],
            pit_constituent_intervals=kwargs["pit_constituent_intervals"],
            constituent_ohlcv=kwargs["constituent_ohlcv"],
        )
        by_date = {out.as_of_date: out for out in timeline.outputs}
        missing = [d for d in v2_dates if d not in by_date]
        if missing:
            raise RuntimeError(
                f"classify_window did not emit outputs for golden dates: {missing!r}. "
                f"Window end={end}, lookback_sessions={lookback_sessions}, "
                f"emitted span={timeline.outputs[0].as_of_date}..{timeline.outputs[-1].as_of_date}"
            )
        outputs.update({d: by_date[d] for d in v2_dates})

    missing_outputs = [d for d in all_golden_dates if d not in outputs]
    if missing_outputs:
        raise RuntimeError(f"golden dates were not classified: {missing_outputs!r}")
    return outputs


def _load_module_for_fixture(name: str, rel_path: str):
    import importlib.util

    script_path = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@pytest.fixture(scope="session")
def walkforward_2023_dec_template(
    tmp_path_factory: pytest.TempPathFactory,
    v2_macro_parquet_path: Path,
) -> Path:
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
    v2_daily_path = _REPO_ROOT / "tests" / "fixtures" / "raw" / "v2" / "daily_ohlcv.csv"
    config_path = _REPO_ROOT / "tests" / "fixtures" / "configs" / "core3-v2-fast.yaml"
    runner.run_walkforward(
        market_data_path=market_data_path,
        output_root=cache_dir,
        start_date=date(2023, 12, 12),
        end_date=date(2023, 12, 14),
        event_calendar_path=_EVENT_CALENDAR_PATH,
        config_path=config_path,
        v2_daily_ohlcv_path=v2_daily_path,
        macro_parquet_path=v2_macro_parquet_path,
    )
    return cache_dir


def _build_real_v2_classify_window(
    *,
    as_of: date,
    v2_market_df_for_asof,
    v2_sector_etf_closes: dict[str, pd.Series],
    v2_cross_asset_closes: dict[str, pd.Series],
    v2_macro_series_by_key: dict[str, pd.Series],
    v2_pit_constituent_intervals: pd.DataFrame,
    v2_constituent_ohlcv_by_symbol: dict[str, pd.DataFrame],
):
    """Compute classify_window once for the real V2 fixture at 2026-05-13
    with the canonical full V2 fixture bundle. The two
    integration tests asserting on this exact engine state (one via
    ``classify_window``, one via ``classify`` which delegates to
    ``classify_window(lookback_days=1).outputs[-1]`` — see
    ``test_classify_delegates_to_classify_window_with_single_day_lookback``)
    can share this result. Now economical because the joblib in-process
    monkeypatch in ``pytest_configure`` cut the build cost from ~90s to
    ~37s — small enough that the cross-worker setup-wait pays off.
    """
    return RegimeEngine().classify_window(
        end_date=as_of,
        market_data=v2_market_df_for_asof(as_of),
        lookback_days=1,
        config=_fast_v2_test_config(),
        event_calendar=load_event_calendar(_EVENT_CALENDAR_PATH),
        sector_etf_closes=v2_sector_etf_closes,
        cross_asset_closes=v2_cross_asset_closes,
        macro_series=v2_macro_series_by_key,
        pit_constituent_intervals=v2_pit_constituent_intervals,
        constituent_ohlcv=v2_constituent_ohlcv_by_symbol,
    )


@pytest.fixture(scope="session")
def real_v2_classify_window_2026_05_13(
    v2_market_df_for_asof,
    v2_sector_etf_closes,
    v2_cross_asset_closes,
    v2_macro_series_by_key,
    v2_pit_constituent_intervals,
    v2_constituent_ohlcv_by_symbol,
    tmp_path_factory: pytest.TempPathFactory,
    worker_id: str,
):
    """Session-scoped, cross-worker pickle-cached classify_window result for
    the real V2 fixture at as_of=2026-05-13 with sector + cross-asset
    closes (no macro). See ``_build_real_v2_classify_window``.
    """
    if worker_id == "master":
        return _build_real_v2_classify_window(
            as_of=date(2026, 5, 13),
            v2_market_df_for_asof=v2_market_df_for_asof,
            v2_sector_etf_closes=v2_sector_etf_closes,
            v2_cross_asset_closes=v2_cross_asset_closes,
            v2_macro_series_by_key=v2_macro_series_by_key,
            v2_pit_constituent_intervals=v2_pit_constituent_intervals,
            v2_constituent_ohlcv_by_symbol=v2_constituent_ohlcv_by_symbol,
        )

    shared_dir = tmp_path_factory.getbasetemp().parent
    cache_path = shared_dir / "real_v2_classify_window_2026_05_13.pkl"
    lock_path = shared_dir / "real_v2_classify_window_2026_05_13.lock"

    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        result = _build_real_v2_classify_window(
            as_of=date(2026, 5, 13),
            v2_market_df_for_asof=v2_market_df_for_asof,
            v2_sector_etf_closes=v2_sector_etf_closes,
            v2_cross_asset_closes=v2_cross_asset_closes,
            v2_macro_series_by_key=v2_macro_series_by_key,
            v2_pit_constituent_intervals=v2_pit_constituent_intervals,
            v2_constituent_ohlcv_by_symbol=v2_constituent_ohlcv_by_symbol,
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
def real_v2_classify_window_2026_05_12(
    v2_market_df_for_asof,
    v2_close_series_by_symbol,
    v2_macro_series_by_key,
    v2_pit_constituent_intervals,
    v2_constituent_ohlcv_by_symbol,
    tmp_path_factory: pytest.TempPathFactory,
    worker_id: str,
):
    """Session-scoped, cross-worker pickle-cached classify_window result for
    the real V2 fixture at as_of=2026-05-12 with the canonical full V2
    fixture bundle."""
    sector_etf_closes = {
        symbol: v2_close_series_by_symbol[symbol] for symbol in SECTOR_ETFS
    }
    # Include KRE (credit_funding) and XLY/XLI/XLP/XLU (inflation_growth) so the
    # ClassifyRequest validator at engine.py:259 sees all required cross_asset
    # inputs when those axes are configured. Without these, fixture build fails
    # with ValueError before the engine can classify.
    cross_asset_closes = {
        symbol: v2_close_series_by_symbol[symbol]
        for symbol in set(CROSS_ASSET_SYMBOLS) | {"KRE", "XLY", "XLI", "XLP", "XLU"}
    }
    if worker_id == "master":
        return _build_real_v2_classify_window(
            as_of=date(2026, 5, 12),
            v2_market_df_for_asof=v2_market_df_for_asof,
            v2_sector_etf_closes=sector_etf_closes,
            v2_cross_asset_closes=cross_asset_closes,
            v2_macro_series_by_key=v2_macro_series_by_key,
            v2_pit_constituent_intervals=v2_pit_constituent_intervals,
            v2_constituent_ohlcv_by_symbol=v2_constituent_ohlcv_by_symbol,
        )

    shared_dir = tmp_path_factory.getbasetemp().parent
    cache_path = shared_dir / "real_v2_classify_window_2026_05_12.pkl"
    lock_path = shared_dir / "real_v2_classify_window_2026_05_12.lock"

    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        result = _build_real_v2_classify_window(
            as_of=date(2026, 5, 12),
            v2_market_df_for_asof=v2_market_df_for_asof,
            v2_sector_etf_closes=sector_etf_closes,
            v2_cross_asset_closes=cross_asset_closes,
            v2_macro_series_by_key=v2_macro_series_by_key,
            v2_pit_constituent_intervals=v2_pit_constituent_intervals,
            v2_constituent_ohlcv_by_symbol=v2_constituent_ohlcv_by_symbol,
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
        "real_v2_classify_window_2026_05_12 build timed out waiting on "
        f"peer worker; cache_path={cache_path}"
    )


@pytest.fixture(scope="session")
def classified_golden_outputs(
    tmp_path_factory: pytest.TempPathFactory,
    golden_rows: list[dict[str, object]],
    v2_market_df_for_asof,
    synthetic_v2_kwargs_for_market_data,
    raw_market_data: pd.DataFrame,
    event_calendar_df: pd.DataFrame,
    worker_id: str,
) -> dict[date, object]:
    """Session-scoped fixture, shared across pytest-xdist workers via a disk
    pickle cache. The single-process path computes inline; the multi-worker
    path elects one worker to build via an exclusive-create lockfile, while
    other workers poll for the pickle to land. This eliminates the per-worker
    rebuild cost (~81s × N workers) that previously dominated wall-clock."""
    if worker_id == "master":
        return _classify_all_golden_rows(
            golden_rows,
            v2_market_df_for_asof,
            synthetic_v2_kwargs_for_market_data,
            raw_market_data,
            event_calendar_df,
        )

    shared_dir = tmp_path_factory.getbasetemp().parent
    cache_path = shared_dir / "classified_golden_outputs.pkl"
    lock_path = shared_dir / "classified_golden_outputs.lock"

    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    try:
        # O_CREAT | O_EXCL: atomic single-winner election across workers.
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        outputs = _classify_all_golden_rows(
            golden_rows,
            v2_market_df_for_asof,
            synthetic_v2_kwargs_for_market_data,
            raw_market_data,
            event_calendar_df,
        )
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
