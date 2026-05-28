from __future__ import annotations

from pathlib import Path

import pandas as pd

from conftest import (
    load_spy_session_index_from_daily_tree,
    MissingLiveDataInput,
    resolve_live_data_inputs,
    write_profile_engine_manifest,
)
from regime_data_fetch.artifact_manifest import (
    ArtifactManifest,
    ManifestArtifact,
    write_manifest,
)
from regime_data_fetch.artifact_store import sha256_file


def _store_uri(root: Path, key: str) -> str:
    return (root.resolve() / key).as_uri()


def _write_symbol_ohlcv(daily_dir: Path, symbol: str, dates: list[str]) -> None:
    symbol_dir = daily_dir / f"symbol={symbol}"
    symbol_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "symbol": pd.Series([symbol] * len(dates), dtype="string"),
            "open": [100.0] * len(dates),
            "high": [101.0] * len(dates),
            "low": [99.0] * len(dates),
            "close": [100.5] * len(dates),
            "volume": [1_000_000] * len(dates),
        }
    ).to_parquet(symbol_dir / "ohlcv.parquet", index=False)


def test_spy_session_index_uses_spy_partition_not_common_market_window(
    tmp_path: Path,
) -> None:
    daily_dir = tmp_path / "daily_ohlcv_762"
    _write_symbol_ohlcv(
        daily_dir,
        "SPY",
        ["2026-05-11", "2026-05-12", "2026-05-13"],
    )
    _write_symbol_ohlcv(daily_dir, "RSP", ["2026-05-11", "2026-05-12"])
    _write_symbol_ohlcv(daily_dir, "VIX", ["2026-05-11", "2026-05-12"])

    sessions = load_spy_session_index_from_daily_tree(daily_dir)

    assert sessions.equals(
        pd.DatetimeIndex(pd.to_datetime(["2026-05-11", "2026-05-12", "2026-05-13"]))
    )


def test_resolve_live_data_inputs_uses_manifest_routed_daily_tree(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data" / "raw"
    manifest_path = write_profile_engine_manifest(tmp_path)

    resolved = resolve_live_data_inputs(
        manifest_path=manifest_path,
        data_root=data_root,
    )

    assert resolved.daily_dir == data_root / "daily_ohlcv_762"
    assert resolved.news_sentiment_parquet == (
        data_root / "news_sentiment" / "sf_fed_news_sentiment.parquet"
    )
    assert resolved.fomc_minutes_parquet is None


def test_resolve_live_data_inputs_reports_missing_materialized_daily_tree(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data" / "raw"
    manifest_path = write_profile_engine_manifest(tmp_path)

    resolved = resolve_live_data_inputs(
        manifest_path=manifest_path,
        data_root=data_root,
    )

    missing = resolved.require_materialized("daily_dir")

    assert missing == MissingLiveDataInput(
        field="daily_dir",
        path=data_root / "daily_ohlcv_762",
        reason="manifest-resolved path is not materialized locally",
    )


def test_resolve_live_data_inputs_reports_missing_optional_parquet(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "store"
    daily_source = (
        store_root / "canonical" / "daily_ohlcv_762" / "symbol=SPY" / "ohlcv.parquet"
    )
    daily_source.parent.mkdir(parents=True)
    daily_source.write_bytes(b"spy")
    news_source = (
        store_root / "canonical" / "news_sentiment" / "sf_fed_news_sentiment.parquet"
    )
    news_source.parent.mkdir(parents=True)
    news_source.write_bytes(b"news")
    manifest = ArtifactManifest(
        artifact_set="live-tests",
        created_at_utc="2026-05-28T00:00:00Z",
        storage_root=str(store_root),
        artifacts=[
            ManifestArtifact.from_dict(
                {
                    "name": "constituent_ohlcv_SPY",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root,
                        "canonical/daily_ohlcv_762/symbol=SPY/ohlcv.parquet",
                    ),
                    "local_path": "data/raw/daily_ohlcv_762/symbol=SPY/ohlcv.parquet",
                    "sha256": sha256_file(daily_source),
                    "required_for": ["profile_engine"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "fred_macro_series",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root, "canonical/macro/fred_macro_series.parquet"
                    ),
                    "local_path": "data/raw/macro/fred_macro_series.parquet",
                    "sha256": sha256_file(daily_source),
                    "required_for": ["profile_engine"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "sp500_pit_constituents",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root,
                        "canonical/pit_constituents/sp500_ticker_intervals.parquet",
                    ),
                    "local_path": "data/raw/pit_constituents/sp500_ticker_intervals.parquet",
                    "sha256": sha256_file(daily_source),
                    "required_for": ["profile_engine"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "event_calendar_us",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root, "canonical/event_calendar/us_events.yaml"
                    ),
                    "local_path": "data/raw/event_calendar/us_events.yaml",
                    "sha256": sha256_file(daily_source),
                    "required_for": ["profile_engine"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "sf_fed_news_sentiment",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root,
                        "canonical/news_sentiment/sf_fed_news_sentiment.parquet",
                    ),
                    "local_path": "data/raw/news_sentiment/sf_fed_news_sentiment.parquet",
                    "sha256": sha256_file(news_source),
                    "required_for": ["profile_engine"],
                }
            ),
            ManifestArtifact.from_dict(
                {
                    "name": "fomc_minutes",
                    "stage": "canonical",
                    "uri": _store_uri(
                        store_root,
                        "canonical/fomc_minutes/fomc_minutes.parquet",
                    ),
                    "local_path": "data/raw/fomc_minutes/fomc_minutes.parquet",
                    "sha256": sha256_file(news_source),
                    "required_for": ["profile_engine"],
                }
            ),
        ],
    )
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, manifest_path)
    data_root = tmp_path / "data" / "raw"
    (data_root / "daily_ohlcv_762" / "symbol=SPY").mkdir(parents=True, exist_ok=True)
    (data_root / "daily_ohlcv_762" / "symbol=SPY" / "ohlcv.parquet").write_bytes(b"spy")
    (data_root / "news_sentiment").mkdir(parents=True, exist_ok=True)
    (data_root / "news_sentiment" / "sf_fed_news_sentiment.parquet").write_bytes(
        b"news"
    )

    resolved = resolve_live_data_inputs(
        manifest_path=manifest_path,
        data_root=data_root,
    )

    missing = resolved.require_materialized("fomc_minutes_parquet")

    assert missing == MissingLiveDataInput(
        field="fomc_minutes_parquet",
        path=data_root / "fomc_minutes" / "fomc_minutes.parquet",
        reason="manifest-resolved path is not materialized locally",
    )
