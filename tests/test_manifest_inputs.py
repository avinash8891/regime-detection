from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from regime_data_fetch.manifest_inputs import (
    ManifestInputResolutionError,
    resolve_runner_input_paths,
)


SHA = "0" * 64


def _artifact(name: str, local_path: str, required_for: list[str] | None = None):
    return {
        "name": name,
        "stage": "canonical",
        "uri": f"s3://bucket/{local_path}",
        "local_path": local_path,
        "sha256": SHA,
        "schema_version": None,
        "rows": 1,
        "min_date": None,
        "max_date": None,
        "required_for": required_for or ["profile_engine_30d"],
    }


def _write_manifest(tmp_path: Path, artifacts: list[dict]) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "artifact_set": "test",
                "created_at_utc": "2026-05-17T00:00:00Z",
                "storage_root": "s3://bucket/root",
                "artifacts": artifacts,
            },
            sort_keys=False,
        )
    )
    return path


def test_resolve_runner_input_paths_uses_manifest_artifact_names(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            _artifact(
                "constituent_ohlcv_AAPL",
                "data/raw/daily_ohlcv_762/symbol=AAPL/ohlcv.parquet",
            ),
            _artifact("fred_macro_series", "data/raw/macro/fred_macro_series.parquet"),
            _artifact(
                "sp500_pit_constituents",
                "data/raw/pit_constituents/sp500_ticker_intervals.parquet",
            ),
            _artifact("event_calendar_us", "data/raw/event_calendar/us_events.yaml"),
            _artifact("ism_pmi_history", "data/raw/pmi/us_ism_pmi_history.parquet"),
            _artifact(
                "sf_fed_news_sentiment",
                "data/raw/news_sentiment/sf_fed_news_sentiment.parquet",
            ),
            _artifact("aaii_sentiment", "data/raw/sentiment/aaii_sentiment.parquet"),
            _artifact("fomc_minutes", "data/raw/fomc_minutes/fomc_minutes.parquet"),
            _artifact(
                "powell_speeches",
                "data/raw/powell_speeches/powell_speeches.parquet",
            ),
            _artifact(
                "cpi_all_items_vintages",
                "data/raw/macro_vintages/cpi_all_items_vintages.parquet",
            ),
        ],
    )
    data_root = tmp_path / "materialized" / "data" / "raw"

    resolved = resolve_runner_input_paths(
        manifest_path=manifest_path,
        data_root=data_root,
        runner_name="profile_engine_30d",
        cli_values={},
        cli_overrides=set(),
        repo_root=tmp_path,
    )

    assert resolved.daily_dir == data_root / "daily_ohlcv_762"
    assert resolved.constituent_tree == data_root / "daily_ohlcv_762"
    assert resolved.macro_parquet == data_root / "macro" / "fred_macro_series.parquet"
    assert resolved.pit_parquet == (
        data_root / "pit_constituents" / "sp500_ticker_intervals.parquet"
    )
    assert resolved.event_calendar == data_root / "event_calendar" / "us_events.yaml"
    assert resolved.pmi_path == data_root / "pmi" / "us_ism_pmi_history.parquet"
    assert resolved.news_sentiment_parquet == (
        data_root / "news_sentiment" / "sf_fed_news_sentiment.parquet"
    )
    assert "news_sentiment_parquet" in resolved.resolved_from_manifest


def test_resolve_runner_input_paths_accepts_emitted_partition_tree_names(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            _artifact(
                "daily_ohlcv_parquet_symbol=SPY_ohlcv.parquet",
                "data/raw/daily_ohlcv_762/symbol=SPY/ohlcv.parquet",
            ),
            _artifact("fred_macro_series", "data/raw/macro/fred_macro_series.parquet"),
            _artifact(
                "sp500_pit_constituents",
                "data/raw/pit_constituents/sp500_ticker_intervals.parquet",
            ),
            _artifact("event_calendar_us", "data/raw/event_calendar/us_events.yaml"),
        ],
    )
    data_root = tmp_path / "materialized" / "data" / "raw"

    resolved = resolve_runner_input_paths(
        manifest_path=manifest_path,
        data_root=data_root,
        runner_name="profile_engine_30d",
        cli_values={},
        cli_overrides=set(),
        repo_root=tmp_path,
    )

    assert resolved.daily_dir == data_root / "daily_ohlcv_762"
    assert resolved.constituent_tree == data_root / "daily_ohlcv_762"


def test_resolve_runner_input_paths_respects_cli_override(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            _artifact(
                "constituent_ohlcv_AAPL",
                "data/raw/daily_ohlcv_762/symbol=AAPL/ohlcv.parquet",
            ),
            _artifact("fred_macro_series", "data/raw/macro/fred_macro_series.parquet"),
            _artifact(
                "sp500_pit_constituents",
                "data/raw/pit_constituents/sp500_ticker_intervals.parquet",
            ),
            _artifact("event_calendar_us", "data/raw/event_calendar/us_events.yaml"),
            _artifact(
                "sf_fed_news_sentiment",
                "data/raw/news_sentiment/sf_fed_news_sentiment.parquet",
            ),
        ],
    )
    override_path = tmp_path / "manual" / "news.parquet"

    resolved = resolve_runner_input_paths(
        manifest_path=manifest_path,
        data_root=tmp_path / "data" / "raw",
        runner_name="profile_engine_30d",
        cli_values={"news_sentiment_parquet": override_path},
        cli_overrides={"news_sentiment_parquet"},
    )

    assert resolved.news_sentiment_parquet == override_path
    assert "news_sentiment_parquet" in resolved.cli_overrides
    assert "news_sentiment_parquet" not in resolved.resolved_from_manifest


def test_resolve_runner_input_paths_fails_without_constituent_tree(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            _artifact("fred_macro_series", "data/raw/macro/fred_macro_series.parquet"),
            _artifact(
                "sp500_pit_constituents",
                "data/raw/pit_constituents/sp500_ticker_intervals.parquet",
            ),
        ],
    )

    with pytest.raises(
        ManifestInputResolutionError,
        match="missing required daily OHLCV artifacts",
    ):
        resolve_runner_input_paths(
            manifest_path=manifest_path,
            data_root=tmp_path / "data" / "raw",
            runner_name="profile_engine_30d",
            cli_values={},
            cli_overrides=set(),
        )
