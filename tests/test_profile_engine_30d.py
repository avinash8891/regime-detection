from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "profile_engine_30d.py"
    spec = importlib.util.spec_from_file_location("profile_engine_30d", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


profile_engine_30d = _load_script_module()


def test_profile_engine_rejects_non_positive_lookback_days(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["profile_engine_30d.py", "--lookback-days", "0"])

    with pytest.raises(SystemExit) as exc_info:
        profile_engine_30d.main()

    assert exc_info.value.code == 2


def test_profile_engine_loads_aaii_sentiment_when_present(tmp_path: Path) -> None:
    parquet_path = tmp_path / "aaii_sentiment.parquet"
    expected = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-07"),
                "publication_date": pd.Timestamp("2026-05-07"),
                "bullish": 0.35,
                "neutral": 0.30,
                "bearish": 0.35,
                "bull_bear_spread": 0.0,
                "bull_bear_spread_8w_ma": 22.0,
            }
        ]
    )
    expected.to_parquet(parquet_path, index=False)

    actual = profile_engine_30d._load_optional_aaii_sentiment(parquet_path)

    assert actual is not None
    pd.testing.assert_frame_equal(actual, expected)


def test_profile_engine_skips_aaii_sentiment_when_absent(tmp_path: Path) -> None:
    assert profile_engine_30d._load_optional_aaii_sentiment(
        tmp_path / "missing.parquet"
    ) is None


def test_profile_engine_loads_news_sentiment_when_present(tmp_path: Path) -> None:
    parquet_path = tmp_path / "sf_fed_news_sentiment.parquet"
    pd.DataFrame(
        [
            {"date": "2026-05-14", "news_sentiment": -0.1},
            {"date": "2026-05-15", "news_sentiment": 0.2},
        ]
    ).to_parquet(parquet_path, index=False)

    actual = profile_engine_30d._load_optional_news_sentiment(parquet_path)

    assert actual is not None
    assert actual.name == "news_sentiment"
    assert actual.index.tolist() == [
        pd.Timestamp("2026-05-14"),
        pd.Timestamp("2026-05-15"),
    ]
    assert actual.tolist() == [-0.1, 0.2]


def test_profile_engine_loads_central_bank_text_when_present(tmp_path: Path) -> None:
    fomc_path = tmp_path / "fomc_minutes.parquet"
    powell_path = tmp_path / "powell_speeches.parquet"
    pd.DataFrame(
        [
            {
                "release_timestamp": pd.Timestamp("2026-05-01T18:00:00Z"),
                "body_text": "inflation restrictive policy",
            }
        ]
    ).to_parquet(fomc_path, index=False)
    pd.DataFrame(
        [
            {
                "publication_timestamp": pd.Timestamp("2026-05-02"),
                "body_text": "growth accommodative policy",
            }
        ]
    ).to_parquet(powell_path, index=False)

    actual = profile_engine_30d._load_optional_central_bank_text_releases(
        fomc_path=fomc_path,
        powell_path=powell_path,
    )

    assert actual is not None
    assert len(actual) == 2
    assert set(actual["source"]) == {"fomc_minutes", "powell_speech"}


def test_profile_engine_loads_cpi_first_release_when_present(tmp_path: Path) -> None:
    parquet_path = tmp_path / "cpi_all_items_vintages.parquet"
    pd.DataFrame(
        [
            {
                "date": "2026-03-01",
                "value": 315.0,
                "realtime_start": "2026-04-10",
                "realtime_end": "2026-05-10",
            },
            {
                "date": "2026-03-01",
                "value": 316.0,
                "realtime_start": "2026-05-10",
                "realtime_end": None,
            },
        ]
    ).to_parquet(parquet_path, index=False)

    actual = profile_engine_30d._load_optional_cpi_first_release(parquet_path)

    assert actual is not None
    assert actual.name == "cpi_first_release"
    assert actual.index.tolist() == [pd.Timestamp("2026-04-10")]
    assert actual.tolist() == [315.0]
