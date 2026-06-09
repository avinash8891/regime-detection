from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

import scripts._v2_calibration_helpers as helpers
from scripts._v2_calibration_helpers import (
    CROSS_ASSET_SYMBOLS,
    default_pmi_path,
    load_close_dict,
    load_macro_series,
    load_market_data,
)
from regime_detection.credit_funding_rules import (
    REQUIRED_CROSS_ASSET_KEYS as CREDIT_FUNDING_CROSS_ASSET_KEYS,
)
from regime_detection.fragility_universe import (
    CROSS_ASSET_SYMBOLS as NETWORK_CROSS_ASSET_SYMBOLS,
)
from regime_detection.inflation_growth_rules import (
    REQUIRED_CROSS_ASSET_KEYS as INFLATION_GROWTH_CROSS_ASSET_KEYS,
)


def test_helper_cross_asset_symbols_are_canonical_network_symbols() -> None:
    assert CROSS_ASSET_SYMBOLS is NETWORK_CROSS_ASSET_SYMBOLS


def test_runner_cross_asset_symbols_are_derived_from_engine_sources() -> None:
    assert helpers.RUNNER_CROSS_ASSET_SYMBOLS == list(
        dict.fromkeys(
            [
                *NETWORK_CROSS_ASSET_SYMBOLS,
                *CREDIT_FUNDING_CROSS_ASSET_KEYS,
                *INFLATION_GROWTH_CROSS_ASSET_KEYS,
            ]
        )
    )


def test_default_pmi_path_uses_history_parquet(tmp_path: Path) -> None:
    assert default_pmi_path(tmp_path) == tmp_path / "pmi" / "us_ism_pmi_history.parquet"


def test_load_market_data_uses_true_vix_symbol(tmp_path: Path) -> None:
    daily_dir = tmp_path / "daily_ohlcv_762"
    for symbol in ("SPY", "RSP", "VIX", "VIXY"):
        symbol_dir = daily_dir / f"symbol={symbol}"
        symbol_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "date": "2026-05-15",
                    "symbol": symbol,
                    "open": 1.0,
                    "high": 2.0,
                    "low": 0.5,
                    "close": 1.5,
                    "volume": 100,
                    "adjusted_close": 1.5,
                }
            ]
        ).to_parquet(symbol_dir / "ohlcv.parquet", index=False)

    market_data = load_market_data(daily_dir)

    assert set(market_data["symbol"]) == {"SPY", "RSP", "VIX"}


def test_profile_input_loaders_are_clean_under_copy_on_write_warning_mode(
    tmp_path: Path,
) -> None:
    daily_dir = tmp_path / "daily_ohlcv_762"
    for symbol in ("SPY", "RSP", "VIX", "XLY"):
        symbol_dir = daily_dir / f"symbol={symbol}"
        symbol_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "date": "2026-05-15",
                    "symbol": symbol,
                    "open": 1.0,
                    "high": 2.0,
                    "low": 0.5,
                    "close": 1.5,
                    "volume": 100,
                    "adjusted_close": 1.5,
                }
            ]
        ).to_parquet(symbol_dir / "ohlcv.parquet", index=False)

    with pd.option_context("mode.copy_on_write", "warn"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            market_data = load_market_data(daily_dir)
            close_dict = load_close_dict(
                daily_dir,
                ["XLY"],
                pd.DatetimeIndex([pd.Timestamp("2026-05-15")]),
            )

    assert market_data["date"].to_list() == [pd.Timestamp("2026-05-15").date()] * 3
    assert close_dict["XLY"].to_list() == [1.5]


def test_load_market_data_clips_to_common_required_symbol_date(
    tmp_path: Path,
) -> None:
    daily_dir = tmp_path / "daily_ohlcv_762"
    rows_by_symbol = {
        "SPY": ["2026-05-15"],
        "RSP": ["2026-05-15"],
        "VIX": ["2026-05-15", "2026-05-22"],
    }
    for symbol, dates in rows_by_symbol.items():
        symbol_dir = daily_dir / f"symbol={symbol}"
        symbol_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "date": date,
                    "symbol": symbol,
                    "open": 1.0,
                    "high": 2.0,
                    "low": 0.5,
                    "close": 1.5,
                    "volume": 100,
                    "adjusted_close": 1.5,
                }
                for date in dates
            ]
        ).to_parquet(symbol_dir / "ohlcv.parquet", index=False)

    market_data = load_market_data(daily_dir)

    assert market_data["date"].max() == pd.Timestamp("2026-05-15").date()
    assert set(market_data["symbol"]) == {"SPY", "RSP", "VIX"}


def test_load_market_data_rejects_required_symbol_calendar_gap(
    tmp_path: Path,
) -> None:
    daily_dir = tmp_path / "daily_ohlcv_762"
    rows_by_symbol = {
        "SPY": ["2026-05-13", "2026-05-14", "2026-05-15"],
        "RSP": ["2026-05-13", "2026-05-15"],
        "VIX": ["2026-05-13", "2026-05-14", "2026-05-15"],
    }
    for symbol, dates in rows_by_symbol.items():
        symbol_dir = daily_dir / f"symbol={symbol}"
        symbol_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "date": date,
                    "symbol": symbol,
                    "open": 1.0,
                    "high": 2.0,
                    "low": 0.5,
                    "close": 1.5,
                    "volume": 100,
                    "adjusted_close": 1.5,
                }
                for date in dates
            ]
        ).to_parquet(symbol_dir / "ohlcv.parquet", index=False)

    try:
        load_market_data(daily_dir)
    except ValueError as exc:
        assert "daily OHLCV calendar coverage gap" in str(exc)
        assert "RSP" in str(exc)
        assert "2026-05-14" in str(exc)
    else:
        raise AssertionError("expected required symbol calendar gap to fail")


def test_load_close_dict_rejects_null_symbol_column_in_partition_file(
    tmp_path: Path,
) -> None:
    daily_dir = tmp_path / "daily_ohlcv_762"
    symbol_dir = daily_dir / "symbol=XLY"
    symbol_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": "2026-05-14",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 100,
                "adjusted_close": 1.5,
                "symbol": None,
            },
            {
                "date": "2026-05-15",
                "open": 2.0,
                "high": 3.0,
                "low": 1.5,
                "close": 2.5,
                "volume": 200,
                "adjusted_close": 2.5,
                "symbol": None,
            },
        ]
    ).to_parquet(symbol_dir / "ohlcv.parquet", index=False)

    try:
        load_close_dict(
            daily_dir,
            ["XLY"],
            pd.DatetimeIndex([pd.Timestamp("2026-05-14"), pd.Timestamp("2026-05-15")]),
        )
    except ValueError as exc:
        assert "daily OHLCV symbol contract violation" in str(exc)
        assert "XLY" in str(exc)
        assert "null" in str(exc)
    else:
        raise AssertionError("expected null symbol column to fail")


def test_load_close_dict_rejects_missing_symbol_column_in_partition_file(
    tmp_path: Path,
) -> None:
    daily_dir = tmp_path / "daily_ohlcv_762"
    symbol_dir = daily_dir / "symbol=XLY"
    symbol_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": "2026-05-14",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 100,
                "adjusted_close": 1.5,
            }
        ]
    ).to_parquet(symbol_dir / "ohlcv.parquet", index=False)

    try:
        load_close_dict(
            daily_dir,
            ["XLY"],
            pd.DatetimeIndex([pd.Timestamp("2026-05-14")]),
        )
    except ValueError as exc:
        assert "daily OHLCV symbol contract violation" in str(exc)
        assert "missing symbol column" in str(exc)
        assert "XLY" in str(exc)
    else:
        raise AssertionError("expected missing symbol column to fail")


def test_load_close_dict_rejects_mismatched_symbol_column_in_partition_file(
    tmp_path: Path,
) -> None:
    daily_dir = tmp_path / "daily_ohlcv_762"
    symbol_dir = daily_dir / "symbol=XLY"
    symbol_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": "2026-05-14",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 100,
                "adjusted_close": 1.5,
                "symbol": "XLU",
            }
        ]
    ).to_parquet(symbol_dir / "ohlcv.parquet", index=False)

    try:
        load_close_dict(
            daily_dir,
            ["XLY"],
            pd.DatetimeIndex([pd.Timestamp("2026-05-14")]),
        )
    except ValueError as exc:
        assert "daily OHLCV symbol contract violation" in str(exc)
        assert "expected XLY" in str(exc)
        assert "XLU" in str(exc)
    else:
        raise AssertionError("expected mismatched symbol column to fail")


def test_load_close_dict_rejects_calendar_gap_between_start_and_end(
    tmp_path: Path,
) -> None:
    daily_dir = tmp_path / "daily_ohlcv_762"
    symbol_dir = daily_dir / "symbol=XLY"
    symbol_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": date,
                "symbol": "XLY",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 100,
                "adjusted_close": 1.5,
            }
            for date in ("2026-05-13", "2026-05-15")
        ]
    ).to_parquet(symbol_dir / "ohlcv.parquet", index=False)

    try:
        load_close_dict(
            daily_dir,
            ["XLY"],
            pd.DatetimeIndex(
                [
                    pd.Timestamp("2026-05-13"),
                    pd.Timestamp("2026-05-14"),
                    pd.Timestamp("2026-05-15"),
                ]
            ),
        )
    except ValueError as exc:
        assert "daily OHLCV calendar coverage gap" in str(exc)
        assert "XLY" in str(exc)
        assert "2026-05-14" in str(exc)
    else:
        raise AssertionError("expected close calendar gap to fail")


def test_load_macro_series_merges_pmi_history_with_latest_parquet(
    tmp_path: Path,
) -> None:
    macro_path = tmp_path / "macro" / "fred_macro_series.parquet"
    macro_path.parent.mkdir()
    pd.DataFrame(
        [
            {
                "date": "2026-05-01",
                "value": 1.0,
                "series_id": "DGS10",
                "logical_name": "10y_yield",
            }
        ]
    ).to_parquet(macro_path, index=False)
    pmi_dir = tmp_path / "pmi"
    pmi_dir.mkdir()
    latest_path = pmi_dir / "us_ism_pmi.parquet"
    pd.DataFrame(
        [
            {
                "series_name": "manufacturing",
                "period": "2026-04",
                "value": 52.7,
                "release_timestamp": "2026-05-01T10:00:00-04:00",
                "source": "live",
                "source_url": "https://example.test/latest",
            }
        ]
    ).to_parquet(latest_path, index=False)
    pd.DataFrame(
        [
            {
                "series_name": "manufacturing",
                "period": "2026-03",
                "value": 50.3,
                "release_timestamp": "2026-04-01T10:00:00-04:00",
                "source": "live",
                "source_url": "https://example.test/history",
            },
            {
                "series_name": "services",
                "period": "2026-03",
                "value": 53.0,
                "release_timestamp": "2026-04-03T10:00:00-04:00",
                "source": "live",
                "source_url": "https://example.test/history",
            },
        ]
    ).to_parquet(pmi_dir / "us_ism_pmi_history.parquet", index=False)

    with pd.option_context("mode.copy_on_write", "warn"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            series = load_macro_series(
                macro_path,
                latest_path,
                cpi_nowcast_parquet=None,
                eps_weekly_history_parquet=None,
            )

    pmi = series["pmi_manufacturing"]
    assert list(pmi.index) == [
        pd.Timestamp("2026-04-01"),
        pd.Timestamp("2026-05-01"),
    ]
    assert list(pmi) == [50.3, 52.7]
