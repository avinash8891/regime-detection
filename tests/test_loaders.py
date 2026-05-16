from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from regime_detection.loaders import (
    load_cpi_nowcast_series,
    load_cpi_vintages_first_release,
    load_cross_asset_closes,
    load_event_calendar,
    load_macro_series,
    load_news_sentiment_series,
    load_sector_etf_closes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SECTOR_ETFS = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")


def _sector_etf_df(*symbols: str) -> pd.DataFrame:
    dates = [date(2026, 1, 2), date(2026, 1, 3)]
    rows = []
    for sym in symbols:
        for d, price in zip(dates, [100.0, 101.5]):
            rows.append({"date": d, "symbol": sym, "close": price})
    return pd.DataFrame(rows)


def _macro_df(*series_ids: str) -> pd.DataFrame:
    rows = []
    for sid in series_ids:
        rows.append({"date": date(2026, 1, 2), "series_id": sid, "value": 4.25})
        rows.append({"date": date(2026, 1, 3), "series_id": sid, "value": 4.30})
    return pd.DataFrame(rows)


def _event_row(
    *,
    event_type: str = "FOMC",
    market: str = "US",
    importance: str = "high",
    event_date: date = date(2026, 1, 29),
) -> dict:
    return {
        "date": event_date,
        "market": market,
        "type": event_type,
        "importance": importance,
    }


# ---------------------------------------------------------------------------
# load_sector_etf_closes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_sector_etf_closes_rejects_malformed_dates() -> None:
    source = pd.DataFrame(
        {
            "date": ["not-a-date"],
            "symbol": ["XLB"],
            "close": [100.0],
        }
    )

    with pytest.raises(ValueError, match="malformed date"):
        load_sector_etf_closes(source)


@pytest.mark.unit
def test_load_sector_etf_closes_rejects_non_numeric_close_values() -> None:
    source = pd.DataFrame(
        {
            "date": [date(2026, 1, 2)],
            "symbol": ["XLB"],
            "close": ["bad-close"],
        }
    )

    with pytest.raises(ValueError, match="non-numeric close"):
        load_sector_etf_closes(source)


@pytest.mark.unit
def test_load_sector_etf_closes_happy_path_returns_series_per_symbol() -> None:
    source = _sector_etf_df("XLB", "XLK")
    result = load_sector_etf_closes(source)

    assert set(result.keys()) == {"XLB", "XLK"}
    assert isinstance(result["XLB"], pd.Series)
    assert len(result["XLB"]) == 2
    assert result["XLB"].iloc[0] == pytest.approx(100.0)


@pytest.mark.unit
def test_load_sector_etf_closes_rejects_missing_columns() -> None:
    source = pd.DataFrame({"date": [date(2026, 1, 2)], "symbol": ["XLB"]})

    with pytest.raises(ValueError, match="missing required columns"):
        load_sector_etf_closes(source)


@pytest.mark.unit
def test_load_sector_etf_closes_universe_filter_raises_on_missing_symbol() -> None:
    source = _sector_etf_df("XLB")

    with pytest.raises(ValueError, match="missing required symbols"):
        load_sector_etf_closes(source, universe=["XLB", "XLK"])


@pytest.mark.unit
def test_load_sector_etf_closes_universe_filter_keeps_only_requested() -> None:
    source = _sector_etf_df("XLB", "XLK", "XLF")
    result = load_sector_etf_closes(source, universe=["XLB", "XLK"])

    assert set(result.keys()) == {"XLB", "XLK"}


@pytest.mark.unit
def test_load_sector_etf_closes_from_csv(tmp_path: "pytest.TempPathFactory") -> None:
    csv_path = tmp_path / "sector_etf.csv"
    _sector_etf_df("XLB", "XLE").to_csv(csv_path, index=False)
    result = load_sector_etf_closes(csv_path)

    assert set(result.keys()) == {"XLB", "XLE"}


@pytest.mark.unit
def test_load_sector_etf_closes_rejects_unsupported_extension(tmp_path) -> None:
    bad_path = tmp_path / "data.xlsx"
    bad_path.write_text("irrelevant")

    with pytest.raises(ValueError, match="Unsupported source"):
        load_sector_etf_closes(bad_path)


# ---------------------------------------------------------------------------
# load_cross_asset_closes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_cross_asset_closes_returns_series_per_symbol() -> None:
    # Use representative cross-asset tickers (TLT, GLD, USO, UUP)
    source = pd.DataFrame(
        [
            {"date": date(2026, 1, 2), "symbol": "TLT", "close": 93.50},
            {"date": date(2026, 1, 2), "symbol": "GLD", "close": 185.20},
            {"date": date(2026, 1, 3), "symbol": "TLT", "close": 94.00},
            {"date": date(2026, 1, 3), "symbol": "GLD", "close": 186.00},
        ]
    )
    result = load_cross_asset_closes(source)

    assert set(result.keys()) == {"TLT", "GLD"}
    assert isinstance(result["TLT"], pd.Series)


# ---------------------------------------------------------------------------
# load_macro_series
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_macro_series_rejects_malformed_dates() -> None:
    source = pd.DataFrame(
        {
            "date": ["not-a-date"],
            "series_id": ["DGS10"],
            "value": [4.25],
        }
    )

    with pytest.raises(ValueError, match="malformed date"):
        load_macro_series(source)


@pytest.mark.unit
def test_load_macro_series_rejects_non_numeric_values() -> None:
    source = pd.DataFrame(
        {
            "date": [date(2026, 1, 2)],
            "series_id": ["DGS10"],
            "value": ["bad-value"],
        }
    )

    with pytest.raises(ValueError, match="non-numeric value"):
        load_macro_series(source)


@pytest.mark.unit
def test_load_macro_series_happy_path_returns_series_keyed_by_series_id() -> None:
    source = _macro_df("DGS10", "DGS2")
    result = load_macro_series(source)

    assert "DGS10" in result
    assert "DGS2" in result
    assert isinstance(result["DGS10"], pd.Series)
    assert len(result["DGS10"]) == 2


@pytest.mark.unit
def test_load_macro_series_adds_lowercase_dgs10_alias() -> None:
    source = _macro_df("DGS10")
    result = load_macro_series(source)

    assert "dgs10" in result
    assert result["dgs10"].name == "dgs10"


@pytest.mark.unit
def test_load_macro_series_adds_lowercase_dgs2_alias() -> None:
    source = _macro_df("DGS2")
    result = load_macro_series(source)

    assert "dgs2" in result


@pytest.mark.unit
def test_load_macro_series_logical_name_column_adds_extra_keys() -> None:
    source = pd.DataFrame(
        [
            {
                "date": date(2026, 1, 2),
                "series_id": "SOFR",
                "value": 5.30,
                "logical_name": "sofr",
            },
            {
                "date": date(2026, 1, 3),
                "series_id": "SOFR",
                "value": 5.32,
                "logical_name": "sofr",
            },
        ]
    )
    result = load_macro_series(source)

    assert "SOFR" in result
    assert "sofr" in result
    assert isinstance(result["sofr"], pd.Series)


@pytest.mark.unit
def test_load_macro_series_series_ids_filter_raises_on_missing() -> None:
    source = _macro_df("DGS10")

    with pytest.raises(ValueError, match="missing required series_ids"):
        load_macro_series(source, series_ids=["DGS10", "DGS2"])


@pytest.mark.unit
def test_load_macro_series_from_csv(tmp_path) -> None:
    csv_path = tmp_path / "fred_macro.csv"
    _macro_df("NFCI").to_csv(csv_path, index=False)
    result = load_macro_series(csv_path)

    assert "NFCI" in result


# ---------------------------------------------------------------------------
# load_cpi_nowcast_series
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_cpi_nowcast_series_happy_path() -> None:
    source = pd.DataFrame(
        {
            "date": [date(2026, 1, 2), date(2026, 1, 3)],
            "cpi_nowcast": [2.8, 2.9],
        }
    )
    result = load_cpi_nowcast_series(source)

    assert isinstance(result, pd.Series)
    assert result.name == "cpi_nowcast"
    assert len(result) == 2
    assert result.iloc[0] == pytest.approx(2.8)


@pytest.mark.unit
def test_load_cpi_nowcast_series_rejects_missing_columns() -> None:
    source = pd.DataFrame({"date": [date(2026, 1, 2)], "inflation": [2.8]})

    with pytest.raises(ValueError, match="cpi_nowcast source missing required columns"):
        load_cpi_nowcast_series(source)


@pytest.mark.unit
def test_load_cpi_nowcast_series_from_csv(tmp_path) -> None:
    source = pd.DataFrame(
        {
            "date": [date(2026, 1, 2), date(2026, 1, 3)],
            "cpi_nowcast": [2.7, 2.8],
        }
    )
    csv_path = tmp_path / "cpi_nowcast.csv"
    source.to_csv(csv_path, index=False)
    result = load_cpi_nowcast_series(csv_path)

    assert isinstance(result, pd.Series)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# load_news_sentiment_series
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_news_sentiment_series_happy_path() -> None:
    source = pd.DataFrame(
        {
            "date": [date(2026, 1, 2), date(2026, 1, 3)],
            "news_sentiment": [0.12, -0.05],
        }
    )
    result = load_news_sentiment_series(source)

    assert isinstance(result, pd.Series)
    assert result.name == "news_sentiment"
    assert result.iloc[1] == pytest.approx(-0.05)


@pytest.mark.unit
def test_load_news_sentiment_series_rejects_missing_columns() -> None:
    source = pd.DataFrame({"date": [date(2026, 1, 2)], "sentiment": [0.1]})

    with pytest.raises(ValueError, match="news_sentiment source missing required columns"):
        load_news_sentiment_series(source)


@pytest.mark.unit
def test_load_news_sentiment_series_from_csv(tmp_path) -> None:
    source = pd.DataFrame(
        {
            "date": [date(2026, 1, 2)],
            "news_sentiment": [0.15],
        }
    )
    csv_path = tmp_path / "news_sentiment.csv"
    source.to_csv(csv_path, index=False)
    result = load_news_sentiment_series(csv_path)

    assert len(result) == 1


# ---------------------------------------------------------------------------
# load_cpi_vintages_first_release
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_cpi_vintages_first_release_picks_earliest_realtime_start() -> None:
    # Reference month 2026-01-01 has two vintages: first released 2026-01-15,
    # revised 2026-02-15. Loader must return the 2026-01-15 value.
    source = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "value": 310.0,
                "realtime_start": "2026-01-15",
                "realtime_end": "2026-02-14",
            },
            {
                "date": "2026-01-01",
                "value": 310.5,
                "realtime_start": "2026-02-15",
                "realtime_end": None,
            },
        ]
    )
    result = load_cpi_vintages_first_release(source)

    assert isinstance(result, pd.Series)
    assert result.name == "cpi_first_release"
    assert len(result) == 1
    assert result.iloc[0] == pytest.approx(310.0)


@pytest.mark.unit
def test_load_cpi_vintages_first_release_returns_empty_series_for_empty_source() -> None:
    source = pd.DataFrame(columns=["date", "value", "realtime_start"])
    result = load_cpi_vintages_first_release(source)

    assert isinstance(result, pd.Series)
    assert len(result) == 0
    assert result.name == "cpi_first_release"


@pytest.mark.unit
def test_load_cpi_vintages_first_release_rejects_missing_columns() -> None:
    source = pd.DataFrame({"date": ["2026-01-01"], "value": [310.0]})

    with pytest.raises(ValueError, match="cpi_vintages source missing required columns"):
        load_cpi_vintages_first_release(source)


@pytest.mark.unit
def test_load_cpi_vintages_first_release_index_is_sorted_by_realtime_start() -> None:
    source = pd.DataFrame(
        [
            {
                "date": "2026-02-01",
                "value": 311.0,
                "realtime_start": "2026-02-13",
                "realtime_end": None,
            },
            {
                "date": "2026-01-01",
                "value": 310.0,
                "realtime_start": "2026-01-15",
                "realtime_end": None,
            },
        ]
    )
    result = load_cpi_vintages_first_release(source)

    assert result.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# load_event_calendar
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_event_calendar_rejects_missing_required_columns() -> None:
    source = pd.DataFrame(
        {
            "date": [date(2026, 1, 2)],
            "market": ["US"],
            "type": ["FOMC"],
        }
    )

    with pytest.raises(ValueError, match=r"event_calendar missing required columns.*importance"):
        load_event_calendar(source)


@pytest.mark.unit
def test_load_event_calendar_happy_path_returns_valid_frame() -> None:
    source = pd.DataFrame([_event_row(event_type="FOMC")])
    result = load_event_calendar(source)

    required_cols = {"date", "market", "type", "importance", "publication_date", "window_days", "approved_label"}
    assert required_cols.issubset(set(result.columns))
    assert len(result) == 1


@pytest.mark.unit
def test_load_event_calendar_filters_to_us_and_global_market_rows() -> None:
    source = pd.DataFrame(
        [
            _event_row(market="US", event_type="FOMC"),
            _event_row(market="EU", event_type="ECB_decision"),
            _event_row(market="GLOBAL", event_type="geopolitical_event"),
        ]
    )
    result = load_event_calendar(source)

    assert set(result["market"].unique()).issubset({"US", "GLOBAL"})


@pytest.mark.unit
def test_load_event_calendar_rejects_unsupported_type() -> None:
    source = pd.DataFrame(
        [_event_row(event_type="UNKNOWN_TYPE")]
    )

    with pytest.raises(ValueError, match="unsupported types"):
        load_event_calendar(source)


@pytest.mark.unit
def test_load_event_calendar_sets_publication_date_90_days_before_for_scheduled_types() -> None:
    event_date = date(2026, 1, 29)
    source = pd.DataFrame([_event_row(event_type="FOMC", event_date=event_date)])
    result = load_event_calendar(source)

    expected_pub_date = event_date - timedelta(days=90)
    assert result.iloc[0]["publication_date"] == expected_pub_date


@pytest.mark.unit
def test_load_event_calendar_non_scheduled_publication_date_equals_event_date() -> None:
    event_date = date(2026, 3, 15)
    source = pd.DataFrame([_event_row(event_type="geopolitical_event", event_date=event_date)])
    result = load_event_calendar(source)

    assert result.iloc[0]["publication_date"] == event_date


@pytest.mark.unit
def test_load_event_calendar_parses_window_days_from_list() -> None:
    source = pd.DataFrame(
        [
            {
                "date": date(2026, 1, 29),
                "market": "US",
                "type": "FOMC",
                "importance": "high",
                "window_days": [-2, 1],
            }
        ]
    )
    result = load_event_calendar(source)

    assert result.iloc[0]["window_days"] == [-2, 1]


@pytest.mark.unit
def test_load_event_calendar_rejects_malformed_window_days() -> None:
    source = pd.DataFrame(
        [
            {
                "date": date(2026, 1, 29),
                "market": "US",
                "type": "FOMC",
                "importance": "high",
                "window_days": [1, 2, 3],  # 3 items — invalid
            }
        ]
    )

    with pytest.raises(ValueError, match="two-item list"):
        load_event_calendar(source)


@pytest.mark.unit
def test_load_event_calendar_from_yaml(tmp_path) -> None:
    import yaml

    data = {
        "events": [
            {
                "date": "2026-01-29",
                "market": "US",
                "type": "FOMC",
                "importance": "high",
            }
        ]
    }
    yaml_path = tmp_path / "events.yaml"
    yaml_path.write_text(yaml.safe_dump(data))
    result = load_event_calendar(yaml_path)

    assert len(result) == 1
    assert result.iloc[0]["type"] == "FOMC"


@pytest.mark.unit
def test_load_event_calendar_from_yaml_list_form(tmp_path) -> None:
    import yaml

    data = [
        {
            "date": "2026-03-19",
            "market": "US",
            "type": "CPI",
            "importance": "high",
        }
    ]
    yaml_path = tmp_path / "events_list.yml"
    yaml_path.write_text(yaml.safe_dump(data))
    result = load_event_calendar(yaml_path)

    assert len(result) == 1


@pytest.mark.unit
def test_load_event_calendar_from_csv(tmp_path) -> None:
    source = pd.DataFrame([_event_row(event_type="NFP")])
    csv_path = tmp_path / "events.csv"
    source.to_csv(csv_path, index=False)
    result = load_event_calendar(csv_path)

    assert len(result) == 1
    assert result.iloc[0]["type"] == "NFP"


@pytest.mark.unit
def test_load_event_calendar_rejects_unsupported_path_extension(tmp_path) -> None:
    bad_path = tmp_path / "events.json"
    bad_path.write_text("{}")

    with pytest.raises(ValueError, match="Unsupported event calendar source"):
        load_event_calendar(bad_path)


@pytest.mark.unit
def test_load_event_calendar_approved_label_preserved_when_provided() -> None:
    source = pd.DataFrame(
        [
            {
                "date": date(2026, 1, 29),
                "market": "US",
                "type": "ad_hoc",
                "importance": "low",
                "approved_label": "risk_off",
            }
        ]
    )
    result = load_event_calendar(source)

    assert result.iloc[0]["approved_label"] == "risk_off"


@pytest.mark.unit
def test_load_event_calendar_malformed_publication_date_raises() -> None:
    source = pd.DataFrame(
        [
            {
                "date": date(2026, 1, 29),
                "market": "US",
                "type": "ad_hoc",
                "importance": "high",
                "publication_date": "not-a-date",
            }
        ]
    )

    with pytest.raises(ValueError, match="malformed publication_date"):
        load_event_calendar(source)
