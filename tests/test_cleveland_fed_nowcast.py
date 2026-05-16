"""Cleveland Fed inflation-nowcast fetcher tests (ADR 0006 / Log #48).

The data source is the Cleveland Fed "Inflation Nowcasting" page's
month-over-month webchart feed (``nowcast_month.json``) — a JSON archive of
one chart object per monthly vintage. These tests exercise the parser, the
parquet merge, and the download+run orchestration against a synthetic feed
shaped like the real one.

Network is never touched: the download path is exercised via ``file://``
source URLs pointing at on-disk fixtures.

Per ~/.claude testing rules: real series key (`cpi_nowcast`), real module
constants, realistic subcaptions / series names / month-over-month values,
no mocks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from regime_data_fetch.cleveland_fed_nowcast import (
    CPI_NOWCAST_PARQUET,
    DEFAULT_VALUE_SCALE,
    MANUAL_REL_PATH,
    NOWCAST_SERIES_NAME,
    ClevelandFedNowcastError,
    download_cleveland_fed_nowcast_json,
    extract_data_vintage,
    parse_cleveland_fed_nowcast_json,
    run_cleveland_fed_nowcast_fetch,
    update_cpi_nowcast_parquet,
)

_DATA_VINTAGE = "2026-05-13 00:00"


def _chart_obj(
    subcaption: str,
    cpi_values: list[str],
    *,
    series_name: str = NOWCAST_SERIES_NAME,
    include_series: bool = True,
    labels: list[str] | None = None,
) -> dict:
    """One FusionCharts-style chart object = one monthly nowcast vintage.

    ``cpi_values`` are the daily nowcast points (strings, possibly "" for
    not-yet-computed days), mirroring the real feed.
    """
    if labels is None:
        match = re.fullmatch(r"(?P<year>\d{4})-(?P<month>\d{1,2})", subcaption)
        assert match is not None
        start = pd.Timestamp(year=int(match.group("year")), month=int(match.group("month")), day=1)
        labels = [(start + pd.Timedelta(days=i)).strftime("%m/%d/%Y") for i in range(len(cpi_values))]
    dataset = [
        {
            "seriesname": "Core CPI Inflation",
            "data": [{"value": ""} for _ in cpi_values],
        },
        {
            "seriesname": "Actual CPI Inflation",
            "data": [{"value": ""} for _ in cpi_values],
        },
    ]
    if include_series:
        dataset.insert(
            0,
            {
                "seriesname": series_name,
                "data": [{"value": v} for v in cpi_values],
            },
        )
    return {
        "chart": {
            "subcaption": subcaption,
            "_comment": _DATA_VINTAGE,
            "caption": "Inflation Nowcasting",
            "yaxisname": "Month-over-month percent change",
        },
        "categories": [
            {"category": [{"label": label} for label in labels]}
        ],
        "dataset": dataset,
    }


def _feed_json(objs: list[dict]) -> str:
    return json.dumps(objs)


# A realistic 3-vintage feed: Nov/Dec 2019 + Jan 2020, month-over-month
# nowcast in percent, with the early daily points still empty.
_FIXTURE_OBJS = [
    _chart_obj("2019-11", ["", "0.21", "0.2284"]),
    _chart_obj("2019-12", ["", "0.2432", "0.2455", "0.2463"]),
    _chart_obj("2020-1", ["", "", "0.1571"]),
]
_FIXTURE_JSON = _feed_json(_FIXTURE_OBJS)


def _write_feed(tmp_path: Path, json_text: str = _FIXTURE_JSON) -> Path:
    json_path = tmp_path / MANUAL_REL_PATH
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text)
    return json_path


# --- parse ------------------------------------------------------------------


def test_parse_takes_last_nonempty_value_per_vintage_keyed_to_publication_date() -> None:
    df = parse_cleveland_fed_nowcast_json(_FIXTURE_JSON)
    assert list(df.columns) == ["date", "cpi_nowcast"]
    assert len(df) == 3
    assert list(df["date"]) == [
        pd.Timestamp("2019-11-03"),
        pd.Timestamp("2019-12-04"),
        pd.Timestamp("2020-01-03"),
    ]
    # The settled (last non-empty) nowcast, percent -> fraction.
    assert df.loc[0, "cpi_nowcast"] == pytest.approx(0.2284 * DEFAULT_VALUE_SCALE)
    assert df.loc[1, "cpi_nowcast"] == pytest.approx(0.2463 * DEFAULT_VALUE_SCALE)
    assert df.loc[2, "cpi_nowcast"] == pytest.approx(0.1571 * DEFAULT_VALUE_SCALE)


def test_parse_keys_nowcast_to_last_nonempty_publication_date() -> None:
    objs = [
        _chart_obj(
            "2024-3",
            ["", "0.30", "0.34"],
            labels=["03/01/2024", "03/07/2024", "03/12/2024"],
        )
    ]

    df = parse_cleveland_fed_nowcast_json(_feed_json(objs))

    assert list(df["date"]) == [pd.Timestamp("2024-03-12")]
    assert df.loc[0, "cpi_nowcast"] == pytest.approx(0.34 * DEFAULT_VALUE_SCALE)


def test_parse_accepts_month_day_category_labels_without_year() -> None:
    objs = [
        _chart_obj(
            "2025-9",
            ["", "0.30", "0.34"],
            labels=["09/01", "09/08", "09/13"],
        )
    ]

    df = parse_cleveland_fed_nowcast_json(_feed_json(objs))

    assert list(df["date"]) == [pd.Timestamp("2025-09-13")]
    assert df.loc[0, "cpi_nowcast"] == pytest.approx(0.34 * DEFAULT_VALUE_SCALE)


def test_parse_skips_vintage_with_no_nonempty_value() -> None:
    """The earliest feed vintages carry only PCE — an all-empty CPI series
    is skipped, not a parse failure."""
    objs = [
        _chart_obj("2013-7", ["", "", ""]),  # no CPI nowcast yet
        _chart_obj("2013-8", ["", "0.1249"]),
    ]
    df = parse_cleveland_fed_nowcast_json(_feed_json(objs))
    assert list(df["date"]) == [pd.Timestamp("2013-08-02")]


def test_parse_honours_custom_series_and_scale() -> None:
    objs = [_chart_obj("2024-3", ["", "0.31"], series_name="Core CPI Inflation")]
    df = parse_cleveland_fed_nowcast_json(
        _feed_json(objs), series_name="Core CPI Inflation", value_scale=1.0
    )
    assert df.loc[0, "cpi_nowcast"] == pytest.approx(0.31)


def test_parse_dedupes_by_date_keeping_last() -> None:
    objs = [
        _chart_obj("2024-3", ["", "0.30"]),
        _chart_obj("2024-3", ["", "0.34"]),  # same target month, later vintage
    ]
    df = parse_cleveland_fed_nowcast_json(_feed_json(objs))
    assert len(df) == 1
    assert df.loc[0, "cpi_nowcast"] == pytest.approx(0.34 * DEFAULT_VALUE_SCALE)


def test_parse_raises_on_invalid_json() -> None:
    with pytest.raises(ClevelandFedNowcastError, match="not valid JSON"):
        parse_cleveland_fed_nowcast_json("{not json")


def test_parse_raises_on_non_list_payload() -> None:
    with pytest.raises(ClevelandFedNowcastError, match="non-empty list"):
        parse_cleveland_fed_nowcast_json(json.dumps({"chart": {}}))


def test_parse_raises_on_chart_object_missing_keys() -> None:
    with pytest.raises(ClevelandFedNowcastError, match="missing 'chart'"):
        parse_cleveland_fed_nowcast_json(json.dumps([{"chart": {}}]))


def test_parse_raises_on_missing_subcaption() -> None:
    obj = _chart_obj("2024-3", ["", "0.30"])
    del obj["chart"]["subcaption"]
    with pytest.raises(ClevelandFedNowcastError, match="no chart.subcaption"):
        parse_cleveland_fed_nowcast_json(_feed_json([obj]))


def test_parse_raises_on_bad_subcaption_format() -> None:
    with pytest.raises(ClevelandFedNowcastError, match="unparseable chart subcaption"):
        parse_cleveland_fed_nowcast_json(
            _feed_json([_chart_obj("March2024", ["0.3"], labels=["03/01/2024"])])
        )


def test_parse_raises_on_missing_series() -> None:
    obj = _chart_obj("2024-3", ["", "0.30"], include_series=False)
    with pytest.raises(ClevelandFedNowcastError, match="no 'CPI Inflation'"):
        parse_cleveland_fed_nowcast_json(_feed_json([obj]))


def test_parse_raises_on_unparseable_value() -> None:
    with pytest.raises(ClevelandFedNowcastError, match="unparseable.*value"):
        parse_cleveland_fed_nowcast_json(_feed_json([_chart_obj("2024-3", ["", "n/a"])]))


def test_parse_raises_when_no_usable_vintages() -> None:
    objs = [_chart_obj("2013-7", ["", ""]), _chart_obj("2013-8", ["", ""])]
    with pytest.raises(ClevelandFedNowcastError, match="no usable"):
        parse_cleveland_fed_nowcast_json(_feed_json(objs))


def test_extract_data_vintage_returns_chart_comment() -> None:
    assert extract_data_vintage(_FIXTURE_JSON) == _DATA_VINTAGE
    assert extract_data_vintage("{not json") is None


# --- download (file:// — no network) ----------------------------------------


def test_download_writes_payload_from_file_url(tmp_path: Path) -> None:
    src = tmp_path / "feed_source.json"
    src.write_text(_FIXTURE_JSON)
    out_path = tmp_path / "downloaded.json"
    download_cleveland_fed_nowcast_json(
        out_path=out_path, source_url=src.as_uri()
    )
    assert out_path.read_text() == _FIXTURE_JSON


def test_download_raises_on_unreachable_source(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(ClevelandFedNowcastError, match="Failed to download"):
        download_cleveland_fed_nowcast_json(
            out_path=tmp_path / "out.json", source_url=missing.as_uri()
        )


# --- update_cpi_nowcast_parquet ---------------------------------------------


def test_update_creates_parquet_when_absent(tmp_path: Path) -> None:
    json_path = _write_feed(tmp_path)
    out_path = tmp_path / "cleveland_fed_nowcast" / CPI_NOWCAST_PARQUET
    df = update_cpi_nowcast_parquet(json_path=json_path, out_path=out_path)
    assert out_path.exists()
    assert len(df) == 3
    reloaded = pd.read_parquet(out_path)
    assert list(reloaded.columns) == ["date", "cpi_nowcast"]


def test_update_merges_and_supersedes_existing(tmp_path: Path) -> None:
    """A re-fetch carries revised values for recent months — the fresh
    parse supersedes same-date rows and appends new ones."""
    out_path = tmp_path / "cleveland_fed_nowcast" / CPI_NOWCAST_PARQUET
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Seed an existing parquet with a stale 2019-12 value.
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2019-11-03"), pd.Timestamp("2019-12-04")],
            "cpi_nowcast": [0.002284, 0.009999],
        }
    ).to_parquet(out_path, index=False)

    json_path = _write_feed(tmp_path)  # Nov/Dec 2019 + Jan 2020
    df = update_cpi_nowcast_parquet(json_path=json_path, out_path=out_path)

    assert len(df) == 3  # Nov superseded, Dec superseded, Jan appended
    dec = df.loc[df["date"] == pd.Timestamp("2019-12-04"), "cpi_nowcast"].iloc[0]
    # Stale 0.009999 replaced by the fresh parse (0.2463 percent -> fraction).
    assert dec == pytest.approx(0.2463 * DEFAULT_VALUE_SCALE)


def test_update_raises_when_json_absent(tmp_path: Path) -> None:
    with pytest.raises(ClevelandFedNowcastError, match="No Cleveland Fed nowcast JSON"):
        update_cpi_nowcast_parquet(
            json_path=tmp_path / "missing.json",
            out_path=tmp_path / "out.parquet",
        )


# --- run_cleveland_fed_nowcast_fetch ----------------------------------------


def test_run_fetch_downloads_parses_and_reports(tmp_path: Path) -> None:
    src = tmp_path / "feed_source.json"
    src.write_text(_FIXTURE_JSON)
    report_path = run_cleveland_fed_nowcast_fetch(
        out_dir=tmp_path, source_url=src.as_uri()
    )

    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["rows"] == 3
    assert report["min_date"] == "2019-11-03"
    assert report["max_date"] == "2020-01-03"
    assert report["data_vintage"] == _DATA_VINTAGE
    assert report["series_name"] == NOWCAST_SERIES_NAME
    assert report["value_scale"] == DEFAULT_VALUE_SCALE

    parquet_path = Path(report["paths"]["cpi_nowcast_parquet"])
    assert parquet_path.exists()
    assert len(pd.read_parquet(parquet_path)) == 3
    # The downloaded feed was persisted at the canonical manual-drop path.
    assert (tmp_path / MANUAL_REL_PATH).exists()


def test_run_fetch_falls_back_to_present_json_on_download_failure(
    tmp_path: Path,
) -> None:
    """If the download fails but a JSON is already on disk (prior download
    or manual drop), the fetch parses that instead of hard-failing."""
    _write_feed(tmp_path)  # pre-place the feed at the canonical path
    missing_src = tmp_path / "unreachable.json"
    report_path = run_cleveland_fed_nowcast_fetch(
        out_dir=tmp_path, source_url=missing_src.as_uri()
    )
    report = json.loads(report_path.read_text())
    assert report["rows"] == 3


def test_run_fetch_raises_when_download_fails_and_no_local_json(
    tmp_path: Path,
) -> None:
    missing_src = tmp_path / "unreachable.json"
    with pytest.raises(ClevelandFedNowcastError, match="Failed to download"):
        run_cleveland_fed_nowcast_fetch(
            out_dir=tmp_path, source_url=missing_src.as_uri()
        )
