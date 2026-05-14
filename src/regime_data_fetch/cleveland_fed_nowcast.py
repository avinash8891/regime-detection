"""Cleveland Fed inflation-nowcast fetcher (ADR 0006 / Ambiguity Log #48).

Produces the ``cpi_nowcast`` series the v2 §2B ``inflation_surprise_zscore``
feature consumes. ADR 0006 picked the free Cleveland Fed inflation nowcast
as the substitute for the (paid) analyst-survey ``consensus_estimate``.

The data source is the Cleveland Fed "Inflation Nowcasting" page's
month-over-month webchart feed — a single JSON file holding the full
historical archive (one chart object per monthly vintage, ~2013-08 to
present). It is fetched directly over HTTPS via ``urllib``; the manual-drop
path is only a fallback for when the network call fails.

JSON shape (FusionCharts export, verified 2026-05):

    [
      {
        "chart": {"subcaption": "2019-12", "_comment": "2026-05-13 00:00", ...},
        "categories": [{"category": [{"label": "12/02"}, ...]}],
        "dataset": [
          {"seriesname": "CPI Inflation", "data": [{"value": "0.243..."}, ...]},
          {"seriesname": "Core CPI Inflation", ...},
          {"seriesname": "PCE Inflation", ...},
          {"seriesname": "Core PCE Inflation", ...},
          {"seriesname": "Actual CPI Inflation", ...},
          ...
        ]
      },
      ...
    ]

Each chart object is one monthly nowcast vintage. ``chart.subcaption`` is
the target month (``YYYY-M``); the ``CPI Inflation`` series holds the daily
evolution of the nowcast within that vintage's nowcasting period. We take
the **last non-empty** value per vintage — the settled nowcast right before
the BLS release — as that month's ``cpi_nowcast``.

Dating convention (ADR 0006): each vintage is keyed to the **1st of its
target month**, matching FRED ``CPIAUCSL``'s reference-date convention, so
that ``compute_inflation_surprise_zscore``'s ``realized - nowcast`` stays
like-for-like (both operands forward-fill on the same monthly anchor).

Unit: the feed publishes month-over-month inflation in *percent*;
``DEFAULT_VALUE_SCALE = 0.01`` converts to the fractional monthly rate the
z-score expects. Parse failures raise ``ClevelandFedNowcastError`` loudly
rather than producing a silently-wrong series.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

SOURCE_NAME = "Cleveland Fed inflation nowcast"
# Month-over-month webchart feed backing the "Inflation Nowcasting" page.
# Verified reachable over urllib/curl (the human-facing HTML page is the
# one that 403s programmatic clients — this media endpoint does not).
SOURCE_URL = (
    "https://www.clevelandfed.org/-/media/files/webcharts/"
    "inflationnowcasting/nowcast_month.json?sc_lang=en"
)
CPI_NOWCAST_PARQUET = "cpi_nowcast.parquet"

# Manual-drop fallback path — same convention as the spdji EPS workbook
# (data/raw/<vendor>/<file>). Only used when the direct download fails: the
# operator saves the nowcast_month.json there and re-runs.
MANUAL_REL_PATH = Path("cleveland_fed_nowcast") / "nowcast_month.json"

# The dataset series carrying the CPI month-over-month nowcast. The feed
# also carries Core CPI / PCE / Core PCE and matching "Actual ..." series;
# the series name is parameterised so an operator can switch measures, but
# the §2B feature consumes headline CPI.
NOWCAST_SERIES_NAME = "CPI Inflation"
# Percent -> fraction. The feed publishes month-over-month inflation in
# percent (e.g. 0.243 = +0.243% m/m); compute_inflation_surprise_zscore
# subtracts the nowcast from a fractional 21-session % change of CPIAUCSL.
DEFAULT_VALUE_SCALE = 0.01

_log = logging.getLogger(__name__)


class ClevelandFedNowcastError(RuntimeError):
    pass


def _parse_subcaption_to_month_start(subcaption: str) -> pd.Timestamp:
    """``"2019-12"`` -> ``Timestamp("2019-12-01")`` (target-month anchor)."""
    parts = subcaption.strip().split("-")
    if len(parts) != 2:
        raise ClevelandFedNowcastError(
            f"Cleveland Fed nowcast: unparseable chart subcaption "
            f"{subcaption!r} (expected 'YYYY-M')"
        )
    try:
        year, month = int(parts[0]), int(parts[1])
        return pd.Timestamp(year=year, month=month, day=1)
    except (ValueError, TypeError) as exc:
        raise ClevelandFedNowcastError(
            f"Cleveland Fed nowcast: unparseable chart subcaption "
            f"{subcaption!r} (expected 'YYYY-M')"
        ) from exc


def parse_cleveland_fed_nowcast_json(
    json_text: str,
    *,
    series_name: str = NOWCAST_SERIES_NAME,
    value_scale: float = DEFAULT_VALUE_SCALE,
) -> pd.DataFrame:
    """Parse the Cleveland Fed month-over-month nowcast webchart JSON into a
    clean two-column DataFrame: ``date`` (Timestamp, 1st of the target
    month) and ``cpi_nowcast`` (float, fractional monthly rate after
    ``value_scale``).

    For each chart object (one monthly vintage): the target month comes from
    ``chart.subcaption`` and the nowcast value is the **last non-empty**
    point of the ``series_name`` dataset series — the settled nowcast right
    before the BLS release. Vintages with no non-empty value for that series
    are skipped (the earliest vintages carry only PCE, no CPI).

    A structurally wrong payload raises ``ClevelandFedNowcastError`` so a
    feed-shape drift fails loudly rather than producing a silently-wrong
    ``cpi_nowcast`` series.
    """
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ClevelandFedNowcastError(
            f"Cleveland Fed nowcast feed was not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, list) or not payload:
        raise ClevelandFedNowcastError(
            "Cleveland Fed nowcast feed was not a non-empty list of chart "
            "objects"
        )

    rows: list[dict[str, object]] = []
    for idx, obj in enumerate(payload):
        if not isinstance(obj, dict) or "chart" not in obj or "dataset" not in obj:
            raise ClevelandFedNowcastError(
                f"Cleveland Fed nowcast feed: chart object {idx} missing "
                f"'chart' / 'dataset' keys"
            )
        subcaption = obj["chart"].get("subcaption")
        if not subcaption:
            raise ClevelandFedNowcastError(
                f"Cleveland Fed nowcast feed: chart object {idx} has no "
                f"chart.subcaption (target month)"
            )
        target_month = _parse_subcaption_to_month_start(str(subcaption))

        series = next(
            (s for s in obj["dataset"] if s.get("seriesname") == series_name),
            None,
        )
        if series is None:
            raise ClevelandFedNowcastError(
                f"Cleveland Fed nowcast feed: chart object {idx} "
                f"({subcaption}) has no {series_name!r} dataset series; "
                f"found: {[s.get('seriesname') for s in obj['dataset']]}"
            )
        non_empty = [
            point["value"]
            for point in series.get("data", [])
            if str(point.get("value", "")).strip() != ""
        ]
        if not non_empty:
            # Earliest vintages carry no CPI nowcast — the series simply
            # starts later. Skip rather than fail.
            continue
        try:
            settled = float(non_empty[-1])
        except (ValueError, TypeError) as exc:
            raise ClevelandFedNowcastError(
                f"Cleveland Fed nowcast feed: chart object {idx} "
                f"({subcaption}) has an unparseable {series_name!r} value "
                f"{non_empty[-1]!r}"
            ) from exc
        rows.append({"date": target_month, "cpi_nowcast": settled * value_scale})

    if not rows:
        raise ClevelandFedNowcastError(
            f"Cleveland Fed nowcast feed held no usable {series_name!r} "
            f"vintages"
        )
    df = pd.DataFrame(rows)
    df = (
        df.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df


def extract_data_vintage(json_text: str) -> str | None:
    """Return the feed's ``chart._comment`` (its generation timestamp) for
    provenance, or None if absent / unparseable."""
    try:
        payload = json.loads(json_text)
        return payload[0]["chart"].get("_comment")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None


def download_cleveland_fed_nowcast_json(
    *,
    out_path: Path,
    source_url: str = SOURCE_URL,
    timeout_seconds: int = 60,
) -> Path:
    """Download the Cleveland Fed month-over-month nowcast webchart JSON to
    ``out_path``.

    The ``nowcast_month.json`` media endpoint is reachable over ``urllib``
    (unlike the human-facing HTML page, which 403s programmatic clients). On
    a network failure this raises ``ClevelandFedNowcastError`` routing the
    operator to the manual-drop fallback:

      1. Open ``SOURCE_URL`` in a browser and save the JSON.
      2. Copy it to ``data/raw/cleveland_fed_nowcast/nowcast_month.json``.
      3. Re-run the fetch — ``run_cleveland_fed_nowcast_fetch`` falls back to
         the already-present file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        source_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except urllib.error.URLError as exc:
        raise ClevelandFedNowcastError(
            f"Failed to download Cleveland Fed nowcast JSON from "
            f"{source_url}: {exc}. To complete the fetch manually: (1) open "
            f"the URL in a browser and save the JSON; (2) copy it to "
            f"data/raw/{MANUAL_REL_PATH}; (3) re-run the fetch — "
            f"run_cleveland_fed_nowcast_fetch falls back to the present file."
        ) from exc
    if not payload:
        raise ClevelandFedNowcastError(
            f"Cleveland Fed nowcast download from {source_url} returned an "
            f"empty payload"
        )
    out_path.write_bytes(payload)
    return out_path


def update_cpi_nowcast_parquet(
    *,
    json_path: Path,
    out_path: Path,
    series_name: str = NOWCAST_SERIES_NAME,
    value_scale: float = DEFAULT_VALUE_SCALE,
) -> pd.DataFrame:
    """Parse the nowcast JSON, merge it with any existing ``cpi_nowcast``
    parquet, dedupe by date, re-save.

    The merge keeps the freshly-parsed row on a date collision — a later
    feed snapshot carries revised nowcast values for recent months. Returns
    the full merged DataFrame, sorted ascending by date.
    """
    if not json_path.exists():
        raise ClevelandFedNowcastError(
            f"No Cleveland Fed nowcast JSON at {json_path}. The download "
            f"step should have written it; see "
            f"download_cleveland_fed_nowcast_json for the manual fallback."
        )
    parsed = parse_cleveland_fed_nowcast_json(
        json_path.read_text(),
        series_name=series_name,
        value_scale=value_scale,
    )
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing["date"] = pd.to_datetime(existing["date"])
        _log.info(
            "cleveland_fed_nowcast: loaded %d existing rows from %s",
            len(existing),
            out_path,
        )
        # Drop existing rows superseded by the fresh parse, then concat.
        existing = existing[~existing["date"].isin(parsed["date"])]
        combined = pd.concat([existing, parsed], ignore_index=True)
    else:
        combined = parsed
    combined = combined.sort_values("date").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    _log.info("cleveland_fed_nowcast: saved %d rows to %s", len(combined), out_path)
    return combined


def run_cleveland_fed_nowcast_fetch(
    *,
    out_dir: Path,
    source_url: str = SOURCE_URL,
    series_name: str = NOWCAST_SERIES_NAME,
    value_scale: float = DEFAULT_VALUE_SCALE,
) -> Path:
    """Orchestrate the Cleveland Fed inflation-nowcast fetch.

    Downloads the month-over-month webchart JSON to
    ``out_dir/cleveland_fed_nowcast/nowcast_month.json``, parses it, merges
    into ``cpi_nowcast.parquet``, and writes a report JSON. If the download
    fails but a JSON is already present (prior download or manual drop),
    that file is parsed instead.

    Cadence: the Cleveland Fed nowcast updates daily and the feed is the
    full archive, so a monthly re-fetch keeps ``cpi_nowcast`` current — the
    engine reads the most-recent monthly value carried forward.
    """
    nowcast_dir = out_dir / "cleveland_fed_nowcast"
    nowcast_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / MANUAL_REL_PATH
    out_path = nowcast_dir / CPI_NOWCAST_PARQUET

    try:
        download_cleveland_fed_nowcast_json(
            out_path=json_path, source_url=source_url
        )
    except ClevelandFedNowcastError:
        if not json_path.exists():
            raise
        _log.warning(
            "cleveland_fed_nowcast: download failed, falling back to the "
            "already-present JSON at %s",
            json_path,
        )

    df = update_cpi_nowcast_parquet(
        json_path=json_path,
        out_path=out_path,
        series_name=series_name,
        value_scale=value_scale,
    )

    report = {
        "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": SOURCE_NAME,
        "source_url": source_url,
        "source_path": str(json_path),
        "data_vintage": extract_data_vintage(json_path.read_text()),
        "series_name": series_name,
        "value_scale": value_scale,
        "rows": int(len(df)),
        "min_date": str(df["date"].min().date()) if not df.empty else None,
        "max_date": str(df["date"].max().date()) if not df.empty else None,
        "paths": {
            "cpi_nowcast_parquet": str(out_path),
        },
    }
    report_path = out_dir / "cleveland_fed_nowcast_fetch_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report_path
