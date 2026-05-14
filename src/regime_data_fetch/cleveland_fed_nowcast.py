"""Cleveland Fed inflation-nowcast fetcher (ADR 0006 / Ambiguity Log #48).

Produces the ``cpi_nowcast`` series the v2 §2B ``inflation_surprise_zscore``
feature consumes. ADR 0006 picked the free Cleveland Fed inflation nowcast
as the substitute for the (paid) analyst-survey ``consensus_estimate`` and
explicitly left the fetch path to the operator:

    "cpi_nowcast is NOT added to V2_FRED_SERIES — the Cleveland Fed nowcast
     is published on the Cleveland Fed site, not cleanly on FRED. The
     operator wires it into macro_series via a dedicated fetch path."

This module IS that dedicated fetch path. It mirrors the manual-drop
fallback architecture proven by ``aggregate_eps`` (the spdji workbook) and
``aaii_sentiment``: try a best-effort download, and on failure route the
operator to drop the file at a known path and re-run.

VERIFICATION NEEDED — the exact CSV schema of the Cleveland Fed export
cannot be confirmed without web access. Two things the operator MUST
verify on first run and pin via the parameters below:

  1. Column names — ``date_column`` / ``value_column`` (the CSV carries
     several inflation measures: CPI / Core CPI / PCE / Core PCE, each
     month-over-month and year-over-year).
  2. Published unit — ``value_scale``. ``compute_inflation_surprise_zscore``
     subtracts the nowcast from a *fractional* monthly % change of CPIAUCSL
     (e.g. 0.003 for +0.3% m/m). The Cleveland Fed publishes its nowcast in
     *percent* m/m, so the default ``value_scale=0.01`` converts percent to
     fraction. If the export is already fractional, pass ``value_scale=1.0``;
     if it is annualised, the operator must pick the m/m column instead.

Parse failures raise ``ClevelandFedNowcastError`` loudly rather than
producing a silently-wrong series.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

SOURCE_NAME = "Cleveland Fed inflation nowcast"
# The Cleveland Fed "Inflation Nowcasting" indicator page. This is the
# human-facing page, not a direct CSV endpoint — the operator passes the
# verified direct-download URL to ``download_cleveland_fed_nowcast_csv``.
SOURCE_URL = "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
CPI_NOWCAST_PARQUET = "cpi_nowcast.parquet"

# Manual-drop path — same convention as the spdji EPS workbook
# (data/raw/<vendor>/<file>). The operator downloads the CSV from the
# Cleveland Fed page and drops it here; ``run_cleveland_fed_nowcast_fetch``
# detects and parses it.
MANUAL_REL_PATH = Path("cleveland_fed_nowcast") / "cleveland_fed_nowcast.csv"

# Parameterised column mapping — see the module docstring's VERIFICATION
# NEEDED note. These defaults are a best guess and will raise loudly on a
# header mismatch.
DEFAULT_DATE_COLUMN = "date"
DEFAULT_VALUE_COLUMN = "CPI"
# Percent -> fraction. See the module docstring's unit note.
DEFAULT_VALUE_SCALE = 0.01

_log = logging.getLogger(__name__)


class ClevelandFedNowcastError(RuntimeError):
    pass


def parse_cleveland_fed_nowcast_csv(
    csv_text: str,
    *,
    date_column: str = DEFAULT_DATE_COLUMN,
    value_column: str = DEFAULT_VALUE_COLUMN,
    value_scale: float = DEFAULT_VALUE_SCALE,
) -> pd.DataFrame:
    """Parse a Cleveland Fed inflation-nowcast CSV export into a clean
    two-column DataFrame: ``date`` (Timestamp) and ``cpi_nowcast`` (float,
    a fractional monthly inflation rate after applying ``value_scale``).

    The column mapping is parameterised because the exact CSV schema of the
    Cleveland Fed export cannot be verified without web access (see the
    module docstring). A missing column or an unparseable cell raises
    ``ClevelandFedNowcastError`` so a schema drift fails loudly rather than
    silently producing a wrong ``cpi_nowcast`` series.

    Rows with a blank date or value cell are skipped (export footers /
    partial rows). Duplicate dates keep the last occurrence — the export is
    cumulative and a later row supersedes an earlier same-date estimate.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        raise ClevelandFedNowcastError("Cleveland Fed nowcast CSV had no header row")
    for required in (date_column, value_column):
        if required not in reader.fieldnames:
            raise ClevelandFedNowcastError(
                f"Cleveland Fed nowcast CSV missing column {required!r}; "
                f"found columns: {list(reader.fieldnames)}"
            )

    rows: list[dict[str, object]] = []
    for line_no, raw in enumerate(reader, start=2):
        raw_date = (raw.get(date_column) or "").strip()
        raw_value = (raw.get(value_column) or "").strip()
        if not raw_date or not raw_value:
            continue
        try:
            parsed_date = pd.Timestamp(raw_date)
        except (ValueError, TypeError) as exc:
            raise ClevelandFedNowcastError(
                f"Cleveland Fed nowcast CSV line {line_no}: unparseable "
                f"date {raw_date!r}"
            ) from exc
        try:
            parsed_value = float(raw_value)
        except ValueError as exc:
            raise ClevelandFedNowcastError(
                f"Cleveland Fed nowcast CSV line {line_no}: unparseable "
                f"value {raw_value!r}"
            ) from exc
        rows.append({"date": parsed_date, "cpi_nowcast": parsed_value * value_scale})

    if not rows:
        raise ClevelandFedNowcastError(
            "Cleveland Fed nowcast CSV contained no usable rows"
        )
    df = pd.DataFrame(rows)
    df = (
        df.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df


def download_cleveland_fed_nowcast_csv(
    *,
    out_path: Path,
    source_url: str,
    timeout_seconds: int = 60,
) -> Path:
    """Best-effort download of the Cleveland Fed inflation-nowcast CSV to
    ``out_path``.

    ``source_url`` MUST be the verified direct-CSV download URL from the
    Cleveland Fed "Inflation Nowcasting" page — the module-level
    ``SOURCE_URL`` is the human-facing HTML page, not a CSV endpoint, so it
    is intentionally not the default here. On any network failure this
    raises ``ClevelandFedNowcastError`` with the manual-drop instructions:

      1. Open the Cleveland Fed inflation-nowcasting page in a browser and
         download the historical CSV.
      2. Copy it to ``data/raw/cleveland_fed_nowcast/cleveland_fed_nowcast.csv``.
      3. Re-run the fetch — ``run_cleveland_fed_nowcast_fetch`` detects the
         manually-dropped file and parses it.

    Same pattern as the spdji EPS workbook and the PMI TSV workflow.
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
            "Accept": "text/csv,application/octet-stream,*/*",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except urllib.error.URLError as exc:
        raise ClevelandFedNowcastError(
            f"Failed to download Cleveland Fed nowcast CSV from {source_url}: "
            f"{exc}. To complete the fetch manually: (1) open the Cleveland "
            f"Fed inflation-nowcasting page and download the historical CSV; "
            f"(2) copy it to data/raw/{MANUAL_REL_PATH}; (3) re-run the "
            f"fetch — run_cleveland_fed_nowcast_fetch detects the "
            f"manually-dropped file and parses it."
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
    csv_path: Path,
    out_path: Path,
    date_column: str = DEFAULT_DATE_COLUMN,
    value_column: str = DEFAULT_VALUE_COLUMN,
    value_scale: float = DEFAULT_VALUE_SCALE,
) -> pd.DataFrame:
    """Parse the manually-dropped (or downloaded) Cleveland Fed CSV, merge
    it with any existing ``cpi_nowcast`` parquet, dedupe by date, re-save.

    The merge keeps the freshly-parsed row on a date collision — a later
    export supersedes an earlier same-date nowcast (the nowcast is revised
    as new daily data arrives within the month). Returns the full merged
    DataFrame, sorted ascending by date.
    """
    if not csv_path.exists():
        raise ClevelandFedNowcastError(
            f"No Cleveland Fed nowcast CSV at {csv_path}. Download the "
            f"historical CSV from the Cleveland Fed inflation-nowcasting "
            f"page and drop it there, then re-run."
        )
    parsed = parse_cleveland_fed_nowcast_csv(
        csv_path.read_text(),
        date_column=date_column,
        value_column=value_column,
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
    date_column: str = DEFAULT_DATE_COLUMN,
    value_column: str = DEFAULT_VALUE_COLUMN,
    value_scale: float = DEFAULT_VALUE_SCALE,
) -> Path:
    """Orchestrate the Cleveland Fed inflation-nowcast fetch.

    Resolution: parse the manually-dropped CSV at
    ``out_dir / cleveland_fed_nowcast / cleveland_fed_nowcast.csv`` (the
    operator downloads it from the Cleveland Fed page — see
    ``download_cleveland_fed_nowcast_csv``), merge into
    ``cleveland_fed_nowcast/cpi_nowcast.parquet``, and write a report JSON.

    Cadence: the Cleveland Fed nowcast updates intra-month as daily data
    arrives. Weekly or monthly re-fetch is sufficient — the engine reads the
    most-recent value carried forward.
    """
    nowcast_dir = out_dir / "cleveland_fed_nowcast"
    nowcast_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / MANUAL_REL_PATH
    out_path = nowcast_dir / CPI_NOWCAST_PARQUET

    df = update_cpi_nowcast_parquet(
        csv_path=csv_path,
        out_path=out_path,
        date_column=date_column,
        value_column=value_column,
        value_scale=value_scale,
    )

    report = {
        "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "source_path": str(csv_path),
        "rows": int(len(df)),
        "min_date": str(df["date"].min().date()) if not df.empty else None,
        "max_date": str(df["date"].max().date()) if not df.empty else None,
        "column_mapping": {
            "date_column": date_column,
            "value_column": value_column,
            "value_scale": value_scale,
        },
        "verification_needed": (
            "The Cleveland Fed CSV schema and published unit could not be "
            "verified without web access. Confirm date_column / value_column "
            "match the export header and that value_scale converts the "
            "published unit to a fractional monthly inflation rate (see "
            "ADR 0006 and the module docstring)."
        ),
        "paths": {
            "cpi_nowcast_parquet": str(out_path),
        },
    }
    report_path = out_dir / "cleveland_fed_nowcast_fetch_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report_path
