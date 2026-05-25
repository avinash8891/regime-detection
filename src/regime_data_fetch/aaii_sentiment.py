from __future__ import annotations

import datetime as dt
import json
import logging
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from regime_data_fetch.acquisition_store import AcquisitionStore

AAII_SENTIMENT_URL = "https://www.aaii.com/sentimentsurvey/sent_results"
AAII_SENTIMENT_PARQUET = "aaii_sentiment.parquet"
AAII_SENTIMENT_SEED_CFB = "aaii_sentiment_historical.cfb"

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML table parser
# ---------------------------------------------------------------------------


class _TableParser(HTMLParser):
    """Extracts all <table> rows as lists of stripped cell strings."""

    def __init__(self) -> None:
        super().__init__()
        self._in_table = False
        self._in_td = False
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "table":
            self._in_table = True
        elif tag == "tr" and self._in_table:
            self._current_row = []
        elif tag in ("td", "th") and self._in_table:
            self._in_td = True
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._in_table:
            self._in_td = False
            self._current_row.append("".join(self._current_cell).strip())
        elif tag == "tr" and self._in_table:
            if any(self._current_row):
                self.rows.append(self._current_row[:])
        elif tag == "table":
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._in_td:
            self._current_cell.append(data)


def _infer_date(month_day: str, today: dt.date) -> dt.date | None:
    """Convert 'May 6' or 'Dec 31' to a full date using today to assign the year."""
    try:
        parsed = dt.datetime.strptime(month_day.strip(), "%b %d").date()
    except ValueError:
        return None
    candidate = parsed.replace(year=today.year)
    if candidate > today:
        candidate = parsed.replace(year=today.year - 1)
    return candidate


def _pct(raw: str) -> float | None:
    """Parse '38.3%' → 0.383, or None on failure."""
    try:
        return float(raw.rstrip("%")) / 100.0
    except (ValueError, AttributeError):
        return None


def _parse_html_table(html_text: str, after_date: dt.date) -> pd.DataFrame:
    parser = _TableParser()
    parser.feed(html_text)

    today = dt.date.today()
    rows = []
    for cells in parser.rows:
        if len(cells) < 4:
            continue
        date = _infer_date(cells[0], today)
        if date is None:
            continue
        bull = _pct(cells[1])
        neut = _pct(cells[2])
        bear = _pct(cells[3])
        if bull is None or neut is None or bear is None:
            continue
        if date <= after_date:
            continue
        rows.append(
            {
                "date": pd.Timestamp(date),
                "bullish": bull,
                "neutral": neut,
                "bearish": bear,
            }
        )

    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["date", "bullish", "neutral", "bearish"])
    )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def parse_historical_cfb(path: Path) -> pd.DataFrame:
    """Parse the AAII historical XLS/CFB seed file into a clean weekly DataFrame."""
    try:
        import xlrd
    except ImportError as exc:
        raise ImportError(
            "xlrd is required to parse the AAII CFB seed file: pip install xlrd"
        ) from exc

    wb = xlrd.open_workbook(str(path))
    ws = wb.sheet_by_name("SENTIMENT")

    rows = []
    for r in range(5, ws.nrows):
        if ws.cell_type(r, 0) != xlrd.XL_CELL_DATE:
            continue
        if ws.cell_type(r, 1) != xlrd.XL_CELL_NUMBER:
            continue
        date = xlrd.xldate_as_datetime(ws.cell_value(r, 0), wb.datemode).date()
        rows.append(
            {
                "date": pd.Timestamp(date),
                "bullish": float(ws.cell_value(r, 1)),
                "neutral": float(ws.cell_value(r, 2)),
                "bearish": float(ws.cell_value(r, 3)),
            }
        )

    df = pd.DataFrame(rows)
    return _compute_derived(df)


def fetch_latest_rows(
    url: str, after_date: dt.date, *, timeout: int = 30
) -> pd.DataFrame:
    """Scrape the AAII sentiment HTML table and return rows newer than after_date."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; regime-engine-fetcher/2.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html_text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to fetch AAII sentiment page: {exc}") from exc

    return _parse_html_table(html_text, after_date=after_date)


def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df["bull_bear_spread"] = df["bullish"] - df["bearish"]
    df["bull_bear_spread_8w_ma"] = (
        df["bull_bear_spread"].rolling(8, min_periods=1).mean()
    )
    return df


def update_aaii_parquet(
    *,
    raw_dir: Path,
    out_path: Path,
    url: str = AAII_SENTIMENT_URL,
) -> pd.DataFrame:
    """Load existing parquet (or parse seed CFB), append new rows from AAII, re-save."""
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing["date"] = pd.to_datetime(existing["date"])
        _log.info(
            "aaii_sentiment: loaded %d existing rows from %s", len(existing), out_path
        )
    else:
        seed_path = raw_dir / "sentiment" / AAII_SENTIMENT_SEED_CFB
        if not seed_path.exists():
            raise FileNotFoundError(
                f"No parquet at {out_path} and no seed CFB at {seed_path}. "
                "Place aaii_sentiment_historical.cfb in data/raw/sentiment/ and re-run."
            )
        _log.info("aaii_sentiment: seeding from CFB %s", seed_path)
        existing = parse_historical_cfb(seed_path)

    last_date = existing["date"].max().date()
    _log.info(
        "aaii_sentiment: last known date %s, fetching newer rows from %s",
        last_date,
        url,
    )

    new_rows = fetch_latest_rows(url, after_date=last_date)

    if new_rows.empty:
        _log.info("aaii_sentiment: no new rows after %s", last_date)
        combined = existing
    else:
        _log.info(
            "aaii_sentiment: appending %d new rows (up to %s)",
            len(new_rows),
            new_rows["date"].max().date(),
        )
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = (
            combined.drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        combined = _compute_derived(
            combined.drop(
                columns=["bull_bear_spread", "bull_bear_spread_8w_ma"], errors="ignore"
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    _log.info("aaii_sentiment: saved %d rows to %s", len(combined), out_path)
    return combined


def run_sentiment_fetch(
    *,
    out_dir: Path,
    url: str = AAII_SENTIMENT_URL,
    acquisition_db_path: Path | None = None,
    artifact_store_root: str | Path | None = None,
    required: bool = True,
) -> Path:
    """Orchestrate the AAII sentiment fetch and write a report JSON."""
    sentiment_dir = out_dir / "sentiment"
    sentiment_dir.mkdir(parents=True, exist_ok=True)
    out_path = sentiment_dir / AAII_SENTIMENT_PARQUET
    seed_path = sentiment_dir / AAII_SENTIMENT_SEED_CFB

    store = (
        AcquisitionStore(acquisition_db_path, artifact_store_root=artifact_store_root)
        if acquisition_db_path
        else None
    )
    fetch_run = (
        store.start_fetch_run(
            fetch_type="sentiment",
            params={"source": "aaii", "url": url},
        )
        if store
        else None
    )

    try:
        try:
            df = update_aaii_parquet(raw_dir=out_dir, out_path=out_path, url=url)
        except FileNotFoundError as exc:
            if required:
                raise
            reason = str(exc)
            report = {
                "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "status": "skipped",
                "reason": reason,
                "materializable": False,
                "paths": {
                    "sentiment_parquet": str(out_path),
                    "seed_cfb": str(seed_path),
                    "acquisition_db": (
                        str(acquisition_db_path) if acquisition_db_path else None
                    ),
                },
            }
            report_path = out_dir / "sentiment_fetch_report.json"
            report_path.write_text(json.dumps(report, indent=2))
            if store and fetch_run:
                store.record_output(
                    run_id=fetch_run.run_id,
                    output_kind="aaii_sentiment_fetch_report",
                    path=report_path,
                    record_artifact=False,
                    notes="AAII sentiment fetch skipped because no bootstrap seed is materialized",
                )
                store.finish_fetch_run(
                    run_id=fetch_run.run_id, status="skipped", notes=reason
                )
            return report_path

        report = {
            "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "rows": int(len(df)),
            "min_date": str(df["date"].min().date()) if not df.empty else None,
            "max_date": str(df["date"].max().date()) if not df.empty else None,
            "paths": {
                "sentiment_parquet": str(out_path),
                "acquisition_db": (
                    str(acquisition_db_path) if acquisition_db_path else None
                ),
            },
        }
        report_path = out_dir / "sentiment_fetch_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if store and fetch_run:
            raw_record = None
            if seed_path.exists():
                raw_artifact = store.record_file_artifact(
                    run_id=fetch_run.run_id,
                    source_name="aaii",
                    artifact_kind="cfb",
                    source_identifier=AAII_SENTIMENT_SEED_CFB,
                    file_path=seed_path,
                    start_date=report["min_date"],
                    end_date=report["max_date"],
                    notes="AAII historical seed file used when canonical parquet is bootstrapped locally",
                )
                raw_record = raw_artifact.artifact_record_id
            canonical_record = store.record_output(
                run_id=fetch_run.run_id,
                output_kind="aaii_sentiment_parquet",
                path=out_path,
                row_count=int(len(df)),
                min_date=report["min_date"],
                max_date=report["max_date"],
                artifact_name="aaii_sentiment",
                source_name="aaii",
                artifact_kind="parquet",
                notes="AAII sentiment canonical parquet",
            )
            if raw_record is not None and canonical_record is not None:
                store.record_artifact_lineage(
                    output_artifact_record_id=canonical_record.artifact_record_id,
                    input_artifact_record_id=raw_record,
                    transform_name="normalize_aaii_sentiment",
                )
            if report["max_date"]:
                store.set_source_checkpoint(
                    source_name="aaii",
                    cursor_key="survey_week",
                    cursor_value=str(report["max_date"]),
                    successful_run_id=fetch_run.run_id,
                )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="aaii_sentiment_fetch_report",
                path=report_path,
                row_count=int(len(df)),
                min_date=report["min_date"],
                max_date=report["max_date"],
                record_artifact=False,
                notes="AAII sentiment fetch report",
            )
            store.finish_fetch_run(run_id=fetch_run.run_id, status="ok")
        return report_path
    except Exception as exc:
        if store and fetch_run:
            store.finish_fetch_run(
                run_id=fetch_run.run_id, status="failed", notes=str(exc)
            )
        raise
