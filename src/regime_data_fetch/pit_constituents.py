from __future__ import annotations

import csv
import datetime as dt
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from regime_data_fetch.acquisition_store import AcquisitionStore


# Community-maintained S&P 500 ticker-membership CSV on GitHub. This is an
# APPROXIMATION of point-in-time S&P 500 membership and is the best free source
# we have today: it may miss short-lived additions/removals, delisted tickers
# whose symbols got reused, and ticker/name changes around mergers. The
# `BIAS_WARNING` row tag is propagated downstream so consumers can decide
# whether to trust it.
#
# TODO: replace with a true point-in-time vendor feed (CRSP / Compustat /
# FactSet / Norgate) when sourcing is approved. The expected vendor format
# matches the same ticker / start_date / end_date interval shape, so the
# parquet schema does not need to change — only `SOURCE_URL`, `SOURCE_NAME`,
# and the `BIAS_WARNING` value (which should become e.g. `"none"` or be
# removed entirely).
SOURCE_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
SOURCE_NAME = "fja05680/sp500"
BIAS_WARNING = "survivorship_biased_constituent_universe"


class PITConstituentFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class PITConstituentInterval:
    ticker: str
    start_date: dt.date
    end_date: dt.date | None
    source: str
    source_url: str
    bias_warning: str


def parse_sp500_ticker_start_end_csv(csv_text: str, *, source_url: str) -> list[PITConstituentInterval]:
    rows: list[PITConstituentInterval] = []
    reader = csv.DictReader(csv_text.splitlines())
    required = ["ticker", "start_date", "end_date"]
    if reader.fieldnames != required:
        raise PITConstituentFetchError(f"Unexpected PIT CSV columns: {reader.fieldnames!r}")

    for idx, raw in enumerate(reader, start=2):
        ticker = (raw.get("ticker") or "").strip()
        if not ticker:
            raise PITConstituentFetchError(f"Row {idx}: missing ticker")

        start_date = _parse_date(raw.get("start_date"), field="start_date", row_number=idx)
        end_date = _parse_optional_date(raw.get("end_date"), field="end_date", row_number=idx)
        if end_date and end_date < start_date:
            raise PITConstituentFetchError(f"Row {idx}: end_date before start_date for {ticker}")

        rows.append(
            PITConstituentInterval(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                source=SOURCE_NAME,
                source_url=source_url,
                bias_warning=BIAS_WARNING,
            )
        )

    if not rows:
        raise PITConstituentFetchError("PIT constituent CSV contained no rows")
    return rows


def fetch_sp500_ticker_start_end_csv() -> str:
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def run_pit_constituents_fetch(
    *,
    out_dir: Path,
    csv_fetcher=fetch_sp500_ticker_start_end_csv,
    acquisition_db_path: Path | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    store = AcquisitionStore(acquisition_db_path) if acquisition_db_path else None
    fetch_run = (
        store.start_fetch_run(
            fetch_type="pit_constituents",
            params={
                "source_url": SOURCE_URL,
            },
        )
        if store
        else None
    )

    try:
        csv_text = csv_fetcher()
        rows = parse_sp500_ticker_start_end_csv(csv_text, source_url=SOURCE_URL)

        if store and fetch_run:
            store.record_text_artifact(
                run_id=fetch_run.run_id,
                source_name="github_raw:sp500_ticker_start_end",
                artifact_kind="csv",
                source_identifier=SOURCE_URL,
                content_text=csv_text,
                timezone="UTC",
                license_note="Raw point-in-time S&P 500 interval CSV fetched from GitHub raw",
                notes="PIT constituent CSV persisted before parquet/report output",
            )

        df = pd.DataFrame(
            [
                {
                    "ticker": row.ticker,
                    "start_date": row.start_date.isoformat(),
                    "end_date": row.end_date.isoformat() if row.end_date else None,
                    "source": row.source,
                    "source_url": row.source_url,
                    "bias_warning": row.bias_warning,
                }
                for row in rows
            ]
        )

        pit_dir = out_dir / "pit_constituents"
        pit_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = pit_dir / "sp500_ticker_intervals.parquet"
        df.to_parquet(parquet_path, index=False)

        report = {
            "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": SOURCE_NAME,
            "source_url": SOURCE_URL,
            "bias_warning": BIAS_WARNING,
            "counts": {
                "rows": int(len(df)),
                "tickers": int(df["ticker"].nunique()),
                "open_intervals": int(df["end_date"].isna().sum()),
            },
            "date_range": {
                "min_start_date": str(df["start_date"].min()) if not df.empty else None,
                "max_start_date": str(df["start_date"].max()) if not df.empty else None,
                "max_end_date": str(df["end_date"].dropna().max()) if not df["end_date"].dropna().empty else None,
            },
            "paths": {
                "pit_constituents_parquet": str(parquet_path),
                "acquisition_db": str(acquisition_db_path) if acquisition_db_path else None,
            },
        }
        report_path = out_dir / "pit_constituents_fetch_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if store and fetch_run:
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="pit_constituents_parquet",
                path=parquet_path,
                row_count=len(df),
                min_date=str(df["start_date"].min()) if not df.empty else None,
                max_date=str(df["end_date"].dropna().max()) if not df["end_date"].dropna().empty else None,
                notes="PIT constituent interval parquet output",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="pit_constituents_report",
                path=report_path,
                row_count=len(df),
                min_date=str(df["start_date"].min()) if not df.empty else None,
                max_date=str(df["end_date"].dropna().max()) if not df["end_date"].dropna().empty else None,
                notes="PIT constituent fetch report",
            )
            store.finish_fetch_run(run_id=fetch_run.run_id, status="ok")
        return report_path
    except Exception as exc:
        if store and fetch_run:
            store.finish_fetch_run(run_id=fetch_run.run_id, status="failed", notes=str(exc))
        raise


def read_pit_intervals(parquet_path: Path) -> pd.DataFrame:
    """Read PIT membership parquet back into a DataFrame.

    Inverse of the writer in ``run_pit_constituents_fetch``. Converts the
    persisted ISO yyyy-mm-dd date strings back to ``datetime.date`` objects
    (``None`` for the null open-interval tail in ``end_date``).
    """
    df = pd.read_parquet(parquet_path)

    def _to_date(value: object) -> dt.date | None:
        if pd.isna(value):
            return None
        if value is None:
            return None
        return dt.date.fromisoformat(str(value))

    df["start_date"] = df["start_date"].map(_to_date)
    df["end_date"] = df["end_date"].map(_to_date)
    return df


def members_on(intervals_df: pd.DataFrame, as_of_date: dt.date) -> frozenset[str]:
    """Return tickers whose interval contains ``as_of_date`` (inclusive both bounds).

    A null ``end_date`` is treated as an open-ended interval. Empty input
    yields ``frozenset()``.
    """
    if intervals_df.empty:
        return frozenset()

    start_ok = intervals_df["start_date"] <= as_of_date
    end_series = intervals_df["end_date"]
    end_ok = end_series.isna() | (end_series >= as_of_date)
    mask = start_ok & end_ok
    return frozenset(intervals_df.loc[mask, "ticker"].tolist())


def _parse_date(value: str | None, *, field: str, row_number: int) -> dt.date:
    raw = (value or "").strip()
    try:
        return dt.date.fromisoformat(raw)
    except ValueError as exc:
        raise PITConstituentFetchError(f"Row {row_number}: invalid {field} {raw!r}") from exc


def _parse_optional_date(value: str | None, *, field: str, row_number: int) -> dt.date | None:
    raw = (value or "").strip()
    if not raw:
        return None
    return _parse_date(raw, field=field, row_number=row_number)
