from __future__ import annotations

import csv
import datetime as dt
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from regime_data_fetch._http import fetch_text
from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_shared.pandas_compat import cow_safe_assign, optional_date
from regime_shared.pit_provenance import BIAS_WARNING, SOURCE_NAME, SOURCE_URL

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
# parquet schema does not need to change — only the shared provenance constants.

# Source-correction overlay for open intervals that the community CSV has not
# closed yet. Dates are the last valid membership/trading date for this feed:
# DAY ceased trading after the 2026-02-04 Thoma Bravo close, HOLX before the
# 2026-04-07 Blackstone/TPG private-company close, and S&P deleted CTRA
# effective before the 2026-05-07 open.
SOURCE_END_DATE_CORRECTIONS: dict[str, dt.date] = {
    "DAY": dt.date(2026, 2, 3),
    "HOLX": dt.date(2026, 4, 6),
    "CTRA": dt.date(2026, 5, 6),
}


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


def parse_sp500_ticker_start_end_csv(
    csv_text: str, *, source_url: str
) -> list[PITConstituentInterval]:
    rows: list[PITConstituentInterval] = []
    reader = csv.DictReader(csv_text.splitlines())
    required = ["ticker", "start_date", "end_date"]
    if reader.fieldnames != required:
        raise PITConstituentFetchError(
            f"Unexpected PIT CSV columns: {reader.fieldnames!r}"
        )

    for idx, raw in enumerate(reader, start=2):
        ticker = (raw.get("ticker") or "").strip()
        if not ticker:
            raise PITConstituentFetchError(f"Row {idx}: missing ticker")

        start_date = _parse_date(
            raw.get("start_date"), field="start_date", row_number=idx
        )
        end_date = _parse_optional_date(
            raw.get("end_date"), field="end_date", row_number=idx
        )
        corrected_end_date = SOURCE_END_DATE_CORRECTIONS.get(ticker)
        if end_date is None and corrected_end_date is not None:
            end_date = corrected_end_date
        if end_date and end_date < start_date:
            raise PITConstituentFetchError(
                f"Row {idx}: end_date before start_date for {ticker}"
            )

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
    return fetch_text(SOURCE_URL, timeout=30)


def run_pit_constituents_fetch(
    *,
    out_dir: Path,
    csv_fetcher=fetch_sp500_ticker_start_end_csv,
    acquisition_db_path: Path | None = None,
    artifact_store_root: str | Path | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    store = (
        AcquisitionStore(acquisition_db_path, artifact_store_root=artifact_store_root)
        if acquisition_db_path
        else None
    )
    run_context = (
        store.run(
            fetch_type="pit_constituents",
            params={
                "source_url": SOURCE_URL,
            },
        )
        if store
        else nullcontext(None)
    )

    with run_context as fetch_run:
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
                "max_end_date": (
                    str(df["end_date"].dropna().max())
                    if not df["end_date"].dropna().empty
                    else None
                ),
            },
            "paths": {
                "pit_constituents_parquet": str(parquet_path),
                "acquisition_db": (
                    str(acquisition_db_path) if acquisition_db_path else None
                ),
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
                max_date=(
                    str(df["end_date"].dropna().max())
                    if not df["end_date"].dropna().empty
                    else None
                ),
                notes="PIT constituent interval parquet output",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="pit_constituents_report",
                path=report_path,
                row_count=len(df),
                min_date=str(df["start_date"].min()) if not df.empty else None,
                max_date=(
                    str(df["end_date"].dropna().max())
                    if not df["end_date"].dropna().empty
                    else None
                ),
                notes="PIT constituent fetch report",
            )
        return report_path


def is_survivorship_biased_universe(intervals: pd.DataFrame) -> bool:
    """True when a constituent universe is survivorship-biased.

    V2 §1D line 326-327: a point-in-time universe INCLUDES removed/delisted
    members (closed membership intervals — a non-null ``end_date``). A
    current-only snapshot, where every interval is still open, is
    survivorship-biased and must be rejected at ingestion unless biased research
    mode is approved.
    """
    if "end_date" not in intervals.columns:
        return True
    return not bool(intervals["end_date"].notna().any())


def read_pit_intervals(
    parquet_path: Path,
    *,
    allow_survivorship_biased_breadth: bool = False,
) -> pd.DataFrame:
    """Read PIT membership parquet back into a DataFrame.

    Inverse of the writer in ``run_pit_constituents_fetch``. Converts the
    persisted ISO yyyy-mm-dd date strings back to ``datetime.date`` objects
    (``None`` for the null open-interval tail in ``end_date``).

    SOURCE_END_DATE_CORRECTIONS are re-applied here as a defensive patch-on-read
    so that stale S3 artifacts (built before corrections were added to the code)
    produce correct membership lookups. The patch is idempotent: rows that
    already carry a non-null end_date are left unchanged.

    V2 §1D line 327 / §10 — fail-closed survivorship-bias gate: corrections are
    applied before membership use, but source corrections alone are not enough
    to prove a point-in-time feed. The raw source must contain at least one
    closed interval unless ``allow_survivorship_biased_breadth`` is True. Real
    point-in-time feeds include delistings and pass; a current-only snapshot does
    not. The V1 ETF-proxy breadth fallback (when no PIT universe is loaded at
    all) is unaffected — this gates the loaded universe, not its absence.
    """
    df = pd.read_parquet(parquet_path)

    df = cow_safe_assign(
        df,
        {
            "start_date": df["start_date"].map(optional_date),
            "end_date": df["end_date"].map(optional_date),
        },
    )
    source_contains_closed_interval = bool(df["end_date"].notna().any())

    # Patch-on-read: apply corrections to any open interval whose ticker has a
    # known correction. This fixes stale artifacts without requiring a re-fetch.
    end_date = df["end_date"].copy()
    for ticker, corrected_end in SOURCE_END_DATE_CORRECTIONS.items():
        mask = (df["ticker"] == ticker) & df["end_date"].isna()
        if mask.any():
            end_date = end_date.mask(mask, corrected_end)
    df = cow_safe_assign(df, {"end_date": end_date})

    # Normalize remaining string columns to object dtype. Newer pandas+pyarrow
    # round-trips parquet strings as StringDtype by default; the reader contract
    # (and downstream consumers comparing dtype == object) requires plain object.
    object_columns = {
        column: object
        for column in ("ticker", "source", "source_url", "bias_warning")
        if column in df.columns
    }
    if object_columns:
        df = df.astype(object_columns)

    if not allow_survivorship_biased_breadth:
        if is_survivorship_biased_universe(df):
            raise ValueError(
                f"PIT constituent universe at {parquet_path} is survivorship-biased "
                "(no removed/delisted members — every membership interval is open). "
                "Refusing to load a current-only universe as point-in-time. Provide a "
                "universe that includes removed members, or pass "
                "allow_survivorship_biased_breadth=True to opt into biased research "
                "mode."
            )
        if not source_contains_closed_interval:
            raise ValueError(
                f"PIT constituent universe at {parquet_path} is survivorship-biased: "
                "source contains no closed intervals before source-specific "
                "corrections. Refusing to treat source-corrected rows alone as "
                "proof of a point-in-time universe."
            )
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
        raise PITConstituentFetchError(
            f"Row {row_number}: invalid {field} {raw!r}"
        ) from exc


def _parse_optional_date(
    value: str | None, *, field: str, row_number: int
) -> dt.date | None:
    raw = (value or "").strip()
    if not raw:
        return None
    return _parse_date(raw, field=field, row_number=row_number)
