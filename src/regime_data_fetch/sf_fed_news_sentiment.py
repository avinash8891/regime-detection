"""SF Fed Daily News Sentiment Index fetcher.

The Federal Reserve Bank of San Francisco publishes a free, daily news
sentiment index built from a lexicon-based scoring of Wall Street
Journal economic articles (Shapiro, Sudhof, Wilson 2020, "Measuring
news sentiment", *Journal of Econometrics*). Coverage: 1980-01 →
present. Updated approximately weekly.

Source:
    https://www.frbsf.org/research-and-insights/data-and-indicators/daily-news-sentiment-index/

The data is published as an XLSX workbook with a single tabular sheet
named "Data" carrying two columns: a date column and a sentiment
column. Schema is stable across publications (the SF Fed has been
publishing this series since 2020 with the same workbook layout).

Used by V2 §1A as **evidence only** alongside the AAII bull-bear
8w-MA — never consumed by the `euphoria` rule predicate. See
`docs/spec_code_data_audit_2026_05_15.md` (audit follow-up for
sentiment concordance, post-#12).

Output parquet schema:

    date              (datetime64)
    news_sentiment    (float64)  — raw SF Fed daily index value
    source            (str)      — "frbsf:daily_news_sentiment"
    source_url        (str)      — the published XLSX URL
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


SF_FED_NEWS_SENTIMENT_URL = (
    "https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx"
)
SF_FED_NEWS_SENTIMENT_PARQUET = "sf_fed_news_sentiment.parquet"
_SOURCE_NAME = "frbsf:daily_news_sentiment"

_log = logging.getLogger(__name__)


class SFFedNewsSentimentFetchError(RuntimeError):
    """Raised when the SF Fed news sentiment fetch fails irrecoverably."""


def fetch_workbook_bytes(*, timeout: int = 30) -> bytes:
    """Download the latest SF Fed news sentiment XLSX as raw bytes."""
    req = urllib.request.Request(
        SF_FED_NEWS_SENTIMENT_URL,
        headers={"User-Agent": "regime-detection-fetch/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        raise SFFedNewsSentimentFetchError(
            f"failed to download {SF_FED_NEWS_SENTIMENT_URL}: {exc}"
        ) from exc


_DATA_SHEET_NAME = "Data"


def parse_workbook(workbook_bytes: bytes) -> pd.DataFrame:
    """Parse the SF Fed XLSX into a normalized long-form DataFrame.

    The published workbook has two sheets — ``Methodology`` (provenance
    text) and ``Data`` (date + News Sentiment columns). We read the
    ``Data`` sheet by name (falling back to position 1 if the name
    changes upstream) and resolve the two columns by position so
    edition-specific renames (`date` / `Date`; `News Sentiment` /
    `news_sentiment`) don't break the loader.
    """
    book = pd.ExcelFile(io.BytesIO(workbook_bytes), engine=None)
    sheet_name: object
    if _DATA_SHEET_NAME in book.sheet_names:
        sheet_name = _DATA_SHEET_NAME
    elif len(book.sheet_names) >= 2:
        sheet_name = book.sheet_names[1]  # fall back to second sheet
    else:
        sheet_name = book.sheet_names[0]
    raw = book.parse(sheet_name)
    if raw.empty or raw.shape[1] < 2:
        raise SFFedNewsSentimentFetchError(
            f"SF Fed workbook sheet {sheet_name!r} unexpected shape: {raw.shape}"
        )
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw.iloc[:, 0], errors="coerce"),
            "news_sentiment": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
        }
    ).dropna(subset=["date", "news_sentiment"])
    out = out.sort_values("date").reset_index(drop=True)
    out["source"] = _SOURCE_NAME
    out["source_url"] = SF_FED_NEWS_SENTIMENT_URL
    return out


def run_sf_fed_news_sentiment_fetch(
    *,
    out_dir: Path,
    workbook_bytes: bytes | None = None,
    workbook_path: str | Path | None = None,
    timeout: int = 30,
) -> dict[str, object]:
    """Materialize ``data/raw/news_sentiment/sf_fed_news_sentiment.parquet``.

    Parameters
    ----------
    out_dir
        Repo-relative path under which ``news_sentiment/`` is created.
    workbook_bytes
        Pre-fetched workbook bytes (test injection). When None, fetch
        from ``SF_FED_NEWS_SENTIMENT_URL``.
    workbook_path
        Alternative — an on-disk XLSX path (useful when running behind
        a firewall and the file was downloaded manually).
    timeout
        Network timeout in seconds when fetching live.

    Returns
    -------
    dict
        Fetch report — written to ``out_dir / sf_fed_news_sentiment_fetch_report.json``.
    """
    if workbook_bytes is None and workbook_path is not None:
        workbook_bytes = Path(workbook_path).read_bytes()
    if workbook_bytes is None:
        _log.info("Fetching SF Fed news sentiment workbook (live)")
        workbook_bytes = fetch_workbook_bytes(timeout=timeout)

    df = parse_workbook(workbook_bytes)
    out_subdir = Path(out_dir) / "news_sentiment"
    out_subdir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_subdir / SF_FED_NEWS_SENTIMENT_PARQUET
    df.to_parquet(parquet_path, index=False)

    report = {
        "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": int(len(df)),
        "min_date": df["date"].min().date().isoformat() if not df.empty else None,
        "max_date": df["date"].max().date().isoformat() if not df.empty else None,
        "parquet": str(parquet_path),
        "source": _SOURCE_NAME,
        "source_url": SF_FED_NEWS_SENTIMENT_URL,
    }
    report_path = Path(out_dir) / "sf_fed_news_sentiment_fetch_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    _log.info(
        "SF Fed news sentiment parquet: %d rows, %s → %s",
        report["rows"],
        report["min_date"],
        report["max_date"],
    )
    return report
