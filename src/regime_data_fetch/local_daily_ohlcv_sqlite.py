from __future__ import annotations

import json
import os
import sqlite3
import datetime as dt
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.alpaca_daily import DailyBarsFetchResult, fetch_daily_bars_alpaca
from regime_data_fetch.daily_bars_provider import (
    DAILY_BARS_PROVIDERS,
    fetch_daily_bars_with_provider,
)
from regime_data_fetch.universe import (
    FIXED_UNIVERSE_LOCAL_PATH,
    FIXED_UNIVERSE_SYMBOL_COUNT,
    FIXED_UNIVERSE_TREE_NAME,
    load_symbols_from_daily_ohlcv_tree,
)
from regime_data_fetch.yahoo_chart_daily import fetch_daily_bars_yahoo_chart


EXPECTED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
]
SYMBOL_TREE_COLUMNS = ["date", "symbol", *EXPECTED_COLUMNS[1:]]


def run_alpaca_constituent_daily_ohlcv_fetch(
    *,
    out_dir: Path,
    pit_parquet_path: Path,
    start: dt.date,
    end: dt.date,
    adjustment: str,
    alpaca_feed: str | None,
    acquisition_db_path: Path,
    artifact_store_root: str | Path | None = None,
    bars_fetcher: Callable[..., DailyBarsFetchResult] | None = None,
    daily_bars_provider: str = "alpaca",
    allow_missing_symbols: bool = False,
    fixed_universe_symbols: list[str] | None = None,
    fixed_universe_dir: Path | None = None,
    allow_pit_universe: bool = False,
    expected_universe_count: int | None = FIXED_UNIVERSE_SYMBOL_COUNT,
    verbose: bool = False,
) -> Path:
    if end < start:
        raise SystemExit("--end must be >= --start")

    if daily_bars_provider not in DAILY_BARS_PROVIDERS:
        raise SystemExit("--daily-bars-provider must be alpaca|yahoo-chart|alpaca-yahoo-fallback")
    if bars_fetcher is None and daily_bars_provider in {"alpaca", "alpaca-yahoo-fallback"}:
        for key in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"):
            if not os.environ.get(key, "").strip():
                raise SystemExit(f"Missing required env var: {key}")

    symbols, universe_source = _resolve_constituent_symbols(
        fixed_universe_symbols=fixed_universe_symbols,
        fixed_universe_dir=fixed_universe_dir,
        pit_parquet_path=pit_parquet_path,
        allow_pit_universe=allow_pit_universe,
    )
    if expected_universe_count is not None and len(symbols) != expected_universe_count:
        raise SystemExit(
            "Constituent OHLCV refresh resolved "
            f"{len(symbols)} symbols from {universe_source}; expected {expected_universe_count}. "
            f"Pass the fixed {FIXED_UNIVERSE_SYMBOL_COUNT}-symbol universe artifact, or override --constituent-universe-expected-count "
            "only for an intentional replay."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_root = out_dir / FIXED_UNIVERSE_TREE_NAME

    store = AcquisitionStore(acquisition_db_path, artifact_store_root=artifact_store_root)
    fetch_run = store.start_fetch_run(
        fetch_type="daily_ohlcv_constituents_alpaca",
        params={
            "pit_parquet_path": str(pit_parquet_path),
            "universe_source": universe_source,
            "symbols_requested": len(symbols),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "adjustment": adjustment,
            "alpaca_feed": alpaca_feed,
            "daily_bars_provider": daily_bars_provider,
            "allow_missing_symbols": allow_missing_symbols,
        },
    )
    try:
        if bars_fetcher is not None:
            bars = bars_fetcher(
                symbols=symbols,
                start_date=start,
                end_date=end,
                adjustment=adjustment,
                feed=alpaca_feed,
                verbose=verbose,
            )
        else:
            bars = fetch_daily_bars_with_provider(
                provider=daily_bars_provider,
                symbols=symbols,
                start_date=start,
                end_date=end,
                adjustment=adjustment,
                feed=alpaca_feed,
                verbose=verbose,
                alpaca_fetcher=fetch_daily_bars_alpaca,
                yahoo_fetcher=fetch_daily_bars_yahoo_chart,
            )
        if bars.df.empty:
            raise RuntimeError(f"{daily_bars_provider} returned no constituent OHLCV rows")
        if bars.missing_symbols and not allow_missing_symbols:
            sample = ", ".join(bars.missing_symbols[:20])
            raise RuntimeError(
                f"{daily_bars_provider} returned no bars for "
                f"{len(bars.missing_symbols)} constituent symbols: {sample}"
            )

        written_files = _write_daily_ohlcv_symbol_tree(bars.df, tree_root=tree_root)
        min_date = str(pd.to_datetime(bars.df["date"]).dt.date.min())
        max_date = str(pd.to_datetime(bars.df["date"]).dt.date.max())

        store.record_output(
            run_id=fetch_run.run_id,
            output_kind="alpaca_constituent_daily_ohlcv_tree",
            path=written_files[0],
            row_count=len(bars.df),
            min_date=min_date,
            max_date=max_date,
            notes=f"profile_tree={tree_root};files={len(written_files)}",
        )
        store.finish_fetch_run(
            run_id=fetch_run.run_id,
            status="ok",
            notes=f"symbols={bars.df['symbol'].nunique()};rows={len(bars.df)};missing={len(bars.missing_symbols)}",
        )

        import_report = run_local_daily_ohlcv_sqlite_import(
            out_dir=out_dir,
            source_dir=tree_root,
            acquisition_db_path=acquisition_db_path,
            artifact_store_root=artifact_store_root,
        )
        report = {
            "source": f"{daily_bars_provider}:daily_bars",
            "pit_parquet_path": str(pit_parquet_path),
            "universe_source": universe_source,
            "requested": {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "adjustment": adjustment,
                "alpaca_feed": alpaca_feed,
            },
            "counts": {
                "symbols_requested": len(symbols),
                "symbols_returned": int(bars.df["symbol"].nunique()),
                "rows": int(len(bars.df)),
                "missing_symbols": len(bars.missing_symbols),
            },
            "missing_symbols_sample": bars.missing_symbols[:50],
            "date_range": {
                "min_date": min_date,
                "max_date": max_date,
            },
            "paths": {
                "profile_constituent_tree": {
                    "path": str(tree_root),
                    "local_path": FIXED_UNIVERSE_LOCAL_PATH,
                },
                "sqlite_import_report": str(import_report),
                "acquisition_db": str(acquisition_db_path),
            },
        }
        report_path = out_dir / "daily_ohlcv_constituents_alpaca_fetch_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        store.record_output(
            run_id=fetch_run.run_id,
            output_kind="alpaca_constituent_daily_ohlcv_report",
            path=report_path,
            row_count=len(bars.df),
            min_date=min_date,
            max_date=max_date,
            notes="Alpaca PIT constituent OHLCV fetch report",
        )
        return report_path
    except Exception as exc:
        store.finish_fetch_run(run_id=fetch_run.run_id, status="failed", notes=str(exc))
        raise


def run_local_daily_ohlcv_sqlite_import(
    *,
    out_dir: Path,
    source_dir: Path,
    acquisition_db_path: Path,
    artifact_store_root: str | Path | None = None,
) -> Path:
    if not source_dir.exists():
        raise SystemExit(f"Missing OHLCV source directory: {source_dir}")

    parquet_files = sorted(source_dir.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No parquet files found under: {source_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    store = AcquisitionStore(acquisition_db_path, artifact_store_root=artifact_store_root)
    fetch_run = store.start_fetch_run(
        fetch_type="daily_ohlcv_local_sqlite",
        params={
            "source_dir": str(source_dir),
            "parquet_files": len(parquet_files),
        },
    )

    imported_rows = 0
    symbol_count = 0
    min_date: str | None = None
    max_date: str | None = None
    artifact_records: list[tuple[Path, str, str, str]] = []

    try:
        with sqlite3.connect(acquisition_db_path) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            _ensure_daily_ohlcv_table(conn)

            for parquet_path in parquet_files:
                symbol = _infer_symbol_from_path(parquet_path)
                frame = pd.read_parquet(parquet_path)
                _validate_symbol_tree_frame(
                    frame=frame,
                    parquet_path=parquet_path,
                    expected_symbol=symbol,
                )
                normalized = frame.copy()
                normalized["date"] = pd.to_datetime(normalized["date"]).dt.date.astype(str)
                rows = [
                    (
                        symbol,
                        row["date"],
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        int(row["volume"]),
                        float(row["adjusted_close"]),
                        str(parquet_path),
                    )
                    for row in normalized.to_dict(orient="records")
                ]
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO daily_ohlcv_rows (
                        symbol,
                        date,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        adjusted_close,
                        source_file
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                imported_rows += len(rows)
                symbol_count += 1
                file_min = normalized["date"].min()
                file_max = normalized["date"].max()
                min_date = file_min if min_date is None else min(min_date, file_min)
                max_date = file_max if max_date is None else max(max_date, file_max)

                artifact_records.append((parquet_path, file_min, file_max, symbol))
            conn.commit()

        for parquet_path, file_min, file_max, symbol in artifact_records:
            store.record_file_artifact(
                run_id=fetch_run.run_id,
                source_name="local:daily_ohlcv",
                artifact_kind="parquet_local",
                source_identifier=str(parquet_path),
                file_path=parquet_path,
                start_date=file_min,
                end_date=file_max,
                timezone="UTC",
                adjustment_policy="raw_or_precomputed_source",
                license_note="Local partitioned OHLCV parquet artifact imported into SQLite row store",
                notes=f"symbol={symbol}",
                store_bytes=False,
            )

        report = {
            "source_dir": str(source_dir),
            "counts": {
                "parquet_files": len(parquet_files),
                "symbols": symbol_count,
                "imported_rows": imported_rows,
            },
            "date_range": {
                "min_date": min_date,
                "max_date": max_date,
            },
            "paths": {
                "acquisition_db": str(acquisition_db_path),
                "profile_constituent_tree": {
                    "path": str(source_dir),
                    "local_path": FIXED_UNIVERSE_LOCAL_PATH,
                },
            },
        }
        report_path = out_dir / "daily_ohlcv_local_sqlite_import_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        store.record_output(
            run_id=fetch_run.run_id,
            output_kind="daily_ohlcv_local_sqlite_import_report",
            path=report_path,
            row_count=imported_rows,
            min_date=min_date,
            max_date=max_date,
            notes="Local OHLCV parquet import report",
        )
        store.finish_fetch_run(
            run_id=fetch_run.run_id,
            status="ok",
            notes=f"symbols={symbol_count};rows={imported_rows}",
        )
        return report_path
    except Exception as exc:
        store.finish_fetch_run(run_id=fetch_run.run_id, status="failed", notes=str(exc))
        raise


def _ensure_daily_ohlcv_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS daily_ohlcv_rows (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            adjusted_close REAL NOT NULL,
            source_file TEXT NOT NULL,
            PRIMARY KEY (symbol, date)
        );
        CREATE INDEX IF NOT EXISTS idx_daily_ohlcv_rows_date
            ON daily_ohlcv_rows (date);
        """
    )


def _symbols_from_pit_parquet(pit_parquet_path: Path) -> list[str]:
    if not pit_parquet_path.exists():
        raise SystemExit(f"Missing PIT constituents parquet: {pit_parquet_path}")
    frame = pd.read_parquet(pit_parquet_path)
    if "ticker" not in frame.columns:
        raise RuntimeError(f"PIT constituents parquet missing ticker column: {pit_parquet_path}")
    symbols = sorted({str(value).strip() for value in frame["ticker"].dropna().tolist() if str(value).strip()})
    if not symbols:
        raise RuntimeError(f"PIT constituents parquet has no ticker values: {pit_parquet_path}")
    return symbols


def _resolve_constituent_symbols(
    *,
    fixed_universe_symbols: list[str] | None,
    fixed_universe_dir: Path | None,
    pit_parquet_path: Path,
    allow_pit_universe: bool,
) -> tuple[list[str], str]:
    if fixed_universe_symbols is not None:
        symbols = sorted({symbol.strip() for symbol in fixed_universe_symbols if symbol.strip()})
        if not symbols:
            raise SystemExit("Fixed constituent universe list is empty")
        return symbols, "fixed_symbol_list"
    if fixed_universe_dir is not None:
        return load_symbols_from_daily_ohlcv_tree(fixed_universe_dir), f"fixed_daily_ohlcv_tree:{fixed_universe_dir}"
    if allow_pit_universe:
        return _symbols_from_pit_parquet(pit_parquet_path), f"pit_constituents_bootstrap:{pit_parquet_path}"
    raise SystemExit(
        "daily-ohlcv-constituents-alpaca requires a fixed constituent universe. "
        f"Pass --universe-json with the {FIXED_UNIVERSE_SYMBOL_COUNT} symbols or --constituent-universe-dir pointing at the "
        f"materialized {FIXED_UNIVERSE_TREE_NAME} tree. Use --allow-pit-constituent-universe only for an explicit bootstrap."
    )


def _write_daily_ohlcv_symbol_tree(frame: pd.DataFrame, *, tree_root: Path) -> list[Path]:
    if "symbol" not in frame.columns:
        raise RuntimeError("Alpaca constituent OHLCV frame missing symbol column")
    _validate_ohlcv_frame(frame=frame[EXPECTED_COLUMNS], parquet_path=tree_root)
    tree_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.date.astype(str)
    for symbol, symbol_frame in normalized.groupby("symbol", sort=True):
        symbol_dir = tree_root / f"symbol={symbol}"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = symbol_dir / "ohlcv.parquet"
        outgoing = symbol_frame[SYMBOL_TREE_COLUMNS].copy()
        outgoing["symbol"] = str(symbol)
        if parquet_path.exists():
            existing = pd.read_parquet(parquet_path)
            _validate_symbol_tree_frame(
                frame=existing,
                parquet_path=parquet_path,
                expected_symbol=str(symbol),
            )
            existing = existing.copy()
            existing["date"] = pd.to_datetime(existing["date"]).dt.date.astype(str)
            if "symbol" not in existing.columns:
                existing["symbol"] = str(symbol)
                existing = existing[SYMBOL_TREE_COLUMNS]
            outgoing = pd.concat([existing, outgoing], ignore_index=True)
        outgoing = (
            outgoing.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
        )
        outgoing.to_parquet(parquet_path, index=False)
        written.append(parquet_path)
    if not written:
        raise RuntimeError("Alpaca constituent OHLCV frame produced no symbol parquet files")
    return written


def _infer_symbol_from_path(parquet_path: Path) -> str:
    parent = parquet_path.parent.name
    if not parent.startswith("symbol="):
        raise RuntimeError(f"Could not infer symbol from parquet path: {parquet_path}")
    return parent.split("=", 1)[1]


def _validate_ohlcv_frame(*, frame: pd.DataFrame, parquet_path: Path) -> None:
    got = list(frame.columns)
    if got != EXPECTED_COLUMNS:
        raise RuntimeError(f"Unexpected OHLCV parquet columns in {parquet_path}: {got!r}")


def _validate_symbol_tree_frame(
    *,
    frame: pd.DataFrame,
    parquet_path: Path,
    expected_symbol: str,
) -> None:
    got = list(frame.columns)
    if got == EXPECTED_COLUMNS:
        return
    if got != SYMBOL_TREE_COLUMNS:
        raise RuntimeError(f"Unexpected OHLCV parquet columns in {parquet_path}: {got!r}")
    if frame["symbol"].isna().any():
        raise RuntimeError(f"Unexpected null symbol in {parquet_path}")
    observed = sorted({str(value) for value in frame["symbol"].unique()})
    if observed != [expected_symbol]:
        raise RuntimeError(
            f"Unexpected symbol values in {parquet_path}: expected {expected_symbol}, got {observed!r}"
        )
