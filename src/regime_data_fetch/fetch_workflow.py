from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.alpaca_daily import fetch_daily_bars_alpaca, verify_min_start_date
from regime_data_fetch.fred import fetch_fred_series_json, parse_fred_series_json

V2_V1_SHARED_ANCHORS = ["SPY", "RSP"]
V2_SECTOR_SYMBOLS = [
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]
V2_CROSS_ASSET_SYMBOLS = [
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "HYG",
    "LQD",
    "GLD",
    "USO",
    "UUP",
    # Bloomberg Commodity Index substitute per Ambiguity Log #48; consumed
    # by v2 §2B `commodity_return_63d` feature in slice 5.
    "DBC",
]
V2_EXTRA_SYMBOLS = ["KRE"] + V2_SECTOR_SYMBOLS + V2_CROSS_ASSET_SYMBOLS
V2_FRED_SERIES = {
    "2y_yield": "DGS2",
    "10y_yield": "DGS10",
    "broad_usd_index": "DTWEXBGS",
    "sofr": "SOFR",
    "nfci": "NFCI",
    "cpi_all_items": "CPIAUCSL",
    "iorb": "IORB",
    # GDPNow nowcast (Atlanta Fed). Free on FRED at series_id GDPNOW.
    # Not consumed by any v2 §2B rule predicate as of slice 5 ship; ingested
    # here for the future-amendment slice that would use it as additional
    # recession_scare / recovery_growth evidence. Listed early in the slice
    # cadence so it lands in archived inputs before any spec amendment.
    "gdp_nowcast": "GDPNOW",
    # ICE BofA Option-Adjusted Spread series — FRED redistributes under
    # license from ICE Indices, free at the FRED endpoint. SINGLE SOURCE
    # for the v2 §2C `hy_oas_63d` / `ig_oas_63d` metrics
    # (Log #49 closure). `credit_funding` lists these in REQUIRED_MACRO_KEYS,
    # so the §2C seam does not build without them — there is no proxy
    # fallback path.
    "hy_oas": "BAMLH0A0HYM2",       # ICE BofA US High Yield Index OAS
    "ig_bbb_oas": "BAMLC0A4CBBB",   # ICE BofA BBB US Corporate Index OAS
    # CBOE VIX — the model-free 30-day implied vol on SPX, free on FRED.
    # §1C `vol_crush` consumes it as `implied_vol_30d = VIXCLS / 100`
    # (decimal-annualized, to match realized_vol units). ADR 0005.
    "implied_vol_30d": "VIXCLS",
}


def _dedupe_preserve_order(symbols: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def build_market_symbols(
    *,
    scope: str,
    stock_symbols: list[str],
    vix_symbol: str,
) -> list[str]:
    if scope not in {"v1", "v2", "all"}:
        raise ValueError(f"Unknown scope: {scope!r}")

    symbols: list[str] = []
    if scope in {"v1", "all"}:
        symbols.extend(stock_symbols)
        symbols.extend(V2_V1_SHARED_ANCHORS)
    if scope in {"v2", "all"}:
        symbols.extend(V2_V1_SHARED_ANCHORS)
        symbols.extend(V2_EXTRA_SYMBOLS)
    symbols.append(vix_symbol)
    return _dedupe_preserve_order(symbols)


def write_event_calendar_template(out_dir: Path) -> Path:
    event_dir = out_dir / "event_calendar"
    event_dir.mkdir(parents=True, exist_ok=True)
    template_path = event_dir / "events.template.yaml"
    template_path.write_text(
        "\n".join(
            [
                "events:",
                '  - date: "2026-01-28"',
                '    market: "US"',
                '    type: "FOMC"',
                '    importance: "high"',
                '  - date: "2026-02-20"',
                '    market: "US"',
                '    type: "monthly_options_expiry"',
                '    importance: "medium"',
                '  - date: "2026-11-03"',
                '    market: "US"',
                '    type: "election"',
                '    importance: "high"',
                "    window_days: [-5, +10]",
                '  - date: "2026-12-10"',
                '    market: "GLOBAL"',
                '    type: "global_rate_decision"',
                '    importance: "medium"',
                '  - date: "2026-06-15"',
                '    market: "GLOBAL"',
                '    type: "geopolitical_event"',
                '    importance: "high"',
            ]
        )
        + "\n"
    )
    return template_path


def run_market_fetch(
    *,
    out_dir: Path,
    scope: str,
    stock_symbols: list[str],
    start: dt.date,
    end: dt.date,
    adjustment: str,
    alpaca_feed: str | None,
    vix_symbol: str,
    allow_vix_proxy: bool,
    verbose: bool,
    acquisition_db_path: Path | None = None,
    artifact_store_root: str | Path | None = None,
) -> Path:
    if end < start:
        raise SystemExit("--end must be >= --start")

    for key in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"):
        if not os.environ.get(key, "").strip():
            raise SystemExit(f"Missing required env var: {key}")

    out_dir.mkdir(parents=True, exist_ok=True)

    all_symbols = build_market_symbols(
        scope=scope,
        stock_symbols=stock_symbols,
        vix_symbol=vix_symbol,
    )

    store = AcquisitionStore(acquisition_db_path, artifact_store_root=artifact_store_root) if acquisition_db_path else None
    fetch_run = (
        store.start_fetch_run(
            fetch_type="market",
            params={
                "scope": scope,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "adjustment": adjustment,
                "alpaca_feed": alpaca_feed,
                "vix_symbol": vix_symbol,
                "allow_vix_proxy": allow_vix_proxy,
                "symbols_requested": len(all_symbols),
            },
        )
        if store
        else None
    )

    try:
        bars = fetch_daily_bars_alpaca(
            symbols=all_symbols,
            start_date=start,
            end_date=end,
            adjustment=adjustment,
            feed=alpaca_feed,
            verbose=verbose,
        )
        df = bars.df

        if store and fetch_run:
            payload = {
                "requested": {
                    "scope": scope,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "adjustment": adjustment,
                    "alpaca_feed": alpaca_feed,
                    "vix_symbol": vix_symbol,
                    "allow_vix_proxy": allow_vix_proxy,
                },
                "symbols_requested": all_symbols,
                "missing_symbols": bars.missing_symbols,
                "rows": [
                    {
                        "date": row["date"].isoformat() if hasattr(row["date"], "isoformat") else str(row["date"]),
                        "symbol": row["symbol"],
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                        "adjusted_close": row["adjusted_close"],
                    }
                    for row in df.to_dict(orient="records")
                ],
            }
            store.record_text_artifact(
                run_id=fetch_run.run_id,
                source_name="alpaca:daily_bars",
                artifact_kind="json",
                source_identifier=f"daily_bars:{scope}:{start.isoformat()}:{end.isoformat()}:{adjustment}:{alpaca_feed or 'default'}",
                content_text=json.dumps(payload, separators=(",", ":")),
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                timezone="UTC",
                adjustment_policy=adjustment,
                license_note="Alpaca market data response normalized at fetch boundary",
                notes="Daily bars fetch result persisted from the Alpaca fetch boundary",
            )

        have_vix = bool(not df.empty and (df["symbol"] == vix_symbol).any())
        if not have_vix and not allow_vix_proxy:
            raise SystemExit(
                f"{vix_symbol} not returned by Alpaca. "
                "If you want to proceed with an Alpaca-tradable proxy, rerun with "
                "`--vix-symbol VIXY --allow-vix-proxy`."
            )

        parquet_dir = out_dir / "daily_ohlcv"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_dir, index=False, partition_cols=["symbol"])

        event_template = write_event_calendar_template(out_dir)

        checks: dict[str, dict[str, object]] = {}
        for symbol in V2_V1_SHARED_ANCHORS:
            min_date, ok = verify_min_start_date(df, symbol=symbol, required_start=start)
            checks[symbol] = {"min_date": str(min_date) if min_date else None, "ok": ok}

        report = {
            "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scope": scope,
            "requested": {
                "start": str(start),
                "end": str(end),
                "adjustment": adjustment,
                "alpaca_feed": alpaca_feed,
            },
            "counts": {
                "rows": int(len(df)),
                "symbols_requested_for_alpaca": len(all_symbols),
                "symbols_returned": int(df["symbol"].nunique()) if not df.empty else 0,
                "missing_symbols": len(bars.missing_symbols),
            },
            "min_date_checks": checks,
            "vix": {
                "source": "alpaca",
                "symbol": vix_symbol,
                "rows": int((df["symbol"] == vix_symbol).sum()),
            },
            "missing_symbols_sample": bars.missing_symbols[:50],
            "paths": {
                "daily_ohlcv_parquet": str(parquet_dir),
                "event_calendar_template": str(event_template),
                "acquisition_db": str(acquisition_db_path) if acquisition_db_path else None,
            },
        }
        report_path = out_dir / "fetch_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if store and fetch_run:
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="alpaca_daily_ohlcv_parquet",
                path=parquet_dir / "_metadata" if (parquet_dir / "_metadata").exists() else next(parquet_dir.rglob("*.parquet")),
                row_count=len(df),
                min_date=min(df["date"]).isoformat() if not df.empty else None,
                max_date=max(df["date"]).isoformat() if not df.empty else None,
                notes="Partitioned Alpaca daily OHLCV parquet output",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="alpaca_market_fetch_report",
                path=report_path,
                row_count=len(df),
                min_date=min(df["date"]).isoformat() if not df.empty else None,
                max_date=max(df["date"]).isoformat() if not df.empty else None,
                notes="Market fetch report",
            )
            store.finish_fetch_run(run_id=fetch_run.run_id, status="ok")
        return report_path
    except Exception as exc:
        if store and fetch_run:
            store.finish_fetch_run(run_id=fetch_run.run_id, status="failed", notes=str(exc))
        raise


def run_macro_fetch(
    *,
    out_dir: Path,
    start: dt.date,
    end: dt.date,
    fred_api_key: str | None,
    include_cpi_vintages: bool,
    acquisition_db_path: Path | None = None,
    artifact_store_root: str | Path | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_fred_api_key = fred_api_key or os.environ.get("FRED_API_KEY", "").strip() or None
    if not effective_fred_api_key:
        raise SystemExit("Missing required FRED API key: pass --fred-api-key or set FRED_API_KEY")

    store = AcquisitionStore(acquisition_db_path, artifact_store_root=artifact_store_root) if acquisition_db_path else None
    fetch_run = (
        store.start_fetch_run(
            fetch_type="macro",
            params={
                "start": start.isoformat(),
                "end": end.isoformat(),
                "include_cpi_vintages": include_cpi_vintages,
            },
        )
        if store
        else None
    )

    try:
        macro_frames: list[pd.DataFrame] = []
        series_meta: dict[str, dict[str, object]] = {}
        for logical_name, series_id in V2_FRED_SERIES.items():
            raw_json = fetch_fred_series_json(
                series_id=series_id,
                start_date=start,
                end_date=end,
                api_key=effective_fred_api_key,
            )
            if store and fetch_run:
                store.record_text_artifact(
                    run_id=fetch_run.run_id,
                    source_name=f"fred:{series_id}",
                    artifact_kind="json",
                    source_identifier=f"series_observations:{series_id}:{start.isoformat()}:{end.isoformat()}",
                    content_text=raw_json,
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    timezone="UTC",
                    license_note="FRED public API response",
                    notes=f"Raw FRED observations for logical_name={logical_name}",
                )
            df = parse_fred_series_json(raw_json, series_id=series_id)
            df["logical_name"] = logical_name
            macro_frames.append(df)
            series_meta[logical_name] = {
                "series_id": series_id,
                "rows": int(len(df)),
                "min_date": str(df["date"].min()) if not df.empty else None,
                "max_date": str(df["date"].max()) if not df.empty else None,
            }

        vintages_path: Path | None = None
        if include_cpi_vintages:
            raw_vintages_json = fetch_fred_series_json(
                series_id=V2_FRED_SERIES["cpi_all_items"],
                start_date=start,
                end_date=end,
                api_key=effective_fred_api_key,
                realtime_start="1776-07-04",
                realtime_end="9999-12-31",
            )
            if store and fetch_run:
                store.record_text_artifact(
                    run_id=fetch_run.run_id,
                    source_name=f"fred:{V2_FRED_SERIES['cpi_all_items']}",
                    artifact_kind="json",
                    source_identifier=f"series_observations_realtime:{V2_FRED_SERIES['cpi_all_items']}:{start.isoformat()}:{end.isoformat()}:1776-07-04:9999-12-31",
                    content_text=raw_vintages_json,
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    timezone="UTC",
                    license_note="FRED public API realtime observations response",
                    notes="Raw FRED realtime observations for CPI vintages",
                )
            vintages = parse_fred_series_json(raw_vintages_json, series_id=V2_FRED_SERIES["cpi_all_items"])
            vintages["logical_name"] = "cpi_all_items_vintages"
            vintages_dir = out_dir / "macro_vintages"
            vintages_dir.mkdir(parents=True, exist_ok=True)
            vintages_path = vintages_dir / "cpi_all_items_vintages.parquet"
            vintages.to_parquet(vintages_path, index=False)
            series_meta["cpi_all_items_vintages"] = {
                "series_id": V2_FRED_SERIES["cpi_all_items"],
                "rows": int(len(vintages)),
                "min_date": str(vintages["date"].min()) if not vintages.empty else None,
                "max_date": str(vintages["date"].max()) if not vintages.empty else None,
            }

        macro_dir = out_dir / "macro"
        macro_dir.mkdir(parents=True, exist_ok=True)
        macro_path = macro_dir / "fred_macro_series.parquet"
        macro_df = pd.concat(macro_frames, ignore_index=True)
        macro_df.to_parquet(macro_path, index=False)

        report = {
            "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "requested": {
                "start": str(start),
                "end": str(end),
                "include_cpi_vintages": include_cpi_vintages,
            },
            "series": series_meta,
            "paths": {
                "macro_parquet": str(macro_path),
                "cpi_vintages_parquet": str(vintages_path) if vintages_path else None,
                "acquisition_db": str(acquisition_db_path) if acquisition_db_path else None,
            },
        }
        report_path = out_dir / "macro_fetch_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if store and fetch_run:
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="fred_macro_parquet",
                path=macro_path,
                row_count=len(macro_df),
                min_date=min(macro_df["date"]).isoformat() if not macro_df.empty else None,
                max_date=max(macro_df["date"]).isoformat() if not macro_df.empty else None,
                notes="Unified FRED macro parquet",
            )
            if vintages_path is not None:
                store.record_output(
                    run_id=fetch_run.run_id,
                    output_kind="fred_cpi_vintages_parquet",
                    path=vintages_path,
                    row_count=series_meta["cpi_all_items_vintages"]["rows"],
                    min_date=series_meta["cpi_all_items_vintages"]["min_date"],
                    max_date=series_meta["cpi_all_items_vintages"]["max_date"],
                    notes="FRED CPI vintages parquet",
                )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="fred_macro_report",
                path=report_path,
                row_count=len(macro_df),
                min_date=min(macro_df["date"]).isoformat() if not macro_df.empty else None,
                max_date=max(macro_df["date"]).isoformat() if not macro_df.empty else None,
                notes="Macro fetch report",
            )
            store.finish_fetch_run(run_id=fetch_run.run_id, status="ok")
        return report_path
    except Exception as exc:
        if store and fetch_run:
            store.finish_fetch_run(run_id=fetch_run.run_id, status="failed", notes=str(exc))
        raise
