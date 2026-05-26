from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
import urllib.error
import urllib.request

import pandas as pd

from regime_data_fetch.acquisition_store import AcquisitionStore
from regime_data_fetch.aggregate_eps_constants import (
    EPS_DIR_NAME,
    EPS_REVISION_LOOKBACK_WEEKS,
    SOURCE_NAME,
    SOURCE_URL,
    SPGLOBAL_EPS_MANUAL_REL_PATH,
    WAYBACK_CDX_URL,
    WAYBACK_DIR_NAME,
    WAYBACK_TIMELINE_FILENAME,
    WEEKLY_HISTORY_FILENAME,
)
from regime_data_fetch.aggregate_eps_models import (
    AggregateEPSFetchError,
    AggregateEPSSnapshot,
    EPSHorizonLabel,
    EPSWaybackSnapshot,
    ParsedAggregateEPSWorkbook,
)
from regime_data_fetch.aggregate_eps_reports import build_aggregate_eps_report
from regime_data_fetch.aggregate_eps_wayback import (
    append_wayback_status as _append_wayback_status,
    filter_wayback_snapshots as _filter_wayback_snapshots,
    parse_wayback_cdx_json as _parse_wayback_cdx_json,
)
from regime_data_fetch.aggregate_eps_workbook import (
    parse_sp500_eps_workbook as _parse_sp500_eps_workbook,
)
from regime_shared.pandas_compat import cow_safe_assign

for _public_type in (
    AggregateEPSFetchError,
    AggregateEPSSnapshot,
    EPSHorizonLabel,
    EPSWaybackSnapshot,
    ParsedAggregateEPSWorkbook,
):
    _public_type.__module__ = __name__
del _public_type


def _close_url_exception(exc: BaseException) -> None:
    close = getattr(exc, "close", None)
    if callable(close):
        close()


def parse_sp500_eps_workbook(workbook_path: Path) -> ParsedAggregateEPSWorkbook:
    return _parse_sp500_eps_workbook(workbook_path, read_excel=pd.read_excel)


def download_spglobal_eps_workbook(
    *,
    out_path: Path,
    source_url: str = SOURCE_URL,
    timeout_seconds: int = 60,
) -> Path:
    """Download the weekly S&P aggregate forward-EPS workbook via direct HTTP.

    Direct S&P requests can hit Akamai 403s; the auto fetcher falls back to a
    browser session or an operator-staged workbook at the canonical raw path.
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
            "Accept": (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
                "application/octet-stream,*/*"
            ),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        try:
            if exc.code == 403:
                raise AggregateEPSFetchError(
                    f"S&P spdji returned HTTP 403 (Akamai bot mitigation) for "
                    f"{source_url}. The URL is browser-only; programmatic clients "
                    f"are blocked. To complete the fetch: (1) open the URL in a "
                    f"browser and download sp-500-eps-est.xlsx; (2) copy it to "
                    f"data/raw/spglobal_eps/sp-500-eps-est.xlsx in this repo; "
                    f"(3) re-run --fetch eps-spglobal-auto — the auto path "
                    f"detects the manually-dropped file and parses it."
                ) from exc
            raise AggregateEPSFetchError(
                f"Failed to download S&P EPS workbook from {source_url}: {exc}"
            ) from exc
        finally:
            _close_url_exception(exc)
    except urllib.error.URLError as exc:
        raise AggregateEPSFetchError(
            f"Failed to download S&P EPS workbook from {source_url}: {exc}"
        ) from exc
    if not payload:
        raise AggregateEPSFetchError(
            f"S&P EPS workbook download from {source_url} returned empty payload"
        )
    out_path.write_bytes(payload)
    return out_path


def download_spglobal_eps_workbook_with_browser(
    *,
    out_path: Path,
    source_url: str = SOURCE_URL,
    timeout_ms: int = 120_000,
    user_data_dir: Path | None = None,
    executable_path: Path | None = None,
    headless: bool = True,
) -> Path:
    """Download the S&P workbook through a persistent browser session."""
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise AggregateEPSFetchError(
            "Playwright is required for browser-backed EPS download. "
            "Install the browser extra and browser runtime, or pre-stage the workbook."
        ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        launch_kwargs: dict[str, object] = {
            "headless": headless,
            "accept_downloads": True,
        }
        if executable_path is not None:
            launch_kwargs["executable_path"] = str(executable_path)

        browser = None
        context = None
        try:
            if user_data_dir is not None:
                context = playwright.chromium.launch_persistent_context(
                    str(user_data_dir),
                    **launch_kwargs,
                )
            else:
                browser = playwright.chromium.launch(
                    headless=headless,
                    executable_path=str(executable_path) if executable_path else None,
                )
                context = browser.new_context(accept_downloads=True)
            page = context.new_page()
            try:
                with page.expect_download(timeout=timeout_ms) as download_info:
                    page.goto(
                        source_url, wait_until="domcontentloaded", timeout=timeout_ms
                    )
                download = download_info.value
                download.save_as(out_path)
            except PlaywrightTimeoutError as exc:
                raise AggregateEPSFetchError(
                    "Browser-backed S&P EPS download did not produce a workbook download before timeout"
                ) from exc
        finally:
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()
    return out_path


def run_aggregate_eps_auto_fetch(
    *,
    out_dir: Path,
    source_url: str = SOURCE_URL,
    acquisition_db_path: Path | None = None,
    artifact_store_root: str | Path | None = None,
    workbook_downloader=download_spglobal_eps_workbook,
    browser_downloader=download_spglobal_eps_workbook_with_browser,
    browser_user_data_dir: Path | None = None,
    browser_executable: Path | None = None,
    browser_headless: bool = True,
    browser_timeout_ms: int = 120_000,
) -> Path:
    """Fetch and parse the latest weekly S&P aggregate-EPS workbook.

    Resolution order is operator-staged workbook, direct HTTP, then browser
    fallback; all paths emit the same parquet and report artifacts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = out_dir / SPGLOBAL_EPS_MANUAL_REL_PATH
    if not workbook_path.exists():
        try:
            workbook_downloader(out_path=workbook_path, source_url=source_url)
        except AggregateEPSFetchError:
            browser_downloader(
                out_path=workbook_path,
                source_url=source_url,
                timeout_ms=browser_timeout_ms,
                user_data_dir=browser_user_data_dir,
                executable_path=browser_executable,
                headless=browser_headless,
            )
    return run_aggregate_eps_fetch(
        out_dir=out_dir,
        workbook_path=workbook_path,
        acquisition_db_path=acquisition_db_path,
        artifact_store_root=artifact_store_root,
    )


def append_weekly_eps_snapshot(
    *,
    eps_dir: Path,
    current_snapshot: AggregateEPSSnapshot,
) -> pd.DataFrame:
    """Append one idempotent weekly snapshot row to the EPS history parquet."""
    history_path = eps_dir / WEEKLY_HISTORY_FILENAME
    new_row = pd.DataFrame(
        [
            {
                "observation_date": current_snapshot.observation_date,
                "observation_label": current_snapshot.observation_label,
                "forward_estimate_value": current_snapshot.forward_estimate_value,
                "source": SOURCE_NAME,
            }
        ]
    )
    if history_path.exists():
        existing = pd.read_parquet(history_path)
        # Drop any prior row for the same observation_date (idempotent
        # re-run), then append the fresh row.
        existing = existing[
            existing["observation_date"] != current_snapshot.observation_date
        ]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined = combined.sort_values("observation_date").reset_index(drop=True)
    combined.to_parquet(history_path, index=False)
    return combined


def seed_weekly_history_from_wayback_timeline(
    *,
    out_dir: Path,
    timeline_path: Path | None = None,
) -> pd.DataFrame:
    """Seed weekly EPS history from Wayback snapshots without clobbering live rows."""
    if timeline_path is None:
        timeline_path = out_dir / WAYBACK_DIR_NAME / WAYBACK_TIMELINE_FILENAME
    if not timeline_path.exists():
        raise AggregateEPSFetchError(
            f"No Wayback EPS timeline at {timeline_path}. Run "
            f"run_wayback_aggregate_eps_fetch first to materialise it."
        )
    timeline = pd.read_parquet(timeline_path)
    if timeline.empty:
        raise AggregateEPSFetchError(
            f"Wayback EPS timeline at {timeline_path} contained no rows"
        )

    seeded = pd.DataFrame(
        {
            "observation_date": timeline["workbook_as_of_date"],
            "observation_label": "wayback_backfill",
            "forward_estimate_value": timeline["forward_estimate_value"],
            "source": "wayback_machine",
        }
    )
    # A single workbook_as_of_date can appear under multiple Wayback
    # snapshots — keep the last (the timeline is sorted ascending by
    # snapshot_date / timestamp, so the last row is the freshest capture).
    seeded = seeded.drop_duplicates(subset=["observation_date"], keep="last")

    eps_dir = out_dir / EPS_DIR_NAME
    eps_dir.mkdir(parents=True, exist_ok=True)
    history_path = eps_dir / WEEKLY_HISTORY_FILENAME
    if history_path.exists():
        existing = pd.read_parquet(history_path)
        # Existing accumulator rows win on collision — a live fetch row is
        # authoritative over a Wayback snapshot for the same date.
        seeded = seeded[~seeded["observation_date"].isin(existing["observation_date"])]
        combined = pd.concat([existing, seeded], ignore_index=True)
    else:
        combined = seeded
    combined = combined.sort_values("observation_date").reset_index(drop=True)
    combined.to_parquet(history_path, index=False)
    return combined


def compute_eps_revision_direction_4w(weekly_history: pd.DataFrame) -> pd.Series:
    """v2 §2B `aggregate_forward_eps_revision_direction_4w` from the
    accumulated weekly history.

    ``revision_4w = (forward_eps[t] - forward_eps[t-4]) / forward_eps[t-4]``
    where ``t-4`` is 4 rows back in the weekly-sorted accumulator (one row
    per weekly fetch). The returned Series is indexed by
    ``observation_date``; values are NaN for the first
    ``EPS_REVISION_LOOKBACK_WEEKS`` rows (cold-start) and wherever the
    4-weeks-prior estimate is NaN or zero.

    Until the accumulator holds more than ``EPS_REVISION_LOOKBACK_WEEKS``
    weekly rows the entire series is NaN — the §2B earnings labels stay
    silent, which is the correct cold-start behaviour (V1 §2.7).
    """
    if weekly_history.empty:
        return pd.Series(
            dtype=float, name="aggregate_forward_eps_revision_direction_4w"
        )
    sorted_history = weekly_history.sort_values("observation_date").reset_index(
        drop=True
    )
    forward_eps = sorted_history["forward_estimate_value"].astype(float)
    prior = forward_eps.shift(EPS_REVISION_LOOKBACK_WEEKS)
    revision = (forward_eps - prior) / prior.where(prior != 0)
    revision.index = pd.DatetimeIndex(
        pd.to_datetime(sorted_history["observation_date"])
    )
    revision.name = "aggregate_forward_eps_revision_direction_4w"
    return revision


def run_aggregate_eps_fetch(
    *,
    out_dir: Path,
    workbook_path: Path,
    acquisition_db_path: Path | None = None,
    artifact_store_root: str | Path | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    store = (
        AcquisitionStore(acquisition_db_path, artifact_store_root=artifact_store_root)
        if acquisition_db_path
        else None
    )
    fetch_run = (
        store.start_fetch_run(
            fetch_type="aggregate_eps",
            params={
                "workbook_path": str(workbook_path),
            },
        )
        if store
        else None
    )

    try:
        if store and fetch_run:
            store.record_file_artifact(
                run_id=fetch_run.run_id,
                source_name=SOURCE_NAME,
                artifact_kind=f"{workbook_path.suffix.lower().lstrip('.')}_manual",
                source_identifier=str(workbook_path),
                file_path=workbook_path,
                timezone="America/New_York",
                license_note="Manually downloaded workbook; public files reported discontinued by source workbook",
                notes="Manual S&P aggregate EPS workbook snapshot",
            )

        parsed = parse_sp500_eps_workbook(workbook_path)

        rows = [*parsed.historical_snapshots, parsed.current_snapshot]
        df = pd.DataFrame(
            [
                {
                    "workbook_as_of_date": parsed.workbook_as_of_date,
                    "observation_date": row.observation_date,
                    "observation_label": row.observation_label,
                    "forward_estimate_label": row.forward_estimate_label,
                    "forward_estimate_value": row.forward_estimate_value,
                    "estimate_2025e": row.estimate_2025e,
                    "estimate_q4_2025e": row.estimate_q4_2025e,
                    "estimate_2026e": row.estimate_2026e,
                    "price": row.price,
                    "pe_2025e": row.pe_2025e,
                    "pe_2026e": row.pe_2026e,
                    "change_vs_prior_observation_2025e": row.change_vs_prior_observation_2025e,
                    "change_vs_prior_observation_q4_2025e": row.change_vs_prior_observation_q4_2025e,
                    "change_vs_prior_observation_2026e": row.change_vs_prior_observation_2026e,
                    "change_vs_prior_observation_price": row.change_vs_prior_observation_price,
                    "change_vs_prior_observation_pe_2025e": row.change_vs_prior_observation_pe_2025e,
                    "change_vs_prior_observation_pe_2026e": row.change_vs_prior_observation_pe_2026e,
                    "source": SOURCE_NAME,
                    "source_path": str(workbook_path),
                    "public_files_discontinued": parsed.public_files_discontinued,
                }
                for row in rows
            ]
        )

        eps_dir = out_dir / EPS_DIR_NAME
        eps_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = eps_dir / "sp500_eps_snapshots.parquet"
        df.to_parquet(parquet_path, index=False)

        # Weekly-snapshot accumulator (documented implementation decision). Append this run's
        # current snapshot to the persistent weekly-history parquet, then
        # compute the 4-week revision direction. The revision series is
        # all-NaN until > EPS_REVISION_LOOKBACK_WEEKS weekly rows have
        # accumulated — the report's availability flag reflects that.
        weekly_history = append_weekly_eps_snapshot(
            eps_dir=eps_dir, current_snapshot=parsed.current_snapshot
        )
        weekly_history_path = eps_dir / WEEKLY_HISTORY_FILENAME
        revision_series = compute_eps_revision_direction_4w(weekly_history)
        revision_available = bool(revision_series.notna().any())
        report = build_aggregate_eps_report(
            as_of_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            workbook_path=workbook_path,
            parsed=parsed,
            weekly_history=weekly_history,
            revision_available=revision_available,
            parquet_path=parquet_path,
            weekly_history_path=weekly_history_path,
            acquisition_db_path=acquisition_db_path,
        )
        report_path = out_dir / "aggregate_eps_fetch_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if store and fetch_run:
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="aggregate_eps_parquet",
                path=parquet_path,
                row_count=len(df),
                min_date=(
                    min(df["observation_date"]).isoformat() if not df.empty else None
                ),
                max_date=(
                    max(df["observation_date"]).isoformat() if not df.empty else None
                ),
                notes="Aggregate EPS workbook snapshots parquet",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="aggregate_eps_report",
                path=report_path,
                row_count=len(df),
                min_date=(
                    min(df["observation_date"]).isoformat() if not df.empty else None
                ),
                max_date=(
                    max(df["observation_date"]).isoformat() if not df.empty else None
                ),
                notes="Aggregate EPS fetch report",
            )
            store.finish_fetch_run(run_id=fetch_run.run_id, status="ok")
        return report_path
    except Exception as exc:
        if store and fetch_run:
            store.finish_fetch_run(
                run_id=fetch_run.run_id, status="failed", notes=str(exc)
            )
        raise


def parse_wayback_cdx_json(
    cdx_json: str, *, target_url: str
) -> list[EPSWaybackSnapshot]:
    return _parse_wayback_cdx_json(cdx_json, target_url=target_url)


def fetch_wayback_cdx(
    target_url: str = SOURCE_URL,
    *,
    max_attempts: int = 3,
    backoff_seconds: float = 2.0,
) -> str:
    query = (
        f"{WAYBACK_CDX_URL}?url={target_url}&output=json"
        "&fl=timestamp,original,statuscode,mimetype&filter=statuscode:200"
    )
    req = urllib.request.Request(query, headers={"User-Agent": "Mozilla/5.0"})
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504}:
                try:
                    raise
                finally:
                    _close_url_exception(exc)
            last_exc = exc
        except urllib.error.URLError as exc:
            last_exc = exc
        if attempt < max_attempts:
            if last_exc is not None:
                _close_url_exception(last_exc)
            time.sleep(backoff_seconds * attempt)
            last_exc = None
    raise AggregateEPSFetchError(
        f"Wayback CDX fetch failed after {max_attempts} attempts: {last_exc}"
    ) from last_exc


def fetch_wayback_snapshot_bytes(snapshot: EPSWaybackSnapshot) -> bytes:
    req = urllib.request.Request(
        snapshot.archive_url, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def run_wayback_aggregate_eps_fetch(
    *,
    out_dir: Path,
    max_snapshots: int | None = None,
    from_date: dt.date | None = None,
    to_date: dt.date | None = None,
    stop_after_first_success: bool = False,
    acquisition_db_path: Path | None = None,
    artifact_store_root: str | Path | None = None,
    cdx_fetcher=fetch_wayback_cdx,
    snapshot_fetcher=fetch_wayback_snapshot_bytes,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    store = (
        AcquisitionStore(acquisition_db_path, artifact_store_root=artifact_store_root)
        if acquisition_db_path
        else None
    )
    fetch_run = (
        store.start_fetch_run(
            fetch_type="aggregate_eps_wayback",
            params={
                "max_snapshots": max_snapshots,
                "from_date": from_date.isoformat() if from_date else None,
                "to_date": to_date.isoformat() if to_date else None,
                "stop_after_first_success": stop_after_first_success,
                "source_url": SOURCE_URL,
            },
        )
        if store
        else None
    )

    try:
        cdx_json = cdx_fetcher()
        snapshots = parse_wayback_cdx_json(cdx_json, target_url=SOURCE_URL)
        snapshots = _filter_wayback_snapshots(
            snapshots,
            from_date=from_date,
            to_date=to_date,
            max_snapshots=max_snapshots,
        )

        wayback_dir = out_dir / WAYBACK_DIR_NAME
        snapshots_dir = wayback_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        snapshot_index_path = wayback_dir / "wayback_snapshot_index.json"
        snapshot_index_path.write_text(
            json.dumps(
                [
                    {
                        "snapshot_date": snapshot.snapshot_date.isoformat(),
                        "timestamp": snapshot.timestamp,
                        "archive_url": snapshot.archive_url,
                    }
                    for snapshot in snapshots
                ],
                indent=2,
            )
        )
        status_path = wayback_dir / "snapshot_status.jsonl"

        if store and fetch_run:
            store.record_text_artifact(
                run_id=fetch_run.run_id,
                source_name="wayback:cdx",
                artifact_kind="json",
                source_identifier=SOURCE_URL,
                content_text=cdx_json,
                start_date=from_date.isoformat() if from_date else None,
                end_date=to_date.isoformat() if to_date else None,
                timezone="UTC",
                license_note="Wayback CDX listing for archived S&P aggregate EPS workbook snapshots",
                notes="Wayback CDX listing persisted before filtered snapshot materialization",
            )

        timeline_rows: list[dict[str, object]] = []
        downloaded = 0
        failures = 0
        parsed_ok = 0
        for snapshot in snapshots:
            workbook_path = snapshots_dir / f"{snapshot.timestamp}.xlsx"
            try:
                if workbook_path.exists():
                    status = "download_reused"
                else:
                    payload = snapshot_fetcher(snapshot)
                    workbook_path.write_bytes(payload)
                    downloaded += 1
                    status = "downloaded"

                if store and fetch_run:
                    store.record_file_artifact(
                        run_id=fetch_run.run_id,
                        source_name="wayback:eps_workbook",
                        artifact_kind="xlsx_wayback",
                        source_identifier=snapshot.timestamp,
                        file_path=workbook_path,
                        effective_date=snapshot.snapshot_date.isoformat(),
                        timezone="UTC",
                        license_note="Archived S&P aggregate EPS workbook snapshot fetched from Wayback Machine",
                        notes=f"Wayback workbook snapshot {status}",
                    )

                parsed = parse_sp500_eps_workbook(workbook_path)
                current = parsed.current_snapshot
                timeline_rows.append(
                    {
                        "snapshot_date": snapshot.snapshot_date,
                        "timestamp": snapshot.timestamp,
                        "archive_url": snapshot.archive_url,
                        "workbook_as_of_date": parsed.workbook_as_of_date,
                        "forward_estimate_label": current.forward_estimate_label,
                        "forward_estimate_value": current.forward_estimate_value,
                        "estimate_2025e": current.estimate_2025e,
                        "estimate_q4_2025e": current.estimate_q4_2025e,
                        "estimate_2026e": current.estimate_2026e,
                        "price": current.price,
                        "pe_2025e": current.pe_2025e,
                        "pe_2026e": current.pe_2026e,
                        "change_vs_prior_observation_2025e": current.change_vs_prior_observation_2025e,
                        "change_vs_prior_observation_q4_2025e": current.change_vs_prior_observation_q4_2025e,
                        "change_vs_prior_observation_2026e": current.change_vs_prior_observation_2026e,
                        "change_vs_prior_observation_price": current.change_vs_prior_observation_price,
                        "change_vs_prior_observation_pe_2025e": current.change_vs_prior_observation_pe_2025e,
                        "change_vs_prior_observation_pe_2026e": current.change_vs_prior_observation_pe_2026e,
                        "public_files_discontinued": parsed.public_files_discontinued,
                        "source": "wayback_machine",
                    }
                )
                parsed_ok += 1
                _append_wayback_status(
                    status_path,
                    snapshot=snapshot,
                    status="parsed_ok",
                    detail=status,
                )
                if stop_after_first_success:
                    break
            except Exception as exc:
                failures += 1
                _append_wayback_status(
                    status_path,
                    snapshot=snapshot,
                    status="failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                continue

        if not timeline_rows:
            raise AggregateEPSFetchError(
                "Wayback EPS backfill produced no parsed timeline rows"
            )

        timeline_df = (
            pd.DataFrame(timeline_rows)
            .sort_values(["snapshot_date", "timestamp"])
            .reset_index(drop=True)
        )
        timeline_path = wayback_dir / WAYBACK_TIMELINE_FILENAME
        timeline_df.to_parquet(timeline_path, index=False)
        weekly_history = seed_weekly_history_from_wayback_timeline(
            out_dir=out_dir,
            timeline_path=timeline_path,
        )
        weekly_history_path = out_dir / EPS_DIR_NAME / WEEKLY_HISTORY_FILENAME
        revision_series = compute_eps_revision_direction_4w(weekly_history)
        revision_available = bool(revision_series.notna().any())

        preview = timeline_df.head(10).copy()
        replacements: dict[str, pd.Series] = {}
        if "snapshot_date" in preview:
            replacements["snapshot_date"] = preview["snapshot_date"].map(
                lambda x: x.isoformat()
            )
        if "workbook_as_of_date" in preview:
            replacements["workbook_as_of_date"] = preview["workbook_as_of_date"].map(
                lambda x: x.isoformat()
            )
        if replacements:
            preview = cow_safe_assign(preview, replacements)

        report = {
            "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": "wayback_machine",
            "source_url": SOURCE_URL,
            "counts": {
                "snapshots_listed": len(snapshots),
                "snapshots_downloaded": downloaded,
                "snapshots_failed": failures,
                "snapshots_parsed_ok": parsed_ok,
                "timeline_rows": int(len(timeline_df)),
                "weekly_history_rows": int(len(weekly_history)),
            },
            "limitations": {
                "aggregate_forward_eps_revision_direction_4w_available": revision_available,
                "revision_cold_start_weeks": EPS_REVISION_LOOKBACK_WEEKS,
            },
            "requested": {
                "max_snapshots": max_snapshots,
                "from_date": from_date.isoformat() if from_date else None,
                "to_date": to_date.isoformat() if to_date else None,
                "stop_after_first_success": stop_after_first_success,
            },
            "timeline_preview": preview.to_dict(orient="records"),
            "paths": {
                "snapshots_dir": str(snapshots_dir),
                "snapshot_index_json": str(snapshot_index_path),
                "snapshot_status_jsonl": str(status_path),
                "timeline_parquet": str(timeline_path),
                "aggregate_eps_weekly_history_parquet": str(weekly_history_path),
                "acquisition_db": (
                    str(acquisition_db_path) if acquisition_db_path else None
                ),
            },
        }
        report_path = out_dir / "aggregate_eps_wayback_fetch_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        if store and fetch_run:
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="aggregate_eps_wayback_snapshot_index",
                path=snapshot_index_path,
                row_count=len(snapshots),
                min_date=(
                    min(snapshot.snapshot_date for snapshot in snapshots).isoformat()
                    if snapshots
                    else None
                ),
                max_date=(
                    max(snapshot.snapshot_date for snapshot in snapshots).isoformat()
                    if snapshots
                    else None
                ),
                notes="Filtered Wayback EPS snapshot index",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="aggregate_eps_wayback_status",
                path=status_path,
                row_count=parsed_ok + failures,
                notes="Wayback EPS per-snapshot status log",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="aggregate_eps_wayback_timeline",
                path=timeline_path,
                row_count=len(timeline_df),
                min_date=(
                    min(timeline_df["snapshot_date"]).isoformat()
                    if not timeline_df.empty
                    else None
                ),
                max_date=(
                    max(timeline_df["snapshot_date"]).isoformat()
                    if not timeline_df.empty
                    else None
                ),
                notes="Wayback EPS historical snapshot timeline parquet",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="aggregate_eps_weekly_history",
                path=weekly_history_path,
                row_count=len(weekly_history),
                min_date=(
                    min(weekly_history["observation_date"]).isoformat()
                    if not weekly_history.empty
                    else None
                ),
                max_date=(
                    max(weekly_history["observation_date"]).isoformat()
                    if not weekly_history.empty
                    else None
                ),
                notes="Aggregate EPS weekly accumulator seeded from Wayback timeline",
            )
            store.record_output(
                run_id=fetch_run.run_id,
                output_kind="aggregate_eps_wayback_report",
                path=report_path,
                row_count=len(timeline_df),
                min_date=(
                    min(timeline_df["snapshot_date"]).isoformat()
                    if not timeline_df.empty
                    else None
                ),
                max_date=(
                    max(timeline_df["snapshot_date"]).isoformat()
                    if not timeline_df.empty
                    else None
                ),
                notes="Wayback EPS fetch report",
            )
            store.finish_fetch_run(run_id=fetch_run.run_id, status="ok")
        return report_path
    except Exception as exc:
        if store and fetch_run:
            store.finish_fetch_run(
                run_id=fetch_run.run_id, status="failed", notes=str(exc)
            )
        raise
