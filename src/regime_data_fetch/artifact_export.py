from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false

import datetime as dt
import json
import re
from pathlib import Path
from typing import Iterable

from regime_data_fetch.artifact_manifest import (
    ArtifactManifest,
    ManifestArtifact,
    strip_data_raw_prefix,
    write_manifest,
)
from regime_data_fetch.artifact_store import build_artifact_store

REPORT_PATH_NAME_TO_ARTIFACT_NAME = {
    "macro_parquet": "fred_macro_series",
    "cpi_vintages_parquet": "cpi_all_items_vintages",
    "pit_constituents_parquet": "sp500_pit_constituents",
    "event_calendar_yaml": "event_calendar_us",
    "us_events_yaml": "event_calendar_us",
    "pmi_history_parquet": "ism_pmi_history",
    "pmi_parquet": "ism_pmi_latest",
    "sentiment_parquet": "aaii_sentiment",
    "news_sentiment_parquet": "sf_fed_news_sentiment",
    "fomc_minutes_parquet": "fomc_minutes",
    "powell_speeches_parquet": "powell_speeches",
    "cpi_nowcast_parquet": "cleveland_fed_cpi_nowcast",
    "aggregate_eps_parquet": "sp500_eps_snapshots",
    "aggregate_eps_weekly_history_parquet": "sp500_eps_weekly_history",
}


def emit_manifest_for_report_paths(
    *,
    report_paths: Iterable[Path],
    out_dir: Path,
    artifact_store_root: str,
    manifest_path: Path,
    artifact_set: str,
    required_for: list[str],
    repo_root: Path | None = None,
) -> ArtifactManifest:
    store = build_artifact_store(artifact_store_root)
    artifacts: list[ManifestArtifact] = []
    seen_local_paths: set[str] = set()
    for report_path in report_paths:
        payload = _load_report_payload(report_path)
        if payload is None:
            continue
        exportable_for_report = 0
        for name, path, local_path_override in _iter_existing_report_files(payload):
            local_path = _local_path_for(
                path=path,
                out_dir=out_dir,
                repo_root=repo_root,
                local_path_override=local_path_override,
            )
            if local_path is None:
                continue
            exportable_for_report += 1
            if local_path in seen_local_paths:
                continue
            seen_local_paths.add(local_path)
            key = _store_key_for(local_path)
            stored = store.put_file(path, key)
            artifact_name = _canonical_artifact_name(name=name, local_path=local_path)
            artifacts.append(
                ManifestArtifact.from_dict(
                    {
                        "name": artifact_name,
                        "stage": "canonical",
                        "uri": stored.uri,
                        "local_path": local_path,
                        "sha256": stored.sha256,
                        "schema_version": None,
                        "rows": None,
                        "min_date": None,
                        "max_date": None,
                        "required_for": required_for,
                    }
                )
            )
        if exportable_for_report == 0 and payload.get("materializable") is not False:
            raise ValueError(f"no exportable artifact files in report: {report_path}")
    if not artifacts:
        raise ValueError("no existing artifact files found in report paths")
    manifest = ArtifactManifest(
        artifact_set=artifact_set,
        created_at_utc=dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        storage_root=artifact_store_root,
        artifacts=artifacts,
    )
    write_manifest(manifest, manifest_path)
    return manifest


def _load_report_payload(report_path: Path) -> dict[str, object] | None:
    if not report_path.exists():
        raise FileNotFoundError(f"manifest report path does not exist: {report_path}")
    if report_path.suffix.lower() != ".json":
        raise ValueError(f"manifest report path must be JSON: {report_path}")
    payload = json.loads(report_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(
            f"manifest report payload must be a JSON object: {report_path}"
        )
    return payload


def _iter_existing_report_files(
    payload: dict[str, object],
) -> Iterable[tuple[str, Path, str | None]]:
    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        return
    for name, value in sorted(paths.items()):
        if name in {"acquisition_db"}:
            continue
        entry = _parse_report_path_entry(value)
        if entry is None:
            continue
        path, local_path_override = entry
        if path.exists() and path.is_file():
            yield name, path, local_path_override
        elif path.exists() and path.is_dir():
            for child in sorted(item for item in path.rglob("*") if item.is_file()):
                child_name = (
                    f"{name}_{child.relative_to(path).as_posix().replace('/', '_')}"
                )
                child_local_path = None
                if local_path_override is not None:
                    child_local_path = str(
                        Path(local_path_override) / child.relative_to(path)
                    )
                yield child_name, child, child_local_path


def _parse_report_path_entry(value: object) -> tuple[Path, str | None] | None:
    if isinstance(value, str):
        return Path(value), None
    if not isinstance(value, dict):
        return None
    path_value = value.get("path")
    local_path_value = value.get("local_path")
    if not isinstance(path_value, str) or not isinstance(local_path_value, str):
        return None
    return Path(path_value), _normalize_manifest_local_path(local_path_value)


def _local_path_for(
    *,
    path: Path,
    out_dir: Path,
    repo_root: Path | None = None,
    local_path_override: str | None = None,
) -> str | None:
    if local_path_override is not None:
        return _normalize_manifest_local_path(local_path_override)
    path = path.resolve()
    out_dir = out_dir.resolve()
    try:
        relative = path.relative_to(out_dir)
    except ValueError:
        if repo_root is None:
            return None
        try:
            repo_relative = path.relative_to(repo_root.resolve())
        except ValueError:
            return None
        if repo_relative == Path("configs") / "events" / "us_events.yaml":
            return str(Path("data") / "raw" / "event_calendar" / "us_events.yaml")
        return str(repo_relative)
    return str(Path("data") / "raw" / relative)


def _store_key_for(local_path: str) -> str:
    return str(Path("canonical") / strip_data_raw_prefix(Path(local_path)))


def _canonical_artifact_name(*, name: str, local_path: str) -> str:
    daily_name = _daily_ohlcv_artifact_name(local_path)
    if daily_name is not None:
        return daily_name
    return REPORT_PATH_NAME_TO_ARTIFACT_NAME.get(name, name)


def _daily_ohlcv_artifact_name(local_path: str) -> str | None:
    parts = Path(local_path).parts
    if len(parts) < 5 or parts[0:2] != ("data", "raw"):
        return None
    if not parts[2].startswith("daily_ohlcv"):
        return None
    symbol_part = parts[3]
    if not symbol_part.startswith("symbol="):
        return None
    symbol = symbol_part.removeprefix("symbol=")
    if not symbol:
        return None
    file_parts = parts[4:]
    if file_parts == ("ohlcv.parquet",):
        return f"constituent_ohlcv_{symbol}"
    suffix = _artifact_name_suffix(file_parts)
    return (
        f"constituent_ohlcv_{symbol}_{suffix}"
        if suffix
        else f"constituent_ohlcv_{symbol}"
    )


def _artifact_name_suffix(parts: tuple[str, ...]) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", "_".join(parts)).strip("_")


def _normalize_manifest_local_path(local_path: str) -> str:
    normalized = Path(local_path)
    if normalized.is_absolute() or normalized == Path("..") or ".." in normalized.parts:
        raise ValueError(
            f"manifest local_path must be relative within the repo: {local_path}"
        )
    return str(normalized)
