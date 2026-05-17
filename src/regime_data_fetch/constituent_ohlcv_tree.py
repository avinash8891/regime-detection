from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


EXPECTED_OHLCV_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
]


@dataclass(frozen=True)
class ConstituentTreeMaterialization:
    report_path: Path
    manifest_path: Path
    output_tree: Path
    requested_symbols: int
    written_symbols: int
    missing_symbols: tuple[str, ...]
    aggregate_sha256: str


def materialize_constituent_ohlcv_tree(
    *,
    source_tree: Path,
    output_tree: Path,
    pit_parquet_path: Path,
    start_date: dt.date,
    end_date: dt.date,
    report_path: Path | None = None,
    allow_missing_symbols: bool = False,
) -> ConstituentTreeMaterialization:
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")
    if source_tree.resolve() == output_tree.resolve():
        raise ValueError("source_tree and output_tree must be different paths")
    if not source_tree.exists():
        raise FileNotFoundError(source_tree)
    if not pit_parquet_path.exists():
        raise FileNotFoundError(pit_parquet_path)

    symbols = _load_pit_overlap_symbols(
        pit_parquet_path=pit_parquet_path,
        start_date=start_date,
        end_date=end_date,
    )
    if not symbols:
        raise ValueError(
            f"PIT constituents have no overlap with {start_date.isoformat()}..{end_date.isoformat()}"
        )

    missing: list[str] = []
    frames: list[tuple[str, pd.DataFrame, Path]] = []
    for symbol in symbols:
        try:
            frame, source_path = _read_symbol_ohlcv(source_tree, symbol)
        except FileNotFoundError:
            missing.append(symbol)
            continue
        frames.append((symbol, frame, source_path))

    if missing and not allow_missing_symbols:
        raise FileNotFoundError(
            "Missing constituent OHLCV symbols in source tree: " + ", ".join(missing[:50])
        )
    if not frames:
        raise ValueError("No constituent OHLCV files were available to materialize")

    output_tree.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_tree.name}.", dir=str(output_tree.parent))
    )
    try:
        file_entries: list[dict[str, object]] = []
        for symbol, frame, source_path in frames:
            symbol_dir = staging / f"symbol={symbol}"
            symbol_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = symbol_dir / "ohlcv.parquet"
            outgoing = frame[EXPECTED_OHLCV_COLUMNS].copy()
            outgoing.to_parquet(parquet_path, index=False)
            file_sha = _sha256_file(parquet_path)
            file_entries.append(
                {
                    "symbol": symbol,
                    "path": str(parquet_path.relative_to(staging)),
                    "source_path": str(source_path),
                    "rows": int(len(outgoing)),
                    "min_date": str(pd.to_datetime(outgoing["date"]).dt.date.min()),
                    "max_date": str(pd.to_datetime(outgoing["date"]).dt.date.max()),
                    "sha256": file_sha,
                }
            )

        aggregate_sha = _aggregate_sha256(file_entries)
        manifest_path = staging / "MANIFEST.sha256.json"
        manifest = {
            "artifact": output_tree.name,
            "source_tree": str(source_tree),
            "pit_parquet_path": str(pit_parquet_path),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "requested_symbols": len(symbols),
            "written_symbols": len(file_entries),
            "missing_symbols": missing,
            "aggregate_sha256": aggregate_sha,
            "files": file_entries,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        if output_tree.exists():
            shutil.rmtree(output_tree)
        staging.replace(output_tree)
        final_manifest_path = output_tree / "MANIFEST.sha256.json"

        final_report_path = report_path or output_tree.parent / f"{output_tree.name}_materialization_report.json"
        final_report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "source": "local:constituent_ohlcv_tree_materialization",
            "counts": {
                "requested_symbols": len(symbols),
                "written_symbols": len(file_entries),
                "missing_symbols": len(missing),
            },
            "missing_symbols": missing,
            "date_range": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            "paths": {
                "source_tree": str(source_tree),
                "profile_constituent_tree": str(output_tree),
                "manifest": str(final_manifest_path),
            },
            "aggregate_sha256": aggregate_sha,
        }
        final_report_path.write_text(json.dumps(report, indent=2) + "\n")
        return ConstituentTreeMaterialization(
            report_path=final_report_path,
            manifest_path=final_manifest_path,
            output_tree=output_tree,
            requested_symbols=len(symbols),
            written_symbols=len(file_entries),
            missing_symbols=tuple(missing),
            aggregate_sha256=aggregate_sha,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _load_pit_overlap_symbols(
    *,
    pit_parquet_path: Path,
    start_date: dt.date,
    end_date: dt.date,
) -> list[str]:
    frame = pd.read_parquet(pit_parquet_path)
    required = {"ticker", "start_date", "end_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{pit_parquet_path} missing required columns: {missing}")
    intervals = frame.copy()
    intervals["start_date"] = pd.to_datetime(intervals["start_date"])
    intervals["end_date"] = pd.to_datetime(intervals["end_date"])
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    mask = (intervals["start_date"] <= end_ts) & (
        intervals["end_date"].isna() | (intervals["end_date"] >= start_ts)
    )
    return sorted(
        {
            str(value).strip()
            for value in intervals.loc[mask, "ticker"].dropna().tolist()
            if str(value).strip()
        }
    )


def _read_symbol_ohlcv(tree_root: Path, symbol: str) -> tuple[pd.DataFrame, Path]:
    symbol_dir = tree_root / f"symbol={symbol}"
    canonical_path = symbol_dir / "ohlcv.parquet"
    if canonical_path.exists():
        source_path = canonical_path
        frame = pd.read_parquet(canonical_path)
    else:
        partition_files = sorted(symbol_dir.glob("*.parquet"))
        if not partition_files:
            raise FileNotFoundError(canonical_path)
        source_path = symbol_dir
        frame = pd.read_parquet(symbol_dir)
    missing = [col for col in EXPECTED_OHLCV_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{source_path} missing required columns: {missing}")
    out = frame[EXPECTED_OHLCV_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True), source_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_sha256(file_entries: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(file_entries, key=lambda entry: str(entry["symbol"])):
        digest.update(f"{item['symbol']}|{item['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()
