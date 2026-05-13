#!/usr/bin/env python3
"""V2 calibration runner.

Fits the v2 §6.1 HMM and §6.2 GMM clustering on the full V2 data
(2018-01-01 to as_of) and dumps:

- ``docs/verification/hmm_state_label_map.candidate.yaml`` — per-state means,
  state-persistence diagonals, and a placeholder ``mappings`` block for
  operator review (per V2 §10 "manual cluster→label review").
- ``docs/verification/cluster_label_map.candidate.yaml`` — per-cluster
  centroids + sample-size + within-cluster scatter; placeholder
  ``mappings`` block.
- ``docs/verification/v2_calibration_summary.md`` — high-level rollup of
  what was fit, training window, dimensions, etc.

These are CANDIDATE artifacts. The operator inspects the state-mean /
centroid tables, decides the economic-label mapping, and renames the
committed ``*.yaml`` files (dropping the ``.candidate`` suffix). The spec
explicitly requires this manual step (§6.1 line 2748, §6.2 line 2842,
§10).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from regime_data_fetch.local_daily_ohlcv_sqlite import EXPECTED_COLUMNS  # noqa: E402

from regime_detection.config import load_default_regime_config  # noqa: E402
from regime_detection.engine import RegimeEngine  # noqa: E402
from regime_detection.feature_store import build_feature_store  # noqa: E402
from regime_detection.fragility_universe import SECTOR_ETFS  # noqa: E402
from regime_detection.market_context import build_market_context  # noqa: E402


def _load_market_data(daily_ohlcv_dir: Path) -> pd.DataFrame:
    """Load the v1-shape (SPY/RSP/VIXY) long-format DataFrame the engine wants."""
    df = pd.read_parquet(daily_ohlcv_dir)
    keep = ["date", "symbol", "open", "high", "low", "close", "volume"]
    out = df[df["symbol"].isin(["SPY", "RSP", "VIXY"])][keep].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def _load_close_dict(daily_ohlcv_dir: Path, symbols: list[str], spy_index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Pivot daily OHLCV parquet into close-series keyed by symbol, reindexed to SPY sessions."""
    df = pd.read_parquet(daily_ohlcv_dir)
    df["date"] = pd.to_datetime(df["date"])
    out: dict[str, pd.Series] = {}
    for sym in symbols:
        sub = df[df["symbol"] == sym].sort_values("date").set_index("date")
        if sub.empty:
            continue
        out[sym] = sub["close"].astype(float).reindex(spy_index).rename(sym)
    return out


def _load_macro_series(macro_parquet: Path, pmi_path: Path | None) -> dict[str, pd.Series]:
    """Load FRED macro series + manually-supplied PMI into a dict keyed by name."""
    macro = pd.read_parquet(macro_parquet)
    # Fetcher long-format columns: date, series_id, value, logical_name, ...
    macro["date"] = pd.to_datetime(macro["date"])
    series_dict: dict[str, pd.Series] = {}
    # Key by logical_name (e.g. 2y_yield, broad_usd_index, cpi_all_items) to
    # match the spec/feature_store seam names. Also key by series_id as alias
    # so downstream consumers that expect DGS2 / DGS10 / CPIAUCSL still work.
    for name, group in macro.groupby("logical_name"):
        s = group.set_index("date")["value"].astype(float).sort_index()
        series_dict[name] = s.rename(name)
    for sid, group in macro.groupby("series_id"):
        s = group.set_index("date")["value"].astype(float).sort_index()
        # don't clobber logical_name keys already present
        series_dict.setdefault(sid, s.rename(sid))
    if pmi_path and pmi_path.exists():
        pmi_df = pd.read_csv(pmi_path, sep="\t")
        # PMI tsv columns: period, release_date_local (DD-MM-YYYY), actual, ...
        if "release_date_local" in pmi_df.columns and "actual" in pmi_df.columns:
            pmi_df["release_date_local"] = pd.to_datetime(pmi_df["release_date_local"], format="%d-%m-%Y")
            pmi = pmi_df.set_index("release_date_local")["actual"].astype(float).sort_index()
            series_dict["pmi_manufacturing"] = pmi.rename("pmi_manufacturing")
    # Lowercase aliases that some axis modules expect:
    if "DGS10" in series_dict and "dgs10" not in series_dict:
        series_dict["dgs10"] = series_dict["DGS10"].rename("dgs10")
    if "DGS2" in series_dict and "dgs2" not in series_dict:
        series_dict["dgs2"] = series_dict["DGS2"].rename("dgs2")
    return series_dict


def _fit_summary_hmm(feature_store: Any, training_window_days: int) -> dict[str, Any]:
    """Produce the candidate HMM state-mean / persistence table from feature_store.hmm."""
    hmm = feature_store.hmm
    if hmm is None:
        return {"status": "unfit", "reason": "feature_store.hmm is None — input gate not lit"}
    state_probs = hmm.state_probabilities.dropna(how="any")
    n_states = hmm.n_states
    summary: dict[str, Any] = {
        "n_states": int(n_states),
        "n_predicted_sessions": int(len(state_probs)),
    }
    # Per-state stats: mean posterior, max-state count, transition diagonal.
    state_counts = {}
    for k in range(n_states):
        mask = state_probs.idxmax(axis=1) == k
        state_counts[int(k)] = int(mask.sum())
    summary["sessions_per_dominant_state"] = state_counts
    # Persistence (probability that argmax stays same on consecutive sessions).
    argmax = state_probs.idxmax(axis=1).to_numpy()
    persistence = {
        int(k): int(((argmax[:-1] == k) & (argmax[1:] == k)).sum()) / max(1, int((argmax[:-1] == k).sum()))
        for k in range(n_states)
    }
    summary["persistence_probability_per_state"] = {k: round(v, 4) for k, v in persistence.items()}
    summary["candidate_mappings"] = {
        int(k): f"<operator review: assign economic label for state {k}>"
        for k in range(n_states)
    }
    return summary


def _fit_summary_clustering(feature_store: Any) -> dict[str, Any]:
    """Produce the candidate cluster centroid / size table from feature_store.clustering."""
    clustering = feature_store.clustering
    if clustering is None:
        return {"status": "unfit", "reason": "feature_store.clustering is None — input gate not lit"}
    n_clusters = clustering.n_clusters
    cluster_id = clustering.cluster_id.dropna()
    distances = clustering.distance_to_centroid.dropna()
    summary: dict[str, Any] = {
        "model_version": clustering.model_version,
        "n_clusters": int(n_clusters),
        "n_predicted_sessions": int(len(cluster_id)),
    }
    cluster_counts = cluster_id.value_counts().sort_index().to_dict()
    summary["sessions_per_cluster"] = {int(k): int(v) for k, v in cluster_counts.items()}
    # Mean distance to centroid per cluster — proxy for cluster tightness.
    per_cluster_distance = {}
    for k in range(n_clusters):
        mask = cluster_id == k
        if mask.any():
            per_cluster_distance[int(k)] = round(float(distances[mask].mean()), 6)
        else:
            per_cluster_distance[int(k)] = None
    summary["mean_distance_to_centroid_per_cluster"] = per_cluster_distance
    summary["candidate_mappings"] = {
        int(k): f"<operator review: assign economic label for cluster {k}>"
        for k in range(n_clusters)
    }
    return summary


def main() -> int:
    data_root = REPO_ROOT / "data" / "raw"
    daily_dir = data_root / "daily_ohlcv"
    macro_parquet = data_root / "macro" / "fred_macro_series.parquet"
    pmi_path = REPO_ROOT / "data" / "manual_inputs" / "pmi" / "ism_manufacturing_pmi.tsv"
    pit_intervals_parquet = data_root / "pit_constituents" / "sp500_ticker_intervals.parquet"
    constituent_db_path = data_root / "constituent_ohlcv.db"
    verification_dir = REPO_ROOT / "docs" / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)

    if not daily_dir.exists():
        raise SystemExit(f"daily_ohlcv directory not found at {daily_dir} — run fetch first")
    if not macro_parquet.exists():
        raise SystemExit(f"macro parquet not found at {macro_parquet} — run macro fetch first")

    market_data = _load_market_data(daily_dir)
    end_date = market_data["date"].max()

    # Load PIT inputs for clustering / pct_above_50dma path. Both are optional
    # — when missing, the breadth_state_v2 PIT seam stays unlit and clustering
    # falls back to None (calibration emits a "deferred" candidate).
    pit_intervals = None
    constituent_ohlcv = None
    if pit_intervals_parquet.exists():
        from regime_data_fetch.pit_constituents import read_pit_intervals, members_on
        pit_intervals = read_pit_intervals(pit_intervals_parquet)
        print(f"PIT intervals: {len(pit_intervals)} rows")
    if constituent_db_path.exists() and pit_intervals is not None:
        from regime_data_fetch.local_daily_ohlcv_sqlite_reader import read_constituent_ohlcv
        # Members on the end_date define the universe whose OHLCV we load.
        # Reader is keyed by ticker; we pass the full distinct-ticker list
        # across the trailing window so newly-listed members have data.
        from regime_data_fetch.pit_constituents import members_on as _members_on
        all_member_tickers = sorted({t for t in pit_intervals["ticker"].unique()})
        # Optimization: only read tickers that DBC-style classifier expects.
        # In practice the universe is ~1200 tickers; read_constituent_ohlcv
        # auto-omits absent ones, so passing the full list is safe.
        constituent_ohlcv = read_constituent_ohlcv(
            constituent_db_path,
            tickers=all_member_tickers,
            start_date=dt.date(2016, 1, 1),
            end_date=end_date,
        )
        print(f"constituent_ohlcv: {len(constituent_ohlcv)} tickers loaded from SQLite")

    # First pass: build a v1-only context to discover the SPY session index.
    config = load_default_regime_config()
    bootstrap_context = build_market_context(
        end_date=end_date,
        market_data=market_data,
        config=config,
    )
    spy_index = bootstrap_context.spy_ohlcv.index

    # Build full V2 inputs.
    cross_asset_symbols = [
        "QQQ", "IWM", "EFA", "EEM", "TLT", "HYG", "LQD", "GLD", "USO", "UUP", "DBC", "KRE",
        # XLY/XLI/XLP/XLU are sector ETFs but §2B inflation_growth reads them
        # from cross_asset_closes (cyclical-vs-defensive ratio). Mirroring the
        # slice-5 test fixtures' convention.
        "XLY", "XLI", "XLP", "XLU",
    ]
    sector_etf_closes = _load_close_dict(daily_dir, list(SECTOR_ETFS), spy_index)
    cross_asset_closes = _load_close_dict(daily_dir, cross_asset_symbols, spy_index)
    macro_series = _load_macro_series(macro_parquet, pmi_path)

    print(f"as_of = {end_date}; spy sessions = {len(spy_index)}")
    print(f"sector_etf symbols: {sorted(sector_etf_closes.keys())}")
    print(f"cross_asset symbols: {sorted(cross_asset_closes.keys())}")
    print(f"macro series: {sorted(macro_series.keys())}")

    # Rebuild context with full V2 inputs + PIT seams.
    context = build_market_context(
        end_date=end_date,
        market_data=market_data,
        config=config,
        sector_etf_closes=sector_etf_closes,
        cross_asset_closes=cross_asset_closes,
        macro_series=macro_series,
        pit_constituent_intervals=pit_intervals,
        constituent_ohlcv=constituent_ohlcv,
    )

    feature_store = build_feature_store(
        context,
        network_fragility_config=config.network_fragility,
        trend_direction_v2_config=config.trend_direction_v2,
        volatility_state_v2_config=config.volatility_state_v2,
        breadth_state_v2_config=config.breadth_state_v2,
        volume_liquidity_v2_config=config.volume_liquidity_v2,
        monetary_pressure_v2_config=config.monetary_pressure_v2,
        credit_funding_config=config.credit_funding,
        inflation_growth_config=config.inflation_growth,
    )

    print(f"feature_store.hmm lit: {feature_store.hmm is not None}")
    print(f"feature_store.clustering lit: {feature_store.clustering is not None}")
    print(f"feature_store.network_fragility lit: {feature_store.network_fragility is not None}")

    hmm_summary = _fit_summary_hmm(feature_store, config.hmm.training_window_days if config.hmm else 0)
    cluster_summary = _fit_summary_clustering(feature_store)

    # Write candidate label maps.
    hmm_doc = {
        "hmm_state_label_map": {
            "version": "0.1-candidate",
            "fitted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "fitted_end_date": end_date.isoformat(),
            "training_window_days": int(config.hmm.training_window_days) if config.hmm else None,
            "random_state": int(config.hmm.random_state) if config.hmm else None,
            "summary": hmm_summary,
            "review_instructions": (
                "Per V2 §6.1 line 2748 + §10: inspect the per-state mean posterior, "
                "sessions-per-state, and persistence diagonal above. Manually assign "
                "an economic label (e.g. calm_bull / choppy_normal / stress_crash) to "
                "each integer state by replacing the placeholder strings in "
                "summary.candidate_mappings. Rename this file to "
                "hmm_state_label_map.yaml (drop .candidate suffix) when done."
            ),
        }
    }
    cluster_doc = {
        "cluster_label_map": {
            "version": "0.1-candidate",
            "fitted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "fitted_end_date": end_date.isoformat(),
            "training_window_days": int(config.clustering.training_window_days) if config.clustering else None,
            "random_state": int(config.clustering.random_state) if config.clustering else None,
            "summary": cluster_summary,
            "review_instructions": (
                "Per V2 §6.2 line 2842 + §10: inspect the per-cluster size and mean "
                "distance to centroid above. Manually assign an economic label "
                "(e.g. calm_low_vol_bull / trending_bull / high_vol_chop / ...) to "
                "each integer cluster by replacing the placeholder strings in "
                "summary.candidate_mappings. Rename this file to cluster_label_map.yaml "
                "(drop .candidate suffix) when done."
            ),
        }
    }

    hmm_path = verification_dir / "hmm_state_label_map.candidate.yaml"
    cluster_path = verification_dir / "cluster_label_map.candidate.yaml"
    hmm_path.write_text(yaml.safe_dump(hmm_doc, sort_keys=False))
    cluster_path.write_text(yaml.safe_dump(cluster_doc, sort_keys=False))

    summary_md = [
        "# V2 Calibration Summary",
        "",
        f"- Fitted at: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"- Data end date: {end_date.isoformat()}",
        f"- SPY sessions: {len(spy_index)}",
        f"- Sector ETFs: {sorted(sector_etf_closes.keys())}",
        f"- Cross-asset: {sorted(cross_asset_closes.keys())}",
        f"- Macro series: {sorted(macro_series.keys())}",
        "",
        "## Feature-store seams",
        "",
        f"- `feature_store.network_fragility` lit: **{feature_store.network_fragility is not None}**",
        f"- `feature_store.volatility_state_v2` lit: **{feature_store.volatility_state_v2 is not None}**",
        f"- `feature_store.breadth_state_v2` lit: **{feature_store.breadth_state_v2 is not None}**",
        f"- `feature_store.volume_liquidity_v2` lit: **{feature_store.volume_liquidity_v2 is not None}**",
        f"- `feature_store.monetary` lit: **{feature_store.monetary is not None}**",
        f"- `feature_store.hmm` lit: **{feature_store.hmm is not None}**",
        f"- `feature_store.clustering` lit: **{feature_store.clustering is not None}**",
        f"- `feature_store.change_point` lit: **{feature_store.change_point is not None}**",
        f"- `feature_store.credit_funding` lit: **{feature_store.credit_funding is not None}**",
        f"- `feature_store.inflation_growth` lit: **{feature_store.inflation_growth is not None}**",
        "",
        "## Candidate artifacts (require operator review per V2 §10)",
        "",
        f"- {hmm_path.relative_to(REPO_ROOT)}",
        f"- {cluster_path.relative_to(REPO_ROOT)}",
    ]
    summary_path = verification_dir / "v2_calibration_summary.md"
    summary_path.write_text("\n".join(summary_md) + "\n")

    print(f"\nWrote candidate label maps:")
    print(f"  {hmm_path}")
    print(f"  {cluster_path}")
    print(f"  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
