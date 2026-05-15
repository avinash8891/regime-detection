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
import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT))

from regime_data_fetch.local_daily_ohlcv_sqlite import EXPECTED_COLUMNS  # noqa: E402
from regime_data_fetch.materialization import materialize_if_requested  # noqa: E402

from regime_detection.config import load_default_regime_config  # noqa: E402
from regime_detection.engine import RegimeEngine  # noqa: E402
from regime_detection.feature_store import build_feature_store  # noqa: E402
from regime_detection.fragility_universe import SECTOR_ETFS  # noqa: E402
from regime_detection.loaders import (  # noqa: E402
    load_central_bank_text_score,
    load_cpi_vintages_first_release,
    load_news_sentiment_series,
)
from regime_detection.market_context import build_market_context  # noqa: E402
from scripts._v2_calibration_helpers import load_macro_series  # noqa: E402


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


def _summarize_central_bank_text(feature_store: Any, config: Any) -> list[str]:
    """Surface the v2 §2A central-bank-text smoothed score distribution.

    Audit M1 — `CentralBankTextConfig.smoothing_window_sessions` defaults
    to 30 (six weeks, AAII-style cadence). This rollup prints the actual
    distribution under the current setting so v2 §9.1 walk-forward
    calibration has the data it needs to retune.
    """
    monetary = feature_store.monetary
    cb_cfg = config.central_bank_text
    if monetary is None or monetary.central_bank_text_score is None:
        return [
            "- `feature_store.monetary.central_bank_text_score` lit: **False** "
            "(no FOMC minutes / Powell speech releases supplied, or config absent)",
        ]
    series = monetary.central_bank_text_score.dropna()
    if series.empty:
        return [
            "- `feature_store.monetary.central_bank_text_score` lit but all NaN "
            "(cold-start — first release is after the SPY index start)",
        ]
    lines = [
        f"- `feature_store.monetary.central_bank_text_score` lit: **True** "
        f"(n={len(series)} sessions)",
        f"- Smoothing window: **{cb_cfg.smoothing_window_sessions}** NYSE "
        f"sessions (CentralBankTextConfig.smoothing_window_sessions; "
        f"v2 §9.1 walk-forward calibration placeholder).",
        f"- max_release_age_days: **{cb_cfg.max_release_age_days}**.",
        f"- Score distribution after smoothing:",
        f"    - min: {float(series.min()):+.3f}",
        f"    - p25: {float(series.quantile(0.25)):+.3f}",
        f"    - median: {float(series.median()):+.3f}",
        f"    - p75: {float(series.quantile(0.75)):+.3f}",
        f"    - max: {float(series.max()):+.3f}",
        f"    - mean: {float(series.mean()):+.3f}",
        f"- Bias-warning code emitted on feature output: "
        f"`central_bank_text_deterministic_lexicon_substitute` (audit M1 / "
        f"docs/spec_code_data_audit_2026_05_15.md §3.1).",
    ]
    return lines


def _summarize_first_release_cpi(
    feature_store: Any, cpi_first_release: pd.Series | None
) -> list[str]:
    """Surface whether the audit M2 first-release CPI seam is in effect."""
    inflation = feature_store.inflation_growth
    if inflation is None:
        return ["- `feature_store.inflation_growth` lit: **False**"]
    bias_codes = (
        set(inflation.bias_warnings["warning_code"].tolist())
        if not inflation.bias_warnings.empty
        else set()
    )
    first_release_active = "cpi_first_release_vintage_replay" in bias_codes
    lines = [
        f"- First-release CPI seam supplied: "
        f"**{cpi_first_release is not None}** "
        f"({'%d releases' % len(cpi_first_release) if cpi_first_release is not None else 'absent'}).",
        f"- First-release substitution in effect: **{first_release_active}** "
        f"(bias-warning row `cpi_first_release_vintage_replay` "
        f"{'present' if first_release_active else 'absent'} on feature output).",
    ]
    return lines


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
    argmax_series = state_probs.idxmax(axis=1)
    for k in range(n_states):
        mask = argmax_series == k
        state_counts[int(k)] = int(mask.sum())
    summary["sessions_per_dominant_state"] = state_counts
    # Persistence (probability that argmax stays same on consecutive sessions).
    argmax = argmax_series.to_numpy()
    persistence = {
        int(k): int(((argmax[:-1] == k) & (argmax[1:] == k)).sum()) / max(1, int((argmax[:-1] == k).sum()))
        for k in range(n_states)
    }
    summary["persistence_probability_per_state"] = {k: round(v, 4) for k, v in persistence.items()}
    # Top-5 most-recent dominant sessions per state — gives the operator a
    # concrete anchor for the manual label mapping (V2 §6.1 line 2748 + §10).
    representative_dates: dict[int, list[str]] = {}
    for k in range(n_states):
        mask = argmax_series == k
        if not mask.any():
            representative_dates[int(k)] = []
            continue
        dates = state_probs.index[mask][-5:].tolist()
        representative_dates[int(k)] = [pd.Timestamp(d).date().isoformat() for d in dates]
    summary["recent_dominant_dates_per_state"] = representative_dates
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
    # Top-5 most-recent + top-5 most-extreme-distance sessions per cluster.
    # Gives the operator concrete date anchors to inspect the historical
    # regime context when assigning the economic label (V2 §6.2 line 2842 + §10).
    recent_dates: dict[int, list[str]] = {}
    extreme_dates: dict[int, list[str]] = {}
    for k in range(n_clusters):
        mask = cluster_id == k
        if not mask.any():
            recent_dates[int(k)] = []
            extreme_dates[int(k)] = []
            continue
        cluster_dates = cluster_id.index[mask]
        recent_dates[int(k)] = [pd.Timestamp(d).date().isoformat() for d in cluster_dates[-5:]]
        cluster_distances = distances.loc[cluster_dates]
        top_extreme = cluster_distances.nlargest(5).index
        extreme_dates[int(k)] = [pd.Timestamp(d).date().isoformat() for d in top_extreme]
    summary["recent_dates_per_cluster"] = recent_dates
    summary["most_extreme_distance_dates_per_cluster"] = extreme_dates
    summary["candidate_mappings"] = {
        int(k): f"<operator review: assign economic label for cluster {k}>"
        for k in range(n_clusters)
    }
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 calibration artifact runner.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional artifact manifest to materialize before calibration.")
    parser.add_argument("--artifact-store", default=None, help="Optional artifact-store root override for --manifest.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data_root = args.data_root
    materialize_if_requested(
        manifest_path=args.manifest,
        local_root=data_root,
        repo_root=REPO_ROOT,
        store_root=args.artifact_store,
        required_for="v2_calibration",
    )
    daily_dir = data_root / "daily_ohlcv"
    macro_parquet = data_root / "macro" / "fred_macro_series.parquet"
    pmi_path = REPO_ROOT / "data" / "manual_inputs" / "pmi" / "ism_manufacturing_pmi.tsv"
    pit_intervals_parquet = data_root / "pit_constituents" / "sp500_ticker_intervals.parquet"
    constituent_db_path = data_root / "constituent_ohlcv.db"
    # v2 §2A central-bank-text + first-release CPI seams (audit M1 / M2).
    # Standard fetch paths under data/raw/; absence is non-fatal — the
    # engine falls through to the existing latest-revision CPI path and
    # the central_bank_text_score stays None.
    fomc_minutes_parquet = data_root / "fomc_minutes" / "fomc_minutes.parquet"
    powell_speeches_parquet = data_root / "powell_speeches" / "powell_speeches.parquet"
    cpi_vintages_parquet = data_root / "macro_vintages" / "cpi_all_items_vintages.parquet"
    news_sentiment_parquet = data_root / "news_sentiment" / "sf_fed_news_sentiment.parquet"
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
    macro_series = load_macro_series(macro_parquet, pmi_path)

    print(f"as_of = {end_date}; spy sessions = {len(spy_index)}")
    print(f"sector_etf symbols: {sorted(sector_etf_closes.keys())}")
    print(f"cross_asset symbols: {sorted(cross_asset_closes.keys())}")
    print(f"macro series: {sorted(macro_series.keys())}")

    # v2 §2A central-bank-text release frame (audit M1). When neither
    # source parquet exists the loader returns an empty frame and the
    # downstream score series is all-NaN.
    central_bank_text_releases = load_central_bank_text_score(
        fomc_minutes_source=(
            fomc_minutes_parquet if fomc_minutes_parquet.exists() else None
        ),
        powell_speeches_source=(
            powell_speeches_parquet if powell_speeches_parquet.exists() else None
        ),
    )
    if central_bank_text_releases.empty:
        print("central_bank_text_releases: empty (no FOMC minutes / Powell speeches parquets)")
    else:
        print(
            f"central_bank_text_releases: {len(central_bank_text_releases)} rows "
            f"({central_bank_text_releases['release_date'].min()} → "
            f"{central_bank_text_releases['release_date'].max()})"
        )

    # v2 §2A first-release CPI for historical replay (audit M2). When the
    # vintages parquet is absent the loader is skipped and the engine
    # falls through to the existing revised CPIAUCSL path.
    cpi_first_release = None
    if cpi_vintages_parquet.exists():
        cpi_first_release = load_cpi_vintages_first_release(cpi_vintages_parquet)
        print(
            f"cpi_first_release: {len(cpi_first_release)} releases "
            f"({cpi_first_release.index.min().date()} → "
            f"{cpi_first_release.index.max().date()})"
        )
    else:
        print(f"cpi_first_release: skipped (no {cpi_vintages_parquet.name})")

    # v2 §1A SF Fed news sentiment evidence (audit post-#12 follow-up).
    news_sentiment = None
    if news_sentiment_parquet.exists():
        news_sentiment = load_news_sentiment_series(news_sentiment_parquet)
        print(
            f"news_sentiment: {len(news_sentiment)} daily rows "
            f"({news_sentiment.index.min().date()} → "
            f"{news_sentiment.index.max().date()})"
        )
    else:
        print(f"news_sentiment: skipped (no {news_sentiment_parquet.name})")

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
        central_bank_text_releases=(
            central_bank_text_releases if not central_bank_text_releases.empty else None
        ),
        cpi_first_release=cpi_first_release,
        news_sentiment=news_sentiment,
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
        # Audit M1 — v2 §9.1 walk-forward calibration placeholder for
        # CentralBankTextConfig.smoothing_window_sessions. The summary
        # report below surfaces the resulting score distribution so the
        # default (30 sessions) can be retuned against actual FOMC
        # tightening / easing cycles.
        central_bank_text_config=config.central_bank_text,
        # Audit post-#12 — SF Fed news sentiment evidence (EVIDENCE ONLY;
        # `euphoria` rule predicate unchanged).
        news_sentiment_config=config.news_sentiment,
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
        "## Audit M1 — central-bank-text score (v2 §9.1 calibration placeholder)",
        "",
        *_summarize_central_bank_text(feature_store, config),
        "",
        "## Audit M2 — first-release CPI provenance",
        "",
        *_summarize_first_release_cpi(feature_store, cpi_first_release),
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
