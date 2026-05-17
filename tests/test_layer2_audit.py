from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from scripts import audit_layer2_30d
from scripts.audit_layer2_30d import build_label_rule_summary, build_wiring_presence_rows


SHA = "0" * 64


def _artifact(name: str, local_path: str) -> dict[str, object]:
    return {
        "name": name,
        "stage": "canonical",
        "uri": f"s3://bucket/{local_path}",
        "local_path": local_path,
        "sha256": SHA,
        "schema_version": None,
        "rows": 1,
        "min_date": None,
        "max_date": None,
        "required_for": ["audit_layer2_30d"],
    }


def _write_audit_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "artifact_set": "audit",
                "created_at_utc": "2026-05-17T00:00:00Z",
                "storage_root": "s3://bucket/root",
                "artifacts": [
                    _artifact(
                        "constituent_ohlcv_AAPL",
                        "data/raw/daily_ohlcv_762/symbol=AAPL/ohlcv.parquet",
                    ),
                    _artifact(
                        "fred_macro_series",
                        "data/raw/macro/fred_macro_series.parquet",
                    ),
                    _artifact(
                        "sp500_pit_constituents",
                        "data/raw/pit_constituents/sp500_ticker_intervals.parquet",
                    ),
                    _artifact(
                        "ism_pmi_history",
                        "data/raw/pmi/us_ism_pmi_history.parquet",
                    ),
                ],
            },
            sort_keys=False,
        )
    )
    return path


def test_layer2_audit_manifest_resolution_replaces_default_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "materialized" / "data" / "raw"
    manifest_path = _write_audit_manifest(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_layer2_30d.py",
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
        ],
    )
    args = audit_layer2_30d._parse_args()

    audit_layer2_30d._apply_manifest_input_paths(
        args, runner_name="audit_layer2_30d"
    )

    assert args.daily_dir == data_root / "daily_ohlcv_762"
    assert args.pmi_path == data_root / "pmi" / "us_ism_pmi_history.parquet"


def test_layer2_wiring_audit_counts_optional_inflation_growth_features() -> None:
    selected = [dt.date(2026, 5, day) for day in range(1, 4)]
    idx = pd.to_datetime(selected)
    feature_store = SimpleNamespace(
        monetary=None,
        credit_funding=None,
        inflation_growth=SimpleNamespace(
            cpi_3m_change_pct=pd.Series([1.0, 1.1, 1.2], index=idx),
            cpi_6m_change_pct=pd.Series([1.0, 1.1, 1.2], index=idx),
            cpi_6m_change_pct_slope_21d=pd.Series([0.1, 0.2, 0.3], index=idx),
            inflation_surprise_zscore=pd.Series([2.0, 2.1, 2.2], index=idx),
            pmi_manufacturing=pd.Series([50.0, 51.0, 52.0], index=idx),
            pmi_manufacturing_slope_21d=pd.Series([0.0, 0.1, 0.2], index=idx),
            aggregate_forward_eps_revision_direction_4w=pd.Series(
                [0.03, 0.04, 0.05], index=idx
            ),
            commodity_return_63d=pd.Series([0.2, 0.3, 0.4], index=idx),
            treasury_10y_yield_slope_21d=pd.Series([0.01, 0.02, 0.03], index=idx),
            cyclical_defensive_ratio=pd.Series([2.0, 2.1, 2.2], index=idx),
            cyclical_defensive_slope_21d=pd.Series([0.01, 0.02, 0.03], index=idx),
            spy_21d_return=pd.Series([0.01, 0.02, 0.03], index=idx),
            tlt_21d_return=pd.Series([-0.01, -0.02, -0.03], index=idx),
        ),
    )

    rows = build_wiring_presence_rows(
        feature_store=feature_store,
        selected_dates=selected,
    )
    by_metric = {row["metric"]: row for row in rows}

    assert by_metric["inflation_surprise_zscore"]["present_days"] == 3
    assert by_metric["inflation_surprise_zscore"]["status"] == "ok"
    assert by_metric["aggregate_forward_eps_revision_direction_4w"]["present_days"] == 3
    assert by_metric["aggregate_forward_eps_revision_direction_4w"]["status"] == "ok"


def test_layer2_label_summary_includes_effective_credit_source_used() -> None:
    selected = [dt.date(2026, 5, 1)]
    output = SimpleNamespace(
        active_label="credit_calm",
        raw_label="credit_calm",
        stable_label="credit_calm",
        classification_status="classified",
        data_quality=SimpleNamespace(status="ok", reason=None),
        evidence={
            "source_used": "oas_confirmed",
            "rule_evidence": {"hy_spread_percentile_504d": 0.25},
        },
    )
    axis_bundle = SimpleNamespace(
        monetary_pressure_state=None,
        credit_funding=None,
        credit_funding_proxy=None,
        credit_funding_effective={selected[0]: output},
        inflation_growth=None,
    )

    summary = build_label_rule_summary(
        axis_bundle=axis_bundle,
        selected_dates=selected,
        missing_constituent_files=0,
    )

    effective = summary["axes"]["credit_funding_effective_state"]
    assert effective["reported"] == {"credit_calm": 1}
    assert effective["active"] == {"credit_calm": 1}
    assert effective["source_used"] == {"oas_confirmed": 1}
    assert effective["rule_evidence_present"] == {"hy_spread_percentile_504d": 1}


def test_layer2_label_summary_counts_reporting_label_separately_from_active() -> None:
    selected = [dt.date(2026, 5, 1)]
    output = SimpleNamespace(
        active_label="unknown",
        raw_label="unknown",
        stable_label="unknown",
        reporting_label="no_rule_fired",
        classification_status="no_rule_fired",
        data_quality=SimpleNamespace(status="ok", reason=None),
        evidence={"rule_evidence": {"hy_spread_percentile_504d": 0.60}},
    )
    axis_bundle = SimpleNamespace(
        monetary_pressure_state=None,
        credit_funding=None,
        credit_funding_proxy={selected[0]: output},
        credit_funding_effective=None,
        inflation_growth=None,
    )

    summary = build_label_rule_summary(
        axis_bundle=axis_bundle,
        selected_dates=selected,
        missing_constituent_files=0,
    )

    proxy = summary["axes"]["credit_funding_state_proxy"]
    assert proxy["reported"] == {"no_rule_fired": 1}
    assert proxy["active"] == {"unknown": 1}
    assert summary["axes"]["monetary_pressure_state"]["reported"] == {"not_wired": 1}


def test_layer2_label_summary_handles_none_and_string_reasons() -> None:
    selected = [dt.date(2026, 5, 1), dt.date(2026, 5, 2)]
    axis_bundle = SimpleNamespace(
        monetary_pressure_state={
            selected[0]: SimpleNamespace(
                active_label="neutral_monetary",
                raw_label="neutral_monetary",
                stable_label="neutral_monetary",
                classification_status="classified",
                data_quality=SimpleNamespace(status="ok", reason=None),
                evidence={"rule_evidence": {"yield_change_zscore_2y_63d": 0.1}},
            ),
            selected[1]: SimpleNamespace(
                active_label="unknown",
                raw_label="unknown",
                stable_label="unknown",
                reporting_label="stale_data",
                classification_status="stale_data",
                data_quality=SimpleNamespace(status="stale_data", reason="pmi_stale"),
                evidence={"reason": "pmi_stale"},
            ),
        },
        credit_funding=None,
        credit_funding_proxy=None,
        credit_funding_effective=None,
        inflation_growth=None,
    )

    summary = build_label_rule_summary(
        axis_bundle=axis_bundle,
        selected_dates=selected,
        missing_constituent_files=0,
    )

    reasons = summary["axes"]["monetary_pressure_state"]["data_quality_reasons"]
    assert reasons == {"None": 1, "pmi_stale": 1}
