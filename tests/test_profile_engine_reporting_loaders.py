from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import profile_engine_reporting
from scripts._v2_calibration_helpers import normalize_datetime_index

from conftest import (
    load_profile_engine_module,
)

profile_engine = load_profile_engine_module()


def _build_profile_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        daily_dir=tmp_path / "daily_ohlcv",
        constituent_tree=tmp_path / "constituents",
        macro_parquet=tmp_path / "macro.parquet",
        event_calendar=tmp_path / "events.yaml",
        aaii_sentiment_parquet=tmp_path / "aaii.parquet",
        news_sentiment_parquet=tmp_path / "news.parquet",
        fomc_minutes_parquet=tmp_path / "fomc.parquet",
        powell_speeches_parquet=tmp_path / "powell.parquet",
        cpi_vintages_parquet=tmp_path / "cpi.parquet",
        pit_parquet=tmp_path / "pit.parquet",
        lookback_days=1,
    )


def _build_profile_inputs(
    *, news_sentiment: pd.Series | None
) -> profile_engine.ProfileInputBundle:
    return profile_engine.ProfileInputBundle(
        market_data=pd.DataFrame({"date": [pd.Timestamp("2026-05-15")]}),
        end_date=pd.Timestamp("2026-05-15").date(),
        required_sessions=252,
        working_start_date=pd.Timestamp("2025-05-16").date(),
        selected_dates=[pd.Timestamp("2026-05-15").date()],
        sector_etf_closes={"XLK": pd.Series([1.0])},
        cross_asset_closes={},
        macro_series={},
        event_calendar=None,
        aaii_sentiment=None,
        news_sentiment=news_sentiment,
        implied_vol_30d=None,
        central_bank_text_releases=pd.DataFrame(
            {"release_date": [pd.Timestamp("2026-05-01")]}
        ),
        cpi_first_release=None,
        pit_constituent_intervals=pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "start_date": [pd.Timestamp("2020-01-01").date()],
                "end_date": [None],
            }
        ),
        constituent_ohlcv={"AAPL": pd.DataFrame({"close": [1.0]})},
        constituent_tickers=["AAPL"],
    )


def test_profile_json_safe_value_converts_nonfinite_floats_to_null() -> None:
    payload = profile_engine_reporting._json_safe_value(
        {
            "cold_start": float("nan"),
            "positive_inf": float("inf"),
            "negative_inf": float("-inf"),
            "nested": [{"ok": 1.25, "missing": float("nan")}],
        }
    )

    assert payload == {
        "cold_start": None,
        "positive_inf": None,
        "negative_inf": None,
        "nested": [{"ok": 1.25, "missing": None}],
    }


def test_input_status_report_counts_series_rows() -> None:
    series = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-05-01", "2026-05-02"]))

    report = profile_engine_reporting.input_status_report("macro", series)

    assert report["status"] == "present"
    assert report["rows"] == 2


def test_normalize_datetime_index_returns_datetime_index() -> None:
    raw = pd.Index(["2026-05-01", "2026-05-02"])

    idx = normalize_datetime_index(raw)

    assert isinstance(idx, pd.DatetimeIndex)
    assert list(idx.strftime("%Y-%m-%d")) == ["2026-05-01", "2026-05-02"]


def test_profile_json_writer_rejects_nonfinite_values(tmp_path: Path) -> None:
    report_path = tmp_path / "profile.json"

    try:
        profile_engine._write_json_report(report_path, {"bad": math.nan})
    except ValueError as exc:
        assert "Out of range float values" in str(exc)
    else:
        raise AssertionError("_write_json_report accepted non-finite JSON")

    assert not report_path.exists()


def test_effective_label_summary_uses_credit_funding_effective_state_for_proxy_fallback() -> (
    None
):
    output = SimpleNamespace(
        credit_funding_state=SimpleNamespace(
            active_label="unknown",
            classification_status="stale_data",
            classification_reason="hy_oas_stale_1000000000d,ig_oas_stale_1000000000d",
            evidence={"spread_source": "ice_bofa_oas"},
            data_quality={"status": "stale_data"},
        ),
        credit_funding_state_proxy=SimpleNamespace(
            active_label="credit_calm",
            classification_status="classified",
            classification_reason=None,
            evidence={"spread_source": "tlt_total_return_differential"},
            data_quality={"status": "ok"},
        ),
        credit_funding_effective_state=SimpleNamespace(
            active_label="credit_calm",
            classification_status="classified",
            classification_reason=None,
            evidence={"source_used": "proxy_fallback"},
            data_quality={"status": "ok"},
        ),
        inflation_growth_state=SimpleNamespace(
            active_label="unknown",
            classification_status="stale_data",
            classification_reason="latest_observation_too_old",
            evidence={"reason": "latest_observation_too_old"},
            data_quality={"status": "stale_data"},
        ),
    )

    summary = profile_engine_reporting._effective_label_summary_report([output])

    assert "credit_funding_state" not in summary
    assert summary["credit_funding_effective_state"]["status"] == {"classified": 1}
    assert summary["inflation_growth_state"]["status"] == {"stale_data": 1}


def test_profile_json_report_emits_machine_readable_sections(tmp_path: Path) -> None:
    def axis(label: str) -> SimpleNamespace:
        return SimpleNamespace(
            active_label=label,
            classification_status="classified",
            classification_reason=None,
            evidence={"rule_evidence": {"sample_metric": 1.25}},
            data_quality={"status": "ok"},
        )

    timer = profile_engine.StageTimer()
    timer.totals["build_feature_store_total"] = 1.5
    timer.counts["build_feature_store_total"] = 1
    timer.totals["feature_store.breadth_state_v2"] = 0.25
    timer.counts["feature_store.breadth_state_v2"] = 1

    report_path = tmp_path / "reports" / "profile.json"
    args = _build_profile_args(tmp_path)
    inputs = _build_profile_inputs(
        news_sentiment=pd.Series([0.1], name="news_sentiment")
    )
    output = SimpleNamespace(
        as_of_date=pd.Timestamp("2026-05-15").date(),
        trend_direction=axis("uptrend"),
        trend_character=axis("trending"),
        volatility_state=axis("low_vol"),
        breadth_state=axis("broadening_breadth"),
        structural_causal_state=SimpleNamespace(
            event_calendar=SimpleNamespace(
                primary_label="fed_week",
                matching_labels=("fed_week", "expiry_week"),
                evidence={"days_to_event": 4},
            ),
            monetary_pressure=SimpleNamespace(
                label="neutral_monetary",
                evidence={"fed_funds_change_63d": 0.0},
                data_quality={"status": "ok"},
            ),
        ),
        transition_risk=SimpleNamespace(
            state="stable",
            score=0.2,
            score_components={"breadth": 0.2},
            primary_drivers=["breadth"],
            triggered_rules=["post_switch_cooldown"],
            evidence={
                "triggered_rules": ["post_switch_cooldown"],
                "axis_switch_count": 1,
                "recent_axis_switch_count": 2,
            },
            data_quality={"status": "ok"},
        ),
        network_fragility=axis("correlation_concentration"),
        volume_liquidity_state=None,
        credit_funding_state=None,
        credit_funding_state_proxy=SimpleNamespace(
            active_label="unknown",
            classification_status="no_rule_fired",
            classification_reason="no_rule_fired",
            evidence={"reason": "no_rule_fired"},
            data_quality={"status": "ok"},
        ),
        credit_funding_effective_state=None,
        inflation_growth_state=SimpleNamespace(
            active_label="unknown",
            classification_status="stale_data",
            classification_reason="latest_observation_too_old",
            evidence={"reason": "latest_observation_too_old"},
            data_quality={"status": "stale_data"},
        ),
        monetary_pressure_state=None,
        cluster=SimpleNamespace(
            cluster_id=3, distance_to_centroid=1.75, model_version="test-cluster"
        ),
        change_point=SimpleNamespace(
            score=0.42, days_since_last_break=7, method="BOCPD"
        ),
        agent_routing=SimpleNamespace(
            active_cohort="risk_on",
            fallback_cohort="neutral",
            blocked_strategy_modes=["short_vol"],
        ),
        strategy_family_constraints={
            "trend_following": SimpleNamespace(
                allowed=True, max_lookback_days=60, reason="classified"
            )
        },
        strategy_response=SimpleNamespace(
            position_size_multiplier=1.0,
            allow_trend_following=True,
            modifiers_applied=["none"],
        ),
    )

    report = profile_engine._build_json_report(
        args=args,
        inputs=inputs,
        timeline=SimpleNamespace(outputs=[output]),
        timer=timer,
        total_wall_clock=2.0,
        per_day_emission_total=0.5,
        per_day_avg_ms=500.0,
        verification_issues=[],
    )
    profile_engine._write_json_report(report_path, report)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["sources"]["market_data"] == str(args.daily_dir)
    assert payload["window"]["selected_window_start"] == "2026-05-15"
    assert payload["inputs"]["constituent_tickers_loaded"] == 1
    assert payload["inputs"]["seams"][7]["kind"] == "dataframe"
    assert payload["inputs"]["seams"][9]["kind"] == "dataframe"
    assert payload["timing"]["stages"][-1] == {
        "call_count": 1,
        "percent_of_total": 25.0,
        "stage_name": "per_day_output_emission_loop_residual",
        "wall_clock_seconds": 0.5,
    }
    assert payload["timeline"] == [
        {
            "activated_v2_seams": {
                "change_point": 0.42,
                "cluster": 3,
                "credit_funding_state_proxy": "no_rule_fired",
                "inflation_growth_state": "stale_data",
                "network_fragility": {
                    "reported": "correlation_concentration",
                    "classification_status": "classified",
                },
                "transition_score": 0.2,
                "transition_risk": {
                    "score": 0.2,
                    "primary_drivers": ["breadth"],
                    "triggered_rules": ["post_switch_cooldown"],
                    "data_quality_status": "ok",
                    "axis_switch_count": 1,
                    "recent_axis_switch_count": 2,
                },
                "event_calendar": {
                    "primary_label": "fed_week",
                    "matching_labels": ["fed_week", "expiry_week"],
                },
            },
            "as_of_date": "2026-05-15",
            "event_calendar_primary_label": "fed_week",
            "event_calendar_matching_labels": ["fed_week", "expiry_week"],
            "transition_risk": "stable",
            "trend_direction": "uptrend",
            "volatility_state": "low_vol",
        }
    ]


def test_profile_json_report_can_include_observability_section(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"

    profile_engine._write_json_report(
        report_path,
        {
            "status": "ok",
            "observability": {
                "trace_id": "trace-123",
                "metrics": {"counters": {"exceptions_total": 1}, "timings_ms": {}},
                "error_tracking": {"enabled": True, "backend": "sentry"},
                "product_analytics": {"enabled": True, "backend": "posthog"},
                "feature_flags": {"shadow_mode": True},
                "deployment_observability": {
                    "dashboard_url": "https://grafana.example/d/abc"
                },
            },
        },
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["observability"]["trace_id"] == "trace-123"
    assert payload["observability"]["metrics"]["counters"]["exceptions_total"] == 1
    assert payload["observability"]["error_tracking"]["backend"] == "sentry"
    assert payload["observability"]["product_analytics"]["backend"] == "posthog"
    assert payload["observability"]["feature_flags"]["shadow_mode"] is True


def test_profile_json_report_uses_loaded_bundle_values_for_input_status(
    tmp_path: Path,
) -> None:
    timer = profile_engine.StageTimer()
    args = _build_profile_args(tmp_path)
    inputs = _build_profile_inputs(news_sentiment=None)
    output = SimpleNamespace(
        as_of_date=pd.Timestamp("2026-05-15").date(),
        trend_direction=SimpleNamespace(
            active_label="uptrend", classification_status="classified"
        ),
        volatility_state=SimpleNamespace(
            active_label="low_vol", classification_status="classified"
        ),
        transition_risk=SimpleNamespace(
            state="stable",
            score=None,
            score_components=None,
            primary_drivers=[],
            triggered_rules=[],
            evidence={},
            data_quality={"status": "ok"},
        ),
        network_fragility=None,
        volume_liquidity_state=None,
        credit_funding_state=None,
        credit_funding_state_proxy=None,
        credit_funding_effective_state=None,
        inflation_growth_state=None,
        monetary_pressure_state=None,
        cluster=None,
        change_point=None,
    )

    report = profile_engine._build_json_report(
        args=args,
        inputs=inputs,
        timeline=SimpleNamespace(outputs=[output]),
        timer=timer,
        total_wall_clock=1.0,
        per_day_emission_total=0.0,
        per_day_avg_ms=0.0,
        verification_issues=[],
    )

    seams = {row["name"]: row for row in report["inputs"]["seams"]}
    assert seams["sector_etf_closes"]["count"] == 1
    assert seams["central_bank_text_releases"]["kind"] == "dataframe"
    assert seams["constituent_ohlcv"]["count"] == 1


def test_profile_engine_loads_aaii_sentiment_when_present(tmp_path: Path) -> None:
    parquet_path = tmp_path / "aaii_sentiment.parquet"
    expected = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-07"),
                "publication_date": pd.Timestamp("2026-05-07"),
                "bullish": 0.35,
                "neutral": 0.30,
                "bearish": 0.35,
                "bull_bear_spread": 0.0,
                "bull_bear_spread_8w_ma": 22.0,
            }
        ]
    )
    expected.to_parquet(parquet_path, index=False)

    actual = profile_engine._load_optional_aaii_sentiment(parquet_path)

    assert actual is not None
    pd.testing.assert_frame_equal(actual, expected)


def test_profile_engine_skips_aaii_sentiment_when_absent(tmp_path: Path) -> None:
    assert (
        profile_engine._load_optional_aaii_sentiment(tmp_path / "missing.parquet")
        is None
    )


def test_profile_engine_loads_event_calendar_when_present(tmp_path: Path) -> None:
    yaml_path = tmp_path / "events.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "events:",
                '  - date: "2026-05-12"',
                '    market: "US"',
                '    type: "CPI"',
                '    importance: "high"',
                "    window_days: [-1, 1]",
            ]
        )
    )

    actual = profile_engine._load_event_calendar(
        yaml_path,
        allow_missing_event_calendar=False,
    )

    assert actual is not None
    assert len(actual) == 1
    assert actual.loc[0, "type"] == "CPI"
    assert actual.loc[0, "window_days"] == [-1, 1]


def test_profile_engine_requires_event_calendar_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="event_calendar"):
        profile_engine._load_event_calendar(
            tmp_path / "missing-events.yaml",
            allow_missing_event_calendar=False,
        )


def test_profile_engine_allows_missing_event_calendar_for_debug(
    tmp_path: Path,
) -> None:
    actual = profile_engine._load_event_calendar(
        tmp_path / "missing-events.yaml",
        allow_missing_event_calendar=True,
    )

    assert actual is None


def test_profile_engine_loads_news_sentiment_when_present(tmp_path: Path) -> None:
    parquet_path = tmp_path / "sf_fed_news_sentiment.parquet"
    pd.DataFrame(
        [
            {"date": "2026-05-14", "news_sentiment": -0.1},
            {"date": "2026-05-15", "news_sentiment": 0.2},
        ]
    ).to_parquet(parquet_path, index=False)

    actual = profile_engine._load_optional_news_sentiment(parquet_path)

    assert actual is not None
    assert actual.name == "news_sentiment"
    assert actual.index.tolist() == [
        pd.Timestamp("2026-05-14"),
        pd.Timestamp("2026-05-15"),
    ]
    assert actual.tolist() == [-0.1, 0.2]


def test_profile_engine_resolves_legacy_news_sentiment_alias(tmp_path: Path) -> None:
    canonical_path = tmp_path / "sf_fed_news_sentiment.parquet"
    legacy_path = tmp_path / "news_sentiment.parquet"
    pd.DataFrame(
        [
            {"date": "2026-05-14", "news_sentiment": -0.1},
            {"date": "2026-05-15", "news_sentiment": 0.2},
        ]
    ).to_parquet(canonical_path, index=False)

    actual = profile_engine._load_optional_news_sentiment(legacy_path)

    assert actual is not None
    assert actual.name == "news_sentiment"
    assert actual.index.tolist() == [
        pd.Timestamp("2026-05-14"),
        pd.Timestamp("2026-05-15"),
    ]
    assert actual.tolist() == [-0.1, 0.2]


def test_profile_engine_loads_central_bank_text_when_present(tmp_path: Path) -> None:
    fomc_path = tmp_path / "fomc_minutes.parquet"
    powell_path = tmp_path / "powell_speeches.parquet"
    pd.DataFrame(
        [
            {
                "release_timestamp": pd.Timestamp("2026-05-01T18:00:00Z"),
                "body_text": "inflation restrictive policy",
            }
        ]
    ).to_parquet(fomc_path, index=False)
    pd.DataFrame(
        [
            {
                "publication_timestamp": pd.Timestamp("2026-05-02"),
                "body_text": "growth accommodative policy",
            }
        ]
    ).to_parquet(powell_path, index=False)

    actual = profile_engine._load_optional_central_bank_text_releases(
        fomc_path=fomc_path,
        powell_path=powell_path,
    )

    assert actual is not None
    assert len(actual) == 2
    assert set(actual["source"]) == {"fomc_minutes", "powell_speech"}


def _make_minimal_output() -> SimpleNamespace:
    """Return a minimal RegimeOutput stand-in for report tests."""
    axis = SimpleNamespace(active_label="uptrend", classification_status="classified")
    return SimpleNamespace(
        as_of_date=pd.Timestamp("2026-05-15").date(),
        trend_direction=axis,
        volatility_state=SimpleNamespace(
            active_label="low_vol", classification_status="classified"
        ),
        transition_risk=SimpleNamespace(
            state="stable",
            score=None,
            score_components=None,
            primary_drivers=[],
            triggered_rules=[],
            evidence={},
            data_quality={"status": "ok"},
        ),
        structural_causal_state=SimpleNamespace(
            event_calendar=SimpleNamespace(
                primary_label="normal_calendar",
                matching_labels=("normal_calendar",),
                evidence={},
            )
        ),
        network_fragility=None,
        volume_liquidity_state=None,
        credit_funding_state=None,
        credit_funding_state_proxy=None,
        credit_funding_effective_state=None,
        inflation_growth_state=None,
        monetary_pressure_state=None,
        cluster=None,
        change_point=None,
    )


def _make_minimal_inputs(
    macro_series: dict | None = None,
) -> "profile_engine.ProfileInputBundle":
    return profile_engine.ProfileInputBundle(
        market_data=pd.DataFrame({"date": [pd.Timestamp("2026-05-15")]}),
        end_date=pd.Timestamp("2026-05-15").date(),
        required_sessions=252,
        working_start_date=pd.Timestamp("2025-05-16").date(),
        selected_dates=[pd.Timestamp("2026-05-15").date()],
        sector_etf_closes={},
        cross_asset_closes={},
        macro_series=macro_series if macro_series is not None else {},
        event_calendar=None,
        aaii_sentiment=None,
        news_sentiment=None,
        implied_vol_30d=None,
        central_bank_text_releases=None,
        cpi_first_release=None,
        pit_constituent_intervals=pd.DataFrame({"ticker": ["AAPL"]}),
        constituent_ohlcv={},
        constituent_tickers=[],
    )


def test_sources_dict_records_cpi_nowcast_path_when_wired(tmp_path: Path) -> None:
    """Regression: cpi_nowcast must appear in sources when the arg is passed
    and the series is present in macro_series."""
    nowcast_path = tmp_path / "cleveland_fed_cpi_nowcast.parquet"
    timer = profile_engine.StageTimer()
    args = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        daily_dir=tmp_path / "daily_ohlcv",
        constituent_tree=tmp_path / "constituents",
        macro_parquet=tmp_path / "macro.parquet",
        event_calendar=tmp_path / "events.yaml",
        aaii_sentiment_parquet=None,
        news_sentiment_parquet=None,
        fomc_minutes_parquet=None,
        powell_speeches_parquet=None,
        cpi_vintages_parquet=None,
        cpi_nowcast_parquet=nowcast_path,
        pit_parquet=tmp_path / "pit.parquet",
        lookback_days=1,
    )
    cpi_nowcast_series = pd.Series(
        [0.025],
        index=pd.DatetimeIndex([pd.Timestamp("2026-05-14")]),
        name="cpi_nowcast",
    )
    inputs = _make_minimal_inputs(macro_series={"cpi_nowcast": cpi_nowcast_series})

    report = profile_engine._build_json_report(
        args=args,
        inputs=inputs,
        timeline=SimpleNamespace(outputs=[_make_minimal_output()]),
        timer=timer,
        total_wall_clock=1.0,
        per_day_emission_total=0.0,
        per_day_avg_ms=0.0,
        verification_issues=[],
    )

    assert (
        "cpi_nowcast" in report["sources"]
    ), "cpi_nowcast key must appear in sources when the parquet arg is provided"
    assert report["sources"]["cpi_nowcast"] == str(
        nowcast_path
    ), f"expected {str(nowcast_path)!r}, got {report['sources']['cpi_nowcast']!r}"


def test_sources_dict_records_cpi_nowcast_none_when_not_wired(tmp_path: Path) -> None:
    """Regression: cpi_nowcast must appear in sources as None when the series
    is absent from macro_series — the key must always be present for audit
    completeness."""
    timer = profile_engine.StageTimer()
    args = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        daily_dir=tmp_path / "daily_ohlcv",
        constituent_tree=tmp_path / "constituents",
        macro_parquet=tmp_path / "macro.parquet",
        event_calendar=tmp_path / "events.yaml",
        aaii_sentiment_parquet=None,
        news_sentiment_parquet=None,
        fomc_minutes_parquet=None,
        powell_speeches_parquet=None,
        cpi_vintages_parquet=None,
        cpi_nowcast_parquet=None,
        pit_parquet=tmp_path / "pit.parquet",
        lookback_days=1,
    )
    inputs = _make_minimal_inputs(macro_series={})

    report = profile_engine._build_json_report(
        args=args,
        inputs=inputs,
        timeline=SimpleNamespace(outputs=[_make_minimal_output()]),
        timer=timer,
        total_wall_clock=1.0,
        per_day_emission_total=0.0,
        per_day_avg_ms=0.0,
        verification_issues=[],
    )

    assert (
        "cpi_nowcast" in report["sources"]
    ), "cpi_nowcast key must always appear in sources for audit completeness"
    assert (
        report["sources"]["cpi_nowcast"] is None
    ), f"expected None when not wired, got {report['sources']['cpi_nowcast']!r}"


def test_sources_dict_records_eps_revision_disabled_by_operator(
    tmp_path: Path,
) -> None:
    timer = profile_engine.StageTimer()
    args = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        daily_dir=tmp_path / "daily_ohlcv",
        constituent_tree=tmp_path / "constituents",
        macro_parquet=tmp_path / "macro.parquet",
        event_calendar=tmp_path / "events.yaml",
        aaii_sentiment_parquet=None,
        news_sentiment_parquet=None,
        fomc_minutes_parquet=None,
        powell_speeches_parquet=None,
        cpi_vintages_parquet=None,
        cpi_nowcast_parquet=None,
        aggregate_forward_eps_weekly_history_parquet=(
            tmp_path / "aggregate_forward_eps" / "sp500_eps_weekly_history.parquet"
        ),
        disable_aggregate_forward_eps_revision=True,
        pit_parquet=tmp_path / "pit.parquet",
        lookback_days=1,
    )
    inputs = _make_minimal_inputs(macro_series={})

    report = profile_engine._build_json_report(
        args=args,
        inputs=inputs,
        timeline=SimpleNamespace(outputs=[_make_minimal_output()]),
        timer=timer,
        total_wall_clock=1.0,
        per_day_emission_total=0.0,
        per_day_avg_ms=0.0,
        verification_issues=[],
    )

    assert report["sources"]["aggregate_forward_eps_revision"] == "disabled_by_operator"


def test_sources_dict_records_cpi_nowcast_none_when_arg_absent_from_namespace(
    tmp_path: Path,
) -> None:
    """Regression: cpi_nowcast must not raise AttributeError when the arg is
    not present on the namespace (legacy callers that don't set cpi_nowcast_parquet)."""
    timer = profile_engine.StageTimer()
    args = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        daily_dir=tmp_path / "daily_ohlcv",
        constituent_tree=tmp_path / "constituents",
        macro_parquet=tmp_path / "macro.parquet",
        event_calendar=tmp_path / "events.yaml",
        aaii_sentiment_parquet=None,
        news_sentiment_parquet=None,
        fomc_minutes_parquet=None,
        powell_speeches_parquet=None,
        cpi_vintages_parquet=None,
        # intentionally omit cpi_nowcast_parquet to simulate legacy callers
        pit_parquet=tmp_path / "pit.parquet",
        lookback_days=1,
    )
    inputs = _make_minimal_inputs(macro_series={})

    report = profile_engine._build_json_report(
        args=args,
        inputs=inputs,
        timeline=SimpleNamespace(outputs=[_make_minimal_output()]),
        timer=timer,
        total_wall_clock=1.0,
        per_day_emission_total=0.0,
        per_day_avg_ms=0.0,
        verification_issues=[],
    )

    assert "cpi_nowcast" in report["sources"]
    assert report["sources"]["cpi_nowcast"] is None


def test_profile_engine_loads_cpi_first_release_when_present(tmp_path: Path) -> None:
    parquet_path = tmp_path / "cpi_all_items_vintages.parquet"
    pd.DataFrame(
        [
            {
                "date": "2026-03-01",
                "value": 315.0,
                "realtime_start": "2026-04-10",
                "realtime_end": "2026-05-10",
            },
            {
                "date": "2026-03-01",
                "value": 316.0,
                "realtime_start": "2026-05-10",
                "realtime_end": None,
            },
        ]
    ).to_parquet(parquet_path, index=False)

    actual = profile_engine._load_optional_cpi_first_release(parquet_path)

    assert actual is not None
    assert actual.name == "cpi_first_release"
    assert actual.index.tolist() == [pd.Timestamp("2026-04-10")]
    assert actual.tolist() == [315.0]
