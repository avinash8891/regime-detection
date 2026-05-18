from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from conftest import (
    load_profile_engine_module,
)

profile_engine = load_profile_engine_module()


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
    args = SimpleNamespace(
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
    inputs = profile_engine.ProfileInputBundle(
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
        news_sentiment=pd.Series([0.1], name="news_sentiment"),
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
    output = SimpleNamespace(
        as_of_date=pd.Timestamp("2026-05-15").date(),
        trend_direction=axis("uptrend"),
        trend_character=axis("trending"),
        volatility_state=axis("low_vol"),
        breadth_state=axis("broadening_breadth"),
        structural_causal_state=SimpleNamespace(
            event_calendar=SimpleNamespace(
                active_label="normal_calendar",
                evidence={"days_to_event": 4},
            ),
            monetary_pressure=SimpleNamespace(
                label="neutral_monetary",
                evidence={"fed_funds_change_63d": 0.0},
                data_quality={"status": "ok"},
            ),
        ),
        transition_risk=SimpleNamespace(
            label="low", score=0.2, score_components={"breadth": 0.2}
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
            },
            "as_of_date": "2026-05-15",
            "transition_risk": "low",
            "trend_direction": "uptrend",
            "volatility_state": "low_vol",
        }
    ]
    assert payload["label_summary"]["inflation_growth_state"] == {
        "active": {"unknown": 1},
        "reported": {"stale_data": 1},
        "status": {"stale_data": 1},
    }
    assert payload["label_summary"]["credit_funding_state_proxy"] == {
        "active": {"unknown": 1},
        "reported": {"no_rule_fired": 1},
        "status": {"no_rule_fired": 1},
    }
    assert payload["trailing_v2_field_status"][-1] == {
        "field": "transition_risk.score_components",
        "status": "present",
    }
    full_output = payload["full_timeline"][0]
    assert full_output["trend_direction"]["classification_status"] == "classified"
    assert full_output["trend_direction"]["evidence"]["rule_evidence"] == {
        "sample_metric": 1.25
    }
    assert full_output["inflation_growth_state"]["classification_status"] == "stale_data"
    assert full_output["inflation_growth_state"]["active_label"] == "unknown"
    assert full_output["inflation_growth_state"]["reporting_label"] == "stale_data"
    assert full_output["credit_funding_state_proxy"]["classification_status"] == (
        "no_rule_fired"
    )
    assert full_output["credit_funding_state_proxy"]["active_label"] == "unknown"
    assert full_output["credit_funding_state_proxy"]["reporting_label"] == (
        "no_rule_fired"
    )
    assert full_output["transition_risk"]["score_components"] == {"breadth": 0.2}
    assert full_output["cluster"] == {
        "cluster_id": 3,
        "distance_to_centroid": 1.75,
        "model_version": "test-cluster",
    }
    assert full_output["change_point"] == {
        "days_since_last_break": 7,
        "method": "BOCPD",
        "score": 0.42,
    }
    assert full_output["agent_routing"]["blocked_strategy_modes"] == ["short_vol"]
    assert full_output["strategy_family_constraints"]["trend_following"] == {
        "allowed": True,
        "max_lookback_days": 60,
        "reason": "classified",
    }


def test_profile_json_report_uses_loaded_bundle_values_for_input_status(
    tmp_path: Path,
) -> None:
    timer = profile_engine.StageTimer()
    args = SimpleNamespace(
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
    inputs = profile_engine.ProfileInputBundle(
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
        news_sentiment=None,
        implied_vol_30d=None,
        central_bank_text_releases=pd.DataFrame({"release_date": [1]}),
        cpi_first_release=None,
        pit_constituent_intervals=pd.DataFrame({"ticker": ["AAPL"]}),
        constituent_ohlcv={"AAPL": pd.DataFrame({"close": [1.0]})},
        constituent_tickers=["AAPL"],
    )
    output = SimpleNamespace(
        as_of_date=pd.Timestamp("2026-05-15").date(),
        trend_direction=SimpleNamespace(
            active_label="uptrend", classification_status="classified"
        ),
        volatility_state=SimpleNamespace(
            active_label="low_vol", classification_status="classified"
        ),
        transition_risk=SimpleNamespace(label="low", score=None, score_components=None),
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

    actual = profile_engine._load_optional_event_calendar(yaml_path)

    assert actual is not None
    assert len(actual) == 1
    assert actual.loc[0, "type"] == "CPI"
    assert actual.loc[0, "window_days"] == [-1, 1]


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
        transition_risk=SimpleNamespace(label="low", score=None, score_components=None),
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

    assert "cpi_nowcast" in report["sources"], (
        "cpi_nowcast key must appear in sources when the parquet arg is provided"
    )
    assert report["sources"]["cpi_nowcast"] == str(nowcast_path), (
        f"expected {str(nowcast_path)!r}, got {report['sources']['cpi_nowcast']!r}"
    )


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

    assert "cpi_nowcast" in report["sources"], (
        "cpi_nowcast key must always appear in sources for audit completeness"
    )
    assert report["sources"]["cpi_nowcast"] is None, (
        f"expected None when not wired, got {report['sources']['cpi_nowcast']!r}"
    )


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
