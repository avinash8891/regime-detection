from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pandas as pd
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "profile_engine_30d.py"
    spec = importlib.util.spec_from_file_location("profile_engine_30d", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


profile_engine_30d = _load_script_module()


def test_profile_engine_rejects_non_positive_lookback_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["profile_engine_30d.py", "--lookback-days", "0"])

    with pytest.raises(SystemExit) as exc_info:
        profile_engine_30d.main()

    assert exc_info.value.code == 2


def test_profile_parse_args_defaults_pmi_to_materialized_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data" / "raw"
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile_engine_30d.py", "--data-root", str(data_root)],
    )

    args = profile_engine_30d._parse_args()

    assert args.pmi_path == data_root / "pmi" / "us_ism_pmi_history.parquet"


def test_profile_parse_args_accepts_json_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "profile_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile_engine_30d.py", "--json-output", str(report_path)],
    )

    args = profile_engine_30d._parse_args()

    assert args.json_output == report_path


def test_profile_parse_args_accepts_operator_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer_file = tmp_path / ".regime-operator.env"
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile_engine_30d.py", "--operator-env-file", str(pointer_file)],
    )

    args = profile_engine_30d._parse_args()

    assert args.operator_env_file == pointer_file


def test_profile_input_bundle_annotations_match_loaded_shapes() -> None:
    hints = get_type_hints(profile_engine_30d.ProfileInputBundle)

    assert hints["central_bank_text_releases"] == pd.DataFrame | None
    assert hints["pit_constituent_intervals"] is pd.DataFrame


def test_profile_reporting_label_uses_granular_status_for_unknown() -> None:
    output = SimpleNamespace(
        active_label="unknown", classification_status="no_rule_fired"
    )

    assert profile_engine_30d._reporting_label(output) == "no_rule_fired"


def test_profile_trailing_status_reports_no_rule_fired_not_unknown() -> None:
    output = SimpleNamespace(
        network_fragility=None,
        volume_liquidity_state=None,
        credit_funding_state=None,
        credit_funding_state_proxy=SimpleNamespace(
            active_label="unknown",
            classification_status="no_rule_fired",
        ),
        credit_funding_effective_state=None,
        inflation_growth_state=None,
        monetary_pressure_state=None,
        cluster=None,
        change_point=None,
        transition_risk=SimpleNamespace(score=None, score_components=None),
    )

    rows = profile_engine_30d._trailing_v2_status(output)

    assert "credit_funding_state_proxy | reported=no_rule_fired" in rows
    assert all("active_label=unknown" not in row for row in rows)


def test_timed_inflation_growth_builder_patches_axis_builder_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import regime_detection.axis_builders.series as axis_builder_series

    calls: list[str] = []

    def assess(*_args: object, **_kwargs: object) -> str:
        calls.append("assess")
        return "assessed"

    def build_inputs(*_args: object, **_kwargs: object) -> str:
        calls.append("build_inputs")
        return "inputs"

    def evaluate(*_args: object, **_kwargs: object) -> str:
        calls.append("evaluate")
        return "label"

    monkeypatch.setattr(axis_builder_series, "assess_series_input_quality", assess)
    monkeypatch.setattr(
        axis_builder_series,
        "build_inflation_growth_rule_inputs_by_date",
        build_inputs,
    )
    monkeypatch.setattr(axis_builder_series, "evaluate_inflation_growth_rules", evaluate)

    def builder(
        context: object,
        feature_store: object,
        credit_funding_active_labels_by_date: object = None,
    ) -> str:
        assert context == "context"
        assert feature_store == "store"
        assert credit_funding_active_labels_by_date == {"2026-05-15": "benign"}
        assert axis_builder_series.assess_series_input_quality() == "assessed"
        assert (
            axis_builder_series.build_inflation_growth_rule_inputs_by_date()
            == "inputs"
        )
        assert axis_builder_series.evaluate_inflation_growth_rules() == "label"
        return "built"

    timer = profile_engine_30d.StageTimer()
    timed_builder = profile_engine_30d._timed_inflation_growth_builder(timer, builder)

    actual = timed_builder(
        "context",
        "store",
        credit_funding_active_labels_by_date={"2026-05-15": "benign"},
    )

    assert actual == "built"
    assert calls == ["assess", "build_inputs", "evaluate"]
    assert timer.counts["axis_series.inflation_growth"] == 1
    assert (
        timer.counts["axis_series.inflation_growth.assess_series_input_quality"] == 1
    )
    assert timer.counts["axis_series.inflation_growth.build_rule_inputs_by_date"] == 1
    assert timer.counts["axis_series.inflation_growth.evaluate_rules"] == 1


def test_profile_json_report_emits_machine_readable_sections(tmp_path: Path) -> None:
    def axis(label: str) -> SimpleNamespace:
        return SimpleNamespace(active_label=label, classification_status="classified")

    timer = profile_engine_30d.StageTimer()
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
    inputs = profile_engine_30d.ProfileInputBundle(
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
        missing_constituent_paths=[tmp_path / "constituents" / "MSFT.parquet"],
        input_kwargs={
            "sector_etf_closes": {"XLK": pd.Series([1.0])},
            "cross_asset_closes": {},
            "macro_series": {},
            "event_calendar": None,
            "aaii_sentiment": None,
            "news_sentiment": pd.Series([0.1], name="news_sentiment"),
            "implied_vol_30d": None,
            "central_bank_text_releases": pd.DataFrame({"release_date": [1]}),
            "cpi_first_release": None,
            "pit_constituent_intervals": pd.DataFrame({"ticker": ["AAPL"]}),
            "constituent_ohlcv": {"AAPL": pd.DataFrame({"close": [1.0]})},
        },
    )
    output = SimpleNamespace(
        as_of_date=pd.Timestamp("2026-05-15").date(),
        trend_direction=axis("uptrend"),
        volatility_state=axis("low_vol"),
        transition_risk=SimpleNamespace(
            label="low", score=0.2, score_components={"breadth": 0.2}
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

    report = profile_engine_30d._build_json_report(
        args=args,
        inputs=inputs,
        timeline=SimpleNamespace(outputs=[output]),
        timer=timer,
        total_wall_clock=2.0,
        per_day_emission_total=0.5,
        per_day_avg_ms=500.0,
        verification_issues=[],
    )
    profile_engine_30d._write_json_report(report_path, report)

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
            "activated_v2_seams": {"transition_score": 0.2},
            "as_of_date": "2026-05-15",
            "transition_risk": "low",
            "trend_direction": "uptrend",
            "volatility_state": "low_vol",
        }
    ]
    assert payload["trailing_v2_field_status"][-1] == {
        "field": "transition_risk.score_components",
        "status": "present",
    }


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

    actual = profile_engine_30d._load_optional_aaii_sentiment(parquet_path)

    assert actual is not None
    pd.testing.assert_frame_equal(actual, expected)


def test_profile_engine_skips_aaii_sentiment_when_absent(tmp_path: Path) -> None:
    assert (
        profile_engine_30d._load_optional_aaii_sentiment(tmp_path / "missing.parquet")
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

    actual = profile_engine_30d._load_optional_event_calendar(yaml_path)

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

    actual = profile_engine_30d._load_optional_news_sentiment(parquet_path)

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

    actual = profile_engine_30d._load_optional_central_bank_text_releases(
        fomc_path=fomc_path,
        powell_path=powell_path,
    )

    assert actual is not None
    assert len(actual) == 2
    assert set(actual["source"]) == {"fomc_minutes", "powell_speech"}


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

    actual = profile_engine_30d._load_optional_cpi_first_release(parquet_path)

    assert actual is not None
    assert actual.name == "cpi_first_release"
    assert actual.index.tolist() == [pd.Timestamp("2026-04-10")]
    assert actual.tolist() == [315.0]
