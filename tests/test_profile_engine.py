from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pandas as pd
import pytest

from conftest import (
    load_profile_engine_module,
    profile_engine_manifest_artifact as _manifest_artifact,
    write_profile_engine_manifest as _write_profile_manifest,
)

profile_engine = load_profile_engine_module()


def test_profile_engine_rejects_non_positive_lookback_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["profile_engine.py", "--lookback-days", "0"])

    with pytest.raises(SystemExit) as exc_info:
        profile_engine.main()

    assert exc_info.value.code == 2


def test_profile_parse_args_defaults_pmi_to_materialized_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data" / "raw"
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile_engine.py", "--data-root", str(data_root)],
    )

    args = profile_engine._parse_args()

    assert args.pmi_path == data_root / "pmi" / "us_ism_pmi_history.parquet"
    assert args.daily_dir == data_root / "daily_ohlcv_762"


def test_profile_manifest_resolution_replaces_default_input_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "materialized" / "data" / "raw"
    manifest_path = _write_profile_manifest(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_engine.py",
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
        ],
    )
    args = profile_engine._parse_args()

    profile_engine._apply_manifest_input_paths(
        args, runner_name="profile_engine"
    )

    assert args.daily_dir == data_root / "daily_ohlcv_762"
    assert args.pmi_path == data_root / "pmi" / "us_ism_pmi_history.parquet"
    assert args.news_sentiment_parquet == (
        data_root / "news_sentiment" / "sf_fed_news_sentiment.parquet"
    )
    assert args.event_calendar == data_root / "event_calendar" / "us_events.yaml"
    assert "news_sentiment_parquet" in args.manifest_resolved_inputs


def test_profile_manifest_resolution_keeps_explicit_cli_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "materialized" / "data" / "raw"
    manifest_path = _write_profile_manifest(tmp_path)
    override_path = tmp_path / "manual" / "news.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_engine.py",
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--event-calendar",
            str(tmp_path / "manual" / "events.yaml"),
            "--news-sentiment-parquet",
            str(override_path),
        ],
    )
    args = profile_engine._parse_args()

    profile_engine._apply_manifest_input_paths(
        args, runner_name="profile_engine"
    )

    assert args.news_sentiment_parquet == override_path
    assert args.event_calendar == tmp_path / "manual" / "events.yaml"
    assert "news_sentiment_parquet" in args.manifest_cli_overrides
    assert "event_calendar" in args.manifest_cli_overrides


def test_manifest_resolution_failure_emits_structured_json(
    tmp_path: Path,
) -> None:
    """If the manifest resolver raises, the runner must still write a
    well-formed JSON record to ``--json-output`` so downstream dashboards
    can distinguish a manifest-shape failure from a missing file. Without
    this, the audit's P2-7 finding stands: the runner crashes silently and
    the consumer of the JSON output has no signal."""
    from regime_data_fetch.manifest_inputs import ManifestInputResolutionError
    import json

    json_path = tmp_path / "profile_30d.json"
    args = SimpleNamespace(
        json_output=json_path,
        manifest=tmp_path / "manifest.yaml",
        data_root=tmp_path / "data" / "raw",
    )
    error = ManifestInputResolutionError(
        "manifest has no artifacts required for profile_engine"
    )

    profile_engine._emit_manifest_resolution_failure(args, error)

    payload = json.loads(json_path.read_text())
    assert payload["status"] == "manifest_resolution_failure"
    assert payload["error_type"] == "ManifestInputResolutionError"
    assert "no artifacts required for profile_engine" in payload["error_message"]
    assert payload["runner_name"] == "profile_engine"
    assert payload["manifest"] == str(args.manifest)
    assert payload["data_root"] == str(args.data_root)


def test_manifest_resolution_failure_no_op_when_json_output_absent(
    tmp_path: Path,
) -> None:
    """When --json-output is not provided, the emitter is a no-op rather than
    crashing — matching the regular runner behavior (JSON write is optional)."""
    from regime_data_fetch.manifest_inputs import ManifestInputResolutionError

    args = SimpleNamespace(
        json_output=None,
        manifest=tmp_path / "manifest.yaml",
        data_root=tmp_path / "data" / "raw",
    )

    profile_engine._emit_manifest_resolution_failure(
        args,
        ManifestInputResolutionError("missing artifact"),
    )

    assert not any(tmp_path.rglob("*.json"))


def test_profile_parse_args_accepts_json_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "profile_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile_engine.py", "--json-output", str(report_path)],
    )

    args = profile_engine._parse_args()

    assert args.json_output == report_path


def test_profile_parse_args_run_timeout_seconds_default_preserves_prior_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["profile_engine.py"])
    args = profile_engine._parse_args()
    assert args.run_timeout_seconds == profile_engine.DEFAULT_RUN_TIMEOUT_SECONDS


def test_profile_parse_args_run_timeout_seconds_accepts_disable_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # P1 #7: passing 0 (or a negative integer) disables the SIGALRM budget so
    # full-history runs of profile_engine.py can let GMM clustering /
    # BOCPD complete instead of silently truncating the trailing sessions.
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile_engine.py", "--run-timeout-seconds", "0"],
    )
    args = profile_engine._parse_args()
    assert args.run_timeout_seconds == 0


def test_profile_parse_args_run_timeout_seconds_accepts_extended_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile_engine.py", "--run-timeout-seconds", "7200"],
    )
    args = profile_engine._parse_args()
    assert args.run_timeout_seconds == 7200


def test_profile_parse_args_accepts_operator_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer_file = tmp_path / ".regime-operator.env"
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile_engine.py", "--operator-env-file", str(pointer_file)],
    )

    args = profile_engine._parse_args()

    assert args.operator_env_file == pointer_file


def test_profile_input_bundle_annotations_match_loaded_shapes() -> None:
    hints = get_type_hints(profile_engine.ProfileInputBundle)

    assert hints["central_bank_text_releases"] == pd.DataFrame | None
    assert hints["pit_constituent_intervals"] is pd.DataFrame


def test_profile_verification_flags_missing_layer1_sentiment_extensions() -> None:
    output = SimpleNamespace(
        as_of_date=pd.Timestamp("2026-05-15").date(),
        trend_direction=SimpleNamespace(active_label="bull"),
        network_fragility=SimpleNamespace(active_label="correlation_concentration"),
        volume_liquidity_state=SimpleNamespace(active_label="normal_volume"),
        credit_funding_state=SimpleNamespace(active_label="credit_calm"),
        credit_funding_effective_state=SimpleNamespace(active_label="credit_calm"),
        inflation_growth_state=SimpleNamespace(active_label="goldilocks"),
        monetary_pressure_state=SimpleNamespace(active_label="neutral_monetary"),
    )
    feature_store = SimpleNamespace(
        network_fragility=object(),
        trend_direction_v2=SimpleNamespace(
            sentiment_score=None,
            news_sentiment_score=None,
            sentiment_concordance=None,
        ),
        volatility_state_v2=object(),
        breadth_state_v2=object(),
        volume_liquidity_v2=object(),
        monetary=object(),
        credit_funding=object(),
        inflation_growth=object(),
        hmm=object(),
        clustering=object(),
        change_point=object(),
    )
    inputs = SimpleNamespace(
        sector_etf_closes={"XLK": pd.Series([1.0])},
        pit_constituent_intervals=pd.DataFrame({"ticker": ["AAPL"]}),
        constituent_ohlcv={"AAPL": pd.DataFrame({"close": [1.0]})},
        cross_asset_closes={"SPY": pd.Series([1.0])},
        macro_series={"10y_yield": pd.Series([1.0])},
        event_calendar=None,
        aaii_sentiment=None,
        news_sentiment=None,
        implied_vol_30d=None,
        central_bank_text_releases=None,
        cpi_first_release=None,
    )

    issues = profile_engine._verify_invariants(
        timeline=SimpleNamespace(outputs=[output]),
        feature_store=feature_store,
        inputs=inputs,
    )

    assert "trend_direction.sentiment_score missing; missing inputs: aaii_sentiment" in issues
    assert (
        "trend_direction.news_sentiment_score missing; missing inputs: news_sentiment"
        in issues
    )
    assert (
        "trend_direction.sentiment_concordance missing; missing inputs: "
        "aaii_sentiment, news_sentiment"
    ) in issues


def test_profile_verification_warns_when_eps_revision_source_is_stale() -> None:
    output = SimpleNamespace(
        as_of_date=pd.Timestamp("2026-05-15").date(),
        trend_direction=SimpleNamespace(active_label="bull"),
        network_fragility=SimpleNamespace(active_label="correlation_concentration"),
        volume_liquidity_state=SimpleNamespace(active_label="normal_volume"),
        credit_funding_state=SimpleNamespace(active_label="credit_calm"),
        credit_funding_effective_state=SimpleNamespace(active_label="credit_calm"),
        inflation_growth_state=SimpleNamespace(active_label="goldilocks"),
        monetary_pressure_state=SimpleNamespace(active_label="neutral_monetary"),
    )
    index = pd.DatetimeIndex(["2026-05-15"])
    feature_store = SimpleNamespace(
        network_fragility=object(),
        trend_direction_v2=SimpleNamespace(
            sentiment_score=pd.Series([1.0], index=index),
            news_sentiment_score=pd.Series([0.1], index=index),
            sentiment_concordance=pd.Series([1.0], index=index),
        ),
        volatility_state_v2=object(),
        breadth_state_v2=object(),
        volume_liquidity_v2=object(),
        monetary=object(),
        credit_funding=object(),
        inflation_growth=object(),
        hmm=object(),
        clustering=object(),
        change_point=object(),
    )
    inputs = SimpleNamespace(
        selected_dates=[pd.Timestamp("2026-05-15").date()],
        sector_etf_closes={"XLK": pd.Series([1.0])},
        pit_constituent_intervals=pd.DataFrame({"ticker": ["AAPL"]}),
        constituent_ohlcv={"AAPL": pd.DataFrame({"close": [1.0]})},
        cross_asset_closes={"SPY": pd.Series([1.0])},
        macro_series={
            "10y_yield": pd.Series([1.0]),
            "aggregate_forward_eps_revision": pd.Series(
                [0.036],
                index=pd.DatetimeIndex(["2026-01-22"]),
                name="aggregate_forward_eps_revision_direction_4w",
            ),
        },
        event_calendar=None,
        aaii_sentiment=pd.DataFrame({"date": [pd.Timestamp("2026-05-15")]}),
        news_sentiment=pd.Series([0.1]),
        implied_vol_30d=None,
        central_bank_text_releases=None,
        cpi_first_release=None,
    )

    issues = profile_engine._verify_invariants(
        timeline=SimpleNamespace(outputs=[output]),
        feature_store=feature_store,
        inputs=inputs,
    )

    assert (
        "aggregate_forward_eps_revision source stale: latest=2026-01-22, "
        "run_end=2026-05-15, age_days=113"
    ) in issues


def test_profile_json_report_emits_layer1_sentiment_metric_summary(
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
        lookback_days=2,
    )
    selected_dates = [
        pd.Timestamp("2026-05-14").date(),
        pd.Timestamp("2026-05-15").date(),
    ]
    inputs = SimpleNamespace(
        end_date=pd.Timestamp("2026-05-15").date(),
        required_sessions=2,
        working_start_date=pd.Timestamp("2026-05-14").date(),
        selected_dates=selected_dates,
        sector_etf_closes={"XLK": pd.Series([1.0])},
        cross_asset_closes={"SPY": pd.Series([1.0])},
        macro_series={"10y_yield": pd.Series([1.0])},
        event_calendar=None,
        aaii_sentiment=pd.DataFrame({"date": selected_dates}),
        news_sentiment=pd.Series([0.1, 0.2]),
        implied_vol_30d=None,
        central_bank_text_releases=None,
        cpi_first_release=None,
        pit_constituent_intervals=pd.DataFrame({"ticker": ["AAPL"]}),
        constituent_tickers=["AAPL"],
        constituent_ohlcv={"AAPL": pd.DataFrame({"close": [1.0]})},
    )
    output = SimpleNamespace(
        as_of_date=pd.Timestamp("2026-05-15").date(),
        trend_direction=SimpleNamespace(active_label="bull"),
        volatility_state=SimpleNamespace(active_label="normal_vol"),
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
    index = pd.DatetimeIndex(selected_dates)
    feature_store = SimpleNamespace(
        trend_direction_v2=SimpleNamespace(
            sentiment_score=pd.Series([12.0, 13.0], index=index),
            news_sentiment_score=pd.Series([0.2, 0.25], index=index),
            sentiment_concordance=pd.Series([1.0, 1.0], index=index),
        )
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
        feature_store=feature_store,
    )

    metrics = report["feature_metrics"]["trend_direction_v2"]
    assert metrics["sentiment_score"]["non_null"] == 2
    assert metrics["news_sentiment_score"]["last_value"] == 0.25
    assert metrics["sentiment_concordance"]["last_value"] == 1.0


def test_read_symbol_ohlcv_accepts_partitioned_parquet_file_name(
    tmp_path: Path,
) -> None:
    symbol_dir = tmp_path / "daily_ohlcv" / "symbol=AAPL"
    symbol_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": "2026-05-15",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "adjusted_close": 10.5,
            }
        ]
    ).to_parquet(symbol_dir / "hash-0.parquet", index=False)

    frame = profile_engine._read_symbol_ohlcv(tmp_path / "daily_ohlcv", "AAPL")

    assert frame[["date", "close"]].to_dict(orient="records") == [
        {"date": pd.Timestamp("2026-05-15"), "close": 10.5}
    ]


def test_load_constituent_ohlcv_accepts_partitioned_parquet_file_name(
    tmp_path: Path,
) -> None:
    symbol_dir = tmp_path / "daily_ohlcv" / "symbol=AAPL"
    symbol_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": "2026-05-15",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100,
                "adjusted_close": 10.5,
            }
        ]
    ).to_parquet(symbol_dir / "hash-0.parquet", index=False)
    intervals = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "start_date": [pd.Timestamp("2020-01-01").date()],
            "end_date": [None],
        }
    )

    loaded, tickers = profile_engine._load_constituent_ohlcv_from_tree(
        tmp_path / "daily_ohlcv",
        intervals,
        start_date=pd.Timestamp("2026-05-01").date(),
        end_date=pd.Timestamp("2026-05-31").date(),
    )

    assert tickers == ["AAPL"]
    assert loaded["AAPL"]["close"].to_list() == [10.5]


def test_profile_reporting_label_uses_granular_status_for_unknown() -> None:
    output = SimpleNamespace(
        active_label="unknown", classification_status="no_rule_fired"
    )

    assert profile_engine._reporting_label(output) == "no_rule_fired"


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

    rows = profile_engine._trailing_v2_status(output)

    assert "credit_funding_state_proxy | reported=no_rule_fired" in rows
    assert all("active_label=unknown" not in row for row in rows)


def test_timed_inflation_growth_builder_patches_axis_builder_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import regime_detection.axis_builders.inflation_growth as inflation_growth_builder

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

    monkeypatch.setattr(inflation_growth_builder, "assess_series_input_quality", assess)
    monkeypatch.setattr(
        inflation_growth_builder,
        "build_inflation_growth_rule_inputs_by_date",
        build_inputs,
    )
    monkeypatch.setattr(
        inflation_growth_builder, "evaluate_inflation_growth_rules", evaluate
    )

    def builder(
        context: object,
        feature_store: object,
        credit_funding_active_labels_by_date: object = None,
    ) -> str:
        assert context == "context"
        assert feature_store == "store"
        assert credit_funding_active_labels_by_date == {"2026-05-15": "benign"}
        assert inflation_growth_builder.assess_series_input_quality() == "assessed"
        assert (
            inflation_growth_builder.build_inflation_growth_rule_inputs_by_date()
            == "inputs"
        )
        assert inflation_growth_builder.evaluate_inflation_growth_rules() == "label"
        return "built"

    timer = profile_engine.StageTimer()
    timed_builder = profile_engine._timed_inflation_growth_builder(timer, builder)

    actual = timed_builder(
        "context",
        "store",
        credit_funding_active_labels_by_date={"2026-05-15": "benign"},
    )

    assert actual == "built"
    assert calls == ["assess", "build_inputs", "evaluate"]
    assert timer.counts["axis_series.inflation_growth"] == 1
    assert timer.counts["axis_series.inflation_growth.assess_series_input_quality"] == 1
    assert timer.counts["axis_series.inflation_growth.build_rule_inputs_by_date"] == 1
    assert timer.counts["axis_series.inflation_growth.evaluate_rules"] == 1
