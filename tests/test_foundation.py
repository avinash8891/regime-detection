from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from regime_detection.calendar import is_nyse_trading_day
from regime_detection.config import RegimeConfig
from regime_detection.engine import ClassifyRequest, RegimeEngine
from regime_detection.versioning import engine_version
from regime_shared.pandas_compat import cow_safe_assign


@pytest.fixture(scope="module")
def baseline_v2_output_2026_05_05(v2_classify_kwargs_for_asof):
    as_of = date(2026, 5, 5)
    engine = RegimeEngine()
    return engine, engine.classify(
        as_of_date=as_of, **v2_classify_kwargs_for_asof(as_of)
    )


def test_engine_version_matches_spec_prefix() -> None:
    assert engine_version().startswith("regime-engine-v")


def test_version_coupling_pyproject_matches_engine_version() -> None:
    # Spec lock: package version and emitted engine_version must stay aligned.
    import tomllib

    repo_root = Path(__file__).resolve().parents[1]
    pyproject = (repo_root / "pyproject.toml").read_bytes()
    version = tomllib.loads(pyproject.decode("utf-8"))["project"]["version"]
    assert engine_version() == f"regime-engine-v{version}"


def test_classify_requires_nyse_trading_day(market_df_for_asof) -> None:
    as_of = date(2017, 1, 1)  # Sunday
    assert not is_nyse_trading_day(as_of)
    engine = RegimeEngine()
    df = market_df_for_asof(date(2017, 1, 3))
    with pytest.raises(ValueError) as excinfo:
        engine.classify(as_of_date=as_of, market_data=df)
    msg = str(excinfo.value)
    assert "Nearest prior trading day" in msg
    assert "Nearest next trading day" in msg


def test_market_data_contract_requires_spy(market_df_for_asof) -> None:
    engine = RegimeEngine()
    as_of = date(2026, 5, 5)
    assert is_nyse_trading_day(as_of)
    df = market_df_for_asof(as_of)
    df = df[df["symbol"] != "SPY"].copy()
    with pytest.raises(ValueError) as excinfo:
        engine.classify(as_of_date=as_of, market_data=df, event_calendar=pd.DataFrame())
    assert "must contain SPY" in str(excinfo.value)


def test_engine_requires_event_calendar(market_df_for_asof) -> None:
    engine = RegimeEngine()
    as_of = date(2026, 5, 5)
    df = market_df_for_asof(as_of)

    with pytest.raises(ValueError, match="event_calendar is required"):
        engine.classify(as_of_date=as_of, market_data=df)

    with pytest.raises(ValueError, match="event_calendar is required"):
        engine.classify_window(
            end_date=as_of,
            market_data=df,
            lookback_days=1,
        )


def test_classify_request_requires_event_calendar(market_df_for_asof) -> None:
    request = ClassifyRequest(
        end_date=date(2026, 5, 5),
        market_data=market_df_for_asof(date(2026, 5, 5)),
        lookback_days=1,
    )

    with pytest.raises(ValueError, match="event_calendar is required"):
        RegimeEngine().classify_request(request)


def test_classify_request_rejects_non_positive_lookback(
    market_df_for_asof, event_calendar_df
) -> None:
    request = ClassifyRequest(
        end_date=date(2023, 12, 14),
        market_data=market_df_for_asof,
        lookback_days=0,
        event_calendar=event_calendar_df,
    )

    with pytest.raises(ValueError, match="lookback_days must be positive"):
        RegimeEngine().classify_request(request)


def test_classify_request_has_no_legacy_breadth_data_field() -> None:
    assert "breadth_data" not in ClassifyRequest.__dataclass_fields__


def test_classify_rejects_legacy_breadth_data_argument(market_df_for_asof) -> None:
    engine = RegimeEngine()

    with pytest.raises(TypeError, match="unexpected keyword argument 'breadth_data'"):
        engine.classify(
            as_of_date=date(2026, 5, 5),
            market_data=market_df_for_asof(date(2026, 5, 5)),
            breadth_data=pd.DataFrame({"legacy": [1]}),
        )


def test_classify_request_rejects_profile_manifest_without_event_calendar_resolution(
    market_df_for_asof, event_calendar_df
) -> None:
    request = ClassifyRequest(
        end_date=date(2023, 12, 14),
        market_data=market_df_for_asof(date(2023, 12, 14)),
        event_calendar=event_calendar_df,
        request_source="profile_manifest",
        manifest_resolved_inputs=frozenset({"news_sentiment_parquet"}),
    )

    with pytest.raises(
        ValueError,
        match="profile_manifest request missing manifest-backed required inputs",
    ):
        RegimeEngine().classify_request(request)


def test_classify_request_accepts_profile_manifest_event_calendar_cli_override(
    market_df_for_asof, event_calendar_df
) -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "regime_detection"
        / "configs"
        / "core3-v1.0.0.yaml"
    )
    request = ClassifyRequest(
        end_date=date(2023, 12, 14),
        market_data=market_df_for_asof(date(2023, 12, 14)),
        event_calendar=event_calendar_df,
        request_source="profile_manifest",
        manifest_cli_overrides=frozenset({"event_calendar"}),
    )

    output = RegimeEngine(config_path=config_path).classify_request(request)

    assert output.outputs[-1].as_of_date == date(2023, 12, 14)


def test_classify_request_rejects_manifest_metadata_in_direct_mode(
    market_df_for_asof, event_calendar_df
) -> None:
    request = ClassifyRequest(
        end_date=date(2023, 12, 14),
        market_data=market_df_for_asof(date(2023, 12, 14)),
        event_calendar=event_calendar_df,
        manifest_resolved_inputs=frozenset({"event_calendar"}),
    )

    with pytest.raises(ValueError, match="manifest metadata requires profile_manifest"):
        RegimeEngine().classify_request(request)


def test_classify_uses_request_object(
    market_df_for_asof,
    event_calendar_df,
) -> None:
    as_of = date(2026, 5, 5)
    config_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "regime_detection"
        / "configs"
        / "core3-v1.0.0.yaml"
    )
    engine = RegimeEngine(config_path=config_path)
    request = ClassifyRequest(
        end_date=as_of,
        market_data=market_df_for_asof(as_of),
        lookback_days=1,
        event_calendar=event_calendar_df,
    )

    out = engine.classify_request(request)

    assert out.outputs[-1].as_of_date == as_of


def test_regime_config_forbids_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        RegimeConfig.model_validate(
            {
                "config_version": "core3-v1.0.0",
                "trading_calendar": "NYSE",
                "hysteresis": {
                    "trend_direction_escalation_days": 1,
                    "trend_direction_deescalation_days": 3,
                    "trend_character_escalation_days": 1,
                    "trend_character_deescalation_days": 3,
                    "volatility_escalation_days": 1,
                    "volatility_deescalation_days": 2,
                    "breadth_escalation_days": 1,
                    "breadth_deescalation_days": 2,
                    "composite_deescalation_days": 3,
                },
                "unknown_key": True,
            }
        )


def test_default_config_is_packaged_and_loadable() -> None:
    # Guard against config drift: default config must be loadable from packaged resources.
    from regime_detection.config import load_default_regime_config

    cfg = load_default_regime_config()
    assert cfg.config_version == "core3-v2.0.0"
    assert cfg.trend_direction.deescalation_days_by_label
    assert cfg.volatility_state.deescalation_days_by_label


def test_classify_emits_regime_output_shape(baseline_v2_output_2026_05_05) -> None:
    as_of = date(2026, 5, 5)
    assert is_nyse_trading_day(as_of)
    engine, out = baseline_v2_output_2026_05_05
    assert out.engine_version == engine_version()
    assert out.config_version == engine.config.config_version
    assert out.as_of_date == as_of
    assert out.market == "SPY"


def test_classify_accepts_timestamp_as_of_date(v2_classify_kwargs_for_asof) -> None:
    engine = RegimeEngine()
    # Common caller input: pandas Timestamp. Must be accepted and normalized.
    out = engine.classify(
        as_of_date=pd.Timestamp("2026-05-05", tz="America/New_York"),
        **v2_classify_kwargs_for_asof(date(2026, 5, 5)),
    )
    assert out.as_of_date == date(2026, 5, 5)


def test_classify_accepts_market_data_with_string_dates(
    v2_classify_kwargs_for_asof,
) -> None:
    engine = RegimeEngine()
    kwargs = v2_classify_kwargs_for_asof(date(2026, 5, 5))
    df = kwargs["market_data"].copy()
    df = cow_safe_assign(
        df, {"date": pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")}
    )
    kwargs["market_data"] = df

    out = engine.classify(
        as_of_date=date(2026, 5, 5),
        **kwargs,
    )

    assert out.as_of_date == date(2026, 5, 5)


def test_engine_rejects_path_based_event_calendar_input(market_df_for_asof) -> None:
    engine = RegimeEngine()
    df = market_df_for_asof(date(2023, 12, 14))
    event_path = (
        Path(__file__).resolve().parent / "fixtures" / "events" / "us_events.yaml"
    )

    with pytest.raises(TypeError, match="event_calendar must be a pandas DataFrame"):
        engine.classify(
            as_of_date=date(2023, 12, 14),
            market_data=df,
            event_calendar=event_path,
        )
