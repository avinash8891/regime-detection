"""Regression: previously-captured V1-wire-shape JSON outputs must continue to
parse through the frozen V1 model shim and round-trip exactly (modulo
engine_version, which Phase A bumped to v2.0.0).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _v1_frozen_models import RegimeOutputV1Frozen  # noqa: E402
from regime_detection.config import load_regime_config
from regime_detection.engine import RegimeEngine

_FROZEN_DIR = Path(__file__).resolve().parent / "fixtures" / "v1_frozen_outputs"
_V1_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "regime_detection"
    / "configs"
    / "core3-v1.0.0.yaml"
)
_V2_EXTENSION_FIELDS = {
    "inflation_growth_state",
    "credit_funding_state",
    "credit_funding_state_proxy",
    "credit_funding_effective_state",
    "volume_liquidity_state",
    "monetary_pressure_state",
    "change_point",
    "hmm",
    "cluster",
    "agent_routing",
    "strategy_family_constraints",
    "effective_strategy_constraints",
    "classification_coverage",
}


def test_v1_frozen_outputs_parse_through_v1_frozen_models() -> None:
    json_paths = sorted(_FROZEN_DIR.glob("*.json"))
    assert json_paths, f"no frozen V1 outputs found under {_FROZEN_DIR}"

    for json_path in json_paths:
        original_text = json_path.read_text()
        original_parsed = json.loads(original_text)

        parsed = RegimeOutputV1Frozen.model_validate_json(original_text)
        round_tripped_text = parsed.model_dump_json(exclude_none=True)
        round_tripped_parsed = json.loads(round_tripped_text)

        original_no_engine = {
            k: v for k, v in original_parsed.items() if k != "engine_version"
        }
        round_tripped_no_engine = {
            k: v for k, v in round_tripped_parsed.items() if k != "engine_version"
        }

        assert (
            round_tripped_no_engine == original_no_engine
        ), f"V1 frozen round-trip drift for {json_path.name}"


def test_v1_frozen_outputs_match_live_engine_replay(
    market_df_for_asof,
    event_calendar_df,
) -> None:
    engine = RegimeEngine()
    config = load_regime_config(_V1_CONFIG_PATH)
    json_paths = sorted(_FROZEN_DIR.glob("*.json"))
    assert json_paths, f"no frozen V1 outputs found under {_FROZEN_DIR}"

    for json_path in json_paths:
        expected = json.loads(json_path.read_text())
        as_of = date.fromisoformat(expected["as_of_date"])
        market_data = market_df_for_asof(as_of)

        actual_output = engine.classify(
            as_of_date=as_of,
            market_data=market_data,
            event_calendar=event_calendar_df,
            config=config,
        )
        actual = json.loads(actual_output.model_dump_json(exclude_none=True))

        expected_no_engine = {
            k: v for k, v in expected.items() if k != "engine_version"
        }
        actual_no_engine = {k: v for k, v in actual.items() if k != "engine_version"}

        assert (
            actual_no_engine == expected_no_engine
        ), f"live V1 frozen replay drift for {json_path.name}"


def test_v1_live_engine_replay_is_independent_of_extra_history_length(
    raw_market_data,
    event_calendar_df,
) -> None:
    engine = RegimeEngine()
    config = load_regime_config(_V1_CONFIG_PATH)
    as_of = date(2024, 4, 15)
    full_market_data = (
        raw_market_data[raw_market_data["date"] <= as_of].copy().reset_index(drop=True)
    )
    spy_sessions = sorted(
        raw_market_data.loc[
            (raw_market_data["symbol"] == "SPY") & (raw_market_data["date"] <= as_of),
            "date",
        ].unique()
    )
    shorter_start = spy_sessions[-320]
    shorter_market_data = (
        raw_market_data[
            (raw_market_data["date"] <= as_of)
            & (raw_market_data["date"] >= shorter_start)
        ]
        .copy()
        .reset_index(drop=True)
    )

    assert shorter_market_data["date"].min() > full_market_data["date"].min()
    assert (
        shorter_market_data.loc[
            shorter_market_data["symbol"] == "SPY", "date"
        ].nunique()
        == 320
    )

    full = engine.classify(
        as_of_date=as_of,
        market_data=full_market_data,
        event_calendar=event_calendar_df,
        config=config,
    )
    shorter = engine.classify(
        as_of_date=as_of,
        market_data=shorter_market_data,
        event_calendar=event_calendar_df,
        config=config,
    )

    full_payload = json.loads(full.model_dump_json(exclude_none=True))
    shorter_payload = json.loads(shorter.model_dump_json(exclude_none=True))

    assert {
        key: value for key, value in shorter_payload.items() if key != "engine_version"
    } == {key: value for key, value in full_payload.items() if key != "engine_version"}


def test_v1_live_replay_omits_all_v2_extension_fields(
    market_df_for_asof,
    event_calendar_df,
) -> None:
    engine = RegimeEngine()
    config = load_regime_config(_V1_CONFIG_PATH)
    json_paths = sorted(_FROZEN_DIR.glob("*.json"))
    assert json_paths, f"no frozen V1 outputs found under {_FROZEN_DIR}"

    for json_path in json_paths:
        expected = json.loads(json_path.read_text())
        as_of = date.fromisoformat(expected["as_of_date"])
        output = engine.classify(
            as_of_date=as_of,
            market_data=market_df_for_asof(as_of),
            event_calendar=event_calendar_df,
            config=config,
        )
        actual = json.loads(output.model_dump_json(exclude_none=True))

        assert not (_V2_EXTENSION_FIELDS & actual.keys()), json_path.name
        assert {
            key: value for key, value in actual.items() if key != "engine_version"
        } == {key: value for key, value in expected.items() if key != "engine_version"}
