from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

def test_breadth_state_matches_pinned_fixtures(classified_golden_outputs) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )
    for row in golden["rows"]:
        as_of = date.fromisoformat(row["as_of_date"])
        out = classified_golden_outputs[as_of]
        assert out.breadth_state.active_label == row["expected"]["breadth_state"]


def test_breadth_state_uses_written_etf_proxy_rules_not_invented_recovery_label(market_df_for_asof) -> None:
    from regime_detection.engine import RegimeEngine

    as_of = date(2023, 12, 14)
    market_data = market_df_for_asof(as_of)
    rsp_recent_idx = (
        market_data[market_data["symbol"] == "RSP"]
        .sort_values("date")
        .tail(20)
        .index
    )
    market_data.loc[rsp_recent_idx, "close"] = market_data.loc[rsp_recent_idx, "close"] * 1.20

    out = RegimeEngine().classify(as_of_date=as_of, market_data=market_data)

    assert out.breadth_state.raw_label == "healthy_breadth"
    assert out.breadth_state.active_label == "healthy_breadth"
    rule_evidence = out.breadth_state.evidence["rule_evidence"]
    assert rule_evidence["healthy_breadth"] is True
    assert "recovery_breadth" not in rule_evidence
