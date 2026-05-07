from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

def test_volatility_state_matches_pinned_fixtures(classified_golden_outputs) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    golden = yaml.safe_load(
        (repo_root / "tests" / "fixtures" / "derived" / "golden_dates.yaml").read_text()
    )
    for row in golden["rows"]:
        as_of = date.fromisoformat(row["as_of_date"])
        out = classified_golden_outputs[as_of]
        assert out.volatility_state.active_label == row["expected"]["volatility_state"]
