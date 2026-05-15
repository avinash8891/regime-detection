from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from regime_data_fetch.event_sources.approvals import load_approval_overlay


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "approve_group_b_candidate.py"
    spec = importlib.util.spec_from_file_location("approve_group_b_candidate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


approve_group_b_candidate = _load_script_module()


def _write_candidates(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def _argv(tmp_path: Path, candidate_id: str) -> list[str]:
    return [
        "approve_group_b_candidate.py",
        "--candidate-id",
        candidate_id,
        "--approver",
        "avinash",
        "--candidates",
        str(tmp_path / "candidates.parquet"),
        "--overlay",
        str(tmp_path / "group_b_approvals.yaml"),
    ]


def test_approve_group_b_candidate_rejects_non_pending_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_candidates(
        tmp_path / "candidates.parquet",
        [
            {
                "candidate_id": "abc123",
                "event_type": "budget",
                "date": "2026-09-30",
                "importance": "medium",
                "source_count": 1,
                "promotion_outcome": "promote",
                "requires_manual_review": False,
            }
        ],
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "abc123"))

    with pytest.raises(SystemExit, match="not pending manual review"):
        approve_group_b_candidate.main()


def test_approve_group_b_candidate_uses_utc_approved_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_candidates(
        tmp_path / "candidates.parquet",
        [
            {
                "candidate_id": "def456",
                "event_type": "geopolitical_event",
                "date": "2022-02-24",
                "importance": "high",
                "source_count": 3,
                "promotion_outcome": "withhold",
                "requires_manual_review": True,
            }
        ],
    )
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "def456"))
    monkeypatch.setattr(approve_group_b_candidate, "_utc_today", lambda: dt.date(2026, 5, 15))

    assert approve_group_b_candidate.main() == 0

    approvals = load_approval_overlay(tmp_path / "group_b_approvals.yaml")
    assert approvals[0].approved_at == dt.date(2026, 5, 15)
