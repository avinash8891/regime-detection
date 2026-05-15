from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from regime_detection.config import load_default_regime_config
from regime_detection.event_calendar import classify_event_calendar
from regime_detection.loaders import load_event_calendar


def test_load_event_calendar_preserves_group_b_approval_marker(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.yaml"
    path.write_text(
        "\n".join(
            [
                "events:",
                '  - date: "2026-11-03"',
                '    market: "US"',
                '    type: "geopolitical_event"',
                '    importance: "high"',
                '    approved_label: "geopolitical_event"',
            ]
        )
        + "\n"
    )

    loaded = load_event_calendar(path, market="US")

    assert loaded.iloc[0]["approved_label"] == "geopolitical_event"


def test_event_calendar_renders_approved_geopolitical_event() -> None:
    cfg = load_default_regime_config()
    events = pd.DataFrame(
        [
            {
                "date": date(2026, 11, 3),
                "market": "US",
                "type": "election",
                "importance": "high",
                "publication_date": date(2026, 8, 1),
            },
            {
                "date": date(2026, 11, 3),
                "market": "US",
                "type": "geopolitical_event",
                "importance": "high",
                "publication_date": date(2026, 11, 3),
                "approved_label": "geopolitical_event",
            },
            {
                "date": date(2026, 11, 3),
                "market": "US",
                "type": "budget",
                "importance": "medium",
                "publication_date": date(2026, 8, 1),
            },
        ]
    )

    out = classify_event_calendar(
        as_of_date=date(2026, 11, 3),
        event_calendar=events,
        config=cfg,
    )

    assert out.active_label == "geopolitical_event"
    assert out.evidence["selected_via_precedence"] == "geopolitical_event"
    assert set(out.evidence["all_matching_events"]) >= {
        "geopolitical_event",
        "election_window",
        "budget_week",
    }


def test_event_calendar_does_not_render_unapproved_geopolitical_event() -> None:
    cfg = load_default_regime_config()
    events = pd.DataFrame(
        [
            {
                "date": date(2026, 11, 3),
                "market": "US",
                "type": "election",
                "importance": "high",
                "publication_date": date(2026, 8, 1),
            },
            {
                "date": date(2026, 11, 3),
                "market": "US",
                "type": "geopolitical_event",
                "importance": "high",
                "publication_date": date(2026, 11, 3),
            },
        ]
    )

    out = classify_event_calendar(
        as_of_date=date(2026, 11, 3),
        event_calendar=events,
        config=cfg,
    )

    assert out.active_label == "election_window"
    assert "geopolitical_event" not in out.evidence["all_matching_events"]
