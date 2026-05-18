from __future__ import annotations

from regime_data_fetch.event_calendar_models import ScheduledEvent


def render_events_yaml(events: list[ScheduledEvent]) -> str:
    lines = ["events:"]
    for event in events:
        lines.extend(
            [
                f'  - date: "{event.date.isoformat()}"',
                f'    release_timestamp_et: "{event.release_timestamp_et.isoformat()}"',
                f'    market: "{event.market}"',
                f'    type: "{event.type}"',
                f'    importance: "{event.importance}"',
                f'    source: "{event.source}"',
            ]
        )
        if event.window_days is not None:
            lines.append(
                f"    window_days: [{event.window_days[0]}, {event.window_days[1]}]"
            )
        if event.approved_label is not None:
            lines.append(f'    approved_label: "{event.approved_label}"')
    return "\n".join(lines) + "\n"
