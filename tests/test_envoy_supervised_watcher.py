from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "envoy_supervised_watcher.py"
    spec = importlib.util.spec_from_file_location("envoy_supervised_watcher", script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_plan_actions_detects_addressed_agents_after_last_cursor() -> None:
    watcher = _load_module()
    history = {
        "messages": [
            {
                "cursor": 4,
                "sender_name": "Claude",
                "body": "Codex, what do you think about the split?",
            },
            {
                "cursor": 5,
                "sender_name": "Owner",
                "body": "Codex and Claude: continue the debate in Envoy.",
            },
        ]
    }

    actions = watcher.plan_actions(
        history=history,
        last_cursor=3,
        participants=("Codex", "Claude"),
    )

    assert [action.participant for action in actions] == ["Codex", "Codex", "Claude"]
    assert actions[0].cursor == 4
    assert actions[1].cursor == 5
    assert "profile codex" in actions[1].prompt
    assert "profile claude" in actions[2].prompt


def test_plan_actions_ignores_messages_from_same_participant() -> None:
    watcher = _load_module()
    history = {
        "messages": [
            {
                "cursor": 5,
                "sender_name": "Codex",
                "body": "Claude, please respond to this.",
            },
            {
                "cursor": 6,
                "sender_name": "Claude",
                "body": "Claude: note to self.",
            },
        ]
    }

    actions = watcher.plan_actions(
        history=history,
        last_cursor=4,
        participants=("Codex", "Claude"),
    )

    assert [action.participant for action in actions] == ["Claude"]
    assert actions[0].cursor == 5


def test_plan_actions_ignores_system_events() -> None:
    watcher = _load_module()
    history = {
        "messages": [
            {
                "cursor": 1,
                "sender_name": "",
                "is_system_event": True,
                "kind": "system",
                "body": "Codex joined the room",
            },
            {
                "cursor": 2,
                "sender_name": "",
                "message_kind": "system",
                "body": "Claude joined the room",
            },
        ]
    }

    actions = watcher.plan_actions(
        history=history,
        last_cursor=0,
        participants=("Codex", "Claude"),
    )

    assert actions == []


def test_cursor_state_round_trips(tmp_path: Path) -> None:
    watcher = _load_module()
    state_path = tmp_path / "watcher-state.json"

    assert watcher.read_last_cursor(state_path) == 0

    watcher.write_last_cursor(state_path, 12)

    assert json.loads(state_path.read_text(encoding="utf-8")) == {"last_cursor": 12}
    assert watcher.read_last_cursor(state_path) == 12


def test_parse_envoy_json_output_accepts_json_lines() -> None:
    watcher = _load_module()
    output = "\n".join(
        [
            json.dumps({"cursor": 3, "body": "first"}),
            json.dumps({"cursor": 4, "body": "second"}),
        ]
    )

    assert watcher.parse_envoy_json_output(output) == [
        {"cursor": 3, "body": "first"},
        {"cursor": 4, "body": "second"},
    ]
