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


def test_plan_actions_honors_explicit_first_responder() -> None:
    watcher = _load_module()
    history = {
        "messages": [
            {
                "cursor": 6,
                "sender_name": "Owner",
                "body": "Codex and Claude: continue. Claude should respond to Codex first, then Codex.",
            },
        ]
    }

    actions = watcher.plan_actions(
        history=history,
        last_cursor=5,
        participants=("Codex", "Claude"),
    )

    assert [action.participant for action in actions] == ["Claude", "Codex"]


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


def test_build_agent_prompt_includes_history_and_instruction_to_return_only_message() -> None:
    watcher = _load_module()
    action = watcher.Action(
        participant="Claude",
        profile="claude",
        cursor=5,
        message_id="msg_5",
        sender_name="Codex",
        prompt="manual prompt",
    )
    history = {
        "messages": [
            {"cursor": 4, "sender_name": "Claude", "body": "Initial position."},
            {"cursor": 5, "sender_name": "Codex", "body": "Claude, your response?"},
        ]
    }

    prompt = watcher.build_agent_prompt(action=action, history=history, space_id="room_123")

    assert "Envoy space: room_123" in prompt
    assert "[4] Claude: Initial position." in prompt
    assert "[5] Codex: Claude, your response?" in prompt
    assert "Return only the message body" in prompt


def test_agent_command_uses_bounded_noninteractive_cli() -> None:
    watcher = _load_module()

    assert watcher.agent_command("Codex", "prompt") == [
        "codex",
        "exec",
        "--sandbox",
        "read-only",
        "prompt",
    ]
    assert watcher.agent_command("Claude", "prompt") == [
        "claude",
        "--print",
        "prompt",
    ]


def test_post_agent_response_sends_as_participant_profile() -> None:
    watcher = _load_module()
    calls: list[list[str]] = []

    def fake_runner(args: list[str]) -> dict[str, str]:
        calls.append(args)
        return {"message_id": "msg_response"}

    result = watcher.post_agent_response(
        profile="codex",
        space_id="room_123",
        body="Codex response",
        envoy_runner=fake_runner,
    )

    assert result == {"message_id": "msg_response"}
    assert calls == [
        [
            "envoy",
            "--profile",
            "codex",
            "--json",
            "send",
            "--space",
            "room_123",
            "Codex response",
        ]
    ]


def test_actions_for_mode_processes_one_autonomous_turn_per_snapshot() -> None:
    watcher = _load_module()
    actions = [
        watcher.Action("Claude", "claude", 6, "msg_6", "Owner", "prompt claude"),
        watcher.Action("Codex", "codex", 6, "msg_6", "Owner", "prompt codex"),
    ]

    assert watcher.actions_for_mode(actions, mode="supervised") == actions
    assert watcher.actions_for_mode(actions, mode="autonomous") == [actions[0]]


def test_next_cursor_for_autonomous_advances_only_processed_action() -> None:
    watcher = _load_module()
    action = watcher.Action("Codex", "codex", 3, "msg_3", "Owner", "prompt")

    assert watcher.next_cursor_for_cycle(
        mode="autonomous",
        latest_cursor=6,
        last_cursor=0,
        processed_actions=[action],
    ) == 3
    assert watcher.next_cursor_for_cycle(
        mode="supervised",
        latest_cursor=6,
        last_cursor=0,
        processed_actions=[action],
    ) == 6
