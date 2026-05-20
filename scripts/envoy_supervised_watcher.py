#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, NamedTuple


DEFAULT_PARTICIPANTS = ("Codex:codex", "Claude:claude")
SUPPORTED_AGENT_COMMANDS = {"Codex", "Claude"}


class Participant(NamedTuple):
    name: str
    profile: str


class Action(NamedTuple):
    participant: str
    profile: str
    cursor: int
    message_id: str
    sender_name: str
    prompt: str


def read_last_cursor(state_path: Path) -> int:
    if not state_path.exists():
        return 0
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return int(payload.get("last_cursor", 0))


def write_last_cursor(state_path: Path, cursor: int) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"last_cursor": cursor}, indent=2) + "\n", encoding="utf-8")


def parse_participants(values: tuple[str, ...]) -> tuple[Participant, ...]:
    participants: list[Participant] = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"participant must be NAME:PROFILE, got {value!r}")
        name, profile = value.split(":", 1)
        name = name.strip()
        profile = profile.strip()
        if not name or not profile:
            raise ValueError(f"participant must be NAME:PROFILE, got {value!r}")
        participants.append(Participant(name=name, profile=profile))
    return tuple(participants)


def _coerce_participants(values: tuple[str | Participant, ...]) -> tuple[Participant, ...]:
    participants: list[Participant] = []
    for value in values:
        if isinstance(value, Participant):
            participants.append(value)
        else:
            participants.append(Participant(name=value, profile=value.lower()))
    return tuple(participants)


def _mentions_participant(body: str, participant: Participant) -> bool:
    return re.search(rf"\b{re.escape(participant.name)}\b", body, flags=re.IGNORECASE) is not None


def _participants_for_message(body: str, participants: tuple[Participant, ...]) -> tuple[Participant, ...]:
    mentioned = [participant for participant in participants if _mentions_participant(body, participant)]
    first_responder: Participant | None = None
    for participant in mentioned:
        pattern = rf"\b{re.escape(participant.name)}\b\s+should\s+respond\b.*\bfirst\b"
        if re.search(pattern, body, flags=re.IGNORECASE | re.DOTALL):
            first_responder = participant
            break
    if first_responder is None:
        return tuple(mentioned)
    return tuple([first_responder, *[participant for participant in mentioned if participant != first_responder]])


def _prompt_for_action(*, participant: Participant, space_id: str, cursor: int, sender_name: str) -> str:
    return (
        f"Read Envoy space {space_id} using profile {participant.profile}. "
        f"Respond to the latest message at cursor {cursor} from {sender_name} if it is actionable for you. "
        "Send your response back into Envoy. If nothing is actionable, stay silent."
    )


def plan_actions(
    *,
    history: dict[str, Any],
    last_cursor: int,
    participants: tuple[str | Participant, ...],
    space_id: str = "<space-id>",
) -> list[Action]:
    parsed_participants = _coerce_participants(participants)
    actions: list[Action] = []
    messages = history.get("messages", [])
    for message in messages:
        cursor = int(message.get("cursor", 0))
        if cursor <= last_cursor:
            continue
        if message.get("is_system_event") or message.get("kind") == "system" or message.get("message_kind") == "system":
            continue

        body = str(message.get("body", ""))
        sender_name = str(message.get("sender_name", ""))
        message_id = str(message.get("id") or message.get("message_id") or "")
        for participant in _participants_for_message(body, parsed_participants):
            if sender_name.casefold() == participant.name.casefold():
                continue
            actions.append(
                Action(
                    participant=participant.name,
                    profile=participant.profile,
                    cursor=cursor,
                    message_id=message_id,
                    sender_name=sender_name,
                    prompt=_prompt_for_action(
                        participant=participant,
                        space_id=space_id,
                        cursor=cursor,
                        sender_name=sender_name or "unknown sender",
                    ),
                )
            )
    return actions


def parse_envoy_json_output(output: str) -> dict[str, Any] | list[Any]:
    stripped = output.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in stripped.splitlines() if line.strip()]
        return records


def _history_excerpt(history: dict[str, Any], *, limit: int = 12) -> str:
    messages = history.get("messages", [])[-limit:]
    lines: list[str] = []
    for message in messages:
        if message.get("is_system_event") or message.get("kind") == "system" or message.get("message_kind") == "system":
            continue
        cursor = int(message.get("cursor", 0))
        sender = str(message.get("sender_name") or "unknown")
        body = str(message.get("body", "")).strip()
        lines.append(f"[{cursor}] {sender}: {body}")
    return "\n".join(lines)


def build_agent_prompt(*, action: Action, history: dict[str, Any], space_id: str) -> str:
    return (
        f"You are {action.participant} participating in an Envoy shared space.\n"
        f"Envoy space: {space_id}\n"
        f"Action cursor: {action.cursor}\n"
        f"Sender: {action.sender_name or 'unknown sender'}\n\n"
        "Recent visible Envoy history:\n"
        f"{_history_excerpt(history)}\n\n"
        "If the latest addressed message is actionable for you, write the response you want posted to Envoy. "
        "If nothing is actionable, return exactly: NO_ACTION\n"
        "Return only the message body. Do not include markdown fences, command transcripts, or explanations outside the Envoy message."
    )


def agent_command(participant: str, prompt: str) -> list[str]:
    if participant == "Codex":
        return ["codex", "exec", "--sandbox", "read-only", prompt]
    if participant == "Claude":
        return ["claude", "--print", "--model", "sonnet", prompt]
    raise ValueError(f"unsupported autonomous participant {participant!r}")


def generate_agent_response(*, participant: str, prompt: str) -> str:
    result = subprocess.run(agent_command(participant, prompt), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"{participant} exited {result.returncode}")
    return result.stdout.strip()


def post_agent_response(
    *,
    profile: str,
    space_id: str,
    body: str,
    envoy_runner=None,
) -> dict[str, Any] | list[Any]:
    if envoy_runner is None:
        envoy_runner = _run_envoy_json
    return envoy_runner(["envoy", "--profile", profile, "--json", "send", "--space", space_id, body])


def actions_for_mode(actions: list[Action], *, mode: str) -> list[Action]:
    if mode == "autonomous":
        return actions[:1]
    return actions


def next_cursor_for_cycle(
    *,
    mode: str,
    latest_cursor: int,
    last_cursor: int,
    processed_actions: list[Action],
) -> int:
    if mode == "autonomous" and processed_actions:
        return max(action.cursor for action in processed_actions)
    if latest_cursor > last_cursor:
        return latest_cursor
    return last_cursor


def _run_envoy_json(args: list[str]) -> dict[str, Any] | list[Any]:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"{args[0]} exited {result.returncode}")
    return parse_envoy_json_output(result.stdout)


def read_history(*, profile: str, space_id: str, limit: int) -> dict[str, Any]:
    payload = _run_envoy_json(["envoy", "--profile", profile, "--json", "history", space_id, "--limit", str(limit)])
    if not isinstance(payload, dict):
        raise RuntimeError("envoy history returned non-object JSON")
    return payload


def read_tasks(*, profile: str, space_id: str) -> list[Any]:
    payload = _run_envoy_json(
        ["envoy", "--profile", profile, "--json", "task", "list", "--space", space_id, "--include-completed"]
    )
    if isinstance(payload, dict):
        return list(payload.get("tasks", []))
    if isinstance(payload, list):
        return payload
    raise RuntimeError("envoy task list returned unsupported JSON")


def read_inbox(*, profile: str, space_id: str) -> list[Any]:
    payload = _run_envoy_json(["envoy", "--profile", profile, "--json", "inbox", "--space", space_id, "read"])
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and not payload:
        return []
    return [payload]


def wait_for_events(*, profile: str, space_id: str, timeout: int) -> None:
    subprocess.run(
        [
            "envoy",
            "--profile",
            profile,
            "--json",
            "events",
            "--space",
            space_id,
            "--types",
            "new_message,epoch_bumped,capability_revoked",
            "--exclude-self",
            "--watch-timeout",
            str(timeout),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _default_state_path(space_id: str) -> Path:
    safe_space = re.sub(r"[^A-Za-z0-9_.-]+", "_", space_id)
    return Path.home() / ".envoy" / f"supervised-watcher-{safe_space}.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded supervised Envoy watcher for manual multi-agent coordination.")
    parser.add_argument("--space", required=True, help="Envoy space id to watch.")
    parser.add_argument("--profile", default="owner", help="Profile used to inspect the space.")
    parser.add_argument(
        "--mode",
        choices=("supervised", "autonomous"),
        default="supervised",
        help="supervised prints prompts; autonomous invokes supported local agent CLIs and posts their responses.",
    )
    parser.add_argument(
        "--participant",
        action="append",
        default=list(DEFAULT_PARTICIPANTS),
        help="Participant mapping as NAME:PROFILE. May be repeated.",
    )
    parser.add_argument("--state-path", type=Path, default=None, help="Path for last-cursor watcher state.")
    parser.add_argument("--history-limit", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=10, help="Maximum watch cycles before exit.")
    parser.add_argument("--max-agent-turns", type=int, default=6, help="Maximum autonomous agent responses before exit.")
    parser.add_argument("--watch-timeout", type=int, default=30, help="Seconds to wait for Envoy events per cycle.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds to sleep after each cycle.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    participants = parse_participants(tuple(args.participant))
    state_path = args.state_path or _default_state_path(args.space)
    last_cursor = read_last_cursor(state_path)

    print(f"Watching Envoy space {args.space} as profile {args.profile}.")
    print(f"Mode: {args.mode}")
    print(f"State file: {state_path}")
    print(f"Starting after cursor {last_cursor}. Press Ctrl+C to stop.")

    agent_turns = 0
    for _ in range(args.iterations):
        inbox_count = len(read_inbox(profile=args.profile, space_id=args.space))
        task_count = len(read_tasks(profile=args.profile, space_id=args.space))
        history = read_history(profile=args.profile, space_id=args.space, limit=args.history_limit)
        actions = plan_actions(
            history=history,
            last_cursor=last_cursor,
            participants=participants,
            space_id=args.space,
        )
        latest_cursor = int(history.get("latest_cursor", last_cursor))

        if actions:
            print(f"\nNew actionable Envoy state: inbox={inbox_count}, tasks={task_count}")
            processed_actions: list[Action] = []
            for action in actions_for_mode(actions, mode=args.mode):
                print(f"\n[{action.participant}] cursor={action.cursor} sender={action.sender_name}")
                if args.mode == "supervised":
                    print(action.prompt)
                    processed_actions.append(action)
                    continue

                if action.participant not in SUPPORTED_AGENT_COMMANDS:
                    print(f"Skipping unsupported autonomous participant {action.participant!r}.")
                    processed_actions.append(action)
                    continue
                if agent_turns >= args.max_agent_turns:
                    print(f"Autonomous turn limit reached ({args.max_agent_turns}); skipping remaining actions.")
                    continue

                agent_prompt = build_agent_prompt(action=action, history=history, space_id=args.space)
                response = generate_agent_response(participant=action.participant, prompt=agent_prompt)
                if response == "NO_ACTION":
                    print(f"{action.participant} returned NO_ACTION.")
                    processed_actions.append(action)
                    continue
                post_agent_response(profile=action.profile, space_id=args.space, body=response)
                agent_turns += 1
                processed_actions.append(action)
                print(f"Posted {action.participant} response via profile {action.profile}.")
        elif latest_cursor > last_cursor:
            print(f"\nObserved Envoy updates through cursor {latest_cursor}; no addressed action needed.")
            processed_actions = []

        next_cursor = next_cursor_for_cycle(
            mode=args.mode,
            latest_cursor=latest_cursor,
            last_cursor=last_cursor,
            processed_actions=processed_actions if actions else [],
        )
        if next_cursor > last_cursor:
            write_last_cursor(state_path, next_cursor)
            last_cursor = next_cursor

        wait_for_events(profile=args.profile, space_id=args.space, timeout=args.watch_timeout)
        time.sleep(args.poll_interval)
        if args.mode == "autonomous" and agent_turns >= args.max_agent_turns:
            break

    print(f"Watcher stopped after {args.iterations} iterations at cursor {last_cursor}; agent_turns={agent_turns}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
