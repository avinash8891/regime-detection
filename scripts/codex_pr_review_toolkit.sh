#!/usr/bin/env bash
set -euo pipefail

base_ref="${1:-origin/main}"
if [[ $# -gt 0 ]]; then
  shift
fi

timeout_seconds="${CODEX_PR_REVIEW_TIMEOUT_SECONDS:-3600}"
if [[ ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "CODEX_PR_REVIEW_TIMEOUT_SECONDS must be a positive integer: $timeout_seconds" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
review_dir="$(mktemp -d "${TMPDIR:-/tmp}/regime-codex-pr-review.XXXXXX")"
prompt_dir="$(mktemp -d "${TMPDIR:-/tmp}/regime-codex-pr-prompts.XXXXXX")"
report_dir="$(mktemp -d "${TMPDIR:-/tmp}/regime-codex-pr-reports.XXXXXX")"

cleanup() {
  rm -rf "$review_dir" "$prompt_dir" "$report_dir"
}
trap cleanup EXIT

strip_frontmatter() {
  local file="$1"
  awk '
    NR == 1 && $0 == "---" { in_frontmatter = 1; next }
    in_frontmatter && $0 == "---" { in_frontmatter = 0; next }
    !in_frontmatter { print }
  ' "$file"
}

agent_file() {
  local agent="$1"
  local agent_dir="${PR_REVIEW_TOOLKIT_AGENT_DIR:-$HOME/.codex/plugins/cache/claude-plugins-official/pr-review-toolkit/local/agents}"
  local primary="${agent_dir}/${agent}.md"
  if [[ -f "$primary" ]]; then
    printf '%s\n' "$primary"
    return 0
  fi
  return 1
}

run_agent() {
  local agent="$1"
  local plugin_file="$2"
  local prompt_file="$prompt_dir/${agent}.prompt.md"
  local output_file="$report_dir/${agent}.md"

  {
    cat <<PROMPT
You are running one installed pr-review-toolkit specialist prompt through Codex.

Specialist: $agent
Repository: $repo_root
Review clone: $review_dir
Base ref: $base_ref

Adaptation rules:
- Treat AGENTS.md as the project instruction file when the plugin text says CLAUDE.md.
- Do not modify files.
- Review only the branch diff against $base_ref.
- Focus only on this specialist role.
- Report only concrete findings with file:line references where possible.
- If no relevant issues exist, say that clearly.

Installed specialist prompt follows.

PROMPT
    strip_frontmatter "$plugin_file"
  } >"$prompt_file"

  local cmd=(codex exec review --base "$base_ref" --ephemeral)
  if [[ -n "${CODEX_REVIEW_MODEL:-}" ]]; then
    cmd+=(--model "$CODEX_REVIEW_MODEL")
  fi
  cmd+=(-)

  echo "=== ${agent} ==="
  if perl -e 'alarm shift @ARGV; exec @ARGV' "$timeout_seconds" "${cmd[@]}" \
    <"$prompt_file" >"$output_file"
  then
    cat "$output_file"
    printf '\n'
    return 0
  fi

  local status=$?
  if [[ "$status" -eq 142 ]]; then
    echo "${agent} timed out after ${timeout_seconds}s; continuing with remaining specialists" >&2
    return 0
  fi
  echo "${agent} failed with exit code $status" >&2
  cat "$output_file" >&2 || true
  return "$status"
}

git clone --shared --no-checkout "$repo_root" "$review_dir" >/dev/null
git -C "$review_dir" checkout --detach -q HEAD
cd "$review_dir"

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found on PATH; skipping pr-review-toolkit" >&2
  exit 0
fi

agents=(
  code-reviewer
  silent-failure-hunter
  pr-test-analyzer
  type-design-analyzer
  comment-analyzer
  code-simplifier
)

missing=0
for agent in "${agents[@]}"; do
  if ! agent_file "$agent" >/dev/null; then
    echo "pr-review-toolkit agent prompt not found: $agent" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "one or more pr-review-toolkit prompts are missing; skipping toolkit" >&2
  exit 0
fi

failed=0
for agent in "${agents[@]}"; do
  plugin_file="$(agent_file "$agent")"
  if ! run_agent "$agent" "$plugin_file"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "codex pr-review-toolkit failed" >&2
  exit 1
fi
