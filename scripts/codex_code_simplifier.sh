#!/usr/bin/env bash
set -euo pipefail

base_ref="${1:-origin/main}"
if [[ $# -gt 0 ]]; then
  shift
fi

timeout_seconds="${CODEX_CODE_SIMPLIFIER_TIMEOUT_SECONDS:-3600}"
if [[ ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "CODEX_CODE_SIMPLIFIER_TIMEOUT_SECONDS must be a positive integer: $timeout_seconds" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
review_dir="$(mktemp -d "${TMPDIR:-/tmp}/regime-codex-simplifier.XXXXXX")"
prompt_file="$(mktemp "${TMPDIR:-/tmp}/regime-codex-simplifier-prompt.XXXXXX")"
output_file="$(mktemp "${TMPDIR:-/tmp}/regime-codex-simplifier-output.XXXXXX")"

cleanup() {
  rm -rf "$review_dir"
  rm -f "$prompt_file" "$output_file"
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

git clone --shared --no-checkout "$repo_root" "$review_dir" >/dev/null
git -C "$review_dir" checkout --detach -q HEAD
cd "$review_dir"

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found on PATH; skipping code simplifier" >&2
  exit 0
fi

plugin_file="${CODEX_CODE_SIMPLIFIER_PROMPT_FILE:-$HOME/.codex/plugins/cache/claude-plugins-official/code-simplifier/1.0.0/agents/code-simplifier.md}"
if [[ ! -f "$plugin_file" ]]; then
  echo "official code-simplifier plugin prompt not found: $plugin_file" >&2
  exit 0
fi

{
  cat <<PROMPT
You are running the installed code-simplifier plugin prompt through Codex.

Repository: $repo_root
Review clone: $review_dir
Base ref: $base_ref

Adaptation rules:
- Treat AGENTS.md as the project instruction file when the plugin text says CLAUDE.md.
- Do not modify files.
- Review only the branch diff against $base_ref.
- Output concise, actionable simplification findings with file:line references where possible.
- If no worthwhile simplifications exist, say that clearly.

Installed plugin prompt follows.

PROMPT
  strip_frontmatter "$plugin_file"
} >"$prompt_file"

cmd=(codex exec review --base "$base_ref" --ephemeral -)
if [[ -n "${CODEX_REVIEW_MODEL:-}" ]]; then
  cmd+=(--model "$CODEX_REVIEW_MODEL")
fi

# IMPORTANT: codex exec review cannot accept a custom [PROMPT] when --base is used.
# We pass instructions via stdin (Codex treats piped stdin as additional context).
if perl -e 'alarm shift @ARGV; exec @ARGV' "$timeout_seconds" "${cmd[@]}" \
  <"$prompt_file" >"$output_file"
then
  cat "$output_file"
else
  status=$?
  if [[ "$status" -eq 142 ]]; then
    echo "codex code simplifier timed out after ${timeout_seconds}s; skipping" >&2
    exit 0
  fi
  echo "codex code simplifier failed with exit code $status" >&2
  cat "$output_file" >&2 || true
  exit "$status"
fi
