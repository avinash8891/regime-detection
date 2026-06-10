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

run_with_timeout() {
  if command -v gtimeout >/dev/null 2>&1; then
    gtimeout -k 5 "$timeout_seconds" "$@"
  elif command -v timeout >/dev/null 2>&1; then
    timeout -k 5 "$timeout_seconds" "$@"
  else
    perl -e 'alarm shift @ARGV; exec @ARGV' "$timeout_seconds" "$@"
  fi
}

is_timeout_status() {
  [[ "$1" -eq 124 || "$1" -eq 137 || "$1" -eq 142 ]]
}

repo_root="$(git rev-parse --show-toplevel)"
review_dir="$(mktemp -d "${TMPDIR:-/tmp}/regime-codex-pr-review.XXXXXX")"
report_dir="$(mktemp -d "${TMPDIR:-/tmp}/regime-codex-pr-reports.XXXXXX")"

cleanup() {
  rm -rf "$review_dir" "$report_dir"
}
trap cleanup EXIT

git clone --shared --no-checkout "$repo_root" "$review_dir" >/dev/null
git -C "$review_dir" checkout --detach -q HEAD
cd "$review_dir"

base_sha="$base_ref"
if git rev-parse -q --verify "$base_ref" >/dev/null 2>&1; then
  base_sha="$(git rev-parse "$base_ref")"
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found on PATH; skipping pr-review-toolkit" >&2
  exit 0
fi

output_file="$report_dir/default.md"
cmd=(codex exec review --base "$base_sha" --ephemeral)
if [[ -n "${CODEX_REVIEW_MODEL:-}" ]]; then
  cmd+=(--model "$CODEX_REVIEW_MODEL")
fi
if run_with_timeout "${cmd[@]}" >"$output_file"; then
  cat "$output_file"
  exit 0
else
  status=$?
fi

if is_timeout_status "$status"; then
  echo "codex pr-review-toolkit timed out after ${timeout_seconds}s; skipping" >&2
  exit 0
fi
if rg -q "failed to get diff|exitCode=128|fatal: working tree '.+' already exists" "$output_file"; then
  echo "codex pr-review-toolkit could not compute diff in this worktree snapshot; skipping" >&2
  exit 0
fi
echo "codex pr-review-toolkit failed with exit code $status" >&2
cat "$output_file" >&2 || true
exit "$status"
