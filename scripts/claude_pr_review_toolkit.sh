#!/usr/bin/env bash
set -euo pipefail

base_ref="${1:-origin/main}"
if [[ $# -gt 0 ]]; then
  shift
fi

timeout_seconds="${CLAUDE_PR_REVIEW_TIMEOUT_SECONDS:-3600}"
if [[ ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "CLAUDE_PR_REVIEW_TIMEOUT_SECONDS must be a positive integer: $timeout_seconds" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
review_dir="$(mktemp -d "${TMPDIR:-/tmp}/regime-claude-pr-review.XXXXXX")"

cleanup() {
  rm -rf "$review_dir"
}
trap cleanup EXIT

git clone --shared --no-checkout "$repo_root" "$review_dir" >/dev/null
git -C "$review_dir" checkout --detach -q HEAD
cd "$review_dir"

if ! command -v claude >/dev/null 2>&1; then
  echo "claude CLI not found on PATH; skipping pr-review-toolkit" >&2
  exit 0
fi

auth_status="$(claude auth status 2>&1 || true)"
if echo "$auth_status" | grep -q '"loggedIn": false'; then
  echo "claude not logged in; skipping pr-review-toolkit" >&2
  exit 0
fi
if echo "$auth_status" | grep -q "Not logged in"; then
  echo "claude not logged in; skipping pr-review-toolkit" >&2
  exit 0
fi

prompt="/pr-review-toolkit:review-pr all"

perl -e 'alarm shift @ARGV; exec @ARGV' \
  "$timeout_seconds" \
  claude -p --bare --output-format text --permission-mode dontAsk --tools "Bash,Read,Glob,Grep,Task" -- "$prompt"
