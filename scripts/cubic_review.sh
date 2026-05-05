#!/usr/bin/env bash
set -euo pipefail

base_ref="${1:-origin/main}"
if [[ $# -gt 0 ]]; then
  shift
fi

# Prevent indefinite pre-push hangs if cubic blocks on network/service calls.
# Default to 60 minutes so large diffs have enough time to complete review
# without requiring a manual override.
timeout_seconds="${CUBIC_REVIEW_TIMEOUT_SECONDS:-3600}"
if [[ ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "CUBIC_REVIEW_TIMEOUT_SECONDS must be a positive integer: $timeout_seconds" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
review_dir="$(mktemp -d "${TMPDIR:-/tmp}/regime-cubic-review.XXXXXX")"

cleanup() {
  rm -rf "$review_dir"
}
trap cleanup EXIT

storage_dir="$HOME/.local/share/cubic/storage"
storage_project_dir="$storage_dir/project"
if ! mkdir -p "$storage_project_dir" >/dev/null 2>&1; then
  echo "cubic storage dir is not writable ($storage_project_dir); skipping cubic review" >&2
  exit 0
fi
if ! (umask 077 && : >"$storage_project_dir/.prepush_write_test") 2>/dev/null; then
  echo "cubic storage dir is not writable ($storage_project_dir); skipping cubic review" >&2
  exit 0
fi
rm -f "$storage_project_dir/.prepush_write_test" >/dev/null 2>&1 || true

git clone --shared --no-checkout "$repo_root" "$review_dir" >/dev/null
git -C "$review_dir" checkout --detach -q HEAD
cd "$review_dir"

# Ensure the base ref exists in this review clone (worktrees sometimes lack remote refs in shared clones).
if [[ "$base_ref" == origin/* ]]; then
  base_branch="${base_ref#origin/}"
  if ! git show-ref --verify --quiet "refs/remotes/origin/${base_branch}"; then
    git fetch -q origin "$base_branch":"refs/remotes/origin/${base_branch}" || git fetch -q origin "$base_branch" || true
  fi
fi

if perl -e 'alarm shift @ARGV; exec @ARGV' \
  "$timeout_seconds" \
  env PATH="$HOME/.superset/bin:$HOME/.cubic/bin:$PATH" cubic review --print-logs --base "$base_ref" "$@"
then
  :
else
  status=$?
  if [[ "$status" -eq 142 ]]; then
    echo "cubic review timed out after ${timeout_seconds}s" >&2
  fi
  exit "$status"
fi
