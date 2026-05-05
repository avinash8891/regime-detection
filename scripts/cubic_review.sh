#!/usr/bin/env bash
set -euo pipefail

base_branch="${1:-main}"
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
cd "$repo_root"

base_ref="$base_branch"
if git show-ref --verify --quiet "refs/heads/${base_branch}"; then
  base_ref="$base_branch"
elif git show-ref --verify --quiet "refs/remotes/origin/${base_branch}"; then
  base_ref="origin/${base_branch}"
else
  # Common on fresh clones: the base branch exists on the remote but hasn't been fetched yet.
  if git fetch -q origin "${base_branch}:refs/remotes/origin/${base_branch}"; then
    base_ref="origin/${base_branch}"
  else
    echo "Base branch not found locally: ${base_branch} (neither local nor origin/*), and fetch failed." >&2
    exit 2
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
