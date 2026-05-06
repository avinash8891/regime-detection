#!/usr/bin/env bash
set -euo pipefail

base_branch="${1:-main}"
if [[ $# -gt 0 ]]; then
  shift
fi

# Accept both `main` and `origin/main`-style inputs (common in hooks / CI).
if [[ "$base_branch" == origin/* ]]; then
  base_branch="${base_branch#origin/}"
fi

# Prevent indefinite pre-push hangs if cubic blocks on network/service calls.
# Default to 60 minutes so large diffs have enough time to complete review
# without requiring a manual override.
timeout_seconds="${CUBIC_REVIEW_TIMEOUT_SECONDS:-3600}"
if [[ ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "CUBIC_REVIEW_TIMEOUT_SECONDS must be a positive integer: $timeout_seconds" >&2
  exit 2
fi

fetch_timeout_seconds="${CUBIC_REVIEW_FETCH_TIMEOUT_SECONDS:-30}"
if [[ ! "$fetch_timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "CUBIC_REVIEW_FETCH_TIMEOUT_SECONDS must be a positive integer: $fetch_timeout_seconds" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
review_dir="$(mktemp -d "${TMPDIR:-/tmp}/regime-cubic-review.XXXXXX")"
head_sha="$(git -C "$repo_root" rev-parse HEAD)"
tmp_xdg_data_home=""

cleanup() {
  rm -rf "$review_dir"
  if [[ -n "${tmp_xdg_data_home}" ]]; then
    rm -rf "${tmp_xdg_data_home}" || true
  fi
}
trap cleanup EXIT

# Resolve base SHA in the original working tree (so any fetch uses the real upstream remote).
base_ref="$base_branch"
if git -C "$repo_root" show-ref --verify --quiet "refs/heads/${base_branch}"; then
  base_ref="$base_branch"
elif git -C "$repo_root" show-ref --verify --quiet "refs/remotes/origin/${base_branch}"; then
  base_ref="origin/${base_branch}"
else
  # Common on fresh clones: the base branch exists on the remote but hasn't been fetched yet.
  if perl -e 'alarm shift @ARGV; exec @ARGV' \
    "$fetch_timeout_seconds" \
    git -C "$repo_root" fetch -q origin "${base_branch}:refs/remotes/origin/${base_branch}"
  then
    base_ref="origin/${base_branch}"
  else
    echo "Base branch not found locally: ${base_branch} (neither local nor origin/*), and fetch failed or timed out; skipping cubic review (non-blocking)." >&2
    exit 0
  fi
fi
base_sha="$(git -C "$repo_root" rev-parse "$base_ref")"

# cubic stores local session state on disk under:
#   $XDG_DATA_HOME/cubic/storage/...
# If $XDG_DATA_HOME isn't set, it defaults to $HOME/.local/share (which may be read-only
# in sandboxed environments like Codex). Prefer the standard location, but fall back to a
# repo-local or /tmp dir so pre-push review still works.
xdg_data_home_candidates=(
  "${XDG_DATA_HOME:-$HOME/.local/share}"
  "__TMP__"
)

selected_xdg_data_home=""
for candidate in "${xdg_data_home_candidates[@]}"; do
  if [[ "$candidate" == "__TMP__" ]]; then
    tmp_xdg_data_home="$(mktemp -d "${TMPDIR:-/tmp}/cubic-xdg-data.XXXXXX")"
    candidate="$tmp_xdg_data_home"
  fi
  candidate_storage_project_dir="$candidate/cubic/storage/project"
  if mkdir -p "$candidate_storage_project_dir" >/dev/null 2>&1; then
    if (umask 077 && : >"$candidate_storage_project_dir/.prepush_write_test") 2>/dev/null; then
      rm -f "$candidate_storage_project_dir/.prepush_write_test" >/dev/null 2>&1 || true
      selected_xdg_data_home="$candidate"
      break
    fi
  fi
done

if [[ -z "$selected_xdg_data_home" ]]; then
  echo "cubic storage dir is not writable (tried XDG_DATA_HOME candidates: ${xdg_data_home_candidates[*]}); skipping cubic review" >&2
  exit 0
fi

git clone --shared --no-checkout "$repo_root" "$review_dir" >/dev/null
git -C "$review_dir" checkout --detach -q "$head_sha"
cd "$review_dir"
set +e
review_output="$(
  perl -e 'alarm shift @ARGV; exec @ARGV' \
    "$timeout_seconds" \
    env XDG_DATA_HOME="$selected_xdg_data_home" PATH="$HOME/.superset/bin:$HOME/.cubic/bin:$PATH" cubic review --print-logs --base "$base_sha" "$@" \
    2>&1
)"
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
  exit 0
fi

# If cubic can't run locally (auth/provider/network/sandbox), don't block the push.
# The repo still gets cloud PR review, which is the primary enforcement mechanism.
if printf '%s' "$review_output" | grep -Eq \
  'No AI provider is configured|No CLI auth found|Token refresh network error|Unable to connect|EPERM: operation not permitted|failed to get diff'
then
  echo "$review_output" >&2
  echo "cubic review could not run locally; skipping (non-blocking)" >&2
  exit 0
fi

echo "$review_output" >&2
if [[ "$status" -eq 142 ]]; then
  echo "cubic review timed out after ${timeout_seconds}s" >&2
fi
exit "$status"
