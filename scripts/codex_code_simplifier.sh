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
output_file="$(mktemp "${TMPDIR:-/tmp}/regime-codex-simplifier-output.XXXXXX")"

cleanup() {
  rm -rf "$review_dir"
  rm -f "$output_file"
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
  echo "codex CLI not found on PATH; skipping code simplifier" >&2
  exit 0
fi

cmd=(codex exec review --base "$base_sha" --ephemeral)
if [[ -n "${CODEX_REVIEW_MODEL:-}" ]]; then
  cmd+=(--model "$CODEX_REVIEW_MODEL")
fi
# NOTE: codex CLI currently does not allow `--base` together with a custom prompt.
# For this pre-push gate we use the default review prompt (still base-diff aware).
if perl -e 'alarm shift @ARGV; exec @ARGV' "$timeout_seconds" "${cmd[@]}" \
  >"$output_file"
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
