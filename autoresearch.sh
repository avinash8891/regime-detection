#!/bin/bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$repo_root"

python3 -m py_compile tests/conftest.py >/dev/null 2>&1 || {
  echo "SYNTAX ERROR"
  exit 1
}

pytest_args=()
if [[ -n "${AUTORESEARCH_PYTEST_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  pytest_args=(${AUTORESEARCH_PYTEST_ARGS})
else
  pytest_args=(-m "" -q -n auto --durations=100)
fi

output="$(
  PYTEST_ADDOPTS='' /usr/bin/time -p python3 -m pytest "${pytest_args[@]}" 2>&1
)"

printf '%s\n' "$output"

real_seconds="$(printf '%s\n' "$output" | awk '/^real /{print $2; exit}')"
summary_line="$(printf '%s\n' "$output" | grep -E '([0-9]+ failed|[0-9]+ passed|[0-9]+ skipped)' | tail -1)"
passed_count="$(printf '%s\n' "$summary_line" | sed -nE 's/(^|.*[^0-9])([0-9]+) passed.*/\2/p')"
skipped_count="$(printf '%s\n' "$summary_line" | sed -nE 's/(^|.*[^0-9])([0-9]+) skipped.*/\2/p')"
failed_count="$(printf '%s\n' "$summary_line" | sed -nE 's/(^|.*[^0-9])([0-9]+) failed.*/\2/p')"
collected_count="$(printf '%s\n' "$output" | sed -nE 's/^collected ([0-9]+) items.*/\1/p' | tail -1)"
slowest_seconds="$(printf '%s\n' "$output" | awk '/^[0-9]+\.[0-9]+s /{gsub(/s/,"",$1); print $1; exit}')"

echo "METRIC wall_seconds=${real_seconds:-0}"
echo "METRIC collected_tests=${collected_count:-0}"
echo "METRIC passed_tests=${passed_count:-0}"
echo "METRIC skipped_tests=${skipped_count:-0}"
echo "METRIC failed_tests=${failed_count:-0}"
echo "METRIC slowest_test_seconds=${slowest_seconds:-0}"
