#!/bin/bash
set -euo pipefail

python3 -m py_compile tests/conftest.py >/dev/null 2>&1
git diff --check
