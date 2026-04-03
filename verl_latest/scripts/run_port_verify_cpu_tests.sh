#!/usr/bin/env bash
# Run CPU-side reward / parquet checks for the ReSearch port (no GPU, no Ray).
# Requires: pip install -e .  (or PYTHONPATH to verl_latest), pytest, and deps from requirements.txt.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
for py in python3 python3.11 python3.12; do
  if command -v "${py}" >/dev/null 2>&1 && "${py}" -m pytest --version >/dev/null 2>&1; then
    exec "${py}" -m pytest tests/utils/reward_score/test_default_compute_score_train_parquet_on_cpu.py -q "$@"
  fi
done
echo "pytest not found for python3 / python3.11 / python3.12. Install: pip install pytest && pip install -e ." >&2
exit 1
