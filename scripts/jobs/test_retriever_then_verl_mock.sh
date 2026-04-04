#!/usr/bin/env bash
# Smoke test: full pipeline with MOCK_RETRIEVER=1 MOCK_TRAIN=1 (no FlashRAG, no verl, no conda).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

bash -n "${SCRIPT_DIR}/run_retriever_then_verl_gpu80.sh"
bash -n "${SCRIPT_DIR}/test_retriever_then_verl_mock.sh"
bash -n "${SCRIPT_DIR}/sbatch_retriever_then_verl_gpu80.sbatch"

RETRIEVER_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"

export MOCK_RETRIEVER=1
export MOCK_TRAIN=1
export RETRIEVER_PORT
export RETRIEVER_HEALTH_TIMEOUT_S=30
unset CONDA_BASE

bash "${SCRIPT_DIR}/run_retriever_then_verl_gpu80.sh"
echo "OK: mock retriever + mock train pipeline exited 0 (repo ${REPO_ROOT})."
