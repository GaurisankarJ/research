#!/usr/bin/env bash
# Start retriever under conda env r_e (or a mock), wait for GET /health, then run
# verl_latest/run_qwen3_0.6b_grpo_vllm_instruct_gpu80.sh under conda env r_t.
#
# Required: CONDA_BASE when not using MOCK_RETRIEVER=1 and MOCK_TRAIN=1 together (path with
#           etc/profile.d/conda.sh), e.g. $HOME/miniconda3
# Optional: RETRIEVER_PORT (default 3005), RETRIEVER_LOG, RETRIEVER_HEALTH_TIMEOUT_S (default 600),
#           MOCK_RETRIEVER=1, MOCK_TRAIN=1, VERBOSE=1

set -euo pipefail

if [ "${VERBOSE:-0}" = "1" ]; then
  set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

RETRIEVER_PORT="${RETRIEVER_PORT:-3005}"
RETRIEVER_LOG="${RETRIEVER_LOG:-${SLURM_TMPDIR:-/tmp}/retriever_serving.log}"
RETRIEVER_HEALTH_TIMEOUT_S="${RETRIEVER_HEALTH_TIMEOUT_S:-600}"
MOCK_RETRIEVER="${MOCK_RETRIEVER:-0}"
MOCK_TRAIN="${MOCK_TRAIN:-0}"

if [ "${MOCK_RETRIEVER}" != "1" ] || [ "${MOCK_TRAIN}" != "1" ]; then
  CONDA_BASE="${CONDA_BASE:?Set CONDA_BASE to your conda install prefix (contains etc/profile.d/conda.sh)}"
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

RETRIEVER_PID=""

cleanup_retriever() {
  if [ -n "${RETRIEVER_PID}" ] && kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
    kill "${RETRIEVER_PID}" 2>/dev/null || true
    wait "${RETRIEVER_PID}" 2>/dev/null || true
  fi
}

trap cleanup_retriever EXIT INT TERM

if [ "${MOCK_RETRIEVER}" = "1" ]; then
  python3 "${SCRIPT_DIR}/mock_retriever_health_server.py" --host 127.0.0.1 --port "${RETRIEVER_PORT}" &
  RETRIEVER_PID=$!
else
  conda activate r_e
  cd "${REPO_ROOT}/scripts/serving"
  python retriever_serving.py --config retriever_config.yaml --num_retriever 1 --port "${RETRIEVER_PORT}" \
    >>"${RETRIEVER_LOG}" 2>&1 &
  RETRIEVER_PID=$!
fi

deadline=$((SECONDS + RETRIEVER_HEALTH_TIMEOUT_S))
while [ "${SECONDS}" -lt "${deadline}" ]; do
  if curl -fsS "http://127.0.0.1:${RETRIEVER_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:${RETRIEVER_PORT}/health" >/dev/null 2>&1; then
  echo "Timed out waiting for retriever health on port ${RETRIEVER_PORT} (${RETRIEVER_HEALTH_TIMEOUT_S}s)." >&2
  exit 1
fi

export SEARCH_URL="http://127.0.0.1:${RETRIEVER_PORT}"

if [ "${MOCK_TRAIN}" = "1" ]; then
  echo "MOCK_TRAIN=1: skipping verl training script."
  exit 0
fi

conda activate r_t
exec bash "${REPO_ROOT}/verl_latest/run_qwen3_0.6b_grpo_vllm_instruct_gpu80.sh" "$@"
