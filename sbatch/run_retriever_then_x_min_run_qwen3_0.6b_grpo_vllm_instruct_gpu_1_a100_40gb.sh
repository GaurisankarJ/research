#!/usr/bin/env bash
# Start the retriever in r_e, wait for /health, then launch the 40GB trainer in r_t.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVING_DIR="${REPO_ROOT}/scripts/serving"
TRAINING_SCRIPT="${REPO_ROOT}/verl_latest/x_min_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb.sh"
TRAINING_LABEL="$(basename "${TRAINING_SCRIPT}" .sh)"

RETRIEVER_PORT="${RETRIEVER_PORT:-3005}"
RETRIEVER_CONFIG="${RETRIEVER_CONFIG:-retriever_config.yaml}"
RETRIEVER_HEALTH_TIMEOUT_S="${RETRIEVER_HEALTH_TIMEOUT_S:-1800}"
RETRIEVER_LOG="${RETRIEVER_LOG:-${SLURM_TMPDIR:-/tmp}/${TRAINING_LABEL}_retriever.log}"
MOCK_RETRIEVER="${MOCK_RETRIEVER:-0}"
MOCK_TRAIN="${MOCK_TRAIN:-0}"
MOCK_SLURM_STEP="${MOCK_SLURM_STEP:-0}"
RUN_TRAIN_DIRECT="${RUN_TRAIN_DIRECT:-0}"
SKIP_ALICE_BOOTSTRAP="${SKIP_ALICE_BOOTSTRAP:-0}"
MOCK_TRAIN_MARKER="${MOCK_TRAIN_MARKER:-}"

RETRIEVER_PID=""
TRAIN_STEP_SCRIPT=""

alice_bootstrap() {
  if [ "${SKIP_ALICE_BOOTSTRAP}" = "1" ]; then
    return
  fi

  module purge
  module load ALICE/default
  module load Miniconda3/24.7.1-0
  module load CUDA/12.4.0
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
}

activate_env() {
  local env_name="$1"
  local env_prefix="${HOME}/.conda/envs/${env_name}"
  if [ -d "${env_prefix}" ]; then
    conda activate "${env_prefix}"
  else
    conda activate "${env_name}"
  fi
  hash -r
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  # These ALICE wrappers target CUDA jobs; ROCm device vars can confuse Ray/verl worker setup.
  unset ROCR_VISIBLE_DEVICES
  unset HIP_VISIBLE_DEVICES
}

cleanup() {
  if [ -n "${RETRIEVER_PID}" ] && kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
    kill "${RETRIEVER_PID}" 2>/dev/null || true
    wait "${RETRIEVER_PID}" 2>/dev/null || true
  fi

  if [ -n "${TRAIN_STEP_SCRIPT}" ] && [ -f "${TRAIN_STEP_SCRIPT}" ]; then
    rm -f "${TRAIN_STEP_SCRIPT}"
  fi
}

trap cleanup EXIT INT TERM

start_retriever() {
  if [ "${MOCK_RETRIEVER}" = "1" ]; then
    python3 "${SCRIPT_DIR}/mock_retriever_health_server.py" --host 127.0.0.1 --port "${RETRIEVER_PORT}" &
    RETRIEVER_PID=$!
    return
  fi

  alice_bootstrap
  cd "${SERVING_DIR}"
  activate_env r_e
  PYTHONUNBUFFERED=1 python retriever_serving.py --config "${RETRIEVER_CONFIG}" --num_retriever 1 --port "${RETRIEVER_PORT}" \
    >>"${RETRIEVER_LOG}" 2>&1 &
  RETRIEVER_PID=$!
}

wait_for_retriever_health() {
  local deadline
  deadline=$((SECONDS + RETRIEVER_HEALTH_TIMEOUT_S))

  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if curl -fsS -X GET "http://127.0.0.1:${RETRIEVER_PORT}/health" >/dev/null 2>&1; then
      return 0
    fi

    if [ -n "${RETRIEVER_PID}" ] && ! kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
      echo "Retriever exited before becoming healthy." >&2
      if [ "${MOCK_RETRIEVER}" != "1" ]; then
        echo "Retriever log: ${RETRIEVER_LOG}" >&2
        if [ -f "${RETRIEVER_LOG}" ]; then
          sed -n '1,120p' "${RETRIEVER_LOG}" >&2 || true
        fi
      fi
      return 1
    fi

    sleep 1
  done

  echo "Timed out waiting for retriever health on port ${RETRIEVER_PORT} (${RETRIEVER_HEALTH_TIMEOUT_S}s)." >&2
  if [ "${MOCK_RETRIEVER}" != "1" ]; then
    echo "Retriever log: ${RETRIEVER_LOG}" >&2
  fi
  return 1
}

create_train_step_script() {
  TRAIN_STEP_SCRIPT="$(mktemp "${SLURM_TMPDIR:-/tmp}/${TRAINING_LABEL}.XXXXXX.sh")"

  cat >"${TRAIN_STEP_SCRIPT}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

alice_bootstrap() {
  if [ "${SKIP_ALICE_BOOTSTRAP:-0}" = "1" ]; then
    return
  fi

  module purge
  module load ALICE/default
  module load Miniconda3/24.7.1-0
  module load CUDA/12.4.0
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
}

activate_env() {
  local env_name="$1"
  local env_prefix="${HOME}/.conda/envs/${env_name}"
  if [ -d "${env_prefix}" ]; then
    conda activate "${env_prefix}"
  else
    conda activate "${env_name}"
  fi
  hash -r
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  # These ALICE wrappers target CUDA jobs; ROCm device vars can confuse Ray/verl worker setup.
  unset ROCR_VISIBLE_DEVICES
  unset HIP_VISIBLE_DEVICES
}

alice_bootstrap
export SEARCH_URL="${SEARCH_URL:?SEARCH_URL must be set before launching training}"

if [ "${MOCK_TRAIN:-0}" = "1" ]; then
  if [ -n "${MOCK_TRAIN_MARKER:-}" ]; then
    printf '%s\n' "${TRAINING_SCRIPT}" >"${MOCK_TRAIN_MARKER}"
  fi
  printf 'MOCK_TRAIN=1: would run %s\n' "${TRAINING_SCRIPT}"
  exit 0
fi

if [ "${PRINT_EFFECTIVE_CONFIG_ONLY:-0}" = "1" ]; then
  cd "${REPO_ROOT}/verl_latest"
  exec bash "${TRAINING_SCRIPT}" "$@"
fi

activate_env r_t
cd "${REPO_ROOT}/verl_latest"
exec bash "${TRAINING_SCRIPT}" "$@"
EOF

  chmod +x "${TRAIN_STEP_SCRIPT}"
}

launch_training_step() {
  export REPO_ROOT
  export SEARCH_URL="http://127.0.0.1:${RETRIEVER_PORT}"
  export SKIP_ALICE_BOOTSTRAP
  export MOCK_TRAIN
  export MOCK_TRAIN_MARKER
  export TRAINING_SCRIPT
  export PRINT_EFFECTIVE_CONFIG_ONLY
  export EFFECTIVE_CONFIG_PATH

  create_train_step_script

  if [ "${MOCK_SLURM_STEP}" = "1" ] || [ "${RUN_TRAIN_DIRECT}" = "1" ]; then
    bash "${TRAIN_STEP_SCRIPT}" "$@"
    return
  fi

  srun --overlap --ntasks=1 bash "${TRAIN_STEP_SCRIPT}" "$@"
}

start_retriever
wait_for_retriever_health
launch_training_step "$@"
