#!/usr/bin/env bash
# Validate the sbatch wrappers without starting the real retriever or trainer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNNERS=(
  "${SCRIPT_DIR}/run_retriever_then_z_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_80gb.sh"
  "${SCRIPT_DIR}/run_retriever_then_x_min_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb.sh"
)

SBATCH_FILES=(
  "${SCRIPT_DIR}/retriever_then_z_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_80gb.sbatch"
  "${SCRIPT_DIR}/retriever_then_x_min_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb.sbatch"
)

free_port() {
  python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

assert_file_contains() {
  local path="$1"
  local expected="$2"
  local content

  content="$(<"${path}")"
  case "${content}" in
    *"${expected}"*) ;;
    *)
      echo "Expected ${path} to contain ${expected}" >&2
      exit 1
      ;;
  esac
}

for file in "${RUNNERS[@]}" "${SBATCH_FILES[@]}" "${SCRIPT_DIR}/test_retriever_then_training_mock.sh"; do
  bash -n "${file}"
done

python3 -m py_compile "${SCRIPT_DIR}/mock_retriever_health_server.py"

for runner in "${RUNNERS[@]}"; do
  port="$(free_port)"
  marker="$(mktemp "${TMPDIR:-/tmp}/$(basename "${runner}" .sh).marker.XXXXXX")"
  retriever_log="$(mktemp "${TMPDIR:-/tmp}/$(basename "${runner}" .sh).retriever.XXXXXX.log")"
  expected_training_script=""

  case "${runner}" in
    *run_retriever_then_z_run*)
      expected_training_script="z_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_80gb.sh"
      ;;
    *run_retriever_then_x_min_run*)
      expected_training_script="x_min_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb.sh"
      ;;
    *)
      echo "Unexpected runner ${runner}" >&2
      exit 1
      ;;
  esac

  (
    export MOCK_RETRIEVER=1
    export MOCK_TRAIN=1
    export MOCK_SLURM_STEP=1
    export SKIP_ALICE_BOOTSTRAP=1
    export RETRIEVER_PORT="${port}"
    export RETRIEVER_HEALTH_TIMEOUT_S=30
    export MOCK_TRAIN_MARKER="${marker}"
    export RETRIEVER_LOG="${retriever_log}"

    bash "${runner}"
  )

  assert_file_contains "${marker}" "${expected_training_script}"

  if curl -fsS -X GET "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "Mock retriever on port ${port} was not cleaned up." >&2
    exit 1
  fi

  rm -f "${marker}" "${retriever_log}"
done

if command -v shellcheck >/dev/null 2>&1; then
  shellcheck "${RUNNERS[@]}" "${SBATCH_FILES[@]}" "${SCRIPT_DIR}/test_retriever_then_training_mock.sh"
fi

if command -v sbatch >/dev/null 2>&1; then
  for sbatch_file in "${SBATCH_FILES[@]}"; do
    if sbatch_output="$(sbatch --test-only "${sbatch_file}" 2>&1)"; then
      printf '%s\n' "${sbatch_output}"
      continue
    fi

    case "${sbatch_output}" in
      *"Unable to contact slurm controller"*|*"connect failure"*)
        printf 'Skipping sbatch --test-only for %s: %s\n' "${sbatch_file}" "${sbatch_output}" >&2
        ;;
      *)
        printf '%s\n' "${sbatch_output}" >&2
        exit 1
        ;;
    esac
  done
fi

echo "OK: mock retriever + mock training validation completed for both sbatch flows."
