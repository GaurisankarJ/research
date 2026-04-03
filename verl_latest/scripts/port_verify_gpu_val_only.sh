#!/usr/bin/env bash
# Optional bounded GPU check: one validation pass, no training steps (cheaper than full GRPO).
# Requires: pip install -e ., CUDA, vLLM, Ray, data + model paths.
#
# Usage:
#   cd verl_latest && ./scripts/port_verify_gpu_val_only.sh
#   SEARCH_URL=  ./scripts/port_verify_gpu_val_only.sh   # single_turn_agent (no HTTP retriever)
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if REPO_ROOT="$(git -C "${VERL_ROOT}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "${VERL_ROOT}/.." && pwd)"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

if [ -f "${VERL_ROOT}/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  . "${VERL_ROOT}/.env"
  set +a
fi

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/Qwen3-0.6B-Base}"
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/musique/train.parquet}"
TEST_FILE="${TEST_FILE:-${REPO_ROOT}/data/musique/test.parquet}"
CKPTS_DIR="${CKPTS_DIR:-${VERL_ROOT}/results/port_verify_val_only}"

SEARCH_URL="${SEARCH_URL:-}"
ROLLOUT_AGENT="${ROLLOUT_AGENT:-single_turn_agent}"
if [ -n "${SEARCH_URL}" ] && [[ ! "${SEARCH_URL}" =~ ^https?:// ]]; then
  SEARCH_URL="http://${SEARCH_URL}"
fi
if [ -n "${SEARCH_URL}" ]; then
  ROLLOUT_AGENT=re_search_agent
fi

mkdir -p "${CKPTS_DIR}"

cd "${VERL_ROOT}"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.prompt_key=question \
  data.prompt_template_name=re_search_template_sys \
  data.train_batch_size=1 \
  data.max_prompt_length=256 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=512 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=512 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.prompt_length=256 \
  actor_rollout_ref.rollout.response_length=256 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEM_UTIL:-0.16}" \
  actor_rollout_ref.rollout.max_model_len="${VLLM_MAX_MODEL_LEN:-1024}" \
  actor_rollout_ref.rollout.max_num_seqs="${VLLM_MAX_NUM_SEQS:-2}" \
  actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-1}" \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-1024}" \
  actor_rollout_ref.rollout.agent.default_agent_loop="${ROLLOUT_AGENT}" \
  +actor_rollout_ref.rollout.search_url="${SEARCH_URL}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1024 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  reward.reward_manager.name=re_search \
  trainer.critic_warmup=0 \
  trainer.project_name=port_verify \
  trainer.experiment_name=gpu_val_only \
  trainer.logger="[console]" \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=1000 \
  trainer.total_epochs=0 \
  trainer.val_before_train=True \
  trainer.val_only=True
