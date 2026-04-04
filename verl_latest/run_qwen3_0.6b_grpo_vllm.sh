#!/usr/bin/env bash
# GRPO with vLLM on a single CUDA GPU (A100 or L4). Override paths with MODEL_PATH, TRAIN_FILE, TEST_FILE.
# Logging: W&B (console + wandb). verl_latest/.env: WANDB_API_KEY, WANDB_ENTITY (team), WANDB_PROJECT (dashboard project),
# optional WANDB_EXPERIMENT_NAME (run name). Or export those vars before launch.
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load verl_latest/.env when present (see header for W&B variable names).
if [ -f "${SCRIPT_DIR}/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  . "${SCRIPT_DIR}/.env"
  set +a
fi
WANDB_API_KEY=${WANDB_API_KEY:-None}
# W&B team + project (verl Tracking uses trainer.project_name → wandb.init(project=...); entity from env).
export WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-verl_grpo_example_musique}"

# Repo checkout root (models/, data/musique/ — same layout as scripts/train/train.sh)
if REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# Single GPU (set e.g. CUDA_VISIBLE_DEVICES=0 before launch if multiple GPUs are visible)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Colocated FSDP actor + ref + vLLM on one GPU: reduce allocator fragmentation during weight sync
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# vLLM v1 engine (optional; omit if your install does not support it)
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

# 1 GPU per node
trainer_n_gpus_per_node=1
trainer_nnodes=1
# Hydra trainer.project_name / experiment_name → wandb.init(project=, name=). Project defaults from WANDB_PROJECT above.
trainer_project_name="${WANDB_PROJECT}"
trainer_experiment_name="${WANDB_EXPERIMENT_NAME:-qwen3_0.6b_base_grpo_1gpu}"
# Checkpoints: ray_trainer only saves when save_freq > 0. -1 disables checkpoint writes (smoke tests).
# Metrics (console + wandb) log every training step regardless. Override e.g. SAVE_FREQ=1 (every step) or -1 (no ckpt).
SAVE_FREQ=-1
# SAVE_FREQ="${SAVE_FREQ:-10}"
RESUME_MODE="${RESUME_MODE:-disable}"

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${REPO_ROOT}/models/Qwen3-0.6B-Base"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${trainer_project_name}/${trainer_experiment_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${REPO_ROOT}/data/musique/train.parquet"}
TEST_FILE=${TEST_FILE:-"${REPO_ROOT}/data/musique/test.parquet"}

mkdir -p "${CKPTS_DIR}"
# Per-step JSONL: prompt, full response (incl. <search>/<result>), raw chat messages, num_turns, num_search_calls.
ROLLOUT_SAVE_PATH="${ROLLOUT_SAVE_PATH:-${CKPTS_DIR}/rollout}"
mkdir -p "${ROLLOUT_SAVE_PATH}"
if [ "${WANDB_API_KEY}" != "None" ] && [ -n "${WANDB_API_KEY}" ]; then
  wandb login --relogin "${WANDB_API_KEY}"
  # Run files / local wandb cache colocated with checkpoints (entity from WANDB_ENTITY in .env if set).
  export WANDB_DIR="${CKPTS_DIR}"
fi

mkdir -p "${RAY_DATA_HOME}/logs/${trainer_project_name}"
LOG_PATH="${RAY_DATA_HOME}/logs/${trainer_project_name}/${trainer_experiment_name}.log"

use_dynamic_bsz=True

# Tuned for one GPU; increase train_batch_size / ppo_mini_batch_size on larger GPUs if memory allows
# train_batch_size=64
# ppo_mini_batch_size=32
train_batch_size=1
ppo_mini_batch_size=1

# vLLM memory (22GB L4 / 40GB A100 colocate): weights are tiny but vLLM reserves
# gpu_memory_utilization * total VRAM for KV/batch; defaults + long max_model_len blow the budget.
#
# L4 smoke-test defaults: aggressively reduce KV/microbatch concurrency.
vllm_gpu_mem_util="${VLLM_GPU_MEM_UTIL:-0.16}"
vllm_max_model_len="${VLLM_MAX_MODEL_LEN:-1024}"
vllm_max_num_seqs="${VLLM_MAX_NUM_SEQS:-2}"
rollout_n="${ROLLOUT_N:-1}"
# Must be ≥ largest single weight tensor. Qwen3 embed_tokens is ~622MB float32 (151936×1024).
# Below that you get AssertionError in bucketed_weight_transfer. Override if you change model.
update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-1024}"

# Live retrieval (legacy vLLMRolloutWithSearch): set SEARCH_URL to HTTP retriever base (POST {url}/batch_search).
# When SEARCH_URL is non-empty, switch to agent loop re_search_agent; otherwise single_turn_agent (no HTTP search).
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:3005}"
ROLLOUT_AGENT="${ROLLOUT_AGENT:-single_turn_agent}"
if [ -n "${SEARCH_URL}" ]; then
  ROLLOUT_AGENT=re_search_agent
fi
# ReSearch: prompt_template_name / re_search_use_chat_format are read by RLHFDataset but are not
# declared in trainer/config/data/legacy_data.yaml; Hydra struct mode requires + to append keys.
# Base checkpoint (Qwen3-*-Base): re_search_use_chat_format=False → dataset builds one formatted string;
# rollout uses tokenizer.encode(text) only (no apply_chat_template), so vLLM does not see extra
# "user"/"assistant" chat wrappers. JSONL "raw_prompt" still stores [{"role":"user","content":...}] for
# the same string; "input" decode matches the tokenized prefix. Optional: +data.re_search_plain_add_special_tokens=True
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.prompt_key=question \
    +data.prompt_template_name=re_search_template \
    +data.re_search_use_chat_format=False \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=512 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=512 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.prompt_length=256 \
    actor_rollout_ref.rollout.response_length=256 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${vllm_gpu_mem_util} \
    actor_rollout_ref.rollout.max_model_len=${vllm_max_model_len} \
    actor_rollout_ref.rollout.max_num_seqs=${vllm_max_num_seqs} \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${update_weights_bucket_megabytes} \
    actor_rollout_ref.rollout.agent.default_agent_loop=${ROLLOUT_AGENT} \
    +actor_rollout_ref.rollout.search_url="${SEARCH_URL}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1024 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    reward.reward_manager.name=re_search \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name=${trainer_project_name} \
    trainer.experiment_name=${trainer_experiment_name} \
    trainer.logger="[console, wandb]" \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.n_gpus_per_node=$trainer_n_gpus_per_node \
    trainer.nnodes=$trainer_nnodes \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=1000 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False 2>&1 | tee ${LOG_PATH}
