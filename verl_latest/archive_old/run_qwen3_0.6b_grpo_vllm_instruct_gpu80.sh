#!/usr/bin/env bash
# GRPO + vLLM — Qwen3-0.6B-Instruct, profile **~80GB** VRAM (A100-80GB / H100 class, single GPU colocated FSDP + vLLM + re_search).
# Same layout as ``run_qwen3_0.6b_grpo_vllm_instruct_gpu40.sh`` with **scaled** batches, vLLM pool, context, and concurrency.
# If CUDA OOM: lower ``VLLM_GPU_MEM_UTIL`` first, then ``TRAIN_BATCH_SIZE`` / ``VLLM_MAX_MODEL_LEN`` / ``ROLLOUT_N``.
# Flags: --add_qwen_chat → +data.re_search_add_qwen_chat=true (manual <|im_start|>…); --add_thinking → +data.re_search_add_thinking=true (\\n only after assistant; default leaves empty <redacted_thinking> block).
# With --add_qwen_chat, rollout JSONL ``input`` is decoded with specials kept (trainer.rollout_prompt_log_skip_special_tokens=false) so it matches vLLM; otherwise specials are stripped and ``input`` looks like plain role lines.
# Override MODEL_PATH, TRAIN_FILE, TEST_FILE. W&B: verl_latest/.env (WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT, WANDB_EXPERIMENT_NAME).
#
# --- vs ``scripts/train/train.sh`` (multi-GPU legacy recipe) ---
# | train.sh | this run (verl_latest) |
# | 4× GPU, TP=2 rollout | 1× GPU, ``tensor_model_parallel_size=1`` |
# | ``train_batch_size=256`` (split across GPUs) | ``TRAIN_BATCH_SIZE=64`` default here (80GB); optional ``GEN_BATCH_SIZE`` |
# | ``Qwen3-0.6B-Base`` + ``data.apply_chat`` | Instruct + ``+data.re_search_use_chat_format`` + ``re_search_template_sys`` |
# | ``reward_model.reward_manager`` | ``reward.reward_manager.name=re_search`` |
# | ``vllm_with_search`` (name may be absent in verl_latest) | ``rollout.name=vllm`` + ``default_agent_loop=re_search_agent`` + ``search_url`` |
# | ``ref`` ``param_offload=True`` | ``param_offload=False`` (single GPU; set True if actor OOM) |
# | ``ppo_max_token_len`` = 2×(P+R) | ``max_seq_tokens_per_gpu`` = ``VLLM_MAX_MODEL_LEN`` (default 24576 here) |
# | ``trainer.rollout_save_path`` | ``trainer.rollout_data_dir`` |
# | ``trainer.val_before_train=True`` | set ``VAL_BEFORE_TRAIN=True`` to mirror |
# To mimic **base** model + plain chat: ``run_qwen3_0.6b_grpo_vllm.sh`` + ``MODEL_PATH=...-Base``, or ``+data.re_search_use_chat_format=False`` (see base script).
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
set -xeuo pipefail

# Optional CLI: --add_qwen_chat (manual <|im_start|>system/user/assistant string, no apply_chat_template),
# --add_thinking (suffix "\\n" only after assistant; default off = empty <redacted_thinking>…</redacted_thinking> block).
RE_SEARCH_ADD_QWEN_CHAT=false
RE_SEARCH_ADD_THINKING=false
ROLLOUT_PROMPT_LOG_SKIP_SPECIAL_TOKENS=true
PASSTHROUGH=()
for _arg in "$@"; do
  case "$_arg" in
    --add_qwen_chat)
      RE_SEARCH_ADD_QWEN_CHAT=true
      ROLLOUT_PROMPT_LOG_SKIP_SPECIAL_TOKENS=false
      ;;
    --add_thinking) RE_SEARCH_ADD_THINKING=true ;;
    *) PASSTHROUGH+=("$_arg") ;;
  esac
done
set -- "${PASSTHROUGH[@]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  . "${SCRIPT_DIR}/.env"
  set +a
fi
WANDB_API_KEY=${WANDB_API_KEY:-None}
export WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-verl_grpo_example_musique}"

if REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

trainer_n_gpus_per_node=1
trainer_nnodes=1
trainer_project_name="${WANDB_PROJECT}"
trainer_experiment_name="${WANDB_EXPERIMENT_NAME:-qwen3_0.6b_instruct_grpo_gpu_1_80gb}"
SAVE_FREQ="${SAVE_FREQ:-10}"
RESUME_MODE="${RESUME_MODE:-disable}"

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${REPO_ROOT}/models/Qwen3-0.6B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${trainer_project_name}/${trainer_experiment_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${REPO_ROOT}/data/musique/train.parquet"}
TEST_FILE=${TEST_FILE:-"${REPO_ROOT}/data/musique/test.parquet"}

mkdir -p "${CKPTS_DIR}"
ROLLOUT_SAVE_PATH="${ROLLOUT_SAVE_PATH:-${CKPTS_DIR}/rollout}"
mkdir -p "${ROLLOUT_SAVE_PATH}"
if [ "${WANDB_API_KEY}" != "None" ] && [ -n "${WANDB_API_KEY}" ]; then
  wandb login --relogin "${WANDB_API_KEY}"
  export WANDB_DIR="${CKPTS_DIR}"
fi

mkdir -p "${RAY_DATA_HOME}/logs/${trainer_project_name}"
LOG_PATH="${RAY_DATA_HOME}/logs/${trainer_project_name}/${trainer_experiment_name}.log"

use_dynamic_bsz=True

# ~80GB: ~2× gpu40 batch headroom (tune down if actor OOM).
train_batch_size="${TRAIN_BATCH_SIZE:-64}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-64}"
gen_batch_size="${GEN_BATCH_SIZE:-${train_batch_size}}"

dataloader_num_workers="${DATALOADER_NUM_WORKERS:-8}"

# vLLM: larger KV pool and concurrency than gpu40; still colocated — leave margin for FSDP + ref.
vllm_gpu_mem_util="${VLLM_GPU_MEM_UTIL:-0.78}"
# Long multi-turn re_search + 8k response; 80GB allows more context than 40GB defaults.
vllm_max_model_len="${VLLM_MAX_MODEL_LEN:-24576}"
max_seq_tokens_per_gpu="${MAX_SEQ_TOKENS_PER_GPU:-${vllm_max_model_len}}"
vllm_max_num_seqs="${VLLM_MAX_NUM_SEQS:-24}"
rollout_n="${ROLLOUT_N:-5}"
rollout_max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}"
# Must divide the per-step prompt batch (typically train_batch_size × rollout_n with GRPO); else DataProto.chunk fails.
rollout_agent_num_workers="${ROLLOUT_AGENT_NUM_WORKERS:-${rollout_n}}"

use_torch_compile="${USE_TORCH_COMPILE:-False}"
gradient_checkpointing="${GRADIENT_CHECKPOINTING:-True}"
update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-1024}"

SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:3005}"
ROLLOUT_AGENT="${ROLLOUT_AGENT:-single_turn_agent}"
if [ -n "${SEARCH_URL}" ]; then
  ROLLOUT_AGENT=re_search_agent
fi
# ReSearch + Instruct: re_search_template_sys + chat format True. Hydra ``+`` for keys not in legacy_data.yaml.
# Do not put ``# ...`` comment-only lines inside the continued ``python3`` block: a ``\``-continued line followed by
# a line that starts with ``#`` ends the command.

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.prompt_key=question \
    +data.prompt_template_name=re_search_template_sys \
    +data.re_search_use_chat_format=True \
    +data.re_search_add_qwen_chat=${RE_SEARCH_ADD_QWEN_CHAT} \
    +data.re_search_add_thinking=${RE_SEARCH_ADD_THINKING} \
    data.train_batch_size=${train_batch_size} \
    +data.gen_batch_size=${gen_batch_size} \
    data.dataloader_num_workers=${dataloader_num_workers} \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    +actor_rollout_ref.model.additional_special_tokens='["<answer>","</answer>","<search>","</search>","<result>","</result>"]' \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=${use_torch_compile} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_seq_tokens_per_gpu} \
    actor_rollout_ref.model.enable_gradient_checkpointing=${gradient_checkpointing} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.agent.num_workers=${rollout_agent_num_workers} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_seq_tokens_per_gpu} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.response_length=8192 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${vllm_gpu_mem_util} \
    actor_rollout_ref.rollout.max_model_len=${vllm_max_model_len} \
    actor_rollout_ref.rollout.max_num_seqs=${vllm_max_num_seqs} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${update_weights_bucket_megabytes} \
    actor_rollout_ref.rollout.agent.default_agent_loop=${ROLLOUT_AGENT} \
    +actor_rollout_ref.rollout.search_url="${SEARCH_URL}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_seq_tokens_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    reward.reward_manager.name=re_search \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name=${trainer_project_name} \
    trainer.experiment_name=${trainer_experiment_name} \
    trainer.logger="[console, wandb]" \
    trainer.rollout_prompt_log_skip_special_tokens=${ROLLOUT_PROMPT_LOG_SKIP_SPECIAL_TOKENS} \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.n_gpus_per_node=$trainer_n_gpus_per_node \
    trainer.nnodes=$trainer_nnodes \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} 2>&1 | tee ${LOG_PATH}
