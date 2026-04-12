# Review report: legacy `verl` (`src/verl_legacy`) vs `verl_latest` (scripts + code)

**Scope:** [scripts/train/train.sh](../../scripts/train/train.sh) vs [verl_latest/run_qwen3_0.6b_grpo_vllm_instruct_gpu40.sh](../../verl_latest/run_qwen3_0.6b_grpo_vllm_instruct_gpu40.sh), plus source-level mapping for the ReSearch path (GRPO + HTTP search + rule-based `re_search` reward).

**Date:** 2026-04-03

---

## Executive summary

- **Scripts:** Both targets implement **GRPO** on MuSiQue-style parquet with **`re_search_template_sys`**, **KL loss** (0.001), **`rollout.n=5`**, and **live retrieval** against `{search_url}/batch_search`. The gpu40 script is explicitly a **single-GPU, memory-constrained** port: smaller batches, lower unified token caps, different validation defaults, and different default **checkpoint** (Base vs Instruct) than `train.sh`.
- **Code:** The **same behavioral pieces** exist but are **split differently**: legacy folds multi-turn search into [`vLLMRolloutWithSearch`](../../src/verl_legacy/workers/rollout/vllm_rollout/vllm_rollout.py); latest uses **vLLM** + [`ReSearchAgentLoop`](../../verl_latest/verl/experimental/agent_loop/re_search_agent_loop.py). Rewards moved from inline `reward_model.reward_manager` to **`RewardLoopManager`** + `reward.reward_manager.name=re_search`.
- **Non-parity risk:** [`verl_latest/verl/utils/reward_score/re_search.py`](../../verl_latest/verl/utils/reward_score/re_search.py) **`compute_score`** is **not** identical to [`src/verl_legacy/utils/reward_score/re_search.py`](../../src/verl_legacy/utils/reward_score/re_search.py): legacy **requires** an EOS suffix after the response span or returns score `0` (`"over length"`); latest allows truncated responses to score and normalizes ground truth differently. **Treat this as a potential training distribution shift**, not just a Qwen2.5 vs Qwen3 artifact.

---

## §1 — Script and Hydra comparison

| Topic | `scripts/train/train.sh` | `run_qwen3_0.6b_grpo_vllm_instruct_gpu40.sh` |
|--------|---------------------------|-----------------------------------------------|
| **Package** | `python3 -m verl.trainer.main_ppo` (expects **`verl`** on path; this repo vendors [`src/verl_legacy`](../../src/verl_legacy) as `verl` when used that way) | Same entrypoint against [`verl_latest`](../../verl_latest) install |
| **Algorithm** | `algorithm.adv_estimator=grpo`, `algorithm.kl_ctrl.kl_coef=0.001` | Same |
| **Data paths** | `TRAIN_FILES` / `TEST_FILES` relative (`../../data/musique/...`) | `TRAIN_FILE` / `TEST_FILE` from `REPO_ROOT` (absolute after resolve) |
| **Prompt** | `data.prompt_key`, `data.apply_chat`, `data.prompt_template_name=re_search_template_sys` | `data.prompt_key=question`, `+data.prompt_template_name=re_search_template_sys`, `+data.re_search_use_chat_format=True` (equivalent chat intent) |
| **Extra data flags** | — | `data.filter_overlong_prompts=True`, `data.truncation=error`, `+data.gen_batch_size`, optional `re_search_add_qwen_chat` / `re_search_add_thinking` |
| **Batch** | `train_batch_size=256`, `ppo_mini_batch_size=256` | Default `16` / `16`, optional `GEN_BATCH_SIZE` |
| **GPUs / TP** | `N_GPUS_PER_NODE=4`, `rollout.tensor_model_parallel_size=2` | `1` GPU, `tensor_model_parallel_size=1` |
| **Token caps** | `ppo_max_token_len_per_gpu=2*(P+R)` → **17408**; `log_prob_max_token_len_per_gpu=4*(P+R)` → **34816** | `max_seq_tokens_per_gpu` tied to `VLLM_MAX_MODEL_LEN` default **8192** for actor, rollout log-prob, ref |
| **Rollout** | `rollout.name=vllm_with_search`, `rollout.n=5`, `gpu_memory_utilization=0.6` | `rollout.name=vllm`, `default_agent_loop=re_search_agent`, `n=5`, tunable `VLLM_GPU_MEM_UTIL`, `max_num_seqs`, `max_num_batched_tokens`, `update_weights_bucket_megabytes` |
| **Search URL** | `SEARCH_URL=127.0.0.1:3005` (no scheme) | `SEARCH_URL=http://127.0.0.1:3005` (documented as **base** URL; `{base}/batch_search`) |
| **Actor** | `lr=1e-6`, `use_kl_loss`, `kl_loss_coef`, `use_dynamic_bsz`, no explicit `entropy_coeff` in script | Same KL settings; adds `entropy_coeff=0.001`, `use_torch_compile`, `override_config.attn_implementation=sdpa` |
| **Ref** | `ref.fsdp_config.param_offload=True` | `param_offload=False` |
| **Reward** | `reward_model.reward_manager=re_search` | `reward.reward_manager.name=re_search` |
| **Trainer** | `test_freq=10`, `val_before_train=True`, `rollout_save_path`, `hydra.run.dir=.../outputs`, `default_hdfs_dir=null` | `test_freq=1000`, `VAL_BEFORE_TRAIN` default **False**, `trainer.rollout_data_dir`, no `hydra.run.dir` in script |
| **Logging** | `WANDB_DIR=${SAVE_PATH}` | `WANDB_DIR=${CKPTS_DIR}`, loads optional `verl_latest/.env` |
| **Env** | `VLLM_ATTENTION_BACKEND=XFORMERS` | `PYTORCH_ALLOC_CONF`, optional `VLLM_USE_V1`, `CUDA_VISIBLE_DEVICES` |
| **Default model** | `Qwen3-0.6B-Base` | `Qwen3-0.6B` (directory name; comments refer to Instruct recipe) |

---

## §2 — Code mapping: search rollout

### Entry and wiring

| Legacy | Latest |
|--------|--------|
| [`src/verl_legacy/workers/fsdp_workers.py`](../../src/verl_legacy/workers/fsdp_workers.py): `rollout.name == 'vllm_with_search'` → `vLLMRolloutWithSearch` | Rollout worker builds vLLM; [`re_search_agent_loop.py`](../../verl_latest/verl/experimental/agent_loop/re_search_agent_loop.py) registered as `re_search_agent`; gpu40 sets `actor_rollout_ref.rollout.agent.default_agent_loop=re_search_agent` |

### HTTP retrieval

- **Legacy:** [`batch_search`](../../src/verl_legacy/workers/rollout/vllm_rollout/vllm_rollout.py) posts to `f'{self.config.search_url}/batch_search'` with JSON `{'query': [...], 'top_n': top_n}`; aggregates `line['contents']` like latest.
- **Latest:** [`_batch_search_http`](../../verl_latest/verl/experimental/agent_loop/re_search_agent_loop.py) uses `search_url.rstrip('/') + '/batch_search'`, same JSON keys, `timeout=120`, `raise_for_status()`.

**URL scheme:** Legacy `train.sh` default **omits `http://`**, producing a malformed URL if passed through unchanged. Latest defaults to a full base URL.

### Multi-turn loop

- **Legacy:** Synchronous loop in `vLLMRolloutWithSearch.generate_sequences`: vLLM `generate` with `stop=['</search>']`, then `batch_search`, then injects `tokenizer.encode(f" <result>\n{result}\n</result>")` (default `encode` special-token behavior), updates `result_mask` (0 on injected tokens). Continues until no active “need search” paths or length cap.
- **Latest:** Async `ReSearchAgentLoop.run`: repeated `server_manager.generate` with `stop=["</tool_call>"]`; **`add_special_tokens=False`** on injected `<tool_response>` chunk; `response_mask` marks LM tokens vs injected. Stops when a segment completes **without** a valid `<tool_call>{"name": "search", "arguments": "..."}</tool_call>` payload with string `arguments`, or budget / max search turns (`search_max_turns` default 32).

**Behavioral nuance:** Latest exposes rich **termination** metadata (`re_search_termination_reason`, `re_search_response_digest`, etc.); legacy does not attach the same `extra_fields`.

---

## §3 — Code mapping: `re_search` reward

| Aspect | Legacy | Latest |
|--------|--------|--------|
| **Class** | [`ReSearchRewardManagerWithSave`](../../src/verl_legacy/workers/reward_manager/re_search.py) (plain callable) | [`ReSearchRewardManagerWithSave`](../../verl_latest/verl/experimental/reward_loop/reward_manager/re_search.py) extends `RewardManagerBase`, `@register("re_search")` |
| **Decode** | `tokenizer.decode(sequences)` (no `skip_special_tokens` argument) | `tokenizer.decode(..., skip_special_tokens=False)` explicitly |
| **Scoring dispatch** | `compute_score` from [`_default_compute_score`](../../src/verl_legacy/utils/reward_score/__init__.py) with `(data_source, tokenizer, solution_str, ground_truth)` | For `data_source` in `{musique, MuSiQue, train}`, calls [`re_search_score.compute_score(solution_str, ground_truth, tokenizer=...)`](../../verl_latest/verl/utils/reward_score/re_search.py) directly; else `default_compute_score` |
| **Wiring** | Trainer passes `reward_fn` from reward worker / manager (legacy `reward_model.reward_manager`) | [`RayPPOTrainer`](../../verl_latest/verl/trainer/ppo/ray_trainer.py) constructs [`RewardLoopManager`](../../verl_latest/verl/experimental/reward_loop/__init__.py); optional **agent-side** streaming when `not use_rm` or RM resource pool |
| **JSONL logging** | `save_path` from trainer rollout path | `data.save_path` via OmegaConf in latest manager; gpu40 script does not set `data.save_path` (optional) |

### `compute_score` parity (important)

| | Legacy [`utils/reward_score/re_search.py`](../../src/verl_legacy/utils/reward_score/re_search.py) | Latest [`utils/reward_score/re_search.py`](../../verl_latest/verl/utils/reward_score/re_search.py) |
|---|----------------|--------|
| **Signature** | `compute_score(tokenizer, solution_str, ground_truth)` | `compute_score(solution_str, ground_truth, tokenizer=None)` |
| **Response extraction** | Split on `<|im_start|>assistant\n` or `Assistant:` | `_extract_response_text` (same markers + fallback) |
| **EOS** | If response does **not** end with `tokenizer.eos_token` → **`return 0, 'over length'`** | EOS stripped only if present; **truncated completions can still be scored** (documented) |
| **Ground truth** | Passed through to F1 | `_normalize_ground_truth` for dict/list |

---

## §4 — Code mapping: data and prompts

| | Legacy | Latest |
|---|--------|--------|
| **Templates** | [`src/verl_legacy/utils/dataset/template.py`](../../src/verl_legacy/utils/dataset/template.py) `re_search_template_sys` | [`verl_latest/verl/utils/dataset/re_search_templates.py`](../../verl_latest/verl/utils/dataset/re_search_templates.py) `re_search_template_sys` |
| **Text parity** | Substantively **the same** instruction string and tag rules; minor escaping differences in the **example** line (`\\boxed{{answer here}}` vs `\\boxed{answer here}`) in the non-`sys` template only. | Same |
| **Dataset** | [`RLHFDataset`](../../src/verl_legacy/utils/dataset/rl_dataset.py): `apply_chat=True` → `apply_chat_template([system, user], add_generation_prompt=True)` | [`RLHFDataset`](../../verl_latest/verl/utils/dataset/rl_dataset.py): `re_search_use_chat_format=True` + `prompt_template_name` → same **system + user** chat pattern for default gpu40 flags |

---

## §5 — Trainer entry (scoped)

| | Legacy | Latest |
|---|--------|--------|
| **Entry** | [`src/verl_legacy/trainer/main_ppo.py`](../../src/verl_legacy/trainer/main_ppo.py) + [`ray_trainer.py`](../../src/verl_legacy/trainer/ppo/ray_trainer.py) | [`verl_latest/verl/trainer/main_ppo.py`](../../verl_latest/verl/trainer/main_ppo.py) + [`ray_trainer.py`](../../verl_latest/verl/trainer/ppo/ray_trainer.py) |
| **Reward** | Reward model worker / `reward_fn` when RM disabled; `rollout_save_path` for val JSONL | **Always** builds `RewardLoopManager`; `reward.reward_manager.name`; `trainer.rollout_data_dir`; ReSearch-specific **metrics** (e.g. `re_search/response_token_count_mean`) |
| **Rollout** | Sync vLLM rollout inside actor worker | **Async** `AgentLoopManager` + checkpoint engine + weight sync to vLLM replicas |

Architecturally, latest **splits** inference (vLLM replicas), **agent loops** (multi-turn tools), and **reward workers** (RewardLoop) with explicit async orchestration; legacy couples vLLM + search loop inside one rollout class.

---

## §6 — Summary table

| Area | Parity? | Notes |
|------|---------|--------|
| GRPO + KL hyperparams in scripts | **Mostly** | gpu40 adds explicit entropy, sdpa, torch.compile toggles |
| Prompt template + chat framing | **Yes** (default flags) | Optional `re_search_add_qwen_chat` only in latest |
| Search HTTP protocol | **Yes** | Same JSON; URL base must include scheme for legacy default |
| Multi-turn search rollout | **Mostly** | Same stop/inject pattern; implementation and masks differ slightly |
| Rule reward `re_search` | **Partial** | **EOS / truncation handling in `compute_score` differs** — see §3 |
| Hardware / memory | **No** | By design (1× GPU, smaller batches, lower token caps) |
| Validation | **No** | `test_freq` and `val_before_train` differ by default |
| Default HF weights | **No** | Base vs default `Qwen3-0.6B` path in scripts |

---

## Appendix — Optional runtime checks (not run in this review)

1. Decode one parquet row with both datasets and compare token IDs / strings.
2. Single rollout step: compare legacy vs latest JSONL fields (prompt, response, scores).
3. If strict reward parity with legacy is required, consider aligning latest `compute_score` EOS rule with [`src/verl_legacy/utils/reward_score/re_search.py`](../../src/verl_legacy/utils/reward_score/re_search.py) or document the intentional relaxation in latest.

---

## References (key files)

**Legacy:** [`workers/fsdp_workers.py`](../../src/verl_legacy/workers/fsdp_workers.py), [`vllm_rollout/vllm_rollout.py`](../../src/verl_legacy/workers/rollout/vllm_rollout/vllm_rollout.py), [`workers/reward_manager/re_search.py`](../../src/verl_legacy/workers/reward_manager/re_search.py), [`utils/dataset/template.py`](../../src/verl_legacy/utils/dataset/template.py), [`utils/reward_score/re_search.py`](../../src/verl_legacy/utils/reward_score/re_search.py).

**Latest:** [`experimental/agent_loop/re_search_agent_loop.py`](../../verl_latest/verl/experimental/agent_loop/re_search_agent_loop.py), [`experimental/reward_loop/reward_manager/re_search.py`](../../verl_latest/verl/experimental/reward_loop/reward_manager/re_search.py), [`utils/dataset/re_search_templates.py`](../../verl_latest/verl/utils/dataset/re_search_templates.py), [`utils/reward_score/re_search.py`](../../verl_latest/verl/utils/reward_score/re_search.py), [`trainer/ppo/ray_trainer.py`](../../verl_latest/verl/trainer/ppo/ray_trainer.py).
