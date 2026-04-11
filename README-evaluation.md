## Evaluation environment

### 1. Create env and install

```bash
conda create -n research-eval python=3.11 -y
conda activate research-eval

pip install -r requirements-evaluation.txt
python setup_evaluation.py develop --no-deps
```

### 2. CUDA (GPU node; before SGLang)

```bash
source scripts/evaluation/env_cuda_alice.sh
```

### 3. Run SGLang

```bash
python -m sglang.launch_server \
  --served-model-name qwen3-0.6b-base \
  --model-path /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B-Base \
  --host 0.0.0.0 \
  --port 3001 \
  --trust-remote-code \
  --disable-cuda-graph
```

#### Base Model

```bash
python -m sglang.launch_server \
  --served-model-name qwen3-0.6b-base \
  --model-path /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B-Base \
  --tp 1 \
  --context-length 8192 \
  --enable-metrics \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 3000 \
  --trust-remote-code \
  --disable-overlap \
  --disable-radix-cache
```

#### Instruct Model

```bash
python -m sglang.launch_server \
  --served-model-name qwen3-0.6b \
  --model-path /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B \
  --tp 1 \
  --context-length 8192 \
  --enable-metrics \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 3000 \
  --trust-remote-code \
  --disable-overlap \
  --disable-radix-cache
```

Change `--model-path` / `--served-model-name` to match your checkpoint.

### 4. Run Evaluation

#### Base Model

Paths below assume the repo root is `/zfsstore/user/s4374886/omega/re-search` and you run from `scripts/evaluation/`. Each run writes a timestamped folder under `results/<dataset_name>/`, e.g. `hotpotqa_2026_03_27_22_00_research_qwen3_base/`; use `--save_note` to tag runs.

Shared flags (same for every dataset):

- `--data_dir` must be the **parent** of the dataset folder: `.../data` (not `.../data/<name>`).
- `--save_dir` is the **parent** for result subfolders: `.../results/<dataset_name>`.

##### Bamboogle (`data/bamboogle/test.jsonl`)

```bash
python run_eval.py \
  --config_path eval_config.yaml \
  --method_name research \
  --data_dir /zfsstore/user/s4374886/omega/re-search/data \
  --dataset_name bamboogle \
  --split test \
  --save_dir /zfsstore/user/s4374886/omega/re-search/results/bamboogle \
  --save_note research_qwen3_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B-Base \
  --apply_chat False
```

##### HotpotQA (`data/hotpotqa/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path eval_config.yaml \
  --method_name research \
  --data_dir /zfsstore/user/s4374886/omega/re-search/data \
  --dataset_name hotpotqa \
  --split dev \
  --save_dir /zfsstore/user/s4374886/omega/re-search/results/hotpotqa \
  --save_note research_qwen3_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B-Base \
  --apply_chat False
```

Use `--split train` to run on `train.jsonl` instead.

##### MuSiQue (`data/musique/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path eval_config.yaml \
  --method_name research \
  --data_dir /zfsstore/user/s4374886/omega/re-search/data \
  --dataset_name musique \
  --split dev \
  --save_dir /zfsstore/user/s4374886/omega/re-search/results/musique \
  --save_note research_qwen3_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B-Base \
  --apply_chat False
```

Use `--split train` for `train.jsonl`.

##### 2WikiMultihopQA (`data/2wikimultihopqa/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path eval_config.yaml \
  --method_name research \
  --data_dir /zfsstore/user/s4374886/omega/re-search/data \
  --dataset_name 2wikimultihopqa \
  --split dev \
  --save_dir /zfsstore/user/s4374886/omega/re-search/results/2wikimultihopqa \
  --save_note research_qwen3_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B-Base \
  --apply_chat False
```

Use `--split train` for `train.jsonl`.

#### Instruct Model

Use the instruct checkpoint for `--generator_model` and `**--apply_chat True**` so prompts use the chat template. Start SGLang with the instruct weights (see §3 **Instruct Model**). Paths and `data_dir` / `save_dir` rules match the base model section; only `generator_model`, `save_note`, and `apply_chat` differ.

Shared defaults here: `--generator_model .../models/Qwen3-0.6B`, `--save_note research_qwen3_instruct`, `--apply_chat True`.

##### Reasoning (optional)

If you want the model’s intermediate reasoning to be included in the generated text, set `--enable_thinking True`. Evaluation/metrics still extract the final answer from `<answer>...</answer>`, so enabling thinking should not change the answer formatting requirements.

##### Bamboogle (`data/bamboogle/test.jsonl`)

```bash
python run_eval.py \
  --config_path eval_config.yaml \
  --method_name research \
  --data_dir /zfsstore/user/s4374886/omega/re-search/data \
  --dataset_name bamboogle \
  --split test \
  --save_dir /zfsstore/user/s4374886/omega/re-search/results/bamboogle \
  --save_note research_qwen3_instruct \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B \
  --apply_chat True

python run_eval.py \
  --config_path eval_config.yaml \
  --method_name research \
  --data_dir /zfsstore/user/s4374886/omega/re-search/data \
  --dataset_name bamboogle \
  --split test \
  --save_dir /zfsstore/user/s4374886/omega/re-search/results/bamboogle \
  --save_note research_qwen3_reasoning_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B \
  --apply_chat True \
  --enable_thinking True
```

##### HotpotQA (`data/hotpotqa/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path eval_config.yaml \
  --method_name research \
  --data_dir /zfsstore/user/s4374886/omega/re-search/data \
  --dataset_name hotpotqa \
  --split dev \
  --save_dir /zfsstore/user/s4374886/omega/re-search/results/hotpotqa \
  --save_note research_qwen3_instruct \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B \
  --apply_chat True

python run_eval.py \
  --config_path eval_config.yaml \
  --method_name research \
  --data_dir /zfsstore/user/s4374886/omega/re-search/data \
  --dataset_name hotpotqa \
  --split dev \
  --save_dir /zfsstore/user/s4374886/omega/re-search/results/hotpotqa \
  --save_note research_qwen3_reasoning \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B \
  --apply_chat True \
  --enable_thinking True
```

Use `--split train` to run on `train.jsonl` instead.

##### MuSiQue (`data/musique/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path eval_config.yaml \
  --method_name research \
  --data_dir /zfsstore/user/s4374886/omega/re-search/data \
  --dataset_name musique \
  --split dev \
  --save_dir /zfsstore/user/s4374886/omega/re-search/results/musique \
  --save_note research_qwen3_instruct \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B \
  --apply_chat True

python run_eval.py \
  --config_path eval_config.yaml \
  --method_name research \
  --data_dir /zfsstore/user/s4374886/omega/re-search/data \
  --dataset_name musique \
  --split dev \
  --save_dir /zfsstore/user/s4374886/omega/re-search/results/musique \
  --save_note research_qwen3_reasoning \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B \
  --apply_chat True \
  --enable_thinking True
```

Use `--split train` for `train.jsonl`.

##### 2WikiMultihopQA (`data/2wikimultihopqa/{dev,train}.jsonl`)

```bash
python run_eval.py \
  --config_path eval_config.yaml \
  --method_name research \
  --data_dir /zfsstore/user/s4374886/omega/re-search/data \
  --dataset_name 2wikimultihopqa \
  --split dev \
  --save_dir /zfsstore/user/s4374886/omega/re-search/results/2wikimultihopqa \
  --save_note research_qwen3_instruct \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B \
  --apply_chat True

python run_eval.py \
  --config_path eval_config.yaml \
  --method_name research \
  --data_dir /zfsstore/user/s4374886/omega/re-search/data \
  --dataset_name 2wikimultihopqa \
  --split dev \
  --save_dir /zfsstore/user/s4374886/omega/re-search/results/2wikimultihopqa \
  --save_note research_qwen3_reasoning \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B \
  --apply_chat True \
  --enable_thinking True
```

Use `--split train` for `train.jsonl`.

### 5. LLM-as-judge (`scripts/evaluation/llm_judge.py`)

After `run_eval.py` finishes, you can score predictions with an **OpenAI-compatible** chat API (OpenAI, Azure OpenAI, vLLM OpenAI server, etc.). The script reads FlashRAG’s `**intermediate_data.json`** in a run directory and writes `**llm_judge.jsonl**` (one judged record per line) plus `**llm_judge_metric.txt**` (aggregate accuracy).

**Inputs**

- `--input_dir`: Path to the **timestamped run folder** that contains `intermediate_data.json` (same folder as `config.yaml` and `metric_score.txt`), e.g.  
`.../results/bamboogle/bamboogle_2026_03_27_23_24_research_qwen3_base`

**Outputs** (created next to `intermediate_data.json`)

- `llm_judge.jsonl` — each line adds an `llm_judge` object with `rationale` and `judgement` (`correct` / `incorrect`)
- `llm_judge_metric.txt` — `llm_judge_metric: <fraction>`

**API**

- `--base_url`: OpenAI-compatible base URL (e.g. `https://api.openai.com/v1` for OpenAI)
- `--api_key`: API key (use an env var; avoid committing secrets)
- `--model_name`: Judge model id (default `gpt-4o-mini`)
- `--max_workers`: Parallel judge calls (default `10`; lower if you hit rate limits)

**Example** (from `scripts/evaluation/`):

```bash
cd scripts/evaluation

python llm_judge.py \
  --input_dir /zfsstore/user/s4374886/omega/re-search/results/bamboogle/bamboogle_2026_03_30_12_00_research_qwen3_reasoning \
  --base_url http://127.0.0.1:11434 \
  --model_name qwen3.5:9b \
  --max_workers 2 \
  --log_path ./llm_judge.log
```

For a **local** OpenAI-compatible server (e.g. vLLM), set `--base_url` to `http://127.0.0.1:11434/v1` and use that server’s model name for `--model_name`.

The judge prompt compares `question`, `golden_answers`, and `output.pred` from each record; ensure `intermediate_data.json` includes those fields (standard FlashRAG eval output).

## Steps to test a VERL-trained model

1. Activate env and go to VERL repo.

```bash
cd /home/s4374886/omega/re-search/verl_latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate r_e
```

2. Merge FSDP actor checkpoint to HF format (replace `global_step_1000` as needed).

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /home/s4374886/verl/ckpts/research/qwen3_0.6b_instruct_grpo_gpu_1_40gb/global_step_1000/actor \
  --target_dir /home/s4374886/verl/ckpts/research/qwen3_0.6b_instruct_grpo_gpu_1_40gb/global_step_1000_hf
```

3. Start SGLang server with the merged model.

```bash
source /home/s4374886/omega/re-search/scripts/evaluation/env_cuda_alice.sh

# Recommended stability guards on ALICE
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_NVML_BASED_CUDA_CHECK=0

python -m sglang.launch_server \
  --served-model-name qwen3-0.6b-ckpt1000 \
  --model-path /home/s4374886/verl/ckpts/research/qwen3_0.6b_instruct_grpo_gpu_1_40gb/global_step_1000_hf \
  --tp 1 \
  --context-length 8192 \
  --enable-metrics \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 3000 \
  --trust-remote-code \
  --disable-overlap \
  --disable-radix-cache
```

4. In another terminal, run evaluation.

```bash
cd /home/s4374886/omega/re-search/scripts/evaluation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate r_e

python run_eval.py \
  --config_path eval_config.yaml \
  --method_name research \
  --data_dir /zfsstore/user/s4374886/omega/re-search/data \
  --dataset_name bamboogle \
  --split test \
  --save_dir /zfsstore/user/s4374886/omega/re-search/results/bamboogle \
  --save_note research_ckpt1000 \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model /home/s4374886/verl/ckpts/research/qwen3_0.6b_instruct_grpo_gpu_1_40gb/global_step_1000_hf \
  --apply_chat True
```

5. Check outputs under:

`/zfsstore/user/s4374886/omega/re-search/results/bamboogle/<dataset>_<timestamp>_research_ckpt1000/`

Expected files:

- `config.yaml`
- `metric_score.txt`
- `intermediate_data.json`