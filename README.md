## 1. Environment setup (split profiles)

Use separate environments for SGLang and vLLM. This avoids pip resolver conflicts (`lm-format-enforcer`, `openai`, `outlines-core`) when trying to satisfy both stacks in one env.

### 1A. Base/SGLang profile

```bash
conda create -n research311 python=3.11 -y
conda activate research311
cd /zfsstore/user/s4374886/omega/re-search
pip install -r requirements.txt
pip install -e .
```

Quick sanity checks:

```bash
python -m pip check
python -c "import torch, sglang, transformers, verl; print(torch.__version__, sglang.__version__, transformers.__version__)"
```

### 1B. vLLM profile (Qwen3)

```bash
conda create -n research311-vllm python=3.11 -y
conda activate research311-vllm
cd /zfsstore/user/s4374886/omega/re-search
pip install -r requirements-vllm.txt
RESEARCH_REQUIREMENTS_FILE=requirements-vllm.txt pip install -e .
```

Quick sanity checks:

```bash
python -m pip check
python -c "import torch, vllm, transformers, verl; print(torch.__version__, vllm.__version__, transformers.__version__)"
```

### Qwen3-capable dependency policy

- **`transformers`**: must be **>= 4.51.0** so Qwen3 configs load (older versions can fail with `KeyError: 'qwen3'`).
- **Serving floors from Qwen docs**: **SGLang >= 0.4.6.post1** or **vLLM >= 0.8.5**.
- **Split profile design**:
  - `requirements.txt`: SGLang-oriented base profile.
  - `requirements-vllm.txt`: vLLM profile pinned to `vllm==0.8.5.post1` with compatible formatter deps.

## 2. SGLang (Qwen3) on ALICE GPU

Inside an active GPU Slurm allocation:

```bash
module load CUDA/12.4.0
conda activate research311
export CUDA_HOME=/easybuild/software/CUDA/12.4.0
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
```

Start server:

```bash
python -m sglang.launch_server \
  --served-model-name qwen3-0.6b-base \
  --model-path /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B-Base \
  --host 0.0.0.0 \
  --port 3001 \
  --trust-remote-code \
  --disable-cuda-graph
```

Smoke call:

```bash
curl -s http://127.0.0.1:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-0.6b-base","messages":[{"role":"user","content":"Say ok"}],"max_tokens":8}'
```

## 3. Retriever serving

```bash
cd scripts/serving
python retriever_serving.py --config retriever_config_mini.yaml --num_retriever 1 --port 3000
```
