# Retriever Serving Guide

This guide covers how to run and call the retriever API in `scripts/serving/retriever_serving.py`.

Use the same Python environment as the repo root `requirements.txt` (including **`transformers>=4.51.0`** for Qwen3-era configs if you touch those code paths).

## 1) Prerequisites

- Python 3.11 conda env (example: `research311`)
- Project installed in editable mode from repo root:

```bash
conda activate research311
cd /zfsstore/user/s4374886/omega/re-search
pip install -r requirements.txt
pip install -e .
```

On ALICE GPU nodes, load CUDA toolkit before launching SGLang:

```bash
module load CUDA/12.4.0
export CUDA_HOME=/easybuild/software/CUDA/12.4.0
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
```

## 2) Configure Retriever

Edit one of these config files before starting:

- `scripts/serving/retriever_config.yaml`
- `scripts/serving/retriever_config_mini.yaml`

Set valid paths for:

- `retrieval_method` (embedding model path)
- `index_path` (FAISS index)
- `corpus_path` (JSONL corpus)
- `faiss_gpu` (`false` for non-CUDA machines)

## 3) Start Server

From `scripts/serving`:

```bash
python retriever_serving.py --config retriever_config.yaml --num_retriever 1 --port 3005
```

Server runs at `http://127.0.0.1:3000`.

## 4) API Endpoints

- `GET /health`
- `POST /search`
- `POST /batch_search`

### Health

```bash
curl -X GET "http://127.0.0.1:3005/health"
```

### Search

```bash
curl -X POST "http://127.0.0.1:3005/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who wrote The Lord of the Rings?",
    "top_n": 3,
    "return_score": false
  }'
```

### Search (with scores)

```bash
curl -X POST "http://127.0.0.1:3000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who wrote The Lord of the Rings?",
    "top_n": 3,
    "return_score": true
  }'
```

### Batch Search

```bash
curl -X POST "http://127.0.0.1:3000/batch_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [
      "capital of France",
      "largest planet in our solar system"
    ],
    "top_n": 2,
    "return_score": false
  }'
```

### Batch Search (with scores)

```bash
curl -X POST "http://127.0.0.1:3000/batch_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [
      "capital of France",
      "largest planet in our solar system"
    ],
    "top_n": 2,
    "return_score": true
  }'
```

## 5) Common Issues

- `ModuleNotFoundError: No module named 'flashrag'`
  - Install project from repo root: `pip install -e .`
- `zsh: command not found: --port`
  - Use one-line command or ensure no trailing spaces after `\` in multiline commands.
- `Torch not compiled with CUDA enabled`
  - You are on a non-CUDA machine. Keep `faiss_gpu: false` and use CPU fallback code path.
- `RuntimeError: Could not find nvcc ... cuda_home='/usr/local/cuda'`
  - On ALICE GPU node, run `module load CUDA/12.4.0` and export `CUDA_HOME=/easybuild/software/CUDA/12.4.0`.
- SGLang exits during graph capture on smaller GPU memory or missing toolkit setup
  - Add `--disable-cuda-graph` to `python -m sglang.launch_server ...`.
