## Training environment (vLLM + Qwen3 + VERL)

Use a **dedicated** conda env (e.g. **`r_t`**) for PPO training with **vLLM** and **Qwen3**. Pins live in **`requirements-training.txt`**.

**`flash-attn`** is **not** in that file on purpose: `pip` builds it in an isolated env where **`torch` is missing**, which fails with `ModuleNotFoundError: No module named 'torch'`. Install it **after** the main stack using **`requirements-training-flashattn.txt`** and **`--no-build-isolation`**.

### Env setup

```bash
conda create -n r_t python=3.11 -y
conda activate r_t
cd /path/to/re-search

# Optional on GPU clusters (needed to compile flash-attn):
# module load CUDA/12.4.0
# export CUDA_HOME=/easybuild/software/CUDA/12.4.0

pip install -r requirements-training.txt
pip check

MAX_JOBS=8 pip install --no-build-isolation --no-cache-dir -r requirements-training-flashattn.txt

python setup_training.py develop --no-deps
```

Do **not** `conda install flash-attn` into this env if you use **pip** `torch==2.6.0` — conda can replace NCCL/CUDA and break PyTorch (e.g. `undefined symbol: ncclCommWindowDeregister`). Prefer the **`pip`** flash-attn step above.

### NCCL / `libtorch_cuda.so` errors

If `import torch` fails with an **undefined NCCL symbol**, an old **`libnccl`** is being loaded. Prefer PyTorch’s bundled NCCL: put `.../site-packages/nvidia/nccl/lib` first on **`LD_LIBRARY_PATH`**, or recreate **`r_t`** and avoid mixing conda CUDA packages with pip torch.

### Training

```bash
cd scripts/train

bash train.sh \
    --train_batch_size 8 \
    --ppo_mini_batch_size 8 \
    --apply_chat True \
    --prompt_template_name re_search_template_sys \
    --actor_model_path /path/to/models/Qwen3-0.6B \
    --search_url 127.0.0.1:3005 \
    --project_name research \
    --experiment_name test \
    --nnodes 1 \
    --n_gpus_per_node 1 \
    --save_freq 5 \
    --test_freq 5 \
    --total_epochs 2 \
    --save_path /path/to/results/run1 \
    --train_files /path/to/data/train.parquet \
    --test_files /path/to/data/test.parquet
```

Set **`WANDB_API_KEY`** in the environment or pass **`--wandb_api_key`** to `train.sh`; do not commit keys to the repo.
