# ReSearch / verl_latest — environment setup

This matches conda env **`r_t`** used for **verl + vLLM** (GRPO/PPO with `actor_rollout_ref.rollout.name=vllm`).

## Reference machine (locked env)

| Item | Value |
|------|--------|
| Python | 3.11.15 |
| `pip` | 26.0.1 |
| GPU (example) | NVIDIA L4 |
| Driver (example) | 580.82.07 |

Use a **Linux** host with an **NVIDIA GPU** and a driver/CUDA stack compatible with the wheels in [`requirements_research.txt`](requirements_research.txt) (PyTorch 2.9.0, vLLM 0.12.0, CUDA 12/13-related NVIDIA packages as pinned).

## Recreate the environment

1. **Create a new conda env** with the same Python minor as `r_t`:

   ```bash
   conda create -n verl_research python=3.11.15 -y
   conda activate verl_research
   ```

2. **Upgrade pip tooling** (reduces resolver edge cases):

   ```bash
   python -m pip install -U pip setuptools wheel
   ```

3. **Install locked dependencies** from this directory:

   ```bash
   REPO_ROOT="/home/s4374886/omega/re-search"   # change to your checkout
   pip install -r "${REPO_ROOT}/verl_latest/requirements_research.txt"
   ```

4. **Install xFormers without re-resolving torch** (matches env `r_t`; omitted from the requirements file because PyPI’s xformers metadata requires `torch==2.6.0`, which conflicts with vLLM’s `torch==2.9.0`):

   ```bash
   pip install "xformers==0.0.29.post2" --no-deps
   ```

5. **Install `verl` from this tree** (editable + vLLM extra). The freeze intentionally **omits** the old `-e git+…#subdirectory=verl_latest` line so paths stay local to your clone:

   ```bash
   cd "${REPO_ROOT}/verl_latest"
   pip install -e ".[vllm]"
   ```

## Runtime environment variables

Aligned with [`run_qwen3_0.6b_grpo_vllm.sh`](run_qwen3_0.6b_grpo_vllm.sh):

- `CUDA_VISIBLE_DEVICES` — e.g. `0` for a single GPU.
- `VLLM_USE_V1` — optional; `1` enables the vLLM v1 engine when supported (`export VLLM_USE_V1=1`).

## Smoke test

After install, run:

```bash
conda activate verl_research   # or your env name

python -c "import numpy, torch, vllm, verl; print('numpy', numpy.__version__); print('torch', torch.__version__); print('vllm', getattr(vllm, '__version__', 'unknown')); print('verl', verl.__file__)"

python -c "import verl.workers.rollout.vllm_rollout; print('vllm_rollout package import OK')"
```

The second command loads the vLLM rollout package (which checks that `vllm` is installed). Expect `numpy 2.1.3` if you followed the lockfile; pip may warn that xFormers metadata wants `torch==2.6.0` while you have `2.9.0` — that is expected with the `--no-deps` xFormers install.

## Troubleshooting

- **`pip install -r requirements_research.txt` fails** on a line with `file://` or another machine-specific URL: regenerate the lock from a working env (`conda activate r_t && pip freeze --all`) or compare with this file; PyTorch/vLLM may need the [official PyTorch pip index](https://pytorch.org/get-started/locally/) if your platform wheels differ.
- **`verl` not found**: run `pip install -e ".[vllm]"` from `verl_latest/` inside the active conda env.
- **NumPy / CUDA mismatches**: this lock uses **`numpy==2.1.3`** and many `nvidia-*` pins as in `r_t`; do not mix with an unrelated conda PyTorch unless you know the stack is compatible.
- **`pip install -e ".[vllm]"` downgrades NumPy / CuPy / OpenCV** (older `verl` pins): this checkout’s [`setup.py`](setup.py) allows NumPy 2.x to match `r_t`. If anything still downgrades them, run: `pip install "numpy==2.1.3" "cupy-cuda12x==14.0.1" "opencv-python-headless==4.13.0.92"`.

## Requirements file notes

[`requirements_research.txt`](requirements_research.txt) is `pip freeze --all` from **`r_t`** with:

- **Omitted:** editable `verl` install (`-e git+…#subdirectory=verl_latest`) — replaced by step 4 above.
- **Omitted:** `outlines==0.1.11` — that release pins `outlines_core==0.1.26`, which conflicts with vLLM’s `outlines_core==0.2.11`. vLLM depends on `outlines_core` only, not the `outlines` package.
- **Omitted:** `xformers==0.0.29.post2` from the lockfile — install separately with `pip install "xformers==0.0.29.post2" --no-deps` after step 3 (see above).
- **Normalized:** `pip @ file:///…` (conda feedstock) → `pip==26.0.1` for portable installs.
