# Slurm Wrappers

This directory contains Slurm launchers for the two-stage ALICE workflow:

1. Bootstrap the ALICE environment.
2. Start the retriever in conda env `r_e`.
3. Wait for `GET /health` on port `3005`.
4. Launch the training script in a separate `srun --overlap` step under conda env `r_t`.
5. Clean up the background retriever when the job exits.

## Files

- `retriever_then_z_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_80gb.sbatch`
- `retriever_then_x_min_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb.sbatch`
- `run_retriever_then_z_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_80gb.sh`
- `run_retriever_then_x_min_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb.sh`
- `test_retriever_then_training_mock.sh`

## Default Resources

The `.sbatch` files currently use these defaults:

- 80GB run: `--partition=gpu-a100-80g`, `--gres=gpu:a100:1`, `--cpus-per-task=8`, `--mem=120g`, `--time=08:00:00`
- 40GB run: `--partition=gpu-mig-40g`, `--gres=gpu:1`, `--cpus-per-task=8`, `--mem=120g`, `--time=08:00:00`

Adjust the `#SBATCH` lines if your target queue or runtime changes.

## Submit

From the repo root:

```bash
sbatch sbatch/retriever_then_z_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_80gb.sbatch

sbatch sbatch/retriever_then_x_min_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb.sbatch
```

## Runtime Details

Each wrapper performs this ALICE bootstrap before starting the retriever, and again inside the training step:

```bash
module purge
module load ALICE/default
module load Miniconda3/24.7.1-0
module load CUDA/12.4.0
source "$(conda info --base)/etc/profile.d/conda.sh"
```

Retriever launch:

```bash
conda activate r_e
cd scripts/serving
python retriever_serving.py --config retriever_config.yaml --num_retriever 1 --port 3005
```

Optional overrides:

- `RETRIEVER_CONFIG=retriever_config_mini.yaml` to test the wrapper flow with the smaller retriever assets
- `RUN_TRAIN_DIRECT=1` to run the training stage directly inside an existing interactive `srun --pty bash` allocation
- `RETRIEVER_HEALTH_TIMEOUT_S=1800` is now the default health wait; increase it further if the full retriever needs more time to initialize

Training launch:

```bash
export SEARCH_URL=http://127.0.0.1:3005
srun --overlap --ntasks=1 ...
```

For testing from inside an existing interactive `srun --pty bash` allocation, skip the nested Slurm step and run the training stage directly:

```bash
RUN_TRAIN_DIRECT=1 RETRIEVER_CONFIG=retriever_config_mini.yaml bash sbatch/retriever_then_z_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_80gb.sbatch
```

## Mock Validation

To validate the orchestration without starting the real retriever or the real trainer:

```bash
bash sbatch/test_retriever_then_training_mock.sh
```

This checks:

- shell syntax for the new `.sh` and `.sbatch` files
- mock `/health` startup and polling
- trainer-stage handoff for both variants
- retriever cleanup after exit
- `shellcheck` when available
- `sbatch --test-only` when available and a Slurm controller is reachable

## Notes

- The wrappers expect the retriever config at `scripts/serving/retriever_config.yaml`.
- The training scripts already default `SEARCH_URL` to `http://127.0.0.1:3005`, but the wrappers export it explicitly.
- After `conda activate`, the wrappers prepend `${CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH` so the compute node uses the env’s OpenSSL libraries instead of the older system copy.
- If `sbatch --test-only` reports controller connectivity errors outside the cluster environment, the mock test script treats that as an environment limitation rather than a wrapper failure.

