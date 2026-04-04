#!/usr/bin/bash
#SBATCH --job-name=retr-verl-gpu80
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
##SBATCH --partition=YOUR_PARTITION
#
# Slurm driver: retriever (conda env r_e) on port 3005, then
# verl_latest/run_qwen3_0.6b_grpo_vllm_instruct_gpu80.sh (conda env r_t).
#
# Before submitting:
#   - Set CONDA_BASE below (or export before sbatch) to your conda prefix.
#   - Adjust #SBATCH lines (partition, time, memory, GPU count) for your site.
#   - On ALICE GPU nodes you may uncomment CUDA module + env (see scripts/evaluation/env_cuda_alice.sh).
#
# Submit from the repo root (or set SLURM_SUBMIT_DIR):
#   export CONDA_BASE=$HOME/miniconda3   # example
#   sbatch scripts/jobs/sbatch_retriever_then_verl_gpu80.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-${PWD}}"

# If the submit directory is not the repo root, set REPO_ROOT explicitly before sbatch:
#   export REPO_ROOT=/path/to/re-search
if [ -n "${REPO_ROOT:-}" ]; then
  cd "${REPO_ROOT}"
fi

# --- Optional: ALICE-style CUDA (uncomment if needed) ---
# module load CUDA/12.4.0
# export CUDA_HOME=/easybuild/software/CUDA/12.4.0
# shellcheck source=/dev/null
# source "${PWD}/scripts/evaluation/env_cuda_alice.sh"

# Conda install prefix (must contain etc/profile.d/conda.sh)
export CONDA_BASE="${CONDA_BASE:?Set CONDA_BASE to your conda prefix, e.g. export CONDA_BASE=\$HOME/miniconda3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_retriever_then_verl_gpu80.sh" "$@"
