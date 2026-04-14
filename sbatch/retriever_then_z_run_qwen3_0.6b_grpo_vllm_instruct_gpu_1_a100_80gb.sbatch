#!/usr/bin/env bash
#SBATCH --job-name=retr-z-qwen3-80g
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120g
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
exec bash "${SCRIPT_DIR}/run_retriever_then_z_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_80gb.sh" "$@"
