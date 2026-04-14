#!/usr/bin/env bash
#SBATCH --job-name=retr-xmin-qwen3-40g
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120g
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-mig-40g
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
exec bash "${SCRIPT_DIR}/run_retriever_then_x_min_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb.sh" "$@"
