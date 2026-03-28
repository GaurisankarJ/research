# Source before `python -m sglang.launch_server` on ALICE GPU nodes.
# Usage: source scripts/evaluation/env_cuda_alice.sh
#
# Adjust CUDA_HOME if your site uses a different toolkit path.

module load CUDA/12.4.0

export CUDA_HOME=/easybuild/software/CUDA/12.4.0
export PATH="${CUDA_HOME}/bin:${PATH}"

if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
fi
