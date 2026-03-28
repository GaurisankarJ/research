## Training Environment

Use this environment for PPO training/rollout with the vLLM-serving path.

### Serving requirements

- Qwen3-compatible Transformers: `transformers>=4.51.0`
- Qwen serving floor for vLLM: `vllm>=0.8.5`
- This profile installs from `requirements-training.txt`

### Env setup

```bash
conda create -n research-train python=3.11 -y
conda activate research-train

pip install -r requirements-training.txt

python setup_training.py develop
```
