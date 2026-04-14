# Alice

## Environment

```bash
module purge
module load ALICE/default
module load Miniconda3/24.7.1-0
module load CUDA/12.4.0
source "$(conda info --base)/etc/profile.d/conda.sh"
```

## Memory

```bash
watch -n 1 nvidia-smi # GPU
htop                  # CPU
```

## SRUN

```bash
sinfo -o "%P %G %l %c %m"

srun \
  --partition=gpu-l4-24g \
  --gres=gpu:l4:4 \
  --cpus-per-task=32 \
  --mem=120g \
  --time=168:00:00 \
  --pty bash

srun \
  --partition=gpu-mig-40g \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=120g \
  --time=168:00:00 \
  --pty bash

srun \
  --partition=gpu-a100-80g \
  --gres=gpu:a100:1 \
  --cpus-per-task=8 \
  --mem=120g \
  --time=168:00:00 \
  --pty bash

srun \
  --partition=gpu-a100-80g \
  --gres=gpu:a100:2 \
  --cpus-per-task=16 \
  --mem=240g \
  --time=168:00:00 \
  --pty bash

srun \
  --partition=gpu-a100-80g \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=90g \
  --time=12:00:00 \
  --pty bash

srun \
  --partition=gpu-short \
  --gres=gpu:a100:1 \
  --cpus-per-task=8 \
  --mem=120g \
  --time=04:00:00 \
  --pty bash

srun \
  --partition=gpu-short \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=8 \
  --mem=120g \
  --time=04:00:00 \
  --pty bash
```

## Slurm

```bash
squeue --start -j 1500344,1500332,1500295,1500293,1501765

sinfo

squeue -u $USER

squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.10L"
squeue -u "$USER" -o "%.18i %.9P %.8j %.8u %.2t %.10M %.10L %.30N"

squeue --start -j JOBID

scontrol show job JOBID

scancel JOBID
```

## Retriever

```bash
python retriever_serving.py --config retriever_config.yaml --num_retriever 1 --port 3005

curl -X GET "http://127.0.0.1:3005/health"
```

