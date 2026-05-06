# Ablation Zero

This folder is a self-contained ablation definition.

The shared execution code still comes from the main repo, but the experiment-specific
inputs for this run live here:

- `config.yaml`
- `prompt.txt`
- `reward.py`

Edit these files:

- `config.yaml`: experiment name, sbatch variant, output root, temperature, top-p
- `prompt.txt`: the system prompt/template body used for the run
- `reward.py`: the experiment-specific reward function

## Submit

Dry-run first to stage the dated run directory and inspect the generated sbatch command:

```bash
python3 scripts/submit_experiment_sbatch.py experiments/ablation_zero/config.yaml --dry-run
```

Submit for real:

```bash
python3 scripts/submit_experiment_sbatch.py experiments/ablation_zero/config.yaml
```

Run a validation that uses the ablation settings, skips the real trainer, uses the mock retriever,
and writes the resolved values:

```bash
python3 scripts/submit_experiment_sbatch.py experiments/ablation_zero/config.yaml --smoke-test
```

## What Gets Created

Each submission creates a dated run directory under:

```text
experiments/ablation_zero/runs/YYYYMMDD_HHMMSS_<experiment_name>/
```

Real submit creates outputs under that run directory, including:

- `inputs/config.yaml`: snapshot of the config used for submission
- `inputs/prompt.txt`: snapshot of the prompt used for submission
- `inputs/reward.py`: snapshot of the reward function used for submission
- `logs/trainer.log`: trainer log written by the launch script
- `logs/retriever.log`: retriever server log
- `logs/slurm-*.out` and `logs/slurm-*.err`: Slurm stdout/stderr
- `checkpoints/`: checkpoint output root
- `rollouts/`: rollout JSONL output root
- `runtime_state/`: Ray/runtime state routed under the run folder
- `tmp/`: temporary runtime files routed under the run folder
- `submission_metadata.json`: resolved paths and the exact sbatch command

`--smoke-test` also creates:

- `logs/effective_launch_config.json`: resolved prompt, reward, temperature, top-p, and output paths from the training shell
- `logs/smoke_test.stdout` and `logs/smoke_test.stderr`: smoke-test command output

`--dry-run` stages the run directory and writes `submission_metadata.json`, but does not launch Slurm.

## Notes

- Set `run.variant` to `z` for the 80GB wrapper or `x_min` for the 40GB wrapper.
- `run.experiment_name` is automatically prefixed with the current date/time when submitted, so the Slurm job name and output folders stay easy to identify.
- `prompt.template_name` should stay `re_search_template_sys` if `prompt.txt` is a full system prompt like this example.
- `reward.function_name` defaults to `compute_score`, which matches the shared reward file and this example.
- Running the commands above uses the settings from this folder's `config.yaml`, `prompt.txt`, and `reward.py`.

