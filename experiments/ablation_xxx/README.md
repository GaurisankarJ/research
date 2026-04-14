# Example Single-Config Ablation

This folder is the template for a self-contained ablation.

Edit these files:

- `config.yaml`: experiment name, sbatch variant, output root, temperature, top-p
- `prompt.txt`: the system prompt/template body used for the run
- `reward.py`: the experiment-specific reward function

## Submit

Dry-run first to stage the dated run directory and inspect the generated sbatch command:

```bash
python3 scripts/submit_experiment_sbatch.py experiments/example_single_config_ablation/config.yaml --dry-run
```

Submit for real:

```bash
python3 scripts/submit_experiment_sbatch.py experiments/example_single_config_ablation/config.yaml
```

Run a no-trainer/no-retriever validation that still exercises the wrapper and writes the resolved values:

```bash
python3 scripts/submit_experiment_sbatch.py experiments/example_single_config_ablation/config.yaml --smoke-test
```

## What Gets Created

Each submission creates a dated run directory under:

```text
experiments/example_single_config_ablation/runs/YYYYMMDD_HHMMSS_<experiment_name>/
```

That run directory contains:

- `inputs/config.yaml`: snapshot of the config used for submission
- `inputs/prompt.txt`: snapshot of the prompt used for submission
- `inputs/reward.py`: snapshot of the reward function used for submission
- `logs/trainer.log`: trainer log written by the launch script
- `logs/retriever.log`: retriever server log
- `logs/slurm-*.out` and `logs/slurm-*.err`: Slurm stdout/stderr
- `logs/effective_launch_config.json`: resolved prompt, reward, temperature, top-p, and output paths from the training shell
- `logs/smoke_test.stdout` and `logs/smoke_test.stderr`: output from `--smoke-test`
- `checkpoints/`: checkpoint output root
- `rollouts/`: rollout JSONL output root
- `runtime_state/`: Ray/runtime state routed under the run folder
- `tmp/`: temporary runtime files routed under the run folder
- `submission_metadata.json`: resolved paths and the exact sbatch command

## Notes

- Set `run.variant` to `z` for the 80GB wrapper or `x_min` for the 40GB wrapper.
- `run.experiment_name` is automatically prefixed with the current date/time when submitted, so the Slurm job name and output folders stay easy to identify.
- `prompt.template_name` should stay `re_search_template_sys` if `prompt.txt` is a full system prompt like this example.
- `reward.function_name` defaults to `compute_score`, which matches the shared reward file and this example.
