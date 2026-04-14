#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only in bare environments
    raise SystemExit("PyYAML is required to read experiment config files.") from exc


WRAPPER_BY_VARIANT = {
    "z": "sbatch/retriever_then_z_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_80gb.sbatch",
    "x_min": "sbatch/retriever_then_x_min_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb.sbatch",
}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "run"


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Expected a mapping in config file: {path}")
    return data


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Expected '{key}' to be a mapping in the experiment config.")
    return value


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _copy_text_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _build_run_layout(config_path: Path, config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    run_cfg = _require_mapping(config, "run")
    prompt_cfg = _require_mapping(config, "prompt")
    reward_cfg = _require_mapping(config, "reward")
    sampling_cfg = _require_mapping(config, "sampling")

    variant = str(run_cfg.get("variant", "")).strip()
    if variant not in WRAPPER_BY_VARIANT:
        valid = ", ".join(sorted(WRAPPER_BY_VARIANT))
        raise SystemExit(f"Unsupported run.variant={variant!r}. Expected one of: {valid}")

    experiment_name = str(run_cfg.get("experiment_name", "")).strip()
    if not experiment_name:
        raise SystemExit("run.experiment_name must be set.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{_slugify(experiment_name)}"

    experiment_dir = config_path.parent.resolve()
    output_root_value = str(run_cfg.get("output_root", "runs"))
    output_root = _resolve_path(output_root_value, experiment_dir)
    run_dir = output_root / run_name
    inputs_dir = run_dir / "inputs"
    logs_dir = run_dir / "logs"
    checkpoints_dir = run_dir / "checkpoints"
    rollouts_dir = run_dir / "rollouts"
    runtime_state_dir = run_dir / "runtime_state"
    tmp_dir = run_dir / "tmp"

    prompt_src = _resolve_path(str(prompt_cfg.get("path", "")), experiment_dir)
    reward_src = _resolve_path(str(reward_cfg.get("path", "")), experiment_dir)
    if not prompt_src.is_file():
        raise SystemExit(f"Prompt file not found: {prompt_src}")
    if not reward_src.is_file():
        raise SystemExit(f"Reward file not found: {reward_src}")

    prompt_template_name = str(prompt_cfg.get("template_name", "re_search_template_sys"))
    reward_function_name = str(reward_cfg.get("function_name", "compute_score"))

    try:
        temperature = float(sampling_cfg["temperature"])
        top_p = float(sampling_cfg["top_p"])
    except KeyError as exc:
        raise SystemExit(f"Missing sampling setting: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise SystemExit("sampling.temperature and sampling.top_p must be numeric.") from exc

    wrapper_path = (repo_root / WRAPPER_BY_VARIANT[variant]).resolve()
    if not wrapper_path.is_file():
        raise SystemExit(f"Could not find sbatch wrapper: {wrapper_path}")

    return {
        "variant": variant,
        "experiment_name": experiment_name,
        "run_name": run_name,
        "run_dir": run_dir,
        "inputs_dir": inputs_dir,
        "logs_dir": logs_dir,
        "checkpoints_dir": checkpoints_dir,
        "rollouts_dir": rollouts_dir,
        "runtime_state_dir": runtime_state_dir,
        "tmp_dir": tmp_dir,
        "prompt_src": prompt_src,
        "reward_src": reward_src,
        "prompt_template_name": prompt_template_name,
        "reward_function_name": reward_function_name,
        "temperature": temperature,
        "top_p": top_p,
        "wrapper_path": wrapper_path,
        "config_copy_path": inputs_dir / "config.yaml",
        "prompt_copy_path": inputs_dir / "prompt.txt",
        "reward_copy_path": inputs_dir / "reward.py",
        "metadata_path": run_dir / "submission_metadata.json",
    }


def _stage_run_inputs(layout: dict[str, Any], config_path: Path) -> None:
    for path in (
        layout["inputs_dir"],
        layout["logs_dir"],
        layout["checkpoints_dir"],
        layout["rollouts_dir"],
        layout["runtime_state_dir"],
        layout["tmp_dir"],
    ):
        Path(path).mkdir(parents=True, exist_ok=True)

    _copy_text_file(config_path, layout["config_copy_path"])
    _copy_text_file(layout["prompt_src"], layout["prompt_copy_path"])
    _copy_text_file(layout["reward_src"], layout["reward_copy_path"])


def _build_env(layout: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "WANDB_EXPERIMENT_NAME": layout["run_name"],
            "ADD_TIMESTAMP_TO_EXPERIMENT_NAME": "false",
            "PROMPT_TEMPLATE_NAME": layout["prompt_template_name"],
            "PROMPT_TEMPLATE_PATH": str(layout["prompt_copy_path"]),
            "RE_SEARCH_REWARD_FUNCTION_PATH": str(layout["reward_copy_path"]),
            "RE_SEARCH_REWARD_FUNCTION_NAME": layout["reward_function_name"],
            "TEMPERATURE": str(layout["temperature"]),
            "TOP_P": str(layout["top_p"]),
            "CKPTS_DIR": str(layout["checkpoints_dir"]),
            "ROLLOUT_SAVE_PATH": str(layout["rollouts_dir"]),
            "LOG_PATH": str(layout["logs_dir"] / "trainer.log"),
            "RETRIEVER_LOG": str(layout["logs_dir"] / "retriever.log"),
            "RAY_DATA_HOME": str(layout["runtime_state_dir"] / "ray"),
            "RAY_TMPDIR": str(layout["tmp_dir"] / "ray"),
            "TMPDIR": str(layout["tmp_dir"]),
        }
    )
    return env


def _write_metadata(layout: dict[str, Any], config_path: Path, command: list[str]) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path.resolve()),
        "experiment_name": layout["experiment_name"],
        "variant": layout["variant"],
        "run_name": layout["run_name"],
        "run_dir": str(layout["run_dir"]),
        "inputs": {
            "config": str(layout["config_copy_path"]),
            "prompt": str(layout["prompt_copy_path"]),
            "reward": str(layout["reward_copy_path"]),
        },
        "outputs": {
            "logs_dir": str(layout["logs_dir"]),
            "trainer_log": str(layout["logs_dir"] / "trainer.log"),
            "retriever_log": str(layout["logs_dir"] / "retriever.log"),
            "slurm_stdout": str(layout["logs_dir"] / "slurm-%x-%j.out"),
            "slurm_stderr": str(layout["logs_dir"] / "slurm-%x-%j.err"),
            "checkpoints_dir": str(layout["checkpoints_dir"]),
            "rollouts_dir": str(layout["rollouts_dir"]),
            "runtime_state_dir": str(layout["runtime_state_dir"]),
            "tmp_dir": str(layout["tmp_dir"]),
        },
        "sampling": {
            "temperature": layout["temperature"],
            "top_p": layout["top_p"],
        },
        "submit_command": command,
    }
    layout["metadata_path"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit an sbatch run from a single experiment config.")
    parser.add_argument("config", help="Path to the experiment config.yaml file.")
    parser.add_argument("--dry-run", action="store_true", help="Stage inputs and print the sbatch command without submitting.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the wrapper locally with mock retriever/trainer behavior and write the resolved launch config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")

    repo_root = Path(__file__).resolve().parents[1]
    config = _load_yaml(config_path)
    layout = _build_run_layout(config_path, config, repo_root)
    _stage_run_inputs(layout, config_path)

    command = [
        "sbatch",
        "--job-name",
        layout["run_name"][:120],
        "--chdir",
        str(layout["run_dir"]),
        "--output",
        str(layout["logs_dir"] / "slurm-%x-%j.out"),
        "--error",
        str(layout["logs_dir"] / "slurm-%x-%j.err"),
        str(layout["wrapper_path"]),
    ]
    _write_metadata(layout, config_path, command)

    if args.dry_run:
        print(f"Staged run directory: {layout['run_dir']}")
        print("SBATCH command:")
        print(" ".join(command))
        return 0

    if args.smoke_test:
        env = _build_env(layout)
        effective_config_path = layout["logs_dir"] / "effective_launch_config.json"
        env.update(
            {
                "MOCK_RETRIEVER": "1",
                "RUN_TRAIN_DIRECT": "1",
                "SKIP_ALICE_BOOTSTRAP": "1",
                "SKIP_WANDB_LOGIN": "1",
                "PRINT_EFFECTIVE_CONFIG_ONLY": "1",
                "EFFECTIVE_CONFIG_PATH": str(effective_config_path),
            }
        )
        smoke = subprocess.run(
            ["bash", str(layout["wrapper_path"])],
            check=False,
            text=True,
            capture_output=True,
            cwd=layout["run_dir"],
            env=env,
        )
        (layout["logs_dir"] / "smoke_test.stdout").write_text(smoke.stdout, encoding="utf-8")
        (layout["logs_dir"] / "smoke_test.stderr").write_text(smoke.stderr, encoding="utf-8")
        if smoke.returncode != 0:
            raise SystemExit(
                f"Smoke test failed with exit code {smoke.returncode}. "
                f"See {(layout['logs_dir'] / 'smoke_test.stderr')}"
            )
        print(f"Smoke test run directory: {layout['run_dir']}")
        print(f"Effective config: {effective_config_path}")
        return 0

    result = subprocess.run(command, check=True, text=True, capture_output=True, env=_build_env(layout))
    sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    print(f"Run directory: {layout['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
