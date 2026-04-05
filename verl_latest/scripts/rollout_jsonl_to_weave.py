#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Upload verl rollout JSONL (``trainer.rollout_data_dir``) to a W&B Weave Dataset.

For **live** logging during training (same row payload as each JSON line, linked to the
``wandb`` run), use ``trainer.weave_rollout_live=true`` or ``WEAVE_ROLLOUT_LIVE=1`` instead;
see ``verl/utils/weave_rollout.py``.

Each line matches ``_dump_generations`` / ``_log_rollout_data`` in
``verl/trainer/ppo/ray_trainer.py``: ``input``, ``output``, ``gts``, ``score``,
``step``, plus optional fields such as ``num_turns``, ``num_search_calls``,
``re_search_*``, etc. Every key in each JSON object is preserved as a column.

Requires: ``pip install weave`` (or ``pip install -e ".[weave]"`` from ``verl_latest``).

Example::

    export WANDB_API_KEY=...
    export WANDB_ENTITY=your-team
    export WANDB_PROJECT=verl_grpo_example_musique

    python verl_latest/scripts/rollout_jsonl_to_weave.py \\
        /path/to/rollout_20260104_120000 \\
        --dataset-name musique_rollout_run42

Weave UI: ``https://wandb.ai/<entity>/<project>/weave`` (see terminal output after publish).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _sort_jsonl_paths(paths: list[Path]) -> list[Path]:
    def key(p: Path) -> tuple[int, str]:
        stem = p.stem
        if stem.isdigit():
            return (0, f"{int(stem):020d}")
        return (1, stem)

    return sorted(paths, key=key)


def _iter_jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def load_rows_from_path(path: Path, max_rows: int | None) -> list[dict]:
    """Load all JSONL rows from a file or every ``*.jsonl`` under a directory."""
    rows: list[dict] = []
    if path.is_file():
        jsonl_files = [path]
    else:
        jsonl_files = _sort_jsonl_paths(list(path.glob("*.jsonl")))

    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found under {path}")

    for jp in jsonl_files:
        rel = jp.name
        for line_in_file, obj in _iter_jsonl_rows(jp):
            if not isinstance(obj, dict):
                raise TypeError(f"{jp}:{line_in_file}: expected JSON object per line, got {type(obj)}")
            row = dict(obj)
            row["_jsonl_file"] = rel
            row["_line_in_file"] = line_in_file
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                return rows
    return rows


def weave_project_from_env() -> str:
    ent = os.environ.get("WANDB_ENTITY", "").strip()
    proj = os.environ.get("WANDB_PROJECT", "verl_grpo_example_musique").strip()
    if ent:
        return f"{ent}/{proj}"
    return proj


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish verl rollout JSONL lines as a W&B Weave Dataset (all keys per row).",
    )
    parser.add_argument(
        "rollout_path",
        type=Path,
        help="Directory containing ``{{step}}.jsonl`` files, or a single ``.jsonl`` file.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Weave Dataset name (letters, numbers, hyphens, underscores).",
    )
    parser.add_argument(
        "--weave-project",
        default=None,
        help="W&B Weave project as ``entity/project`` or ``project``. "
        "Default: WANDB_ENTITY + WANDB_PROJECT from the environment.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after this many rows (useful for smoke tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load JSONL and print row count only; do not call weave.",
    )
    args = parser.parse_args()

    rollout_path: Path = args.rollout_path
    if not rollout_path.exists():
        print(f"error: path does not exist: {rollout_path}", file=sys.stderr)
        return 1

    try:
        rows = load_rows_from_path(rollout_path, max_rows=args.max_rows)
    except (FileNotFoundError, TypeError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"Loaded {len(rows)} rows from {rollout_path}")
    if args.dry_run:
        return 0

    try:
        import weave
        from weave import Dataset
    except ImportError:
        print("error: install weave:  pip install weave", file=sys.stderr)
        return 1

    project = args.weave_project or weave_project_from_env()
    weave.init(project)

    dataset = Dataset(name=args.dataset_name, rows=rows)
    ref = weave.publish(dataset)
    print(f"Published Weave Dataset '{args.dataset_name}' to project {project}")
    print(f"Reference: {ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
