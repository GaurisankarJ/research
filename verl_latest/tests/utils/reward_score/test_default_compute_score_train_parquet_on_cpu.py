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

"""Exercise ``default_compute_score`` on rows from ``train.parquet`` (MuSiQue / GRPO layout).

Path resolution (first match wins):

1. Environment variable ``VERL_TRAIN_PARQUET`` or ``TRAIN_FILE``
2. ``<verl_latest>/data/musique/train.parquet``
3. ``<parent of verl_latest>/data/musique/train.parquet`` (monorepo layout)

Run::

    cd verl_latest && pip install -e .  # if needed
    VERL_TRAIN_PARQUET=/path/to/train.parquet pytest tests/utils/reward_score/test_default_compute_score_train_parquet_on_cpu.py -v

Or a few rows only::

    pytest ... -k train_parquet --maxfail=1
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from verl.utils.reward_score import default_compute_score


def _verl_latest_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _train_parquet_candidates() -> list[Path]:
    env = os.environ.get("VERL_TRAIN_PARQUET") or os.environ.get("TRAIN_FILE")
    out: list[Path] = []
    if env:
        out.append(Path(os.path.expanduser(env)))
    root = _verl_latest_root()
    out.append(root / "data" / "musique" / "train.parquet")
    out.append(root.parent / "data" / "musique" / "train.parquet")
    return out


def _resolve_train_parquet() -> Path | None:
    for p in _train_parquet_candidates():
        if p.is_file():
            return p
    return None


def _to_py(obj):
    """HF datasets / numpy scalars -> plain Python for asserts."""
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def _parse_reward_model(raw):
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


def _ground_truth_from_row(row: dict):
    rm = _parse_reward_model(row.get("reward_model"))
    if not rm:
        return None
    gt = rm.get("ground_truth")
    return _to_py(gt)


def _data_source_from_row(row: dict):
    return _to_py(row.get("data_source"))


def _first_reference_answer(ground_truth) -> str:
    """Pick one string label for ReSearch ``\\boxed{}`` smoke checks."""
    if isinstance(ground_truth, dict) and "target" in ground_truth:
        t = ground_truth["target"]
    else:
        t = ground_truth
    if isinstance(t, str):
        return t
    if isinstance(t, list) and len(t) > 0:
        return str(t[0])
    return "unknown"


def _re_search_response_with_boxed(boxed_inner: str) -> str:
    """Minimal valid ReSearch-formatted response (matches ``re_search.validate_format``)."""
    return (
        "<think> reasoning </think>"
        f"<answer> \\boxed{{{boxed_inner}}} </answer>"
    )


@pytest.fixture(scope="module")
def train_parquet_path():
    p = _resolve_train_parquet()
    if p is None:
        pytest.skip(
            "No train.parquet found. Set VERL_TRAIN_PARQUET or TRAIN_FILE, or place data at "
            "data/musique/train.parquet under the repo (see run_qwen3_0.6b_grpo_vllm.sh)."
        )
    return p


def test_default_compute_score_train_parquet_rows(train_parquet_path):
    """Same kwargs shape as ``NaiveRewardManager``: data_source, solution_str, ground_truth."""
    import datasets

    ds = datasets.load_dataset("parquet", data_files=str(train_parquet_path))["train"]
    max_rows = int(os.environ.get("VERL_REWARD_TEST_MAX_ROWS", "32"))
    n = min(len(ds), max_rows)

    for i in range(n):
        row = ds[i]
        data_source = _data_source_from_row(row)
        ground_truth = _ground_truth_from_row(row)
        assert data_source is not None and str(data_source) != "", f"row {i}: missing data_source"
        data_source = str(data_source)
        assert ground_truth is not None, f"row {i}: missing reward_model.ground_truth"

        ref = _first_reference_answer(ground_truth)
        solution_match = _re_search_response_with_boxed(ref)
        solution_wrong = _re_search_response_with_boxed("___definitely_not_the_label___")

        score_match = default_compute_score(
            data_source,
            solution_match,
            ground_truth,
        )
        score_wrong = default_compute_score(
            data_source,
            solution_wrong,
            ground_truth,
        )

        assert isinstance(score_match, float), f"row {i}: expected float, got {type(score_match)}"
        assert isinstance(score_wrong, float), f"row {i}: expected float, got {type(score_wrong)}"
        assert 0.0 <= score_match <= 1.0, f"row {i}: score_match out of range: {score_match}"
        assert 0.0 <= score_wrong <= 1.0, f"row {i}: score_wrong out of range: {score_wrong}"
        assert score_match >= score_wrong, f"row {i}: matching answer should score >= wrong answer"


def test_search_r1_branch_dict_ground_truth_unchanged():
    """searchR1_* path expects dict with target; no TypeError."""
    gt = {"target": ["Paris"]}
    s = "x\n<answer>Paris</answer>"
    r = default_compute_score("searchR1_nq", s, gt)
    assert r == 1.0


def test_train_branch_list_ground_truth_normalized():
    """train / musique branch: list-shaped labels; ReSearch format + token F1."""
    gt = ["Paris"]
    s = _re_search_response_with_boxed("Paris")
    r = default_compute_score("train", s, gt)
    assert r == 1.0
