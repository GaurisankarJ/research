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

"""ReSearch reward manager: full prompt+response decode, optional JSONL logging (legacy verl port)."""

from __future__ import annotations

import inspect
import json
import os
from typing import Any

import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.import_utils import load_extern_object
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import re_search as re_search_score

# Keep in sync with ``verl.utils.reward_score.default_compute_score`` ReSearch branch.
_RESEARCH_DATA_SOURCES = frozenset({"musique", "MuSiQue", "train"})


@register("re_search")
class ReSearchRewardManagerWithSave(RewardManagerBase):
    """Rule-based reward with optional JSONL logging (legacy ``ReSearchRewardManagerWithSave``).

    - Decodes **prompt + response** (same as legacy) so ``re_search.compute_score`` sees a full chat string.
    - For MuSiQue-style ``data_source`` values, calls ``re_search.compute_score`` to preserve ``reason`` strings.
    - If ``data.save_path`` is set, appends one JSON object per sample (thread-safe enough for Ray append).
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        self.save_path = OmegaConf.select(config, "data.save_path")
        self.num_examine = OmegaConf.select(config, "reward.num_examine")
        self.re_search_function_path = OmegaConf.select(config, "reward.re_search_function.path") or None
        self.re_search_function_name = OmegaConf.select(config, "reward.re_search_function.name") or "compute_score"
        if self.re_search_function_path:
            self.re_search_compute_score = load_extern_object(
                module_path=self.re_search_function_path,
                object_name=self.re_search_function_name,
            )
        else:
            self.re_search_compute_score = re_search_score.compute_score
        if self.num_examine is None:
            self.num_examine = 0
        self._examine_counts: dict[str, int] = {}

    def _score_one(
        self,
        data_source: Any,
        solution_str: str,
        ground_truth: Any,
        extra_info: dict,
    ) -> tuple[float, str, dict]:
        """Returns (reward float, reason string, reward_extra_info dict)."""
        extra_reward_kwargs: dict[str, Any] = {"tokenizer": self.tokenizer}
        if self.reward_router_address is not None:
            extra_reward_kwargs["reward_router_address"] = self.reward_router_address
            extra_reward_kwargs["reward_model_tokenizer"] = self.reward_model_tokenizer

        if data_source in _RESEARCH_DATA_SOURCES:
            try:
                score_t = self.re_search_compute_score(
                    solution_str,
                    ground_truth,
                    tokenizer=self.tokenizer,
                )
            except TypeError as exc:
                if self.re_search_function_path and "tokenizer" in str(exc):
                    score_t = self.re_search_compute_score(solution_str, ground_truth)
                else:
                    raise
            if isinstance(score_t, tuple):
                reward_f, reason = float(score_t[0]), str(score_t[1])
            else:
                reward_f, reason = float(score_t), ""
            extra: dict[str, Any] = {"acc": reward_f, "reason": reason}
            return reward_f, reason, extra

        if self.is_async_reward_score:
            raise RuntimeError("Async compute_score is not supported in re_search reward manager.")
        result = self.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **extra_reward_kwargs,
        )
        if isinstance(result, dict):
            reward_f = float(result["score"])
            extra = dict(result)
            reason = str(extra.get("reason", ""))
            return reward_f, reason, extra
        reward_f = float(result)
        return reward_f, "", {"acc": reward_f}

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        prompt_ids = data_item.batch["prompts"]
        if prompt_ids.ndim == 2:
            prompt_ids = prompt_ids[0]
        attention_mask = data_item.batch["attention_mask"]
        if attention_mask.ndim == 2:
            attention_mask = attention_mask[0]

        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = int(attention_mask[:prompt_length].sum().item())
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        if response_ids.ndim == 2:
            response_ids = response_ids[0]
        valid_response_length = int(attention_mask[prompt_length:].sum().item())
        valid_response_ids = response_ids[:valid_response_length]

        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(sequences, skip_special_tokens=False),
        )

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch["data_source"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info = {**extra_info, **tool_extra_fields}

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        reward_f, reason, reward_extra_info = await self.loop.run_in_executor(
            None,
            lambda: self._score_one(data_source, sequences_str, ground_truth, extra_info),
        )

        if self.save_path is not None:
            line = {
                "data_source": data_source,
                "sequences_str": sequences_str,
                "ground_truth": ground_truth,
                "score": reward_f,
                "reason": reason,
            }
            await self.loop.run_in_executor(
                None,
                lambda ln=line, p=self.save_path: _append_jsonl(p, ln),
            )

        if self.num_examine > 0:
            ds_key = str(data_source)
            cnt = self._examine_counts.get(ds_key, 0)
            if cnt < self.num_examine:
                self._examine_counts[ds_key] = cnt + 1
                print("-" * 20)
                print(f"data_source:\n{data_source}")
                print(f"sequences_str:\n{sequences_str}")
                print(f"ground_truth:\n{ground_truth}")
                print(f"score:\n{reward_f}")
                print(f"reason:\n{reason}")
                print("-" * 20)

        reward_extra_info.setdefault("acc", reward_f)
        return {"reward_score": reward_f, "reward_extra_info": reward_extra_info}


def _append_jsonl(path: str, obj: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
