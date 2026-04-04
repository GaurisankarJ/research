# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""ReSearch rollout: stop at ``</search>``, call HTTP retriever, inject ``<result>...</result>`` (legacy ``vLLMRolloutWithSearch``).

Text-only (no multimodal processor). Configure ``actor_rollout_ref.rollout.search_url`` and set
``actor_rollout_ref.rollout.agent.default_agent_loop=re_search_agent``.
"""

from __future__ import annotations

import logging
import os
import time
from functools import wraps
from typing import Any
from uuid import uuid4

from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def retry(max_attempts: int = 5, sleep_s: int = 1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if i == max_attempts - 1:
                        logger.exception("%s failed after %s attempts", func.__name__, max_attempts)
                        raise
                    time.sleep(sleep_s)

        return wrapper

    return decorator


@retry(max_attempts=5, sleep_s=1)
def _batch_search_http(search_url: str, queries: list[str], top_n: int) -> list[str]:
    import requests

    if len(queries) == 0:
        return []
    url = f"{search_url.rstrip('/')}/batch_search"
    resp = requests.post(url, json={"query": queries, "top_n": top_n}, timeout=120)
    resp.raise_for_status()
    result_list: list[str] = []
    for item in resp.json():
        curr = ""
        for line in item:
            curr += f"{line['contents']}\n\n"
        result_list.append(curr.strip())
    return result_list


def extract_search_content(text: str) -> str:
    try:
        start_tag = "<search>"
        end_tag = "</search>"
        end_pos = text.rindex(end_tag)
        start_pos = text.rindex(start_tag, 0, end_pos)
        return text[start_pos + len(start_tag) : end_pos].strip()
    except ValueError:
        return ""


def _decode_token_pieces(tokenizer: Any, token_ids: list[int]) -> list[str]:
    """One decoded string per token id (same order as ``token_ids``)."""
    pieces: list[str] = []
    for tid in token_ids:
        try:
            pieces.append(tokenizer.decode([tid], skip_special_tokens=False))
        except Exception:
            pieces.append(f"<decode_err:{tid}>")
    return pieces


def classify_last_segment_no_valid_search(seg_text: str) -> str:
    """Why ``need_search`` is false for this LM segment (paired ``<search>q</search>`` with non-empty ``q``)."""
    has_close = "</search>" in seg_text
    has_open = "<search>" in seg_text
    if not has_close:
        return "open_search_without_close" if has_open else "no_search_close_tag"
    raw = extract_search_content(seg_text)
    if raw:
        return "valid_search_pair_unexpected"
    if has_open:
        return "malformed_or_empty_search_query"
    return "close_search_without_prior_open_pair"


def _digest_response_text(text: str, *, preview_chars: int = 160) -> dict[str, Any]:
    """Structure / format hints for the full decoded rollout response (prompt excluded)."""
    n = len(text)

    def paired(open_tag: str, close_tag: str) -> bool:
        return open_tag in text and close_tag in text

    tail = text[-preview_chars:] if n > preview_chars else text
    head = text[:preview_chars] if n > preview_chars else text
    return {
        "response_chars": n,
        "n_lm_search_close": text.count("</search>"),
        "n_injected_result_close": text.count("</result>"),
        "thinking_paired": paired("<redacted_thinking>", "</redacted_thinking>"),
        "answer_paired": paired("<answer>", "</answer>"),
        "answer_open_only": "<answer>" in text and "</answer>" not in text,
        "answer_close_only": "</answer>" in text and "<answer>" not in text,
        "search_tags_present": "<search>" in text or "</search>" in text,
        "result_tags_present": "<result>" in text or "</result>" in text,
        "has_boxed": "\\boxed" in text,
        "head_preview": head,
        "tail_preview": tail,
    }


def _merge_digest(
    base: dict[str, Any],
    *,
    response_tokens_emitted: int,
    response_budget_tokens: int,
    search_http_calls: int,
    last_lm_segment_chars: int | None,
    last_segment_no_search_class: str | None,
    termination_code: str,
) -> dict[str, Any]:
    out = {
        **base,
        "termination_code": termination_code,
        "response_tokens_emitted": response_tokens_emitted,
        "response_budget_tokens": response_budget_tokens,
        "at_response_budget_cap": response_tokens_emitted >= response_budget_tokens,
        "search_http_calls": search_http_calls,
    }
    if last_lm_segment_chars is not None:
        out["last_lm_segment_chars"] = last_lm_segment_chars
    if last_segment_no_search_class is not None:
        out["last_segment_no_valid_search_class"] = last_segment_no_search_class
    return out


def _format_termination_reason(code: str, digest: dict[str, Any]) -> str:
    """Single log line: code + compact facts (full detail lives in ``re_search_response_digest``)."""
    parts = [
        code,
        f"tok={digest['response_tokens_emitted']}/{digest['response_budget_tokens']}",
        f"lm_tok={digest.get('lm_output_token_count')}",
        f"injected_tok={digest.get('injected_result_token_count')}",
        f"search_calls={digest['search_http_calls']}",
        f"answer_paired={digest.get('answer_paired')}",
        f"thinking_paired={digest.get('thinking_paired')}",
        f"chars={digest.get('response_chars')}",
    ]
    if digest.get("last_segment_no_valid_search_class"):
        parts.append(f"last_seg_no_search={digest['last_segment_no_valid_search_class']}")
    if digest.get("last_lm_end_token_id") is not None:
        parts.append(f"end_tok={digest['last_lm_end_token_id']}")
    if digest.get("last_lm_stop_reason") is not None:
        parts.append(f"lm_stop={digest['last_lm_stop_reason']}")
    if digest.get("at_response_budget_cap"):
        parts.append("at_budget_cap")
    return " | ".join(parts)


@register("re_search_agent")
class ReSearchAgentLoop(AgentLoopBase):
    """Multi-segment generation with ``</search>`` stops and HTTP retrieval (legacy port)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.search_url = OmegaConf.select(self.config, "actor_rollout_ref.rollout.search_url", default=None)
        self.search_top_n = int(OmegaConf.select(self.config, "actor_rollout_ref.rollout.search_top_n", default=5))
        self.search_max_turns = int(
            OmegaConf.select(self.config, "actor_rollout_ref.rollout.search_max_turns", default=32)
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        if self.processor is not None:
            raise NotImplementedError(
                "re_search_agent currently supports tokenizer-only (text) rollouts; use single_turn_agent for VLM."
            )
        if not self.search_url:
            raise ValueError(
                "re_search_agent requires actor_rollout_ref.rollout.search_url (HTTP retriever base URL, "
                "same as legacy config.search_url)."
            )

        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        initial_prompt_ids = await self.tokenize_prompt_for_rollout(messages, images=images, videos=videos)
        full_context: list[int] = list(initial_prompt_ids)

        response_tokens: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []

        timing: dict[str, Any] = {}
        request_id = str(kwargs.get("uid", uuid4().hex))
        search_segments = 0
        num_preempted = -1

        base_sp = {**sampling_params}

        # Short code for why the loop stopped; finalized into ``re_search_termination_reason`` + digest after decode.
        termination_code = "in_progress"
        last_lm_segment_text: str | None = None
        last_segment_no_search_class: str | None = None
        last_lm_end_token_id: int | None = None
        last_lm_stop_reason: str | None = None
        last_lm_segment_token_ids: list[int] | None = None

        with simple_timer("generate_sequences", timing):
            while len(response_tokens) < self.response_length and search_segments < self.search_max_turns:
                remaining = self.response_length - len(response_tokens)
                if remaining <= 0:
                    termination_code = "response_budget_exhausted_before_generate"
                    break

                seg_params = {
                    **base_sp,
                    "max_tokens": remaining,
                    "stop": ["</search>"],
                }

                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=full_context,
                    sampling_params=seg_params,
                    image_data=images,
                    video_data=videos,
                )
                if output.num_preempted is not None:
                    num_preempted = output.num_preempted

                new_ids = list(output.token_ids)
                if not new_ids:
                    termination_code = "empty_model_output"
                    break

                last_lm_end_token_id = new_ids[-1]
                last_lm_stop_reason = output.stop_reason
                last_lm_segment_token_ids = list(new_ids)

                seg_text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
                last_lm_segment_text = seg_text
                need_search = "</search>" in seg_text and extract_search_content(seg_text) != ""

                response_tokens.extend(new_ids)
                response_mask.extend([1] * len(new_ids))
                if output.log_probs:
                    response_logprobs.extend(output.log_probs)
                else:
                    response_logprobs.extend([0.0] * len(new_ids))
                full_context = full_context + new_ids

                if not need_search:
                    termination_code = "segment_completed_without_valid_search"
                    last_segment_no_search_class = classify_last_segment_no_valid_search(seg_text)
                    break

                queries = [extract_search_content(seg_text)]
                results = await self.loop.run_in_executor(
                    None,
                    lambda q=queries: _batch_search_http(self.search_url, q, self.search_top_n),
                )
                result_text = results[0] if results else ""
                result_suffix = f" <result>\n{result_text}\n</result>"
                result_ids = self.tokenizer.encode(result_suffix, add_special_tokens=False)

                response_tokens.extend(result_ids)
                response_mask.extend([0] * len(result_ids))
                response_logprobs.extend([0.0] * len(result_ids))
                full_context = full_context + result_ids
                search_segments += 1

        if termination_code == "in_progress":
            if len(response_tokens) >= self.response_length:
                termination_code = "hit_response_length_cap_after_loop"
            elif search_segments >= self.search_max_turns:
                termination_code = "hit_search_max_turns"
            else:
                termination_code = "loop_finished"

        if len(response_tokens) > self.response_length:
            response_tokens = response_tokens[: self.response_length]
            response_mask = response_mask[: self.response_length]
            response_logprobs = response_logprobs[: self.response_length]

        clipped = response_tokens
        full_response_text = self.tokenizer.decode(clipped, skip_special_tokens=False)
        text_digest = _digest_response_text(full_response_text)
        merged_digest = _merge_digest(
            text_digest,
            response_tokens_emitted=len(clipped),
            response_budget_tokens=int(self.response_length),
            search_http_calls=int(search_segments),
            last_lm_segment_chars=len(last_lm_segment_text) if last_lm_segment_text is not None else None,
            last_segment_no_search_class=last_segment_no_search_class,
            termination_code=termination_code,
        )
        lm_tok = int(sum(response_mask))
        merged_digest["lm_output_token_count"] = lm_tok
        merged_digest["injected_result_token_count"] = len(clipped) - lm_tok
        if last_lm_segment_text:
            merged_digest["last_lm_segment_head_preview"] = last_lm_segment_text[:220]
            merged_digest["last_lm_segment_tail_preview"] = last_lm_segment_text[-220:]
        if last_lm_end_token_id is not None:
            merged_digest["last_lm_end_token_id"] = last_lm_end_token_id
        if last_lm_stop_reason is not None:
            merged_digest["last_lm_stop_reason"] = last_lm_stop_reason
        if last_lm_end_token_id is not None:
            try:
                piece = self.tokenizer.decode([last_lm_end_token_id], skip_special_tokens=False)
                merged_digest["last_lm_end_token_piece"] = piece if len(piece) <= 64 else piece[:61] + "..."
            except Exception:
                merged_digest["last_lm_end_token_piece"] = None
        if last_lm_segment_token_ids is not None:
            merged_digest["last_lm_segment_token_ids"] = last_lm_segment_token_ids
            merged_digest["last_lm_segment_decoded_pieces"] = _decode_token_pieces(
                self.tokenizer, last_lm_segment_token_ids
            )

        termination_reason = _format_termination_reason(termination_code, merged_digest)

        metrics = AgentLoopMetrics(
            generate_sequences=float(timing.get("generate_sequences", 0.0)),
            num_preempted=num_preempted,
        )

        return AgentLoopOutput(
            prompt_ids=initial_prompt_ids,
            response_ids=response_tokens[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=None,
            multi_modal_data=multi_modal_data,
            num_turns=2 + search_segments,
            metrics=metrics,
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                # HTTP /batch_search invocations this trajectory (aggregated in train metrics as tool_call_counts/*).
                "tool_call_counts": int(search_segments),
                "re_search_termination_reason": termination_reason,
                "re_search_response_digest": merged_digest,
                "re_search_response_token_count": len(response_tokens),
                "re_search_budget_tokens": int(self.response_length),
                "re_search_search_segments": int(search_segments),
            },
        )
