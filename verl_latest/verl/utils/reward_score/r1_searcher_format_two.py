from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Block / payload patterns
# ---------------------------------------------------------------------------
_BLOCK_PATTERN = re.compile(r"<(think|tool_call|tool_response|answer)>(.*?)</\1>", re.DOTALL)
_TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>.*?</tool_response>", re.DOTALL)
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*)\}")


# ---------------------------------------------------------------------------
# Reward shape
# ---------------------------------------------------------------------------
# Stage-1 format-only reward, weighted to emphasise behaviours the agent loop
# does NOT teacher-force. Each ratchet is gated to require genuine model output;
# perfect trace = 1.00.
SCORE_FLOOR = -0.20
SCORE_CEIL = 1.00

# Positive milestones (sum to 1.00 at perfect trace)
FIRST_THINK_REWARD = 0.20             # first parsed block is <think> with non-empty body, closed
VALID_TOOL_CALL_REWARD = 0.20         # >= 1 tool_call AND every tool_call is valid search-JSON
POST_TOOL_THINK_CLOSED_REWARD = 0.30  # a non-empty <think>...</think> appears after a </tool_response>
BOXED_ANSWER_REWARD = 0.30            # exactly one <answer>, last block, after <think>, non-empty \boxed{}

# Small penalties
LANGUAGE_MIXING_PENALTY_MAX = 0.10    # continuous: scale * non-Latin letter fraction (model text only)
NO_BLOCKS_PENALTY = 0.10              # flat: zero structural blocks parsed at all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_response_text(solution_str: str) -> str:
    """Accept full chat decodes and response-only strings."""
    if "<|im_start|>assistant\n" in solution_str:
        return solution_str.split("<|im_start|>assistant\n", 1)[1]
    if "Assistant:" in solution_str:
        return solution_str.split("Assistant:", 1)[1]
    return solution_str.strip()


def _strip_eos(response: str, tokenizer: Optional[Any]) -> tuple[Optional[str], str]:
    """Hard gate: response must end with tokenizer EOS token. Truncated rollouts get 0."""
    if tokenizer is None:
        return response, ""

    eos_token = getattr(tokenizer, "eos_token", None)
    if not eos_token:
        return response, ""

    if not response.endswith(eos_token):
        return None, "no_eos"
    return response[: -len(eos_token)], ""


def _restore_prefilled_think(response: str) -> str:
    """
    If the rollout begins inside a prefilled <think> body and later closes it
    with </think>, prepend <think> so parsing sees a well-formed first block.
    No-op when the response already opens with <think> or never closes one.
    """
    text = response.lstrip()
    if text.startswith("<think>"):
        return response
    open_idx = response.find("<think>")
    close_idx = response.find("</think>")
    if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
        return "<think>" + response
    return response


def _validate_search_tool_call(payload: str) -> tuple[bool, str]:
    payload = payload.strip()
    if not payload:
        return False, "empty_payload"
    try:
        function_call = json.loads(payload)
    except json.JSONDecodeError:
        return False, "invalid_json"
    if not isinstance(function_call, dict):
        return False, "not_object"
    if function_call.get("name") != "search":
        return False, "name_not_search"
    arguments = function_call.get("arguments")
    if not isinstance(arguments, dict):
        return False, "arguments_not_object"
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return False, "query_empty_or_non_string"
    return True, "ok"


def _language_mixing_fraction(text: str) -> float:
    """
    Fraction of letter characters outside Latin / Latin-Extended blocks,
    measured ONLY over model-natural-language regions. Tool call JSON and
    tool response payloads are stripped first, so the penalty reflects what
    the model wrote in <think> / <answer> rather than the retriever's output.
    """
    cleaned = _TOOL_RESPONSE_PATTERN.sub(" ", text)
    cleaned = _TOOL_CALL_PATTERN.sub(" ", cleaned)

    total_letters = 0
    non_latin_letters = 0
    for ch in cleaned:
        if not unicodedata.category(ch).startswith("L"):
            continue
        total_letters += 1
        if ord(ch) > 0x024F:
            non_latin_letters += 1

    if total_letters == 0:
        return 0.0
    return non_latin_letters / total_letters


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Trace analysis
# ---------------------------------------------------------------------------
@dataclass
class _Analysis:
    block_types: list[str]
    block_contents: list[str]

    # raw counts (kept so validate_format can reuse this dataclass)
    think_open: int
    think_close: int
    tool_call_open: int
    tool_call_close: int
    tool_response_open: int
    tool_response_close: int
    answer_open: int
    answer_close: int
    has_non_tag_text: bool

    # used by validate_format
    has_first_think_closed: bool
    has_valid_tool_call: bool
    has_tool_response_after_tool_call: bool
    has_post_tool_think: bool
    answer_is_last: bool
    answer_after_think: bool
    has_boxed_answer: bool
    boxed_content_nonempty: bool

    # used by compute_score (the simplified reward)
    first_block_is_nonempty_think: bool
    has_post_tool_think_closed: bool


def _analyze(text: str) -> _Analysis:
    def _count(tag: str) -> tuple[int, int]:
        return text.count(f"<{tag}>"), text.count(f"</{tag}>")

    think_open, think_close = _count("think")
    tool_call_open, tool_call_close = _count("tool_call")
    tool_response_open, tool_response_close = _count("tool_response")
    answer_open, answer_close = _count("answer")

    matches = list(_BLOCK_PATTERN.finditer(text))
    block_types = [m.group(1) for m in matches]
    block_contents = [m.group(2) for m in matches]

    cursor = 0
    has_non_tag_text = False
    for m in matches:
        if text[cursor : m.start()].strip():
            has_non_tag_text = True
            break
        cursor = m.end()
    if not has_non_tag_text and text[cursor:].strip():
        has_non_tag_text = True

    # Tool call validity (every tool_call must validate)
    tool_call_valid_count = 0
    for bt, content in zip(block_types, block_contents):
        if bt == "tool_call":
            ok, _reason = _validate_search_tool_call(content)
            if ok:
                tool_call_valid_count += 1
    has_valid_tool_call = tool_call_open > 0 and tool_call_valid_count == tool_call_open

    # Answer structure
    answer_indices = [i for i, bt in enumerate(block_types) if bt == "answer"]
    answer_count_is_one = len(answer_indices) == 1
    answer_is_last = answer_count_is_one and answer_indices[0] == len(block_types) - 1
    answer_after_think = (
        answer_count_is_one
        and answer_indices[0] > 0
        and block_types[answer_indices[0] - 1] == "think"
    )
    answer_content = block_contents[answer_indices[0]] if answer_count_is_one else ""
    boxed_match = _BOXED_PATTERN.search(answer_content)
    has_boxed_answer = boxed_match is not None
    boxed_content_nonempty = boxed_match is not None and bool(boxed_match.group(1).strip())

    has_first_think_closed = think_close >= 1

    # Open-only ordering (kept for validate_format)
    has_tool_response_after_tool_call = False
    has_post_tool_think = False
    for i in range(len(block_types) - 1):
        if block_types[i] == "tool_call" and block_types[i + 1] == "tool_response":
            has_tool_response_after_tool_call = True
        if (
            i + 2 < len(block_types)
            and block_types[i] == "tool_call"
            and block_types[i + 1] == "tool_response"
            and block_types[i + 2] == "think"
        ):
            has_post_tool_think = True

    # Reward-side gates
    first_block_is_nonempty_think = bool(
        block_types
        and block_types[0] == "think"
        and block_contents[0].strip()
    )

    # Post-tool-response think CLOSED with non-empty body. The agent loop
    # injects <think> after </tool_response> (when ``post_tool_think_prefill``
    # is on), but it does NOT emit </think> or fill the body. So this gate
    # rewards only what the model itself produced.
    first_tr_close = text.find("</tool_response>")
    has_post_tool_think_closed = False
    if first_tr_close >= 0:
        for m in matches:
            if (
                m.group(1) == "think"
                and m.start() > first_tr_close
                and m.group(2).strip()
            ):
                has_post_tool_think_closed = True
                break

    return _Analysis(
        block_types=block_types,
        block_contents=block_contents,
        think_open=think_open,
        think_close=think_close,
        tool_call_open=tool_call_open,
        tool_call_close=tool_call_close,
        tool_response_open=tool_response_open,
        tool_response_close=tool_response_close,
        answer_open=answer_open,
        answer_close=answer_close,
        has_non_tag_text=has_non_tag_text,
        has_first_think_closed=has_first_think_closed,
        has_valid_tool_call=has_valid_tool_call,
        has_tool_response_after_tool_call=has_tool_response_after_tool_call,
        has_post_tool_think=has_post_tool_think,
        answer_is_last=answer_is_last,
        answer_after_think=answer_after_think,
        has_boxed_answer=has_boxed_answer,
        boxed_content_nonempty=boxed_content_nonempty,
        first_block_is_nonempty_think=first_block_is_nonempty_think,
        has_post_tool_think_closed=has_post_tool_think_closed,
    )


# ---------------------------------------------------------------------------
# Eval-time binary checker (kept for downstream validators)
# ---------------------------------------------------------------------------
def validate_format(text: str) -> tuple[bool, str]:
    a = _analyze(text)

    if not a.block_types:
        return False, "no_blocks"
    if a.think_open != a.think_close or a.think_open == 0:
        return False, "<think> tags are missing or not paired"
    if a.tool_call_open != a.tool_call_close:
        return False, "<tool_call> tags are not paired"
    if a.tool_response_open != a.tool_response_close:
        return False, "<tool_response> tags are not paired"
    if a.answer_open != 1 or a.answer_close != 1:
        return False, "<answer> must appear exactly once"
    if a.has_non_tag_text:
        return False, "text outside tags is not allowed"
    if not a.has_first_think_closed:
        return False, "first think is not closed"
    if not a.has_valid_tool_call:
        return False, "tool call is missing or invalid"
    if not a.has_tool_response_after_tool_call:
        return False, "tool_response must follow tool_call"
    if not a.has_post_tool_think:
        return False, "tool_response must be followed by think"
    if not a.answer_is_last:
        return False, "answer must be the final block"
    if not a.answer_after_think:
        return False, "answer must come after think"
    if not a.has_boxed_answer or not a.boxed_content_nonempty:
        return False, "answer must contain non-empty \\boxed{}"
    return True, "format is correct"


# ---------------------------------------------------------------------------
# Training-time reward
# ---------------------------------------------------------------------------
def compute_score(
    solution_str: str,
    ground_truth: Any,
    tokenizer: Optional[Any] = None,
) -> tuple[float, str]:
    """
    Stage-1 format-only reward, simplified.

    Aligned to:
      1. small penalty for language mixing
      2. small penalty for empty / no-block responses
      3. first <think> completed and closed (with non-empty body) -> reward
      4. tool_call valid (every <tool_call> validates) -> reward
      5. <think> closed AFTER <tool_response> (model-emitted, not teacher-forced) -> reward
      6. exactly one <answer>, last block, after <think>, non-empty \\boxed{} -> reward

    EOS gate: a rollout that does not end with the tokenizer EOS scores 0
    (truncated traces are not credited).
    """
    del ground_truth

    response = _extract_response_text(solution_str)
    response, eos_reason = _strip_eos(response, tokenizer)
    if response is None:
        return 0.0, f"no_reward:{eos_reason}"

    response = _restore_prefilled_think(response)

    if not response.strip():
        return SCORE_FLOOR, "empty_response"

    a = _analyze(response)

    score = 0.0
    reasons: list[str] = []

    # (2) Empty-block penalty: zero structural blocks at all
    if not a.block_types:
        score -= NO_BLOCKS_PENALTY
        reasons.append("no_blocks")

    # (3) First think completed and closed
    if a.first_block_is_nonempty_think:
        score += FIRST_THINK_REWARD
    else:
        reasons.append("no_first_think")

    # (4) Tool call valid
    if a.has_valid_tool_call:
        score += VALID_TOOL_CALL_REWARD
    else:
        reasons.append("invalid_or_missing_tool_call")

    # (5) Post-tool-response think CLOSED (the only model-emitted part of the cycle)
    if a.has_post_tool_think_closed:
        score += POST_TOOL_THINK_CLOSED_REWARD
    else:
        reasons.append("no_post_tool_close_think")

    # (6) Boxed answer at the end, after a think
    if (
        a.answer_is_last
        and a.answer_after_think
        and a.has_boxed_answer
        and a.boxed_content_nonempty
    ):
        score += BOXED_ANSWER_REWARD
    else:
        reasons.append("no_terminal_boxed_answer")

    # (1) Language mixing penalty (continuous, capped at LANGUAGE_MIXING_PENALTY_MAX)
    mixing_fraction = _language_mixing_fraction(response)
    mixing_penalty = LANGUAGE_MIXING_PENALTY_MAX * mixing_fraction
    score -= mixing_penalty
    if mixing_fraction > 0.05:
        reasons.append("language_mixing")

    score = _clamp(score, SCORE_FLOOR, SCORE_CEIL)

    return (
        score,
        (
            f"score={score:.3f} "
            f"mix_frac={mixing_fraction:.3f} "
            f"mix_pen={mixing_penalty:.3f} "
            f"reason={','.join(reasons) if reasons else 'ok'}"
        ),
    )
