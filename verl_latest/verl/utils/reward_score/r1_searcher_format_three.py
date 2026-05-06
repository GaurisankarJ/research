from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Block / payload patterns
# ---------------------------------------------------------------------------
_BLOCK_PATTERN = re.compile(r"<(think|tool_call|tool_response|answer)>(.*?)</\1>", re.DOTALL)
_BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*)\}")


# ---------------------------------------------------------------------------
# Reward shape
# ---------------------------------------------------------------------------
# Hybrid-instruct format reward.
#
# Design goals:
#   * Hybrid instruct already thinks reliably and does not mix languages, so
#     there is no first-think reward and no language-mixing penalty.
#   * Every component is BINARY and invariant to the number of
#     (tool_call -> tool_response -> post-think) cycles. 1 well-formed cycle
#     and N well-formed cycles earn the same score.
#   * Every gate is "all cycles well-formed", never "any". That makes spam
#     strictly bad: adding a single malformed cycle to an otherwise valid
#     trace collapses the gate, so the model cannot inflate reward by
#     producing more (or noisier) cycles.
#
# Perfect trace = 1.00. Floor = 0.00.
SCORE_FLOOR = 0.0
SCORE_CEIL = 1.00

# Positive milestones (sum to 1.00 at perfect trace)
CYCLE_REWARD = 0.30           # >= 1 cycle, every cycle is think -> tool_call -> tool_response
POST_TOOL_THINK_REWARD = 0.30 # every tool_response is immediately followed by a non-empty closed <think>
BOXED_ANSWER_REWARD = 0.40    # exactly one <answer>, last block, after non-empty <think>, non-empty \boxed{}


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

    Needed even though we do not reward the first think directly: the
    boxed-answer gate checks that the block immediately preceding <answer>
    is a <think>, and on single-think traces that think may be the prefilled one.
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


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Trace analysis
# ---------------------------------------------------------------------------
@dataclass
class _Analysis:
    block_types: list[str]
    block_contents: list[str]

    # Raw tag counts
    think_open: int
    think_close: int
    tool_call_open: int
    tool_call_close: int
    tool_response_open: int
    tool_response_close: int
    answer_open: int
    answer_close: int
    has_non_tag_text: bool

    # Strict checks / reusable facts
    has_first_think_closed: bool
    all_tool_calls_valid: bool
    answer_is_last: bool
    answer_after_think: bool
    has_boxed_answer: bool
    boxed_content_nonempty: bool

    # Reward / validator gates
    cycle_well_formed: bool
    every_tool_response_has_post_think: bool
    terminal_boxed_answer: bool


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
        if text[cursor:m.start()].strip():
            has_non_tag_text = True
            break
        cursor = m.end()
    if not has_non_tag_text and text[cursor:].strip():
        has_non_tag_text = True

    # Every tool_call must validate as search-JSON.
    tool_call_valid_count = 0
    for bt, content in zip(block_types, block_contents):
        if bt == "tool_call":
            ok, _reason = _validate_search_tool_call(content)
            if ok:
                tool_call_valid_count += 1
    all_tool_calls_valid = tool_call_open > 0 and tool_call_valid_count == tool_call_open

    # Gate A: cycle_well_formed
    #   * at least one tool_call
    #   * every tool_call valid
    #   * every tool_call is immediately preceded by think
    #   * every tool_call is immediately followed by tool_response
    #   * every tool_response is immediately preceded by tool_call
    cycle_pairing_ok = True
    for i, bt in enumerate(block_types):
        if bt == "tool_call":
            if i == 0 or block_types[i - 1] != "think":
                cycle_pairing_ok = False
                break
            if i + 1 >= len(block_types) or block_types[i + 1] != "tool_response":
                cycle_pairing_ok = False
                break
        elif bt == "tool_response":
            if i == 0 or block_types[i - 1] != "tool_call":
                cycle_pairing_ok = False
                break

    cycle_well_formed = (
        all_tool_calls_valid
        and tool_call_open > 0
        and tool_response_open == tool_call_open
        and cycle_pairing_ok
    )

    # Gate B: every tool_response is immediately followed by a non-empty think.
    every_tool_response_has_post_think = cycle_well_formed
    if every_tool_response_has_post_think:
        for i, bt in enumerate(block_types):
            if bt != "tool_response":
                continue
            if i + 1 >= len(block_types) or block_types[i + 1] != "think":
                every_tool_response_has_post_think = False
                break
            if not block_contents[i + 1].strip():
                every_tool_response_has_post_think = False
                break

    # Answer structure
    answer_indices = [i for i, bt in enumerate(block_types) if bt == "answer"]
    answer_count_is_one = len(answer_indices) == 1
    answer_is_last = answer_count_is_one and answer_indices[0] == len(block_types) - 1
    answer_after_think = (
        answer_count_is_one
        and answer_indices[0] > 0
        and block_types[answer_indices[0] - 1] == "think"
        and bool(block_contents[answer_indices[0] - 1].strip())
    )
    answer_content = block_contents[answer_indices[0]] if answer_count_is_one else ""
    boxed_match = _BOXED_PATTERN.search(answer_content)
    has_boxed_answer = boxed_match is not None
    boxed_content_nonempty = boxed_match is not None and bool(boxed_match.group(1).strip())

    # Optional strictness: answer reward only fires if the trace already contains
    # a real well-formed tool cycle.
    terminal_boxed_answer = (
        cycle_well_formed
        and answer_is_last
        and answer_after_think
        and has_boxed_answer
        and boxed_content_nonempty
    )

    has_first_think_closed = think_close >= 1

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
        all_tool_calls_valid=all_tool_calls_valid,
        answer_is_last=answer_is_last,
        answer_after_think=answer_after_think,
        has_boxed_answer=has_boxed_answer,
        boxed_content_nonempty=boxed_content_nonempty,
        cycle_well_formed=cycle_well_formed,
        every_tool_response_has_post_think=every_tool_response_has_post_think,
        terminal_boxed_answer=terminal_boxed_answer,
    )


# ---------------------------------------------------------------------------
# Eval-time binary checker
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
    if not a.cycle_well_formed:
        return False, "all tool_call/tool_response cycles must be well formed"
    if not a.every_tool_response_has_post_think:
        return False, "every tool_response must be followed by a non-empty think"
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
    Hybrid-instruct format-only reward.

    Three binary, multi-chain-invariant gates:
      A. cycle_well_formed (0.30):
           >= 1 <tool_call>, every <tool_call> valid search-JSON,
           every <tool_call> is immediately preceded by <think>,
           every <tool_call> is immediately followed by <tool_response>,
           and every <tool_response> is immediately preceded by <tool_call>.
      B. every_tool_response_has_post_think (0.30):
           every <tool_response> is immediately followed by a non-empty
           closed <think> block.
      C. terminal_boxed_answer (0.40):
           exactly one <answer>, last block, preceded by a non-empty <think>,
           containing a non-empty \\boxed{...}, and only after a real cycle.

    Invariance:
      1 valid cycle and N valid cycles earn the same score.
      Adding one malformed cycle collapses gate A and/or B.

    Additional strictness:
      text outside tags gets 0 reward immediately.

    EOS gate:
      a rollout without final EOS gets 0.
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

    if a.has_non_tag_text:
        return 0.0, "text_outside_tags"

    score = 0.0
    earned: list[str] = []
    missed: list[str] = []

    if a.cycle_well_formed:
        score += CYCLE_REWARD
        earned.append("cycle")
    else:
        missed.append("cycle")

    if a.every_tool_response_has_post_think:
        score += POST_TOOL_THINK_REWARD
        earned.append("post_tool_think")
    else:
        missed.append("post_tool_think")

    if a.terminal_boxed_answer:
        score += BOXED_ANSWER_REWARD
        earned.append("boxed_answer")
    else:
        missed.append("boxed_answer")

    score = _clamp(score, SCORE_FLOOR, SCORE_CEIL)

    return (
        score,
        (
            f"score={score:.3f} "
            f"cycles={a.tool_call_open} "
            f"earned={','.join(earned) if earned else 'none'} "
            f"missed={','.join(missed) if missed else 'none'}"
        ),
    )