from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Optional

_BLOCK_PATTERN = re.compile(r"<(think|tool_call|tool_response|answer)>(.*?)</\1>", re.DOTALL)
_TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>.*?</tool_response>", re.DOTALL)
_BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*)\}")

TIER1_FIRST_THINK = 0.10
TIER2_TOOL_CALL = 0.20
TIER3_POST_TOOL_THINK = 0.20
TIER4_ANSWER_BLOCK = 0.20
TIER5_BOXED_CONTENT = 0.30

MIXING_PENALTY_MAX = 0.50
MIXING_FLOOR = -0.50
SCORE_FLOOR = -0.50
SCORE_CEIL = 1.00


def _extract_response_text(solution_str: str) -> str:
    """Accept full chat decodes and response-only strings."""
    if "<|im_start|>assistant\n" in solution_str:
        return solution_str.split("<|im_start|>assistant\n", 1)[1]
    if "Assistant:" in solution_str:
        return solution_str.split("Assistant:", 1)[1]
    return solution_str.strip()


def _strip_eos(response: str, tokenizer: Optional[Any]) -> tuple[Optional[str], str]:
    """Hard gate: response must end with the tokenizer EOS token (ReSearch-style)."""
    if tokenizer is None:
        return response, ""

    eos_token = getattr(tokenizer, "eos_token", None)
    if not eos_token:
        return response, ""

    if not response.endswith(eos_token):
        return None, "no_eos"
    return response[: -len(eos_token)], ""


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
    if not isinstance(query, str):
        return False, "query_not_string"

    if not query.strip():
        return False, "query_empty"

    return True, "ok"


def _language_mixing_fraction(text: str) -> float:
    """Fraction of characters that belong to non-Latin, non-symbol scripts.

    Tool response content is stripped first so retriever-injected text does not
    get blamed on the model. We only count characters with Unicode category
    starting with ``L`` (letters); punctuation, digits, and whitespace do not
    contribute to either numerator or denominator. Latin / Latin-Extended
    letters count as English-compatible; anything else (CJK, Cyrillic, Arabic,
    Devanagari, Hangul, Thai, Hebrew, Greek beyond ASCII, ...) is "mixed".
    """
    cleaned = _TOOL_RESPONSE_PATTERN.sub(" ", text)
    total_letters = 0
    non_latin_letters = 0
    for ch in cleaned:
        category = unicodedata.category(ch)
        if not category.startswith("L"):
            continue
        total_letters += 1
        cp = ord(ch)
        if cp <= 0x024F:
            continue
        non_latin_letters += 1
    if total_letters == 0:
        return 0.0
    return non_latin_letters / total_letters


def _analyze(text: str) -> dict[str, Any]:
    blocks = list(_BLOCK_PATTERN.finditer(text))
    block_types = [m.group(1) for m in blocks]
    block_contents = [m.group(2) for m in blocks]

    result: dict[str, Any] = {
        "has_first_think": False,
        "has_valid_tool_call": False,
        "tool_call_reason": "absent",
        "has_post_tool_think": False,
        "has_answer_block": False,
        "answer_reason": "absent",
        "has_boxed": False,
        "boxed_content": "",
    }

    if not block_types:
        return result

    if block_types[0] == "think" and block_contents[0].strip():
        result["has_first_think"] = True

    if not result["has_first_think"]:
        return result

    first_tool_call_idx = None
    for idx, bt in enumerate(block_types):
        if bt == "tool_call":
            first_tool_call_idx = idx
            break

    if first_tool_call_idx is None:
        result["tool_call_reason"] = "absent"
        return result

    if first_tool_call_idx != 1:
        result["tool_call_reason"] = "not_directly_after_first_think"
        return result

    ok, why = _validate_search_tool_call(block_contents[first_tool_call_idx])
    if not ok:
        result["tool_call_reason"] = why
        return result

    if first_tool_call_idx + 1 >= len(block_types) or block_types[first_tool_call_idx + 1] != "tool_response":
        result["tool_call_reason"] = "no_tool_response_after"
        return result

    result["has_valid_tool_call"] = True
    result["tool_call_reason"] = "ok"

    post_tool_think_idx = first_tool_call_idx + 2
    if post_tool_think_idx >= len(block_types):
        return result
    if block_types[post_tool_think_idx] != "think":
        return result
    if not block_contents[post_tool_think_idx].strip():
        return result
    result["has_post_tool_think"] = True

    answer_indices = [i for i, bt in enumerate(block_types) if bt == "answer"]
    if len(answer_indices) != 1:
        result["answer_reason"] = "not_unique" if len(answer_indices) > 1 else "absent"
        return result
    answer_idx = answer_indices[0]
    if answer_idx != len(block_types) - 1:
        result["answer_reason"] = "not_terminal"
        return result
    if block_types[answer_idx - 1] != "think":
        result["answer_reason"] = "not_after_think"
        return result
    result["has_answer_block"] = True
    result["answer_reason"] = "ok"

    boxed_match = _BOXED_PATTERN.search(block_contents[answer_idx])
    if boxed_match is None:
        return result
    boxed_content = boxed_match.group(1).strip()
    if not boxed_content:
        return result
    result["has_boxed"] = True
    result["boxed_content"] = boxed_content
    return result


def compute_score(
    solution_str: str,
    ground_truth: Any,
    tokenizer: Optional[Any] = None,
) -> tuple[float, str]:
    """Cascaded format reward for ReSearch-style base-model training.

    Tiers (strict cascade; each tier only scores if all previous tiers scored):
      1. First ``<think>…</think>`` block with non-empty content.
      2. Valid ``<tool_call>`` (JSON, name="search", arguments.query non-empty)
         directly following the first think and followed by ``<tool_response>``.
      3. A second ``<think>…</think>`` block immediately after that
         ``<tool_response>`` with non-empty content.
      4. Exactly one ``<answer>…</answer>`` block that is the final block and
         preceded by a think block.
      5. ``\\boxed{…}`` with non-empty content inside the answer.

    A language-mixing penalty (proportional to the fraction of non-Latin letters
    in the model-generated text, tool_response excluded) is subtracted. Missing
    EOS is a hard 0.0 — no tiers, no penalty.
    """
    del ground_truth

    response = _extract_response_text(solution_str)
    response, eos_reason = _strip_eos(response, tokenizer)
    if response is None:
        return 0.0, f"no_reward:{eos_reason}"

    analysis = _analyze(response)

    tier_score = 0.0
    tiers_passed: list[str] = []
    stop_reason: Optional[str] = None

    if analysis["has_first_think"]:
        tier_score += TIER1_FIRST_THINK
        tiers_passed.append("t1")
    else:
        stop_reason = "missing_first_think"

    if stop_reason is None:
        if analysis["has_valid_tool_call"]:
            tier_score += TIER2_TOOL_CALL
            tiers_passed.append("t2")
        else:
            stop_reason = f"tool_call:{analysis['tool_call_reason']}"

    if stop_reason is None:
        if analysis["has_post_tool_think"]:
            tier_score += TIER3_POST_TOOL_THINK
            tiers_passed.append("t3")
        else:
            stop_reason = "missing_post_tool_think"

    if stop_reason is None:
        if analysis["has_answer_block"]:
            tier_score += TIER4_ANSWER_BLOCK
            tiers_passed.append("t4")
        else:
            stop_reason = f"answer:{analysis['answer_reason']}"

    if stop_reason is None:
        if analysis["has_boxed"]:
            tier_score += TIER5_BOXED_CONTENT
            tiers_passed.append("t5")
        else:
            stop_reason = "missing_boxed"

    mixing_fraction = _language_mixing_fraction(response)
    mixing_penalty = MIXING_PENALTY_MAX * mixing_fraction
    raw_score = tier_score - mixing_penalty
    score = max(SCORE_FLOOR, min(SCORE_CEIL, raw_score))

    passed_str = ",".join(tiers_passed) if tiers_passed else "none"
    stop_str = stop_reason if stop_reason is not None else "ok"
    reason = (
        f"tiers={passed_str} stop={stop_str} "
        f"mix_frac={mixing_fraction:.3f} mix_pen={mixing_penalty:.3f} "
        f"tier_score={tier_score:.3f} score={score:.3f}"
    )
    return score, reason
