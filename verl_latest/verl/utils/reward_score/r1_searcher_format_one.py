from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

_BLOCK_PATTERN = re.compile(r"<(think|tool_call|tool_response|answer)>(.*?)</\1>", re.DOTALL)
_TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>.*?</tool_response>", re.DOTALL)
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*)\}")


# Final reward range
SCORE_FLOOR = -0.50
SCORE_CEIL = 1.00

# Language-mixing penalty
MIXING_PENALTY_MAX = 0.50

# Component weights
PAIRING_WEIGHT = 0.12
TRANSITION_WEIGHT = 0.34
CYCLE_WEIGHT = 0.22
TOOL_JSON_WEIGHT = 0.12
ANSWER_WEIGHT = 0.10
THINK_QUALITY_WEIGHT = 0.10

# Degeneration penalties
REPEATED_QUERY_PENALTY = 0.04
REPEATED_THINK_PENALTY = 0.04
EXCESSIVE_LOOP_PENALTY = 0.04

# Cold-start bonus: isolated credit for emitting the very first structural token.
# Added on top of the weighted format_score (still capped by SCORE_CEIL).
FIRST_THINK_BONUS = 0.05

# Close-tag bonus: under a prefilled opening ``<think>`` (both the prompt prefill
# and the post-``</tool_response>`` teacher-forced prefill), the actual
# structural discriminator is whether the model CLOSES the think block with
# ``</think>``. Fires per ``</think>`` found in the response up to
# ``CLOSE_THINK_BONUS_MAX_COUNT`` occurrences, giving graded credit for:
#   - 1 close  → completed first think (0.05)
#   - 2 closes → completed post-tool-response think too (0.10)
# Independent of the block-match path so it still discriminates when the model
# produces a malformed outer shell but a clean inner close.
CLOSE_THINK_BONUS = 0.05
CLOSE_THINK_BONUS_MAX_COUNT = 2


def _extract_response_text(solution_str: str) -> str:
    """Accept full chat decodes and response-only strings."""
    if "<|im_start|>assistant\n" in solution_str:
        return solution_str.split("<|im_start|>assistant\n", 1)[1]
    if "Assistant:" in solution_str:
        return solution_str.split("Assistant:", 1)[1]
    return solution_str.strip()


def _strip_eos(response: str, tokenizer: Optional[Any]) -> tuple[Optional[str], str]:
    """Hard gate: response must end with tokenizer EOS token."""
    if tokenizer is None:
        return response, ""

    eos_token = getattr(tokenizer, "eos_token", None)
    if not eos_token:
        return response, ""

    if not response.endswith(eos_token):
        return None, "no_eos"
    return response[: -len(eos_token)], ""


def _restore_prefilled_think(response: str) -> str:
    """If the prompt prefilled an opening ``<think>`` (e.g. base-model cold-start),
    the rollout output starts inside the think body and contains a ``</think>``
    with no preceding ``<think>``. Prepend ``<think>`` so the parser sees a
    well-formed first block. Idempotent if the response already opens with
    ``<think>``.
    """
    text = response.lstrip()
    if text.startswith("<think>"):
        return response
    open_idx = response.find("<think>")
    close_idx = response.find("</think>")
    if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
        return "<think>" + response
    return response


def _validate_search_tool_call(payload: str) -> tuple[bool, str, str]:
    payload = payload.strip()
    if not payload:
        return False, "empty_payload", ""

    try:
        function_call = json.loads(payload)
    except json.JSONDecodeError:
        return False, "invalid_json", ""

    if not isinstance(function_call, dict):
        return False, "not_object", ""

    if function_call.get("name") != "search":
        return False, "name_not_search", ""

    arguments = function_call.get("arguments")
    if not isinstance(arguments, dict):
        return False, "arguments_not_object", ""

    query = arguments.get("query")
    if not isinstance(query, str):
        return False, "query_not_string", ""

    query = query.strip()
    if not query:
        return False, "query_empty", ""

    return True, "ok", query


def _language_mixing_fraction(text: str) -> float:
    """Fraction of letter characters outside Latin / Latin-Extended blocks.

    Both ``<tool_response>`` (retriever text) and ``<tool_call>`` (ASCII JSON
    payload the model is forced to emit) are stripped before counting so the
    fraction reflects the model's natural-language letters only and is not
    diluted by JSON ASCII when only ``<think>`` bodies drift multilingual.
    Only letters count toward the fraction.
    """
    cleaned = _TOOL_RESPONSE_PATTERN.sub(" ", text)
    cleaned = _TOOL_CALL_PATTERN.sub(" ", cleaned)
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


def _normalize_ws(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class _Analysis:
    block_types: list[str]
    block_contents: list[str]

    think_open: int
    think_close: int
    tool_call_open: int
    tool_call_close: int
    tool_response_open: int
    tool_response_close: int
    answer_open: int
    answer_close: int

    has_non_tag_text: bool

    starts_with_think: bool
    answer_count_is_one: bool
    answer_is_last: bool
    answer_after_think: bool
    has_boxed_answer: bool
    boxed_content_nonempty: bool

    tool_call_valid_count: int
    tool_call_invalid_reasons: list[str]
    tool_queries: list[str]

    think_blocks: list[str]
    empty_think_count: int
    repeated_think_count: int
    repeated_query_count: int

    valid_transition_count: int
    total_transition_checks: int
    sequence_violations: list[str]

    complete_cycle_count: int


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

    # Detect text outside valid blocks
    cursor = 0
    has_non_tag_text = False
    for m in matches:
        if text[cursor:m.start()].strip():
            has_non_tag_text = True
            break
        cursor = m.end()
    if not has_non_tag_text and text[cursor:].strip():
        has_non_tag_text = True

    think_blocks = [content for t, content in zip(block_types, block_contents) if t == "think"]

    empty_think_count = sum(1 for t in think_blocks if not t.strip())

    normalized_thinks = [_normalize_ws(t) for t in think_blocks if _normalize_ws(t)]
    repeated_think_count = 0
    if normalized_thinks:
        c = Counter(normalized_thinks)
        repeated_think_count = sum(max(0, v - 1) for v in c.values())

    tool_call_valid_count = 0
    tool_call_invalid_reasons: list[str] = []
    tool_queries: list[str] = []

    for t, content in zip(block_types, block_contents):
        if t != "tool_call":
            continue
        ok, reason, query = _validate_search_tool_call(content)
        if ok:
            tool_call_valid_count += 1
            tool_queries.append(query.strip().lower())
        else:
            tool_call_invalid_reasons.append(reason)

    repeated_query_count = 0
    if tool_queries:
        c = Counter(tool_queries)
        repeated_query_count = sum(max(0, v - 1) for v in c.values())

    starts_with_think = len(block_types) > 0 and block_types[0] == "think"

    answer_indices = [i for i, bt in enumerate(block_types) if bt == "answer"]
    answer_count_is_one = len(answer_indices) == 1
    answer_is_last = answer_count_is_one and answer_indices[0] == len(block_types) - 1
    answer_after_think = answer_count_is_one and answer_indices[0] > 0 and block_types[answer_indices[0] - 1] == "think"

    answer_content = block_contents[answer_indices[0]] if answer_count_is_one else ""
    boxed_match = _BOXED_PATTERN.search(answer_content)
    has_boxed_answer = boxed_match is not None
    boxed_content_nonempty = boxed_match is not None and bool(boxed_match.group(1).strip())

    valid_transition_count = 0
    total_transition_checks = 0
    sequence_violations: list[str] = []

    for i, bt in enumerate(block_types):
        prev_bt = block_types[i - 1] if i > 0 else None
        next_bt = block_types[i + 1] if i + 1 < len(block_types) else None

        if bt == "think":
            total_transition_checks += 1
            if next_bt in {"tool_call", "answer"}:
                valid_transition_count += 1
            else:
                sequence_violations.append("think_not_followed_by_tool_call_or_answer")

        elif bt == "tool_call":
            total_transition_checks += 2
            if prev_bt == "think":
                valid_transition_count += 1
            else:
                sequence_violations.append("tool_call_not_after_think")

            if next_bt == "tool_response":
                valid_transition_count += 1
            else:
                sequence_violations.append("tool_call_not_followed_by_tool_response")

        elif bt == "tool_response":
            total_transition_checks += 2
            if prev_bt == "tool_call":
                valid_transition_count += 1
            else:
                sequence_violations.append("tool_response_not_after_tool_call")

            if next_bt == "think":
                valid_transition_count += 1
            else:
                sequence_violations.append("tool_response_not_followed_by_think")

        elif bt == "answer":
            total_transition_checks += 1
            if prev_bt == "think":
                valid_transition_count += 1
            else:
                sequence_violations.append("answer_not_after_think")

    complete_cycle_count = 0
    for i in range(len(block_types) - 3):
        if block_types[i:i + 4] == ["think", "tool_call", "tool_response", "think"]:
            complete_cycle_count += 1

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
        starts_with_think=starts_with_think,
        answer_count_is_one=answer_count_is_one,
        answer_is_last=answer_is_last,
        answer_after_think=answer_after_think,
        has_boxed_answer=has_boxed_answer,
        boxed_content_nonempty=boxed_content_nonempty,
        tool_call_valid_count=tool_call_valid_count,
        tool_call_invalid_reasons=tool_call_invalid_reasons,
        tool_queries=tool_queries,
        think_blocks=think_blocks,
        empty_think_count=empty_think_count,
        repeated_think_count=repeated_think_count,
        repeated_query_count=repeated_query_count,
        valid_transition_count=valid_transition_count,
        total_transition_checks=total_transition_checks,
        sequence_violations=sequence_violations,
        complete_cycle_count=complete_cycle_count,
    )


def _pairing_score(a: _Analysis) -> tuple[float, list[str]]:
    """Require at least one well-paired block of each kind to earn credit.

    Earlier versions awarded credit when ``open == close`` even when both were
    zero, which gave any non-empty text a ~0.048 free floor after weighting.
    All bonuses now require ``open > 0`` so absence cannot be rewarded.
    """
    reasons: list[str] = []
    score = 0.0

    if a.think_open > 0 and a.think_open == a.think_close:
        score += 0.30
    else:
        reasons.append("bad_think_pairing")

    if a.tool_call_open > 0 and a.tool_call_open == a.tool_call_close:
        score += 0.20
    else:
        reasons.append("bad_tool_call_pairing")

    if a.tool_response_open > 0 and a.tool_response_open == a.tool_response_close:
        score += 0.20
    else:
        reasons.append("bad_tool_response_pairing")

    if a.answer_open == a.answer_close == 1:
        score += 0.20
    else:
        reasons.append("answer_count_not_one")

    # Only credit "no stray text" once at least one valid block exists, otherwise
    # an empty / blank response would collect this bonus too.
    if a.block_types and not a.has_non_tag_text:
        score += 0.10
    else:
        reasons.append("text_outside_tags")

    return _clamp(score, 0.0, 1.0), reasons


def _transition_score(a: _Analysis) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if not a.block_types:
        return 0.0, ["no_blocks"]

    score = 0.0

    if a.starts_with_think:
        score += 0.10
    else:
        reasons.append("not_start_with_think")

    if a.total_transition_checks > 0:
        local = a.valid_transition_count / a.total_transition_checks
        score += 0.90 * local
    else:
        reasons.append("no_transition_checks")

    if a.sequence_violations:
        reasons.extend(a.sequence_violations[:4])

    return _clamp(score, 0.0, 1.0), reasons


def _cycle_score(a: _Analysis) -> tuple[float, list[str]]:
    reasons: list[str] = []

    if a.complete_cycle_count == 0:
        return 0.0, ["no_complete_cycle"]

    score = 0.0
    score += 0.75  # strong reward for at least one full cycle

    # allow 2–3 cycles naturally; slight saturation after that
    if a.complete_cycle_count == 1:
        score += 0.15
    elif a.complete_cycle_count in {2, 3}:
        score += 0.25
    else:
        score += 0.20
        reasons.append("many_cycles")

    # if there are many cycles and repetition appears, treat as likely looping
    if a.complete_cycle_count >= 4 and (a.repeated_query_count > 0 or a.repeated_think_count > 0):
        score -= 0.15
        reasons.append("looping_pattern")

    return _clamp(score, 0.0, 1.0), reasons


def _tool_json_score(a: _Analysis) -> tuple[float, list[str]]:
    reasons: list[str] = []

    if a.tool_call_open == 0:
        return 0.0, ["no_tool_calls"]

    score = a.tool_call_valid_count / max(1, a.tool_call_open)
    if score < 1.0 and a.tool_call_invalid_reasons:
        reasons.append(a.tool_call_invalid_reasons[0])

    return _clamp(score, 0.0, 1.0), reasons


def _answer_score(a: _Analysis) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    if a.answer_count_is_one:
        score += 0.35
    else:
        reasons.append("answer_not_unique")

    if a.answer_is_last:
        score += 0.20
    else:
        reasons.append("answer_not_terminal")

    if a.answer_after_think:
        score += 0.15
    else:
        reasons.append("answer_not_after_think")

    if a.has_boxed_answer:
        score += 0.15
    else:
        reasons.append("missing_boxed")

    if a.boxed_content_nonempty:
        score += 0.15
    else:
        reasons.append("empty_boxed")

    return _clamp(score, 0.0, 1.0), reasons


def _think_quality_score(a: _Analysis) -> tuple[float, list[str]]:
    reasons: list[str] = []

    if not a.think_blocks:
        return 0.0, ["no_think_blocks"]

    score = 1.0

    if a.empty_think_count > 0:
        score -= min(0.50, 0.25 * a.empty_think_count)
        reasons.append("empty_think")

    very_short_count = sum(1 for t in a.think_blocks if 0 < _count_words(t.strip()) <= 2)
    if very_short_count > 0:
        score -= min(0.20, 0.05 * very_short_count)
        reasons.append("very_short_think")

    very_long_count = sum(1 for t in a.think_blocks if _count_words(t) >= 35)
    if very_long_count > 0:
        score -= min(0.15, 0.04 * very_long_count)
        reasons.append("very_long_think")

    if a.repeated_think_count > 0:
        score -= min(0.30, 0.10 * a.repeated_think_count)
        reasons.append("repeated_think")

    return _clamp(score, 0.0, 1.0), reasons


def validate_format(text: str) -> tuple[bool, str]:
    a = _analyze(text)

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

    if not a.starts_with_think:
        return False, "response must start with <think>"

    if a.tool_call_open < 1 or a.tool_response_open < 1:
        return False, "at least one tool cycle is required"

    if a.tool_call_valid_count != a.tool_call_open:
        return False, a.tool_call_invalid_reasons[0]

    if a.sequence_violations:
        return False, a.sequence_violations[0]

    if a.complete_cycle_count < 1:
        return False, "missing think->tool_call->tool_response->think cycle"

    if not a.answer_is_last:
        return False, "answer must be the final block"

    if not a.answer_after_think:
        return False, "answer must come after <think>"

    if not a.has_boxed_answer or not a.boxed_content_nonempty:
        return False, "answer must contain non-empty \\boxed{}"

    return True, "format is correct"


def compute_score(
    solution_str: str,
    ground_truth: Any,
    tokenizer: Optional[Any] = None,
) -> tuple[float, str]:
    """
    Stage-1 format-only reward for reasoning-over-search.

    Optimizes:
    - whole-trace protocol correctness
    - at least one full think->tool_call->tool_response->think cycle
    - valid search tool JSON
    - final boxed answer closure
    - language-mixing penalty retained

    ground_truth is unused in this stage.
    """
    del ground_truth

    response = _extract_response_text(solution_str)
    response, eos_reason = _strip_eos(response, tokenizer)
    if response is None:
        return 0.0, f"no_reward:{eos_reason}"
    response = _restore_prefilled_think(response)

    a = _analyze(response)

    pairing_score, pairing_reasons = _pairing_score(a)
    transition_score, transition_reasons = _transition_score(a)
    cycle_score, cycle_reasons = _cycle_score(a)
    tool_score, tool_reasons = _tool_json_score(a)
    answer_score, answer_reasons = _answer_score(a)
    think_score, think_reasons = _think_quality_score(a)

    format_score = (
        PAIRING_WEIGHT * pairing_score
        + TRANSITION_WEIGHT * transition_score
        + CYCLE_WEIGHT * cycle_score
        + TOOL_JSON_WEIGHT * tool_score
        + ANSWER_WEIGHT * answer_score
        + THINK_QUALITY_WEIGHT * think_score
    )

    # Cold-start bonus: isolated credit for the very first structural token
    # being a non-empty <think>. Drowned out otherwise inside transition_score.
    first_think_bonus = 0.0
    if (
        a.block_types
        and a.block_types[0] == "think"
        and a.block_contents
        and a.block_contents[0].strip()
    ):
        first_think_bonus = FIRST_THINK_BONUS

    # Cold-start bonus: graded credit per ``</think>`` close token, capped at
    # ``CLOSE_THINK_BONUS_MAX_COUNT``. Under prefilled ``<think>`` (prompt and
    # post-tool-response), the close tag is the earliest structural signal the
    # model can emit, and per-close counting additionally rewards multi-cycle
    # traces. Counted from the raw response so it is robust to block-parse
    # edge cases (unclosed outer shell, extra text between blocks, etc.).
    close_think_count = min(response.count("</think>"), CLOSE_THINK_BONUS_MAX_COUNT)
    close_think_bonus = CLOSE_THINK_BONUS * close_think_count

    degeneration_penalty = 0.0
    degeneration_reasons: list[str] = []

    if a.repeated_query_count > 0:
        degeneration_penalty += min(REPEATED_QUERY_PENALTY, 0.02 * a.repeated_query_count)
        degeneration_reasons.append("repeated_tool_query")

    if a.repeated_think_count > 0:
        degeneration_penalty += min(REPEATED_THINK_PENALTY, 0.02 * a.repeated_think_count)
        degeneration_reasons.append("repeated_think")

    if len(a.block_types) > 12 and (a.repeated_query_count > 0 or a.repeated_think_count > 0):
        degeneration_penalty += EXCESSIVE_LOOP_PENALTY
        degeneration_reasons.append("excessive_looping")

    mixing_fraction = _language_mixing_fraction(response)
    mixing_penalty = MIXING_PENALTY_MAX * mixing_fraction

    raw_score = (
        format_score
        + first_think_bonus
        + close_think_bonus
        - degeneration_penalty
        - mixing_penalty
    )
    score = _clamp(raw_score, SCORE_FLOOR, SCORE_CEIL)

    reasons = (
        pairing_reasons[:2]
        + transition_reasons[:2]
        + cycle_reasons[:2]
        + tool_reasons[:1]
        + answer_reasons[:2]
        + think_reasons[:2]
        + degeneration_reasons[:2]
    )
    reasons = [r for r in reasons if r]
    reason_str = ",".join(reasons) if reasons else "ok"

    return (
        score,
        (
            f"format={format_score:.3f} "
            f"first_think_bonus={first_think_bonus:.3f} "
            f"close_think_bonus={close_think_bonus:.3f} "
            f"close_count={close_think_count} "
            f"deg_pen={degeneration_penalty:.3f} "
            f"mix_frac={mixing_fraction:.3f} "
            f"mix_pen={mixing_penalty:.3f} "
            f"cycles={a.complete_cycle_count} "
            f"score={score:.3f} "
            f"reason={reason_str}"
        ),
    )