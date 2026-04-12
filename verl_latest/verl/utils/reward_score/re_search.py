import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional


def _validate_search_tool_call(payload: str) -> tuple[bool, str, str]:
    payload = payload.strip()
    if not payload:
        return False, "tool_call payload is empty", ""

    try:
        function_call = json.loads(payload)
    except json.JSONDecodeError:
        return False, "tool_call payload is not valid JSON", ""

    if not isinstance(function_call, dict):
        return False, "tool_call payload must be a JSON object", ""

    if function_call.get("name") != "search":
        return False, 'tool_call name must be "search"', ""

    arguments = function_call.get("arguments")
    if not isinstance(arguments, dict):
        return False, "tool_call arguments must be an object", ""

    query = arguments.get("query")
    if not isinstance(query, str):
        return False, "tool_call arguments.query must be a string", ""

    query = query.strip()
    if not query:
        return False, "tool_call arguments.query cannot be empty", ""

    return True, "ok", query


def _extract_response_text(solution_str: str) -> str:
    """Model response text: full decode (prompt+response) or response-only (verl NaiveRewardManager)."""
    if "<|im_start|>assistant\n" in solution_str:
        return solution_str.split("<|im_start|>assistant\n", 1)[1]
    if "Assistant:" in solution_str:
        return solution_str.split("Assistant:", 1)[1]
    return solution_str.strip()


def _normalize_ground_truth(ground_truth: Any) -> str | list[str]:
    """Align with Search-R1 / MuSiQue parquet labels (str, list, or dict with target)."""
    if isinstance(ground_truth, dict) and "target" in ground_truth:
        t = ground_truth["target"]
    else:
        t = ground_truth
    if isinstance(t, list):
        return [str(x) for x in t]
    return str(t)


_BLOCK_PATTERN = re.compile(r"<(think|tool_call|tool_response|answer)>(.*?)</\1>", re.DOTALL)


def _normalize_ws(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class _FormatAnalysis:
    think_open: int
    think_close: int
    tool_call_open: int
    tool_call_close: int
    tool_response_open: int
    tool_response_close: int
    answer_open: int
    answer_close: int

    block_types: list[str]

    think_blocks: list[str]
    tool_queries: list[str]
    tool_call_valid_count: int
    tool_call_invalid_reasons: list[str]
    empty_query_count: int
    repeated_query_count: int

    sequence_violations: list[str]
    local_transition_score: float
    complete_cycle_count: int

    answer_content: str
    answer_has_boxed: bool
    answer_is_last_block: bool
    has_nonempty_answer: bool
    non_tag_text_present: bool

    repeated_think_count: int
    empty_think_count: int
    very_short_think_count: int
    very_long_think_count: int

    tool_call_count: int
    tool_response_count: int


def _analyze_format(text: str) -> _FormatAnalysis:
    def _count(tag: str) -> tuple[int, int]:
        return text.count(f"<{tag}>"), text.count(f"</{tag}>")

    think_open, think_close = _count("think")
    tool_call_open, tool_call_close = _count("tool_call")
    tool_response_open, tool_response_close = _count("tool_response")
    answer_open, answer_close = _count("answer")

    blocks = list(_BLOCK_PATTERN.finditer(text))
    block_types = [m.group(1) for m in blocks]

    think_blocks: list[str] = []
    tool_queries: list[str] = []
    tool_call_valid_count = 0
    tool_call_invalid_reasons: list[str] = []
    empty_query_count = 0
    answer_content = ""

    cursor = 0
    non_tag_text_present = False
    for m in blocks:
        if text[cursor:m.start()].strip():
            non_tag_text_present = True
            break
        cursor = m.end()
    if not non_tag_text_present and text[cursor:].strip():
        non_tag_text_present = True

    for m in blocks:
        block_type = m.group(1)
        content = m.group(2)

        if block_type == "think":
            think_blocks.append(content)

        elif block_type == "tool_call":
            valid, reason, query = _validate_search_tool_call(content)
            if valid:
                tool_call_valid_count += 1
                tool_queries.append(query.strip().lower())
            else:
                tool_call_invalid_reasons.append(reason)
                if reason == "tool_call arguments.query cannot be empty":
                    empty_query_count += 1

        elif block_type == "answer":
            answer_content = content

    answer_has_boxed = "\\boxed{" in answer_content and "}" in answer_content
    answer_is_last_block = len(block_types) > 0 and block_types[-1] == "answer"
    has_nonempty_answer = bool(answer_content.strip())

    repeated_query_count = 0
    if tool_queries:
        query_counts = Counter(tool_queries)
        repeated_query_count = sum(max(0, c - 1) for c in query_counts.values())

    normalized_thinks = [_normalize_ws(t) for t in think_blocks if _normalize_ws(t)]
    repeated_think_count = 0
    if normalized_thinks:
        think_counts = Counter(normalized_thinks)
        repeated_think_count = sum(max(0, c - 1) for c in think_counts.values())

    empty_think_count = 0
    very_short_think_count = 0
    very_long_think_count = 0
    for t in think_blocks:
        wc = _count_words(t)
        if wc == 0:
            empty_think_count += 1
        elif wc <= 2:
            very_short_think_count += 1
        elif wc >= 35:
            very_long_think_count += 1

    sequence_violations: list[str] = []
    valid_transition_count = 0
    total_transition_checks = 0
    complete_cycle_count = 0

    for idx, block_type in enumerate(block_types):
        prev_type = block_types[idx - 1] if idx > 0 else None
        next_type = block_types[idx + 1] if idx + 1 < len(block_types) else None

        if block_type == "tool_call":
            total_transition_checks += 1
            if prev_type == "think":
                valid_transition_count += 1
            else:
                sequence_violations.append("tool_call_not_after_think")

            total_transition_checks += 1
            if next_type == "tool_response":
                valid_transition_count += 1
            else:
                sequence_violations.append("tool_call_not_followed_by_tool_response")

        elif block_type == "tool_response":
            total_transition_checks += 1
            if prev_type == "tool_call":
                valid_transition_count += 1
            else:
                sequence_violations.append("tool_response_not_after_tool_call")

            total_transition_checks += 1
            if next_type == "think":
                valid_transition_count += 1
            else:
                sequence_violations.append("tool_response_not_followed_by_think")

        elif block_type == "answer":
            total_transition_checks += 1
            if prev_type == "think":
                valid_transition_count += 1
            else:
                sequence_violations.append("answer_not_after_think")

        if idx + 3 < len(block_types):
            if block_types[idx:idx + 4] == ["think", "tool_call", "tool_response", "think"]:
                complete_cycle_count += 1

    local_transition_score = (
        valid_transition_count / total_transition_checks if total_transition_checks > 0 else 0.0
    )

    return _FormatAnalysis(
        think_open=think_open,
        think_close=think_close,
        tool_call_open=tool_call_open,
        tool_call_close=tool_call_close,
        tool_response_open=tool_response_open,
        tool_response_close=tool_response_close,
        answer_open=answer_open,
        answer_close=answer_close,
        block_types=block_types,
        think_blocks=think_blocks,
        tool_queries=tool_queries,
        tool_call_valid_count=tool_call_valid_count,
        tool_call_invalid_reasons=tool_call_invalid_reasons,
        empty_query_count=empty_query_count,
        repeated_query_count=repeated_query_count,
        sequence_violations=sequence_violations,
        local_transition_score=local_transition_score,
        complete_cycle_count=complete_cycle_count,
        answer_content=answer_content,
        answer_has_boxed=answer_has_boxed,
        answer_is_last_block=answer_is_last_block,
        has_nonempty_answer=has_nonempty_answer,
        non_tag_text_present=non_tag_text_present,
        repeated_think_count=repeated_think_count,
        empty_think_count=empty_think_count,
        very_short_think_count=very_short_think_count,
        very_long_think_count=very_long_think_count,
        tool_call_count=tool_call_open,
        tool_response_count=tool_response_open,
    )


def _reasoning_quality_score(analysis: _FormatAnalysis) -> tuple[float, list[str]]:
    """
    Reward concise but non-empty, decision-oriented thinking.
    Do not punish multiple valid cycles.
    Penalize repeated/empty/degenerate thinking.
    """
    reasons: list[str] = []
    thinks = analysis.think_blocks
    if not thinks:
        return 0.0, ["no_think_blocks"]

    score = 1.0

    if analysis.empty_think_count > 0:
        score -= min(0.50, 0.25 * analysis.empty_think_count)
        reasons.append("empty_think")

    if analysis.very_short_think_count > 0:
        score -= min(0.25, 0.06 * analysis.very_short_think_count)
        reasons.append("very_short_think")

    if analysis.very_long_think_count > 0:
        score -= min(0.20, 0.05 * analysis.very_long_think_count)
        reasons.append("very_long_think")

    if analysis.repeated_think_count > 0:
        score -= min(0.35, 0.10 * analysis.repeated_think_count)
        reasons.append("repeated_think")

    return _clamp01(score), reasons


def _query_quality_score(analysis: _FormatAnalysis) -> tuple[float, list[str]]:
    """
    Reward valid, non-empty, reasonably short factual queries.
    Penalize repeated identical queries.
    Do not penalize multiple distinct queries.
    """
    reasons: list[str] = []

    if analysis.tool_call_count == 0:
        return 0.0, ["no_tool_calls"]

    if analysis.tool_call_valid_count == 0:
        return 0.0, ["no_valid_tool_calls"]

    score = analysis.tool_call_valid_count / max(1, analysis.tool_call_count)

    long_or_bad = 0
    for q in analysis.tool_queries:
        wc = _count_words(q)
        if wc < 1:
            long_or_bad += 1
        elif wc > 12:
            long_or_bad += 1
        if "\n" in q or "<" in q or ">" in q:
            long_or_bad += 1

    if long_or_bad > 0:
        score -= min(0.25, 0.05 * long_or_bad)
        reasons.append("query_quality_issue")

    if analysis.repeated_query_count > 0:
        score -= min(0.35, 0.12 * analysis.repeated_query_count)
        reasons.append("repeated_tool_query")

    return _clamp01(score), reasons


def _cycle_score(analysis: _FormatAnalysis) -> tuple[float, list[str]]:
    """
    Main structural reward:
    think -> tool_call -> tool_response -> think
    repeated valid cycles are good.
    """
    reasons: list[str] = []

    if analysis.tool_call_count == 0 or analysis.tool_response_count == 0:
        return 0.0, ["missing_tool_cycle"]

    if analysis.complete_cycle_count == 0:
        return 0.0, ["no_complete_reasoning_cycle"]

    score = 0.0

    # strong reward for at least one full cycle
    score += 0.55

    # reward local structural correctness
    score += 0.30 * analysis.local_transition_score

    # allow 2-3 valid cycles without penalty
    if analysis.complete_cycle_count == 1:
        score += 0.10
    elif analysis.complete_cycle_count == 2:
        score += 0.12
    elif analysis.complete_cycle_count == 3:
        score += 0.12
    else:
        # very mild saturation, not a harsh penalty
        score += 0.08
        reasons.append("many_cycles")

    # but repeated identical queries/thinks make extra cycles suspicious
    if analysis.complete_cycle_count >= 4 and (
        analysis.repeated_query_count > 0 or analysis.repeated_think_count > 0
    ):
        score -= 0.10
        reasons.append("looping_pattern")

    return _clamp01(score), reasons


def _format_reward(analysis: _FormatAnalysis) -> tuple[float, list[str]]:
    """
    Total shaped format reward in [0, 0.45].

    Priorities:
    1. Well-formed tags
    2. At least one valid reasoning-over-search cycle
    3. Correct local transitions
    4. Valid tool JSON
    5. Non-empty, non-degenerate thinking
    6. Proper final answer closure
    """
    reasons: list[str] = []
    score = 0.0

    # 1) Tag pairing and basic block hygiene
    pair_score = 0.0
    if analysis.think_open == analysis.think_close and analysis.think_open > 0:
        pair_score += 0.25
    else:
        reasons.append("bad_think_pairing")

    if analysis.answer_open == analysis.answer_close == 1:
        pair_score += 0.20
    else:
        reasons.append("answer_count_not_one")

    if analysis.tool_call_open == analysis.tool_call_close:
        pair_score += 0.10
    else:
        reasons.append("tool_call_not_paired")

    if analysis.tool_response_open == analysis.tool_response_close:
        pair_score += 0.10
    else:
        reasons.append("tool_response_not_paired")

    if not analysis.non_tag_text_present:
        pair_score += 0.10
    else:
        reasons.append("text_outside_tags")

    score += 0.10 * _clamp01(pair_score)

    # 2) Reasoning-over-search cycles
    cycle_score, cycle_reasons = _cycle_score(analysis)
    score += 0.18 * cycle_score
    reasons.extend(cycle_reasons)

    # 3) Tool payload quality
    query_score, query_reasons = _query_quality_score(analysis)
    score += 0.07 * query_score
    reasons.extend(query_reasons)

    # 4) Thinking quality
    think_score, think_reasons = _reasoning_quality_score(analysis)
    score += 0.05 * think_score
    reasons.extend(think_reasons)

    # 5) Sequence compliance
    if analysis.sequence_violations:
        reasons.extend(analysis.sequence_violations[:3])
    else:
        score += 0.03

    # 6) Final answer closure
    if analysis.has_nonempty_answer:
        score += 0.01
    else:
        reasons.append("empty_answer")

    if analysis.answer_is_last_block:
        score += 0.01
    else:
        reasons.append("answer_not_terminal")

    # 7) Light degeneration penalties
    penalty = 0.0
    if analysis.empty_query_count > 0:
        penalty += min(0.04, 0.02 * analysis.empty_query_count)
        reasons.append("empty_tool_query")

    # Only mildly penalize excessive calls, and only if obviously looping
    if analysis.tool_call_count > 5 and (
        analysis.repeated_query_count > 0 or analysis.repeated_think_count > 0
    ):
        penalty += min(0.05, 0.01 * (analysis.tool_call_count - 5))
        reasons.append("excessive_looping_tool_calls")

    final_score = max(0.0, min(0.45, score - penalty))
    return final_score, reasons


def validate_format(text: str) -> tuple[bool, str]:
    analysis = _analyze_format(text)

    if analysis.think_open != analysis.think_close or analysis.think_open == 0:
        return False, "<think> tags are missing or not paired"

    if analysis.answer_open != 1 or analysis.answer_close != 1:
        return False, "<answer> must appear exactly once"

    if analysis.tool_call_open != analysis.tool_call_close:
        return False, "<tool_call> tags are not paired"

    if analysis.tool_response_open != analysis.tool_response_close:
        return False, "<tool_response> tags are not paired"

    if analysis.tool_call_count < 1 or analysis.tool_response_count < 1:
        return False, "at least one complete tool cycle is required"

    if analysis.tool_call_valid_count != analysis.tool_call_count:
        return False, analysis.tool_call_invalid_reasons[0]

    if analysis.sequence_violations:
        return False, analysis.sequence_violations[0]

    if analysis.complete_cycle_count < 1:
        return False, "missing think->tool_call->tool_response->think cycle"

    if not analysis.answer_has_boxed:
        return False, "answer is missing \\boxed{} format"

    if not analysis.answer_is_last_block:
        return False, "answer must be the final block"

    if analysis.non_tag_text_present:
        return False, "text outside tags is not allowed"

    return True, "format is correct"


def extract_answer(text: str):
    text = text.strip()
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    return match.group(1)


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"
    assert s[:len(left)] == left
    assert s[-1] == "}"
    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_f1_score(prediction: str, ground_truths: str | list[str]):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])

    return final_metric["f1"]


def compute_score(
    solution_str: str,
    ground_truth: Any,
    tokenizer: Optional[Any] = None,
) -> tuple[float, str]:
    """
    ReSearch reward aligned to reasoning-over-search:
    - shaped reward for valid think -> tool_call -> tool_response -> think cycles
    - answer must still be boxed
    - final score combines answer F1 and structured reasoning reward
    """
    response = _extract_response_text(solution_str)

    if tokenizer is not None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token:
            if not response.endswith(eos_token):
                return 0.0, "over length"
            response = response[: -len(eos_token)]

    analysis = _analyze_format(response)
    format_score, format_reasons = _format_reward(analysis)

    if not analysis.answer_has_boxed:
        near_zero = min(0.02, format_score * 0.1)
        return near_zero, f"missing_boxed: format={format_score:.3f}, reasons={','.join(format_reasons[:3])}"

    answer_part = extract_answer(response)
    if answer_part is None:
        near_zero = min(0.02, format_score * 0.1)
        return near_zero, f"cannot_extract_answer: format={format_score:.3f}"

    try:
        boxed = last_boxed_only_string(answer_part)
        if boxed is None:
            return 0.0, f"cannot_find_boxed: format={format_score:.3f}"
        answer = remove_boxed(boxed)
    except Exception as e:
        return 0.0, f"find box error: {e}"

    gt = _normalize_ground_truth(ground_truth)
    f1_score = get_f1_score(answer, gt)

    # Main combination:
    # keep semantic correctness primary,
    # but give meaningful reward for the desired trajectory behavior.
    if f1_score > 0:
        final = min(1.0, (0.55 * float(f1_score)) + format_score)
        return final, f"f1={f1_score:.3f}, format={format_score:.3f}, cycles={analysis.complete_cycle_count}, answer={answer}"

    # Small floor only if the model followed the desired protocol.
    floor = min(0.12, 0.01 + 0.25 * format_score)
    return floor, (
        f"f1=0.000, format={format_score:.3f}, cycles={analysis.complete_cycle_count}, "
        f"answer={answer}, reasons={','.join(format_reasons[:3])}"
    )