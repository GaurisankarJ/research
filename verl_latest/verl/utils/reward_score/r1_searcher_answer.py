from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Optional


def _extract_response_text(solution_str: str) -> str:
    """Accept full chat decodes and response-only strings."""
    if "<|im_start|>assistant\n" in solution_str:
        return solution_str.split("<|im_start|>assistant\n", 1)[1]
    if "Assistant:" in solution_str:
        return solution_str.split("Assistant:", 1)[1]
    return solution_str.strip()


def _strip_eos(response: str, tokenizer: Optional[Any]) -> tuple[Optional[str], str]:
    if tokenizer is None:
        return response, ""

    eos_token = getattr(tokenizer, "eos_token", None)
    if not eos_token:
        return response, ""

    if not response.endswith(eos_token):
        return None, "over length"
    return response[: -len(eos_token)], ""


def _normalize_ground_truth(ground_truth: Any) -> str | list[str]:
    if isinstance(ground_truth, dict) and "target" in ground_truth:
        target = ground_truth["target"]
    else:
        target = ground_truth
    if isinstance(target, list):
        return [str(item) for item in target]
    return str(target)


def _extract_answer(text: str) -> str | None:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in value)

    def lower(value: str) -> str:
        return value.lower()

    def replace_underscore(value: str) -> str:
        return value.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(text)))))


def _f1_score(prediction: str, ground_truth: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0.0
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def get_f1_score(prediction: str, ground_truths: str | list[str]) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    return max((_f1_score(prediction, ground_truth) for ground_truth in ground_truths), default=0.0)


def compute_score(
    solution_str: str,
    ground_truth: Any,
    tokenizer: Optional[Any] = None,
) -> tuple[float, str]:
    """
    Standalone R1-Searcher answer reward.

    Mirrors the upstream stage-2 answer component: token-level F1 on the final
    ``<answer>...</answer>`` text, without mixing in the separate format penalty.
    """
    response = _extract_response_text(solution_str)
    response, eos_reason = _strip_eos(response, tokenizer)
    if response is None:
        return 0.0, eos_reason

    answer = _extract_answer(response)
    if answer is None:
        return 0.0, "cannot extract answer"

    references = _normalize_ground_truth(ground_truth)
    f1 = get_f1_score(answer, references)
    return float(f1), f"f1={f1:.3f}, answer={answer}"
