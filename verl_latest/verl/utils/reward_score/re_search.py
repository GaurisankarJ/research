import json
import re
import string
from collections import Counter
from typing import Any, Optional


def _validate_search_tool_call(payload: str) -> tuple[bool, str]:
    payload = payload.strip()
    if not payload:
        return False, "tool_call payload is empty"

    try:
        function_call = json.loads(payload)
    except json.JSONDecodeError:
        return False, "tool_call payload is not valid JSON"

    if not isinstance(function_call, dict):
        return False, "tool_call payload must be a JSON object"

    if function_call.get("name") != "search":
        return False, 'tool_call name must be "search"'

    arguments = function_call.get("arguments")
    if not isinstance(arguments, str):
        return False, "tool_call arguments must be a string"

    if not arguments.strip():
        return False, "tool_call arguments cannot be empty"

    return True, "ok"


def _extract_response_text(solution_str: str) -> str:
    """Model response text: full decode (prompt+response) or response-only (verl NaiveRewardManager)."""
    if "<|im_start|>assistant\n" in solution_str:
        return solution_str.split("<|im_start|>assistant\n", 1)[1]
    if "Assistant:" in solution_str:
        return solution_str.split("Assistant:", 1)[1]
    return solution_str.strip()


def _normalize_ground_truth(ground_truth: Any) -> str | list[str]:
    """Align with Search-R1 / MuSiQue parquet labels (str, list, or dict with ``target``)."""
    if isinstance(ground_truth, dict) and "target" in ground_truth:
        t = ground_truth["target"]
    else:
        t = ground_truth
    if isinstance(t, list):
        return [str(x) for x in t]
    return str(t)

def validate_format(text: str) -> tuple[bool, str]:
    # check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired"
    
    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"
    
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"        
    
    # check the order of tool_call/tool_response
    current_pos = 0
    while True:
        tool_call_pos = text.find('<tool_call>', current_pos)
        if tool_call_pos == -1:
            break
            
        tool_response_pos = text.find('<tool_response>', tool_call_pos)
        tool_call_end_pos = text.find('</tool_call>', tool_call_pos)
        tool_response_end_pos = text.find('</tool_response>', tool_response_pos)
        
        if -1 in (tool_response_pos, tool_call_end_pos, tool_response_end_pos):
            return False, "tool_call/tool_response tags are incomplete"
            
        if not (tool_call_pos < tool_call_end_pos < tool_response_pos < tool_response_end_pos):
            return False, "tool_call/tool_response tags are nested in the wrong order"

        payload = text[tool_call_pos + len('<tool_call>'):tool_call_end_pos]
        valid_payload, payload_reason = _validate_search_tool_call(payload)
        if not valid_payload:
            return False, payload_reason
            
        current_pos = tool_response_end_pos

    if any(tag in text for tag in ('<search>', '</search>', '<result>', '</result>')):
        return False, "legacy search/result tags are not allowed"
    
    # check if \boxed{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"
    
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
    
    return final_metric['f1']

def compute_score(
    solution_str: str,
    ground_truth: Any,
    tokenizer: Optional[Any] = None,
) -> tuple[float, str]:
    """ReSearch-style reward: strict tags + token F1 vs labels.

    ``solution_str`` may be a full chat decode or **response-only** (as passed by ``NaiveRewardManager``).

    If ``tokenizer`` is set and defines ``eos_token``, the response must end with EOS (legacy behavior);
    otherwise reward is zero (``over length``). EOS is stripped before answer extraction.
    """
    response = _extract_response_text(solution_str)
    valid_template, reason = validate_format(response)
    if not valid_template:
        return 0.0, f"bad format: {reason}"

    if tokenizer is not None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token:
            if not response.endswith(eos_token):
                return 0.0, "over length"
            response = response[: -len(eos_token)]

    gt = _normalize_ground_truth(ground_truth)
    answer_part = extract_answer(response)
    if answer_part is not None:
        try:
            answer = remove_boxed(last_boxed_only_string(answer_part))
        except Exception as e:
            return 0, f'find box error: {e}'
    else:
        return 0, 'cannot extract answer'

    f1_score = get_f1_score(answer, gt)
    if f1_score > 0:
        return float(f1_score), f"correct answer, get f1 score: {f1_score}"
    return 0.1, f"wrong answer but good format: {answer}"