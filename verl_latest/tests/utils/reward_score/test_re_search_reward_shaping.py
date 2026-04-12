from verl.utils.reward_score import re_search


def _score(solution: str, ground_truth: str = "Paris") -> tuple[float, str]:
    return re_search.compute_score(solution, ground_truth)


def test_minimal_valid_trace_gets_positive_format_credit():
    no_tool_trace = (
        "<think>I can answer directly.</think>"
        "<answer>The final answer is \\[ \\boxed{Rome} \\]</answer>"
    )

    score, reason = _score(no_tool_trace)

    assert score > 0.0
    assert "format=" in reason


def test_valid_multiturn_trace_scores_higher_than_no_tool_trace():
    no_tool_trace = (
        "<think>I can answer directly.</think>"
        "<answer>The final answer is \\[ \\boxed{Rome} \\]</answer>"
    )
    one_search_trace = (
        "<think>I need one fact.</think>"
        '<tool_call>{"name":"search","arguments":{"query":"capital of france"}}</tool_call>'
        "<tool_response>Paris is the capital of France.</tool_response>"
        "<think>Now I can answer.</think>"
        "<answer>The final answer is \\[ \\boxed{Rome} \\]</answer>"
    )

    no_tool_score, _ = _score(no_tool_trace)
    one_search_score, _ = _score(one_search_trace)

    assert one_search_score > no_tool_score


def test_invalid_tool_payload_loses_score_against_valid_payload():
    valid_trace = (
        "<think>I should search.</think>"
        '<tool_call>{"name":"search","arguments":{"query":"capital of france"}}</tool_call>'
        "<tool_response>Paris is the capital of France.</tool_response>"
        "<think>Now I can answer.</think>"
        "<answer>The final answer is \\[ \\boxed{Rome} \\]</answer>"
    )
    invalid_payload_trace = (
        "<think>I should search.</think>"
        '<tool_call>{"name":"search","arguments":"capital of france"}</tool_call>'
        "<tool_response>Paris is the capital of France.</tool_response>"
        "<think>Now I can answer.</think>"
        "<answer>The final answer is \\[ \\boxed{Rome} \\]</answer>"
    )

    valid_score, _ = _score(valid_trace)
    invalid_score, reason = _score(invalid_payload_trace)

    assert invalid_score < valid_score
    assert "format=" in reason


def test_missing_think_after_tool_response_loses_sequence_credit():
    valid_sequence = (
        "<think>I should search.</think>"
        '<tool_call>{"name":"search","arguments":{"query":"capital of france"}}</tool_call>'
        "<tool_response>Paris is the capital of France.</tool_response>"
        "<think>Now I can answer.</think>"
        "<answer>The final answer is \\[ \\boxed{Rome} \\]</answer>"
    )
    broken_sequence = (
        "<think>I should search.</think>"
        '<tool_call>{"name":"search","arguments":{"query":"capital of france"}}</tool_call>'
        "<tool_response>Paris is the capital of France.</tool_response>"
        "<answer>The final answer is \\[ \\boxed{Rome} \\]</answer>"
    )

    valid_score, _ = _score(valid_sequence)
    broken_score, _ = _score(broken_sequence)

    assert broken_score < valid_score


def test_missing_boxed_is_near_zero():
    missing_boxed_trace = (
        "<think>I can answer now.</think>"
        "<answer>The final answer is Paris.</answer>"
    )

    score, reason = _score(missing_boxed_trace)

    assert score <= 0.02
    assert reason.startswith("missing_boxed")


def test_correct_then_wrong_then_malformed_ranking():
    correct_compliant = (
        "<think>I can answer directly.</think>"
        "<answer>The final answer is \\[ \\boxed{Paris} \\]</answer>"
    )
    wrong_compliant = (
        "<think>I can answer directly.</think>"
        "<answer>The final answer is \\[ \\boxed{Rome} \\]</answer>"
    )
    malformed = (
        "<think>I can answer directly.</think>"
        '<tool_call>{"name":"search","arguments":"capital of france"}</tool_call>'
        "<answer>The final answer is Paris.</answer>"
    )

    correct_score, _ = _score(correct_compliant)
    wrong_score, _ = _score(wrong_compliant)
    malformed_score, _ = _score(malformed)

    assert correct_score > wrong_score > malformed_score
