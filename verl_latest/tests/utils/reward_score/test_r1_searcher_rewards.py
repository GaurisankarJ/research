from verl.utils.reward_score import r1_searcher_format, r_1_searcher_answer


def _valid_r1_searcher_trace(answer: str = "Paris") -> str:
    return (
        "<think>Search first.</think>"
        '<tool_call>{"name":"search","arguments":{"query":"capital of france"}}</tool_call>'
        "<tool_response>Paris is the capital of France.</tool_response>"
        "<think>Now I can answer.</think>"
        f"<answer>\\boxed{{{answer}}}</answer>"
    )


class _FakeTokenizer:
    def __init__(self, eos_token: str = "<|endoftext|>"):
        self.eos_token = eos_token


def test_full_cascade_gives_one_point():
    score, reason = r1_searcher_format.compute_score(_valid_r1_searcher_trace(), ground_truth=None)

    assert score == 1.0
    assert "tiers=t1,t2,t3,t4,t5" in reason
    assert "stop=ok" in reason


def test_tier1_only_when_no_tool_call():
    score, reason = r1_searcher_format.compute_score(
        "<think>I already know this.</think><answer>\\boxed{Paris}</answer>",
        ground_truth=None,
    )

    assert score == r1_searcher_format.TIER1_FIRST_THINK
    assert "tiers=t1" in reason
    assert "stop=tool_call:absent" in reason


def test_tier1_and_tier2_stop_when_missing_post_tool_think():
    trace = (
        "<think>Search first.</think>"
        '<tool_call>{"name":"search","arguments":{"query":"capital of france"}}</tool_call>'
        "<tool_response>Paris is the capital of France.</tool_response>"
        "<answer>\\boxed{Paris}</answer>"
    )
    score, reason = r1_searcher_format.compute_score(trace, ground_truth=None)

    expected = r1_searcher_format.TIER1_FIRST_THINK + r1_searcher_format.TIER2_TOOL_CALL
    assert score == expected
    assert "tiers=t1,t2" in reason
    assert "stop=missing_post_tool_think" in reason


def test_cascade_stops_before_boxed_when_answer_missing_boxed():
    trace = (
        "<think>Search first.</think>"
        '<tool_call>{"name":"search","arguments":{"query":"capital of france"}}</tool_call>'
        "<tool_response>Paris is the capital of France.</tool_response>"
        "<think>Now I can answer.</think>"
        "<answer>Paris</answer>"
    )
    score, reason = r1_searcher_format.compute_score(trace, ground_truth=None)

    expected = (
        r1_searcher_format.TIER1_FIRST_THINK
        + r1_searcher_format.TIER2_TOOL_CALL
        + r1_searcher_format.TIER3_POST_TOOL_THINK
        + r1_searcher_format.TIER4_ANSWER_BLOCK
    )
    assert abs(score - expected) < 1e-9
    assert "tiers=t1,t2,t3,t4" in reason
    assert "stop=missing_boxed" in reason


def test_invalid_tool_payload_stops_after_tier1():
    trace = (
        "<think>Search first.</think>"
        '<tool_call>{"name":"search","arguments":"capital of france"}</tool_call>'
        "<tool_response>Paris is the capital of France.</tool_response>"
        "<think>Now I can answer.</think>"
        "<answer>\\boxed{Paris}</answer>"
    )
    score, reason = r1_searcher_format.compute_score(trace, ground_truth=None)

    assert score == r1_searcher_format.TIER1_FIRST_THINK
    assert "stop=tool_call:arguments_not_object" in reason


def test_language_mixing_penalty_subtracts_from_score():
    trace = (
        "<think>搜索首都信息 first.</think>"
        '<tool_call>{"name":"search","arguments":{"query":"capital of france"}}</tool_call>'
        "<tool_response>Paris is the capital of France.</tool_response>"
        "<think>现在 I can answer.</think>"
        "<answer>\\boxed{Paris}</answer>"
    )
    score, reason = r1_searcher_format.compute_score(trace, ground_truth=None)

    assert score < 1.0
    assert "mix_frac=" in reason
    assert "mix_pen=" in reason


def test_tool_response_content_is_not_penalised():
    trace = (
        "<think>Search first.</think>"
        '<tool_call>{"name":"search","arguments":{"query":"capital of france"}}</tool_call>'
        "<tool_response>巴黎 is the capital of France.</tool_response>"
        "<think>Now I can answer.</think>"
        "<answer>\\boxed{Paris}</answer>"
    )
    score, reason = r1_searcher_format.compute_score(trace, ground_truth=None)

    assert score == 1.0
    assert "mix_frac=0.000" in reason


def test_missing_eos_is_hard_zero():
    tokenizer = _FakeTokenizer(eos_token="<|endoftext|>")
    trace = _valid_r1_searcher_trace()
    score, reason = r1_searcher_format.compute_score(trace, ground_truth=None, tokenizer=tokenizer)

    assert score == 0.0
    assert reason == "no_reward:no_eos"


def test_eos_present_is_full_credit():
    tokenizer = _FakeTokenizer(eos_token="<|endoftext|>")
    trace = _valid_r1_searcher_trace() + tokenizer.eos_token
    score, reason = r1_searcher_format.compute_score(trace, ground_truth=None, tokenizer=tokenizer)

    assert score == 1.0
    assert "stop=ok" in reason


def test_multiple_tool_loops_still_full_credit():
    trace = (
        "<think>Search first.</think>"
        '<tool_call>{"name":"search","arguments":{"query":"capital of france"}}</tool_call>'
        "<tool_response>Paris is the capital of France.</tool_response>"
        "<think>I need one more fact.</think>"
        '<tool_call>{"name":"search","arguments":{"query":"france continent"}}</tool_call>'
        "<tool_response>France is in Europe.</tool_response>"
        "<think>Now I can answer.</think>"
        "<answer>\\boxed{Paris}</answer>"
    )
    score, reason = r1_searcher_format.compute_score(trace, ground_truth=None)

    assert score == 1.0
    assert "stop=ok" in reason


def test_r1_searcher_answer_scores_exact_and_partial_match():
    exact_score, exact_reason = r_1_searcher_answer.compute_score(
        _valid_r1_searcher_trace("New York City"),
        {"target": ["New York City"]},
    )
    partial_score, partial_reason = r_1_searcher_answer.compute_score(
        _valid_r1_searcher_trace("New York"),
        {"target": ["New York City"]},
    )

    assert exact_score == 1.0
    assert 0.0 < partial_score < exact_score
    assert "f1=1.000" in exact_reason
    assert "f1=" in partial_reason
