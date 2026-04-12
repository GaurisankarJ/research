# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import pytest

from verl.experimental.agent_loop.re_search_agent_loop import (
    _digest_response_text,
    classify_last_segment_no_valid_search,
    extract_search_content,
    extract_search_tool_call,
)


def test_extract_search_tool_call_accepts_nested_query_arguments():
    text = """
<think>Need a fact.</think>
<tool_call>
{"name": "search", "arguments": {"query": "  Hamlet author  "}}
</tool_call>
""".strip()

    query, status = extract_search_tool_call(text)

    assert status == "valid_search_tool_call"
    assert query == "Hamlet author"
    assert extract_search_content(text) == "Hamlet author"


@pytest.mark.parametrize(
    ("text", "expected_status"),
    [
        (
            """
<tool_call>
{"name": "search", "arguments": "missing brace"
</tool_call>
""".strip(),
            "malformed_tool_call_json",
        ),
        (
            """
<tool_call>
{"name": "lookup", "arguments": "Hamlet"}
</tool_call>
""".strip(),
            "tool_name_not_search",
        ),
        (
            """
<tool_call>
{"name": "search", "arguments": "Hamlet"}
</tool_call>
""".strip(),
            "tool_arguments_not_object",
        ),
        (
            """
<tool_call>
{"name": "search", "arguments": {}}
</tool_call>
""".strip(),
            "tool_arguments_query_not_string",
        ),
        (
            """
<tool_call>
{"name": "search", "arguments": {"query": 123}}
</tool_call>
""".strip(),
            "tool_arguments_query_not_string",
        ),
        (
            """
<tool_call>
{"name": "search", "arguments": {"query": "   "}}
</tool_call>
""".strip(),
            "empty_search_query",
        ),
    ],
)
def test_extract_search_tool_call_rejects_invalid_payloads(text, expected_status):
    query, status = extract_search_tool_call(text)

    assert query == ""
    assert status == expected_status
    assert extract_search_content(text) == ""


@pytest.mark.parametrize(
    ("text", "expected_status"),
    [
        ("<tool_call>", "open_tool_call_without_close"),
        ("</tool_call>", "close_tool_call_without_prior_open_pair"),
        ("plain text only", "no_tool_call_close_tag"),
    ],
)
def test_classify_last_segment_no_valid_search_reports_tool_call_shape(text, expected_status):
    assert classify_last_segment_no_valid_search(text) == expected_status


def test_digest_response_text_tracks_qwen_tool_tags():
    text = """
<tool_call>
{"name": "search", "arguments": {"query": "Hamlet author"}}
</tool_call> <tool_response>
Hamlet was written by William Shakespeare.
</tool_response>
<answer>The final answer is \\[ \\boxed{William Shakespeare} \\]</answer>
""".strip()

    digest = _digest_response_text(text)

    assert digest["n_lm_tool_call_close"] == 1
    assert digest["n_injected_tool_response_close"] == 1
    assert digest["tool_call_tags_present"] is True
    assert digest["tool_response_tags_present"] is True
