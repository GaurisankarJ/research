# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ReSearch / Search-R1 style prompt strings (legacy ``prompt_template_dict`` names).

Use with ``{prompt}`` filled from the dataset question field. Tags match
``verl.utils.reward_score.re_search`` format validation.
"""

re_search_template = """A conversation between User and Assistant. \
The user asks a question, and the assistant solves it. \
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. \
During thinking, the assistant can invoke the wikipedia search tool to search for fact information about specific topics if needed. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{{answer here}} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{{}} with latex format. \
User: {prompt}. Assistant:"""

re_search_template_sys = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
During thinking, you can invoke the wikipedia search tool to search for fact information about specific topics if needed. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."""

prompt_template_dict = {}
prompt_template_dict["re_search_template"] = re_search_template
prompt_template_dict["re_search_template_sys"] = re_search_template_sys

# Qwen2+ chat end-of-turn special (tokenizer); built with concat so tooling does not alter the literal.
QWEN_CHAT_IM_END = "<|" + "im_end" + "|>"


def build_qwen_manual_chat_prompt(
    messages: list[dict],
    *,
    add_thinking: bool,
    im_end: str = QWEN_CHAT_IM_END,
) -> str:
    """Qwen chat-style prefix without ``tokenizer.apply_chat_template`` (rollout / length filter).

    Builds::

        <|im_start|>system\\n{system_msg}{im_end}\\n
        <|im_start|>user\\n{user_msg}{im_end}\\n
        <|im_start|>assistant

    Only initial consecutive ``system`` / ``user`` turns are consumed; stops at any other role.

    After the assistant header: if ``add_thinking`` is True, append ``\\n`` only; if False (default),
    append ``\\n<redacted_thinking>\\n\\n</redacted_thinking>\\n\\n`` so generation continues after
    an empty thinking section (matches common Qwen3 thinking flows).
    """
    im_start = "<|im_start|>"
    parts: list[str] = []
    for m in messages:
        role = m.get("role")
        if role not in ("system", "user"):
            break
        content = m.get("content", "")
        if not isinstance(content, str):
            raise TypeError(f"Expected string content for role={role!r}, got {type(content)}")
        parts.append(f"{im_start}{role}\n{content}{im_end}\n")
    parts.append(f"{im_start}assistant")
    suffix = "\n" if add_thinking else "\n<thinking>\n\n</thinking>\n\n"
    return "".join(parts) + suffix


__all__ = [
    "re_search_template",
    "re_search_template_sys",
    "prompt_template_dict",
    "QWEN_CHAT_IM_END",
    "build_qwen_manual_chat_prompt",
]
