re_search_template = """A conversation between User and Assistant. \
The user asks a question, and the assistant solves it. \
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. \
During thinking, the assistant can invoke the wikipedia search tool to search for factual information when needed. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and tool calls and tool responses are enclosed within <tool_call> </tool_call> and <tool_response> </tool_response> tags respectively. \
Every <tool_call> must contain valid JSON with this schema: {"name": "search", "arguments": {"query": "short factual query"}}. \
For example, <think>Need the key fact.</think> <tool_call>{"name": "search", "arguments": {"query": "entity founding year"}}</tool_call> <tool_response>search result here</tool_response> \
<think>I now have the needed fact.</think> <answer>The final answer is \\[ \\boxed{{answer here}} \\]</answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{{}} with latex format. \
User: {prompt}. Assistant:"""

re_search_template_sys = """You are a reasoning assistant with one tool: search.

Use only these tags in assistant output: <think>, <tool_call>, <tool_response>, <answer>.

Use multi-step tool calling when needed:
<think>...</think>
<tool_call>{"name":"search","arguments":{"query":"short factual query"}}</tool_call>
<tool_response>...</tool_response>
<think>...</think>

Repeat this loop until you have enough evidence, then finish once with:
<answer>The final answer is \\[ \\boxed{...} \\]</answer>

Rules:
- Every <tool_call> must be valid JSON with arguments as an object.
- Do not stop after <think>, <tool_call>, or <tool_response>.
- After </answer>, stop immediately.

<tools>
{"name":"search","description":"Search Wikipedia for factual information.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Short factual search query."}},"required":["query"]}}
</tools>
"""

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
    suffix = "\n" if add_thinking else "\n<think>\n\n</think>\n\n"
    return "".join(parts) + suffix


__all__ = [
    "re_search_template",
    "re_search_template_sys",
    "prompt_template_dict",
    "QWEN_CHAT_IM_END",
    "build_qwen_manual_chat_prompt",
]
