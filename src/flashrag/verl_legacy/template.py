re_search_template = """Who founded Microsoft?
<think>Need the founders.</think>
<tool_call>{{"name":"search","arguments":{{"query":"Microsoft founders"}}}}</tool_call>
<tool_response>Microsoft was founded by Bill Gates and Paul Allen.</tool_response>
<think>I have the founders.</think>
<answer>The final answer is \\[ \\boxed{{Bill Gates and Paul Allen}} \\]</answer>

What year was Google founded?
<think>Need the founding year.</think>
<tool_call>{{"name":"search","arguments":{{"query":"Google founded year"}}}}</tool_call>
<tool_response>Google was founded in 1998.</tool_response>
<think>I have the year.</think>
<answer>The final answer is \\[ \\boxed{{1998}} \\]</answer>

What is the capital of France?
<think>Need the capital.</think>
<tool_call>{{"name":"search","arguments":{{"query":"France capital city"}}}}</tool_call>
<tool_response>The capital city of France is Paris.</tool_response>
<think>I have the capital.</think>
<answer>The final answer is \\[ \\boxed{{Paris}} \\]</answer>

What year did Apollo 11 land on the Moon?
<think>Need the year.</think>
<tool_call>{{"name":"search","arguments":{{"query":"Apollo 11 Moon landing year"}}}}</tool_call>
<tool_response>Apollo 11 landed on the Moon in 1969.</tool_response>
<think>I have the year.</think>
<answer>The final answer is \\[ \\boxed{{1969}} \\]</answer>

Who wrote Hamlet?
<think>Need the author.</think>
<tool_call>{{"name":"search","arguments":{{"query":"Hamlet author"}}}}</tool_call>
<tool_response>Hamlet was written by William Shakespeare.</tool_response>
<think>I have the author.</think>
<answer>The final answer is \\[ \\boxed{{William Shakespeare}} \\]</answer>

What is the capital of Italy?
<think>Need the capital.</think>
<tool_call>{{"name":"search","arguments":{{"query":"Italy capital city"}}}}</tool_call>
<tool_response>The capital city of Italy is Rome.</tool_response>
<think>I have the capital.</think>
<answer>The final answer is \\[ \\boxed{{Rome}} \\]</answer>

What is the largest planet in the Solar System?
<think>Need the planet.</think>
<tool_call>{{"name":"search","arguments":{{"query":"largest planet in the Solar System"}}}}</tool_call>
<tool_response>The largest planet in the Solar System is Jupiter.</tool_response>
<think>I have the planet.</think>
<answer>The final answer is \\[ \\boxed{{Jupiter}} \\]</answer>

{prompt}
"""

re_search_template_sys = """You are a reasoning agent with one tool named search.

Use the native Qwen chat behavior: think first, then act with the tool, then think again, then answer.

# Output format
Output only these blocks and nothing else:
<think>...</think>
<tool_call>...</tool_call>
<tool_response>...</tool_response>
<answer>...</answer>

# Tool
The available tool is:
<tools>
{"name":"search","description":"Search Wikipedia for factual information about specific topics.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"The search query."}},"required":["query"]}}
</tools>

Every tool call must be valid JSON with this exact schema:
<tool_call>{"name":"search","arguments":{"query":"short factual query"}}</tool_call>

# Required loop
This runtime requires reasoning over search, not answering from memory.
For every user question, you must use the search tool at least once before the final answer.

Required block order:
1. Start with <think>...</think>
2. Immediately after the first <think>, output a <tool_call>...</tool_call>
3. After each <tool_call>, a <tool_response>...</tool_response> will be provided
4. Immediately after each <tool_response>, output a new <think>...</think>
5. After that new <think>, output either another <tool_call>...</tool_call> or the final <answer>...</answer>
6. Finish with exactly one <answer>...</answer>

# Hard rules
- Never answer without a prior search tool call
- Never stop after <think>
- Never stop after <tool_call>
- Never stop after <tool_response>
- Do not output ChatML or special end tokens such as <|im_end|> or <|endoftext|>
- Do not describe a tool call inside <think>; perform it in the next block
- Keep each <think> short, concrete, and action-oriented
- Do not put plain text outside the required tags
- Do not put the final answer inside <think>

# Final answer
The final answer must be exactly one block:
<answer>The final answer is \[ \boxed{...} \]</answer>

Do not output anything after </answer>."""

prompt_template_dict = {}
prompt_template_dict["re_search_template"] = re_search_template
prompt_template_dict["re_search_template_sys"] = re_search_template_sys
