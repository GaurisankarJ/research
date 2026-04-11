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

re_search_template_sys_iter_1 = """You are a helpful assistant who can answer questions using multiple Wikipedia search tool calls. \
You can call the search tool by writing: \
<search> your query </search> \
You will receive the result in: \
<result> your search result </result> \
Use the search tool to obtain the information needed for the answer. Answers should be based on the search results. \
You may use the search tool multiple times if needed before giving the final answer. \
Provide the final answer in the format: \
<answer>The final answer is \\[ \\boxed{answer here} \\]</answer>. \
For example: \
Question: What is the nationality of the author of Hamlet? \
<search>Hamlet</search> \
<result>The Tragedy of Hamlet was written by William Shakespeare.</result> \
<search>William Shakespeare</search> \
<result>William Shakespeare was an English playwright.</result> \
<answer>The final answer is \\[ \\boxed{English} \\]</answer>"""

prompt_template_dict = {}
prompt_template_dict['re_search_template'] = re_search_template
# prompt_template_dict['re_search_template_sys'] = re_search_template_sys
prompt_template_dict['re_search_template_sys'] = re_search_template_sys_iter_1
