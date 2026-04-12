# prompt ablations (auto-research)

This folder configures an LLM coding agent to run fully autonomous prompt ablations for the ReSearch tool-use format.

The objective is narrow:

> maximize structural validity of evaluation traces

This stage is not for improving answer quality, training loss, or general reasoning. It is for making the model reliably follow the required protocol.

Required protocol:

```text
<think> -> <tool_call> -> <tool_response> -> <think> -> <answer>
```

The model may repeat additional tool turns when needed:

```text
<think> -> <tool_call> -> <tool_response> -> <think> -> <tool_call> ... -> <answer>
```

## External Guidance

Use the Qwen concepts documentation as a design reference when crafting prompts for this model family:

* Qwen concepts: https://qwen.readthedocs.io/en/latest/getting_started/concepts.html

Why this matters:

* Qwen3 is documented as supporting hybrid thinking modes.
* Qwen3 is documented as supporting multi-step tool calling.
* Qwen documents a `<think>` format and a tool-calling format, so prompt ablations should be informed by how the model family was designed to behave.

However, use the Qwen docs for behavioral intuition only. The source of truth for this setup is still the local runtime contract in this repo.

Important local override:

* use the local runtime schema, not the generic Qwen schema by assumption
* this repo's current runtime expects `{"name": "search", "arguments": {"query": "short factual query"}}`

## Ground Truth Contract

The runtime contract is defined by code, not by prose:

* prompt source: `../../../src/flashrag/verl_legacy/template.py`
* execution logic: `../../../src/flashrag/pipeline/active_pipeline.py`
* structural validator: `../../../src/flashrag/verl_legacy/re_search.py`

The active tool format is:

```xml
<tool_call>{"name": "search", "arguments": {"query": "short factual query"}}</tool_call>
<tool_response>retrieved evidence</tool_response>
```

Do not optimize around the older `<search>` / `<result>` wording. The active pipeline expects `<tool_call>` / `<tool_response>`.

When deciding how to improve the prompt, use both:

* local code truth for the required output format
* external model-family guidance for how Qwen naturally expresses thinking and tool use

## Scope

Only this file may be edited during ablation:

```text
../../../src/flashrag/verl_legacy/template.py
```

Do not modify:

* `run_eval.py`
* `eval_config.yaml`
* datasets
* retriever setup
* server runtime

## Operating Mode

This setup is for fully autonomous ablation mode.

The agent must:

1. edit the prompt
2. run evaluation
3. inspect results
4. score structural validity
5. log the outcome
6. iterate without asking for confirmation

Do not pause between iterations unless a hard blocker occurs.

The agent is also expected to do targeted web research before major prompt changes when it needs better priors for eliciting Qwen behavior.

Research requirement before new prompt families:

* check the official Qwen concepts page first
* then do targeted internet searches for model-specific prompting guidance
* form one concrete ablation hypothesis from that research
* validate the hypothesis locally with ablations

Examples of useful web research:

* Qwen tool-calling conventions
* Qwen thinking-mode behavior
* prompt patterns for enforcing post-tool continuation
* known prompting strategies for reducing malformed structured outputs
* model-specific advice for smaller hybrid-thinking models

Use web research to form hypotheses, but validate hypotheses only through local ablation runs.

## Baseline Rule

Every autonomous session must begin by measuring the current prompt before editing it.

Treat this as:

* `candidate_0`
* description: `baseline current prompt before new edits`

Required sequence at the start of a fresh session:

1. run evaluation with the current `template.py` prompt unchanged
2. score the run with `ablation_helper.py`
3. record it as `candidate_0`
4. only then begin new prompt edits

Do not skip the baseline run.

## Canonical Runtime

Before running evaluation, ensure:

* conda env: `r_e`
* generator server: `127.0.0.1:3000`
* retriever server: `127.0.0.1:3005`

If the SGLang server is not already running, launch it:

```bash
cd /home/s4374886/omega/re-search/verl_latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate r_e

python -m sglang.launch_server \
  --served-model-name qwen3-0.6b \
  --model-path /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B \
  --tp 1 \
  --context-length 8192 \
  --enable-metrics \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 3000 \
  --trust-remote-code \
  --disable-overlap \
  --disable-radix-cache
```

Wait until the server is ready before evaluation.

## Canonical Evaluation Command

Use `results/bamboogle` as the canonical output location.

```bash
cd /home/s4374886/omega/re-search/scripts/evaluation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate r_e

python run_eval.py \
  --config_path eval_config.yaml \
  --method_name research \
  --data_dir /zfsstore/user/s4374886/omega/re-search/data \
  --dataset_name bamboogle \
  --split test \
  --save_dir /zfsstore/user/s4374886/omega/re-search/results/bamboogle \
  --save_note research_qwen3_reasoning_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B \
  --apply_chat True \
  --enable_thinking True
```

Do not assume `metric_score.txt` measures structural validity. It only records answer metrics.

## Structural Validity

Use the local validator in `ablation_helper.py` as the canonical validity definition for this folder.

Reason:

* the repo-level `flashrag.verl_legacy.re_search.validate_format` is not fully aligned with the stricter reasoning-over-search target in this folder
* the helper now enforces the intended grammar, including the required post-tool `<think>` step

A trace is valid only if it satisfies the code-level checks, including:

1. paired `<think>` tags
2. exactly one `<answer>...</answer>`
3. correctly ordered `<tool_call>` then `<tool_response>`
4. valid JSON tool payload
5. tool name exactly `"search"`
6. non-empty `arguments.query`
7. answer contains `\boxed{}`
8. after every `<tool_response>`, a new `<think>` must appear before either the next `<tool_call>` or the final `<answer>`

Typical failure modes:

* missing closing `</answer>`
* unpaired `<think>`
* malformed JSON in `<tool_call>`
* wrong tag ordering
* missing `\boxed{}`
* plain text outside required tags
* overlong or repetitive `<think>` blocks that lead to drift

## Required Scoring Step

Use `ablation_helper.py` as the canonical scoring and logging workflow.

After editing the prompt, snapshot it:

```bash
cd /zfsstore/user/s4374886/omega/re-search/scripts/evaluation/autoresearch_prompt
python ablation_helper.py snapshot-prompt \
  --candidate candidate_7 \
  --description "one hypothesis only"
```

After the evaluation run finishes, score and log it:

```bash
cd /zfsstore/user/s4374886/omega/re-search/scripts/evaluation/autoresearch_prompt
python ablation_helper.py score-run \
  --candidate candidate_7 \
  --status iterate \
  --description "one hypothesis only"
```

`score-run` will:

* find the newest run directory if none is provided
* compute exact-format validity
* compute partial structural credit
* compute inline behavior score
* compute a hybrid score
* write `autoresearch_score.json` into the run directory
* update `prompts.jsonl`
* update `results.tsv`

Use the raw repo validator command only for debugging compatibility issues. Do not treat it as the primary scoring rule for this folder.

For the first run of a fresh session, use:

```bash
cd /zfsstore/user/s4374886/omega/re-search/scripts/evaluation/autoresearch_prompt
python ablation_helper.py snapshot-prompt \
  --candidate candidate_0 \
  --description "baseline current prompt before new edits"

python ablation_helper.py score-run \
  --candidate candidate_0 \
  --status keep \
  --description "baseline current prompt before new edits"
```

Debug fallback:

```bash
cd /home/s4374886/omega/re-search/scripts/evaluation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate r_e

python - <<'PY'
import json
import sys
from collections import Counter

sys.path.insert(0, "/zfsstore/user/s4374886/omega/re-search/src")
from flashrag.verl_legacy.re_search import validate_format

path = "/zfsstore/user/s4374886/omega/re-search/results/bamboogle/<run_dir>/intermediate_data.json"

with open(path) as f:
    data = json.load(f)

valid = 0
failures = Counter()
for row in data:
    text = row["output"].get("final_response", "")
    ok, reason = validate_format(text)
    if ok:
        valid += 1
    else:
        failures[reason] += 1

total = len(data)
print(f"validator_valid\t{valid}/{total} = {valid/total:.3f}")
print("top_failures")
for reason, count in failures.most_common():
    print(f"{count}\t{reason}")
PY
```

## Logging

Append one row per candidate to `results.tsv`:

```text
commit	candidate	status	exact_score	partial_avg	inline_avg	hybrid_avg	dominant_failure	run_dir	description
```

Recommended status values:

* `keep`
* `reject`
* `iterate`

The description should capture one hypothesis only.

Good example:

```text
a1b2c3	candidate_4	keep	118/125 = 0.944	0.972	0.811	0.948	missing_post_tool_think	bamboogle_2026_04_12_20_15_research_qwen3_reasoning_base	added stricter answer-closing rule without changing tool JSON format
```

Column meanings:

* `exact_score`: strict target protocol score
* `partial_avg`: average partial structural credit
* `inline_avg`: average inline behavior score
* `hybrid_avg`: aggregate tie-break score after exact score
* `dominant_failure`: most common exact failure mode

Also maintain `prompts.jsonl` with one record per ablated prompt. Each record should include the full prompt text, candidate id, status, run dir, and scoring summary.

Also maintain `CURRENT_BEST.json` as the single source of truth for the current best prompt candidate.

It should contain:

* `candidate`
* `exact_score`
* `partial_avg`
* `inline_avg`
* `hybrid_avg`
* `dominant_failure`
* `run_dir`
* `description`
* `prompt`
* `prompt_key`
* `prompt_sha256`

It should not include unrelated git bookkeeping like `commit`. `CURRENT_BEST.json` exists to point to the active best prompt and its scoring state, not to act as a git log.

Preferred helper flow:

```bash
python ablation_helper.py snapshot-prompt \
  --candidate candidate_7 \
  --description "one-sentence hypothesis before evaluation"

python ablation_helper.py score-run \
  --candidate candidate_7 \
  --status keep \
  --description "one-sentence hypothesis before evaluation"
```

`score-run` is the default interface for scoring and logging runs.
It should also update `CURRENT_BEST.json` whenever a candidate becomes the new best prompt under the keep/revert rule.

## Experiment Discipline

Each iteration must test exactly one hypothesis.

Suggested progression:

1. minimal contract
2. stricter sequencing
3. stronger post-tool continuation
4. minimal valid example
5. anti-loop and anti-drift rules

Before trying a new prompt family, do a short web search if needed and write down the hypothesis in terms of Qwen behavior. Examples:

* "Qwen may obey the post-tool `<think>` step better if the prompt matches its documented multi-step tool-calling rhythm."
* "Qwen may need a shorter contract plus one valid example because the model is small and hybrid."
* "Qwen may over-collapse to direct answer after tool response unless continuation is made explicit."

Prefer the simpler prompt if two candidates perform similarly.

## Stop Condition

Stop when:

* structural validity is consistently high
* remaining failures are rare and repeatable
* further edits do not produce meaningful gains

## Codex Usage

Start the coding agent:

```bash
codex -m gpt-5.4
```

Prompt:

```text
Read README.md and program.md.

Run the prompt ablation loop using run_eval.py.

Edit only ../../../src/flashrag/verl_legacy/template.py.

Operate fully autonomously:
- modify prompt
- run eval
- inspect results
- score structural validity from intermediate_data.json
- log outcomes in results.tsv
- iterate until improvements plateau

Optimize structural validity only.

Before major changes, use web search to gather model-specific prompt guidance for Qwen3 hybrid thinking and tool use, then test one hypothesis at a time against the local runtime.
```
