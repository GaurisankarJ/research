# program.md

You are an autonomous research agent running prompt ablations for the ReSearch format.

Your objective is:

> maximize structural validity of evaluation traces

Do not optimize for answer quality, reward, or general reasoning quality in this stage.

## External Research Rule

Use the Qwen concepts documentation as a standing reference:

* https://qwen.readthedocs.io/en/latest/getting_started/concepts.html

Reasoning:

* Qwen3 is documented as supporting hybrid thinking.
* Qwen3 is documented as supporting multi-step tool calling.
* The model family therefore should be capable of the required behavior if prompted correctly.

But obey the local runtime over the generic docs.

Local override:

* follow the local runtime schema instead of assuming the generic Qwen schema
* this runtime expects `{"name": "search", "arguments": {"query": "short factual query"}}`

Do not drift from the local schema while ablation is focused on prompt changes only.

## Hard Scope

Only edit:

```text
../../../src/flashrag/verl_legacy/template.py
```

Do not modify:

* evaluation scripts
* configs
* datasets
* retriever setup
* runtime setup

## Hard Runtime Contract

The active protocol is:

```text
<think> -> <tool_call> -> <tool_response> -> <think> -> <answer>
```

Tool format:

```xml
<tool_call>{"name": "search", "arguments": {"query": "short factual query"}}</tool_call>
<tool_response>retrieved evidence</tool_response>
```

Use the local validator inside `ablation_helper.py` as the source of truth for validity in this folder.

Do not rely on `flashrag.verl_legacy.re_search.validate_format` as the primary scorer here, because it is not fully aligned with the stricter reasoning-over-search target.

Do not target the older `<search>` / `<result>` format.

## Autonomy Rule

Operate fully autonomously.

Do not ask for confirmation between steps.
Do not stop after one candidate.
Continue until improvements plateau or a hard blocker appears.

Do targeted web searches when needed to improve prompt hypotheses.

Research rule:

* start with the official Qwen concepts page
* then use targeted internet searches to gather model-specific prompting guidance
* turn that research into one concrete ablation hypothesis at a time

Search for information like:

* Qwen hybrid thinking behavior
* Qwen tool-calling templates
* prompt ablation strategies for structured output
* ways to enforce a post-tool reasoning continuation step

Always do this loop:

1. edit prompt
2. run evaluation
3. inspect newest run output
4. score structural validity
5. log result
6. decide next hypothesis

Before introducing a substantially new prompt strategy, search for relevant model-specific guidance and convert it into one concrete ablation hypothesis.

## Baseline Rule

The first run of every fresh autonomous session must be a baseline run with no prompt edits.

Log it as:

* candidate: `candidate_0`
* description: `baseline current prompt before new edits`

Required order:

1. run evaluation with the current prompt unchanged
2. snapshot and score it as `candidate_0`
3. compare all future candidates against that baseline

## Keep/Revert Rule

Treat prompt search as a ratcheting loop.

After each evaluated candidate:

* keep the prompt in `template.py` only if it improves the chosen score
* otherwise restore `template.py` to the previous best prompt before the next experiment

Default comparison rule:

1. exact score first
2. hybrid score second
3. simplicity as tie-breaker

The prompt stored in `template.py` should always be the best known prompt so far.
Use `CURRENT_BEST.json` as the canonical pointer to that best known prompt.
It should store the actual prompt text and scoring metadata, not unrelated git bookkeeping like `commit`.

## Canonical Evaluation Command

Run:

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

Assume:

* SGLang server is already running on `127.0.0.1:3000`
* retriever server is already running on `127.0.0.1:3005`
* env is `r_e`

## Scoring Procedure

Use `ablation_helper.py` as the default scoring and logging interface.

After editing the prompt:

```bash
cd /zfsstore/user/s4374886/omega/re-search/scripts/evaluation/autoresearch_prompt
python ablation_helper.py snapshot-prompt \
  --candidate candidate_7 \
  --description "one hypothesis only"
```

After the run:

```bash
cd /zfsstore/user/s4374886/omega/re-search/scripts/evaluation/autoresearch_prompt
python ablation_helper.py score-run \
  --candidate candidate_7 \
  --status iterate \
  --description "one hypothesis only"
```

This helper will:

1. locate the newest run directory under `/zfsstore/user/s4374886/omega/re-search/results/bamboogle/` unless one is provided
2. score exact structural validity
3. score partial structural validity
4. score inline behavioral quality
5. compute a hybrid score
6. update `prompts.jsonl`
7. update `results.tsv`
8. write `autoresearch_score.json` into the run directory

Use the raw repo validator workflow only as a debug fallback for inspecting compatibility issues manually.

For the baseline run of a fresh session:

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

Do not use `metric_score.txt` as the primary selection signal. It is secondary in this stage.

## Logging

Append to:

```text
results.tsv
```

Format:

```text
commit	candidate	status	exact_score	partial_avg	inline_avg	hybrid_avg	dominant_failure	run_dir	description
```

Status meanings:

* `keep`: clearly better
* `reject`: clearly worse or not useful
* `iterate`: promising but incomplete

The description must describe exactly one tested hypothesis.

Column meanings:

* `exact_score`: strict `think -> tool_call -> tool_response -> think -> answer` score
* `partial_avg`: average partial structural score
* `inline_avg`: average inline behavior score
* `hybrid_avg`: secondary comparison score after exact score
* `dominant_failure`: most common exact failure mode

Use `ablation_helper.py` to keep logging consistent.

Recommended flow:

```bash
python ablation_helper.py snapshot-prompt \
  --candidate candidate_7 \
  --description "one hypothesis only"

python ablation_helper.py score-run \
  --candidate candidate_7 \
  --status iterate \
  --description "one hypothesis only"
```

The helper updates both `results.tsv` and `prompts.jsonl`, writes per-run scoring summaries, and updates `CURRENT_BEST.json` when a candidate becomes the best known prompt.

## Failure-Driven Iteration

Map failures to changes:

* missing `</answer>` -> strengthen completion and closing-tag rules
* unpaired `<think>` -> simplify or shorten thought requirements
* malformed tool JSON -> reduce wording around tool format and add one minimal example
* wrong ordering -> restate the sequence in a shorter, stricter form
* missing `\boxed{}` -> harden final answer rule
* repeated long thoughts -> constrain brevity and discourage repetition

## Strategy

Use this search order:

1. minimal contract
2. strict format enforcement
3. post-tool continuation rule
4. one minimal valid example
5. anti-loop and anti-drift rules

When choosing the next hypothesis, prefer ones motivated by either:

* local failure data from `autoresearch_score.json`
* web research about Qwen prompting behavior

Change one thing at a time.

Prefer the simpler prompt when scores are similar.

## Stop Condition

Stop when:

* structural validity is high and stable
* remaining failures are few and repeatable
* new prompt edits stop improving `good_traces`
