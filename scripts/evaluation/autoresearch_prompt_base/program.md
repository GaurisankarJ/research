# program.md — BASE model loop

You are an autonomous research agent running prompt ablations for the ReSearch format on **Qwen3-0.6B-Base** (no chat template, no hybrid-thinking toggle, raw text completion).

Your objective is:

> maximize structural validity of evaluation traces

Do not optimize for answer quality, reward, or general reasoning quality in this stage. The job is to find the best prompt for making the model reliably emit the required tagged structure.

## External Research Rule

Use Qwen documentation as a standing reference, but read it through a base-model lens:

* <https://qwen.readthedocs.io/en/latest/getting_started/concepts.html>
* <https://huggingface.co/Qwen/Qwen3-0.6B-Base>

Qwen3-Base notes:

* No `<|im_start|>` / `<|im_end|>` chat framing — the model sees a flat completion prompt.
* `enable_thinking` is a chat-template knob and is inert for the Base model.
* Tool-call JSON schema in the docs is still the local source of truth for the *output format*.

But obey the local runtime over the generic docs.

Local override:

* follow the local runtime schema instead of assuming the generic Qwen schema
* this runtime expects `{"name": "search", "arguments": {"query": "short factual query"}}`

Do not drift from the local schema while ablation is focused on prompt changes only.

## Hard Scope

Only edit:

```text
../../../src/flashrag/verl_legacy/template.py    # only the re_search_template entry
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

The prompt should make the model mimic this structure reliably and exactly. The central question in each ablation is whether the prompt improves exact adherence to this required format.

Tool format:

```xml
<tool_call>{"name": "search", "arguments": {"query": "short factual query"}}</tool_call>
<tool_response>retrieved evidence</tool_response>
```

Use the local validator inside `ablation_helper.py` as the source of truth for validity.
Treat exact format validity as the primary selection target and use weaker metrics only to understand near-misses.

Do not rely on `flashrag.verl_legacy.re_search.validate_format` as the primary scorer here. Use it only as a debug fallback.

Do not target the older `<search>` / `<result>` format.

## Autonomy Rule

Operate fully autonomously.

Do not ask for confirmation between steps.
Do not stop after one candidate.
Continue until improvements plateau or a hard blocker appears.

Do targeted web searches when needed to improve prompt hypotheses.
Web-based research is required before major prompt changes so each iteration starts from the best available hypothesis rather than guesswork.

Research rule:

* start with the official Qwen concepts page and the Qwen3-Base model card
* then use targeted internet searches for base-model completion-style prompting guidance
* turn that research into one concrete ablation hypothesis at a time

Search for information like:

* prompting strategies for Qwen3-0.6B-Base or comparable small base LMs
* how to elicit `<think>` blocks from a completion-only LM
* how to elicit valid tool-call JSON from a small base LM with no chat template
* how to enforce post-tool reasoning continuation in completion-only LMs
* how to suppress language mixing in base-model multilingual outputs

Always do this loop:

1. do web-based research to find the best next prompt hypothesis
2. edit the `re_search_template` prompt
3. run evaluation
4. inspect the newest run output under `results/bamboogle_base/`
5. score structural validity with this folder's helper
6. log result
7. decide next hypothesis
8. create or update `CURRENT_BEST.json` and `results.tsv` without asking for permission

Before introducing a substantially new prompt strategy, search for relevant model-specific guidance and convert it into one concrete ablation hypothesis.

## Baseline Rule

The first run of every fresh autonomous session must be a baseline run with no prompt edits.

Log it as:

* candidate: `candidate_0`
* description: `baseline current base prompt before new edits`

Required order:

1. run evaluation with the current `re_search_template` unchanged
2. stop the run after 20 completed examples
3. snapshot and score it as `candidate_0`
4. compare all future candidates against that baseline

## Evaluation Budget

Use only 20 completed examples per `run_eval.py` iteration during prompt search.

`run_eval.py` does not expose a local sample-count flag here, so the working rule is:

1. start the eval run
2. let it produce 20 completed examples in `intermediate_data.json`
3. stop the eval process
4. score exactly those 20 completed examples with `ablation_helper.py --max-rows 20`

Use this 20-example budget for baseline runs and candidate comparisons. Do not wait for a full-dataset pass during normal prompt iteration.

## Keep/Revert Rule

Treat prompt search as a ratcheting loop.

After each evaluated candidate:

* keep the prompt in `template.py:re_search_template` only if it improves the chosen score
* otherwise restore `template.py:re_search_template` to the previous best prompt before the next experiment

Default comparison rule:

1. exact score first
2. hybrid score second
3. simplicity as tie-breaker (prefer the shorter prompt)

The prompt stored in `re_search_template` should always be the best known base-model prompt so far.
Use `CURRENT_BEST.json` (this folder) as the canonical pointer.
Best means best at generating the required format, not best at answer correctness.

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
  --save_dir /zfsstore/user/s4374886/omega/re-search/results/bamboogle_base \
  --save_note research_qwen3_base \
  --sgl_remote_url 127.0.0.1:3000 \
  --remote_retriever_url 127.0.0.1:3005 \
  --generator_model /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B-Base \
  --apply_chat False \
  --enable_thinking False
```

Assume:

* if you are already on a node with `r_e` active, check whether SGLang and the retriever are already running before starting anything
* reuse the existing services if SGLang is healthy on `127.0.0.1:3000` with **Qwen3-0.6B-Base** and the retriever is healthy on `127.0.0.1:3005`
* env is `r_e`

If the SGLang server has a different model loaded, restart it with the Base path before running. Mixing models silently invalidates scores.
If either service is missing, start only the missing one. Do not relaunch healthy existing services.
During prompt search, stop each evaluation run after 20 completed examples rather than waiting for the full dataset.

## Scoring Procedure

Use this folder's `ablation_helper.py`.

After editing the prompt:

```bash
cd /home/s4374886/omega/re-search/scripts/evaluation/autoresearch_prompt_base
python ablation_helper.py snapshot-prompt \
  --candidate candidate_7 \
  --prompt-key re_search_template \
  --description "one hypothesis only"
```

After the run:

```bash
cd /home/s4374886/omega/re-search/scripts/evaluation/autoresearch_prompt_base
python ablation_helper.py score-run \
  --candidate candidate_7 \
  --prompt-key re_search_template \
  --max-rows 20 \
  --status iterate \
  --description "one hypothesis only"
```

`--prompt-key re_search_template` is the default in this folder's helper but pass it explicitly — it documents intent and prevents accidental cross-contamination.

This helper will:

1. locate the newest run directory under `/zfsstore/user/s4374886/omega/re-search/results/bamboogle_base/` unless one is provided
2. score the completed examples already present in `intermediate_data.json`
3. optionally cap scoring to the first `N` completed examples via `--max-rows`
4. score exact structural validity
5. score partial structural validity
6. score inline behavioral quality
7. compute a hybrid score
8. update `prompts.jsonl` (this folder)
9. update `results.tsv` (this folder)
10. write `autoresearch_score.json` into the run directory
11. update `CURRENT_BEST.json` (this folder) only when a `keep` candidate is the new best

For the baseline run of a fresh session:

```bash
cd /home/s4374886/omega/re-search/scripts/evaluation/autoresearch_prompt_base
python ablation_helper.py snapshot-prompt \
  --candidate candidate_0 \
  --prompt-key re_search_template \
  --description "baseline current base prompt before new edits"

python ablation_helper.py score-run \
  --candidate candidate_0 \
  --prompt-key re_search_template \
  --max-rows 20 \
  --status keep \
  --description "baseline current base prompt before new edits"
```

Stop the baseline run after 20 completed examples before scoring it.

Do not use `metric_score.txt` as the primary selection signal. It is secondary in this stage.

## Logging

Append to:

```text
results.tsv      # in this folder
prompts.jsonl    # in this folder
CURRENT_BEST.json  # in this folder
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

## Failure-Driven Iteration (base-model bias)

Map common failures to prompt changes:

* no `<think>` ever emitted → end the prompt with an explicit `Assistant: <think>` prefill, or strengthen the structural template ahead of the user line
* `<think>` opened but never closed → add an explicit closing-tag rule and one minimal in-line example
* `<tool_call>` malformed JSON → reduce wording, give one minimal valid JSON example, drop natural-language tool descriptions
* wrong block ordering → restate the sequence in a shorter, stricter form
* missing `\boxed{}` → harden the final-answer rule with one explicit example
* repeated long thoughts / drift → constrain brevity (e.g. "each `<think>` ≤ N words")
* language mixing / non-English tokens → add an explicit "respond only in English" rule and an English-only worked example; only after that, consider the reward-side `MIXING_PENALTY` as a follow-up
* model echoes the prompt or runs few-shot continuation → trim or reformat the example block; consider replacing few-shot with a single contract paragraph
* model emits `<|endoftext|>` mid-trace → check that EOS is being used as the genuine terminator, not as a structural separator inside a `<think>`

## Strategy

Use this search order:

1. minimal contract + `<think>` cold-start prefill (`Assistant: <think>` ending)
2. one minimal valid worked example
3. strict format enforcement
4. post-tool continuation rule
5. anti-loop and anti-drift rules
6. anti-language-mixing rule

When choosing the next hypothesis, prefer ones motivated by either:

* local failure data from `autoresearch_score.json`
* web research about Qwen base-model prompting behavior

Change one thing at a time.

Prefer the simpler prompt when scores are similar.

## Stop Condition

Stop when:

* structural validity is high and stable across repeated 20-example runs (target > 0.5)
* remaining failures are few and repeatable
* new prompt edits stop improving `good_traces`
