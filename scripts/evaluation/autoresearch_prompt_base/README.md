# prompt ablations (auto-research) — BASE model

This folder configures an LLM coding agent to run fully autonomous prompt ablations for the ReSearch tool-use format on **Qwen3-0.6B-Base** using raw text completion.

The objective is:

> maximize structural validity of evaluation traces

This stage is not for improving answer quality, training loss, or general reasoning. It is for finding the best prompt that makes the model reliably follow the required protocol from a cold base prompt. Format validity is the primary success criterion.

Required protocol:

```text
<think> -> <tool_call> -> <tool_response> -> <think> -> <answer>
```

The model may repeat additional tool turns when needed:

```text
<think> -> <tool_call> -> <tool_response> -> <think> -> <tool_call> ... -> <answer>
```

The prompt search objective is to make the model mimic this structure reliably. The best prompt is the one that most consistently reproduces the required format across evaluation traces.

## Model Notes

Base models behave very differently from chat-tuned ones:

* No `<|im_start|>` / `<|im_end|>` framing — the model sees a flat completion prompt.
* No hybrid-thinking knob (`enable_thinking`) — `<think>` blocks must be elicited entirely from prompt cues.
* Instruction following is weaker — schemas in the prompt translate into output format much less reliably.
* Tendency to drift, language-mix, or echo the prompt rather than execute on it.
* Cold-start matters: tiny prompt changes (e.g. ending the prompt with `Assistant: <think>`) can flip the model from 0% structural validity to non-trivial structural validity.

## External Guidance

Use the Qwen documentation as a design reference, but read it through a base-model lens:

* Qwen concepts: <https://qwen.readthedocs.io/en/latest/getting_started/concepts.html>
* Qwen3 model card (Hugging Face): <https://huggingface.co/Qwen/Qwen3-0.6B-Base>

What is and is not transferrable:

* Qwen3's documented tool-call JSON schema is still the local source of truth for the *output format*.
* Qwen3's hybrid-thinking control flow is **not** available on the Base variant — do not include `enable_thinking` instructions in the prompt.
* Chat-format examples in the docs (`<|im_start|>system ...`) are inert for the Base model — do not include them.

Local override:

* this runtime expects `{"name": "search", "arguments": {"query": "short factual query"}}`

## Ground Truth Contract

* prompt source: `../../../src/flashrag/verl_legacy/template.py` (key: `re_search_template`)
* execution logic: `../../../src/flashrag/pipeline/active_pipeline.py` (`apply_chat=False` branch)
* structural validator: `../../../src/flashrag/verl_legacy/re_search.py`

The active tool format is:

```xml
<tool_call>{"name": "search", "arguments": {"query": "short factual query"}}</tool_call>
<tool_response>retrieved evidence</tool_response>
```

Do not optimize around the older `<search>` / `<result>` wording.

## Scope

Only this file may be edited during ablation:

```text
../../../src/flashrag/verl_legacy/template.py   # only the re_search_template entry
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

1. edit the `re_search_template` entry in `template.py`
2. do web-based research to find the strongest next prompt hypothesis before any major iteration
3. run evaluation with the base-model command below
4. inspect results
5. score structural validity with the helper in this folder (`--prompt-key re_search_template`)
6. log the outcome
7. iterate without asking for confirmation
8. create or update `CURRENT_BEST.json` and `results.tsv` without asking for permission

Do not pause between iterations unless a hard blocker occurs.

The agent must use targeted web research to form the best next hypothesis before major prompt changes. Useful queries:

* prompting strategies for Qwen3-0.6B-Base
* how to elicit structured tool-call JSON from a small base LM with no chat template
* prompt patterns for cold-start `<think>` elicitation in completion-only LLMs
* how to enforce post-tool reasoning continuation in completion-only LMs
* known failure modes of small base LMs on multi-step structured generation

Use web research to form hypotheses, but validate hypotheses only through local ablation runs.

## Baseline Rule

Every autonomous session must begin by measuring the current `re_search_template` before editing it.

Treat this as:

* `candidate_0`
* description: `baseline current base prompt before new edits`

Required sequence at the start of a fresh session:

1. run evaluation with the current `template.py:re_search_template` unchanged
2. stop the run after 20 completed examples
3. score the run with `ablation_helper.py`
4. record it as `candidate_0`
5. only then begin new prompt edits

Do not skip the baseline run.

## Evaluation Budget

For prompt search, each `run_eval.py` iteration only needs 20 completed examples.

`run_eval.py` does not expose a local sample-count flag here, so the working rule is:

1. start the eval run
2. wait until the active run has 20 completed examples in `intermediate_data.json`
3. stop the eval process
4. score exactly those 20 completed examples with `ablation_helper.py --max-rows 20`

Use this 20-example budget for the baseline and for all prompt comparisons during normal iteration.

## Canonical Runtime

Before running evaluation, ensure:

* conda env: `r_e`
* generator server: `127.0.0.1:3000` serving **Qwen3-0.6B-Base**
* retriever server: `127.0.0.1:3005`

Environment note for this workspace:

* `conda activate r_e` may leave `python` pointing at the base Miniconda install instead of the real env interpreter
* for ablation runs, prefer the explicit interpreter `/home/s4374886/.conda/envs/r_e/bin/python`
* set `CUDA_VISIBLE_DEVICES=0` before `run_eval.py` so Torch initializes against the same single visible GPU from process start
* once the generator and retriever are healthy, routine ablation runs can proceed autonomously with that interpreter + GPU setting
* ablation-specific commands in this loop should run autonomously without asking for additional permission each iteration

If you are already on a node with `r_e` active, check whether the generator and retriever are already running before starting anything. If both services are healthy, do not launch them again.

If the SGLang server is not already running with the **Base** model, launch it:

```bash
cd /home/s4374886/omega/re-search/verl_latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate r_e

python -m sglang.launch_server \
  --served-model-name qwen3-0.6b-base \
  --model-path /zfsstore/user/s4374886/omega/re-search/models/Qwen3-0.6B-Base \
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

If the SGLang server currently has a different model loaded, restart it with the Base model path before any run. Mixing models will silently invalidate scores.
If the retriever is already healthy on `127.0.0.1:3005`, reuse it instead of starting a second copy.

## Canonical Evaluation Command

Use `results/bamboogle_base` as the canonical output location.

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

Notes:

* `--apply_chat False` causes `active_pipeline` to use `re_search_template` (this loop's prompt key) and to skip `tokenizer.apply_chat_template`.
* `--enable_thinking False` is a no-op for the Base model but kept explicit to avoid accidentally toggling on a chat-formatted run.
* `do_sample: False` in `eval_config.yaml` means greedy decoding — runs are deterministic per prompt, so any score change between consecutive runs is attributable to the prompt edit alone.
* stop each prompt-search run after 20 completed examples instead of waiting for the full dataset

Do not assume `metric_score.txt` measures structural validity. It only records answer metrics.

## Structural Validity

Use the local validator in `ablation_helper.py` as the canonical validity definition:

1. paired `<think>` tags
2. exactly one `<answer>...</answer>`
3. correctly ordered `<tool_call>` then `<tool_response>`
4. valid JSON tool payload
5. tool name exactly `"search"`
6. non-empty `arguments.query`
7. answer contains `\boxed{}`
8. after every `<tool_response>`, a new `<think>` must appear before either the next `<tool_call>` or the final `<answer>`

This validity definition is the main optimization target for the project. Partial credit is useful for diagnosis, but prompt selection should prioritize exact reproduction of the required structure.

Typical base-model failure modes:

* the model never emits `<think>` at all (whole response is plain prose)
* the model emits `<think>` but never closes it
* the model emits `<think>` and `<tool_call>` but produces invalid JSON or wrong key names
* the model emits the structural tags but in wrong order or interleaved with non-tag chatter
* language mixing (Chinese / French tokens drifting into the response)
* the model echoes the prompt or produces a few-shot continuation instead of acting

If language-mixing dominates the failure ledger, prefer prompt-level mitigations (an explicit "Output only English" rule, English-only worked example) before reaching for the reward-side `MIXING_PENALTY` knob.

## Required Scoring Step

Use this folder's `ablation_helper.py`.

After editing the prompt:

```bash
cd /home/s4374886/omega/re-search/scripts/evaluation/autoresearch_prompt_base
python ablation_helper.py snapshot-prompt \
  --candidate candidate_7 \
  --prompt-key re_search_template \
  --description "one hypothesis only"
```

After the evaluation run finishes:

```bash
cd /home/s4374886/omega/re-search/scripts/evaluation/autoresearch_prompt_base
python ablation_helper.py score-run \
  --candidate candidate_7 \
  --prompt-key re_search_template \
  --max-rows 20 \
  --status iterate \
  --description "one hypothesis only"
```

`--prompt-key re_search_template` is the default in this folder's helper but pass it explicitly to document intent.

`score-run` will:

* find the newest run directory under `results/bamboogle_base/` if none is provided
* score the completed examples already present in `intermediate_data.json`
* optionally cap scoring to the first `N` completed examples via `--max-rows`
* compute exact-format validity
* compute partial structural credit
* compute inline behavior score
* compute a hybrid score
* write `autoresearch_score.json` into the run directory
* update `prompts.jsonl` (this folder)
* update `results.tsv` (this folder)
* update `CURRENT_BEST.json` (this folder) only when a `keep` candidate becomes the new best

For the first run of a fresh session, use:

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

## Logging

Append one row per candidate to `results.tsv` (this folder):

```text
commit	candidate	status	exact_score	partial_avg	inline_avg	hybrid_avg	dominant_failure	run_dir	description
```

Recommended status values:

* `keep`
* `reject`
* `iterate`

The description should capture one hypothesis only.

`CURRENT_BEST.json` is promoted only by `keep` candidates.

`prompts.jsonl` and `CURRENT_BEST.json` are written by the helper and act as the canonical local run log and best-pointer state.

## Stop Condition

Stop when:

* structural validity is consistently > 0.5 on bamboogle
  Use repeated 20-example runs as the comparison unit during prompt search.
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

Run the prompt ablation loop for the BASE model using run_eval.py.

Edit only the re_search_template entry inside ../../../src/flashrag/verl_legacy/template.py.
Do not modify prompt keys other than `re_search_template`.

Operate fully autonomously:
- modify the re_search_template prompt
- run eval with --apply_chat False --enable_thinking False against Qwen3-0.6B-Base on 127.0.0.1:3000
- stop each eval run after 20 completed examples
- inspect the newest run under results/bamboogle_base/
- score structural validity with ablation_helper.py from this folder
- log outcomes in this folder's results.tsv
- iterate until improvements plateau

Optimize exact format validity first, then use partial and inline scores only as secondary tie-breakers.

Before major changes, use web search to gather model-specific prompt guidance for Qwen3 base-model completion behavior, then test one hypothesis at a time against the local runtime.
```
