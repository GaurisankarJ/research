# autoresearch (prompt-only ablation)

Goal:
Find a system prompt that makes Qwen3-0.6B reliably produce the required ReSearch interaction format.

We are NOT optimizing training performance.
We are ONLY optimizing output structure.

---

## Target behavior

Each response must follow one of these valid patterns:

### With search:
<think>...</think>
<search>...</search>
<result>...</result>
<think>...</think>
<answer> \boxed{...} </answer>

### Multi-hop:
<think>...</think>
<search>...</search>
<result>...</result>
<think>...</think>
<search>...</search>
<result>...</result>
<think>...</think>
<answer>...</answer>

All tags must be properly opened and closed.

---

## Allowed changes

You may ONLY edit:

- ../verl/utils/dataset/re_search_templates.py

Specifically:
- Add new `re_search_template_sys_iter_*` prompt candidates
- Switch `prompt_template_dict["re_search_template_sys"]` to one candidate at a time

---

## Forbidden changes

Do NOT modify:

- reward logic (`reward_score/re_search.py`)
- agent loop (`re_search_agent_loop.py`)
- launcher script
- datasets
- evaluation code
- any other files

---

## Fixed experiment command

Run exactly:

```bash
cd ..
bash x_min_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb.sh
````

Do not change this command.

---

## What to optimize

We optimize **format correctness**, not answer accuracy.

A "good trace" satisfies ALL:

* at least one `<think>...</think>` block
* exactly one `<answer>...</answer>` block
* `<search>` (if present) is non-empty and properly closed
* `<search>` → `<result>` ordering is correct
* final answer contains `\boxed{...}`

---

## Metrics to log

Record each run in `results.tsv`:

```text
commit	candidate	status	good_traces	search_calls_mean	description
```

Where:

* `commit`: short git hash
* `candidate`: prompt name (e.g. re_search_template_sys_iter_5)
* `status`: keep / discard / crash
* `good_traces`: e.g. "7/10"
* `search_calls_mean`: average tool calls
* `description`: hypothesis being tested

---

## How to inspect a run

After running:

1. Locate rollout JSONL directory from logs
2. Inspect entries:

   * `response`
   * `tool_call_counts`
   * `num_turns`
3. Manually count good traces
4. Identify failure modes:

   * missing `<think>`
   * malformed `<search>`
   * no `<answer>`
   * no `\boxed{}`
   * early termination

---

## Prompt design strategy

Test small, controlled variations:

1. Minimal contract prompt
2. Explicit format enforcement
3. Loop-enforcing prompt
4. Compact few-shot example
5. Anti-degeneration prompt

Each experiment = ONE clear hypothesis.

---

## Experiment loop

Repeat:

1. Inspect current best prompt
2. Add or modify ONE candidate
3. Set it active
4. Commit change
5. Run experiment
6. Inspect outputs
7. Log results in `results.tsv`
8. Keep or revert

---

## Decision rule

Prefer prompts that:

* produce valid structure consistently
* use `<search>` correctly when needed
* terminate cleanly with `<answer>`

If two prompts are similar → prefer simpler one.

---

## Important constraints

* Do NOT optimize for correctness of answers
* Do NOT change multiple variables at once
* Keep prompts short and interpretable
* Prefer strict instructions over verbose explanations

---

## Stop condition

Stop when:

* ≥80% traces are structurally valid

Then move to next stage (reward tuning / RL).

---

## Core principle

We are not training intelligence.

We are enforcing a protocol.

The goal is to make the model reliably follow:

think → search → result → think → answer
