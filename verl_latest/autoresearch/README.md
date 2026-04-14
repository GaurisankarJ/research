# autoresearch (ReSearch prompt ablation)

This setup uses an LLM coding agent (e.g. Codex) to perform **prompt ablations** for the ReSearch pipeline.

The goal is to find a system prompt that reliably induces the structured interaction pattern:

```

<think> → <search> → <result> → <think> → <answer>

```

or, when more evidence is needed:

```

<think> → <search> → <result> → <think> → <search>

````

---

## Objective

We are NOT optimizing:

- training loss
- answer accuracy
- benchmark performance

We are ONLY optimizing:

> **format and loop compliance**

The model must reliably follow the ReSearch protocol.

---

## Core files

### Editable (ONLY target)

- `../verl/utils/dataset/re_search_templates.py`

This is where prompt candidates are defined.

---

### Fixed (DO NOT MODIFY)

- `../x_min_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb.sh`  
  → experiment launcher

- `../verl/utils/reward_score/re_search.py`  
  → strict validator + reward logic

- `../verl/experimental/agent_loop/re_search_agent_loop.py`  
  → search loop (injects `<result>` after `<search>`)

---

## How the system works

1. The model emits `<search>query</search>`
2. The agent loop intercepts it
3. A `<result>...</result>` is injected
4. The model must continue reasoning
5. The trajectory ends with `<answer> \boxed{...} </answer>`

If the format is incorrect, the trace is invalid.

---

## Running an experiment

Always use the fixed launcher:

```bash
cd ../
bash x_min_run_qwen3_0.6b_grpo_vllm_instruct_gpu_1_a100_40gb.sh
````

Important properties:

* uses `re_search_template_sys`
* chat format is enabled
* search loop activates if `SEARCH_URL` is set
* rollout JSONL is saved
* short run (`trainer.total_training_steps=100`)

---

## What counts as a good trace

A valid trajectory must:

* include ≥1 `<think>...</think>` (prefer ≥2)
* use `<search>query</search>` correctly (non-empty)
* respect `<search>` → `<result>` ordering
* continue reasoning after `<result>`
* end with exactly one `<answer>...</answer>`
* include `\boxed{...}` in the final answer

---

## Failure modes

Common prompt failures:

* missing `<think>`
* empty or malformed `<search>`
* incorrect tag ordering
* no `<answer>`
* no `\boxed{}`
* early termination
* infinite or degenerate loops

---

## Prompt design strategy

Test small, controlled variations:

1. Minimal contract prompt
2. Strict format enforcement
3. Loop-enforcing prompt
4. Compact few-shot example
5. Anti-degeneration prompt

Each experiment must test **one hypothesis only**.

---

## Evaluating a run

After execution:

1. Locate rollout JSONL (`trainer.rollout_data_dir`)
2. Inspect:

   * `response`
   * `tool_call_counts`
   * `num_turns`
   * `re_search_termination_reason`
3. Count how many traces are structurally valid

---

## Logging results

Record results in `results.tsv`:

```text
commit	candidate	status	good_traces	search_calls_mean	description
```

Example:

```text
a1b2c3	iter_5	keep	7/10	1.4	added explicit loop instruction
```

---

## Experiment loop

Repeat:

1. Modify ONE prompt candidate
2. Set it as active
3. Commit change
4. Run experiment
5. Inspect outputs
6. Log results
7. Keep or revert

---

## Decision rule

Prefer prompts that:

* consistently produce valid structure
* correctly use `<search>`
* terminate cleanly with `<answer>`

If two prompts are similar → prefer simpler one.

---

## Stop condition

Stop when:

* ≥80% of traces are structurally valid

Then proceed to reward tuning / RL.

---

## Running with Codex

Start Codex in the repo:

```bash
codex -m gpt-5.4
```

Then prompt:

```
Read README.md and program.md.
Run the prompt ablation loop.
Edit only re_search_templates.py.
Keep the launcher fixed.
Select the prompt that best induces the ReSearch interaction pattern.
```

---

## Core principle

This stage is not about intelligence.

It is about enforcing a protocol:

```
think → search → result → think → answer
```
