# ReSearch reward (v1)

This document describes [`re_search.py`](re_search.py): how assistant text is parsed, how **format / trajectory** credit is shaped, and how that combines with **answer F1**.

## Entry point

`compute_score(solution_str, ground_truth, tokenizer=None) -> tuple[float, str]`

1. **`_extract_response_text`**: take the assistant segment from a full chat decode (`<|im_start|>assistant\n…`) or use the string as-is.
2. **EOS (optional)**: if `tokenizer.eos_token` is set, the response must end with it; otherwise score `0.0` with reason `over length`, then strip EOS.
3. **`_analyze_format`**: scan the string for paired blocks (`<redacted_thinking>`, `<tool_call>`, `<tool_response>`, `<answer>`), count tags, validate each `search` tool JSON, record transition violations, count full **`redacted_thinking → tool_call → tool_response → redacted_thinking`** cycles, and simple degeneration stats (repeated/empty/short/long thinks, repeated queries).
4. **`_format_reward`**: turn that analysis into a **bounded format score** in **`[0, 0.45]`** plus a reason list (first few reasons may appear in the returned string).

## How format reward is shaped (`_format_reward`)

Weights are chosen so **semantic F1 stays primary**, but **multi-step search loops** still get a clear learning signal.

| Piece | Role |
|--------|------|
| **Tag pairing & hygiene** | Reward matched open/close counts, exactly one `<answer>`, paired tool tags, and **no non-whitespace outside** the four block types. Contributes via `0.10 × clamp01(pair_score)`. |
| **Cycles** (`0.18 × _cycle_score`) | Main structural prior: at least one real tool round (`tool_call` + `tool_response`), a completed **redacted_thinking → tool_call → tool_response → redacted_thinking** window, and **local transition** correctness. Extra cycles are slightly rewarded up to a few rounds; **many cycles + repetition** can be discounted. |
| **Query quality** (`0.07 × _query_quality_score`) | Fraction of `tool_call` payloads that are valid `search` JSON; light penalties for **overlong / messy queries** and **repeated identical** queries. |
| **Thinking quality** (`0.05 × _reasoning_quality_score`) | Prefer **non-empty**, not trivially short, not extremely long, and **non-repeated** thinking blocks. |
| **Sequence bonus** | `+0.03` if there are **no** recorded transition violations. |
| **Answer closure** | `+0.01` non-empty answer body; `+0.01` if `<answer>` is the **last** parsed block. |
| **Penalties** | e.g. empty search queries; mild **excessive tool use** when loop/repetition signals are present. |

Result is **clamped to `[0, 0.45]`**.

## Boxed answer gate

Training still treats **`\boxed{...}` inside `<answer>`** as mandatory for full credit:

- If the answer region lacks a `\boxed{…}` substring, return a **small** score `≤ 0.02` (scaled down from `format_score`) so malformed finals do not look “good enough.”
- Otherwise extract the boxed span, normalize against `ground_truth` (string, list, or `{"target": …}`), and compute **token F1** (`get_f1_score`).

## Final score mix

- **Correct (F1 > 0):** `min(1.0, 0.55 * f1 + format_score)` — F1 dominates; format adds up to the cap above.
- **Wrong (F1 = 0):** small **protocol floor** `min(0.12, 0.01 + 0.25 * format_score)` so well-formed search loops still get a non-zero gradient without beating correct answers.

The second return value is a short diagnostic string (F1, format, cycle count, optional reasons).

## `validate_format` (strict checklist)

Separate from `compute_score`: a **binary** validator for “fully valid” trajectories (paired tags, ≥1 complete tool cycle, all tool calls valid JSON, no transition violations, boxed answer, answer last, no text outside tags). Use this for **audits** or offline pass/fail; **RL shaping** uses `compute_score` as above.
