#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SRC_ROOT = REPO_ROOT / "src"
TEMPLATE_PATH = SRC_ROOT / "flashrag" / "verl_legacy" / "template.py"
RESULTS_ROOT = REPO_ROOT / "results" / "bamboogle_base"
RESULTS_TSV = HERE / "results.tsv"
PROMPTS_JSONL = HERE / "prompts.jsonl"
CURRENT_BEST_JSON = HERE / "CURRENT_BEST.json"
PROMPT_KEY = "re_search_template"
RESULTS_HEADER = (
    "commit\tcandidate\tstatus\texact_score\tpartial_avg\tinline_avg\thybrid_avg\tdominant_failure\trun_dir\tdescription\n"
)

BLOCK_RE = re.compile(
    r"<(think|tool_call|tool_response|answer)>(.*?)</\1>",
    re.DOTALL,
)


@dataclass
class TraceScore:
    exact_valid: bool
    exact_reason: str
    validator_valid: bool
    validator_reason: str
    partial_score: float
    inline_score: float
    hybrid_score: float
    block_types: list[str]
    failure_labels: list[str]
    component_scores: dict[str, float]
    stats: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_prompt(prompt_key: str) -> str:
    spec = importlib.util.spec_from_file_location("autoresearch_template", TEMPLATE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load template module from {TEMPLATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt = module.prompt_template_dict[prompt_key]
    if not isinstance(prompt, str):
        raise TypeError(f"prompt {prompt_key!r} is not a string")
    return prompt


def git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_ws(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def parse_blocks(text: str) -> tuple[list[dict[str, Any]], bool]:
    blocks = []
    pos = 0
    stray_text = False
    for match in BLOCK_RE.finditer(text):
        if text[pos:match.start()].strip():
            stray_text = True
        blocks.append(
            {
                "type": match.group(1),
                "content": match.group(2),
                "start": match.start(),
                "end": match.end(),
            }
        )
        pos = match.end()
    if text[pos:].strip():
        stray_text = True
    return blocks, stray_text


def tool_payload_valid(payload: str) -> tuple[bool, str, str]:
    try:
        obj = json.loads(payload.strip())
    except Exception:
        return False, "malformed_tool_json", ""
    if not isinstance(obj, dict):
        return False, "tool_call_not_object", ""
    if obj.get("name") != "search":
        return False, "tool_name_not_search", ""
    args = obj.get("arguments")
    if not isinstance(args, dict):
        return False, "tool_arguments_not_object", ""
    query_value = args.get("query")
    if not isinstance(query_value, str):
        return False, "tool_arguments_query_not_string", ""
    query = query_value.strip()
    if not query:
        return False, "empty_search_query", ""
    return True, "ok", query


def has_well_formed_boxed(answer_text: str) -> bool:
    start = answer_text.find("\\boxed{")
    while start != -1:
        idx = start + len("\\boxed{")
        depth = 1
        while idx < len(answer_text):
            char = answer_text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return True
            idx += 1
        start = answer_text.find("\\boxed{", start + 1)
    return False


def exact_sequence_reason(block_types: list[str], stray_text: bool) -> tuple[bool, str]:
    if stray_text:
        return False, "text_outside_tags"
    if not block_types:
        return False, "no_blocks"
    if block_types[0] != "think":
        return False, "not_start_with_think"
    if block_types[-1] != "answer":
        return False, "not_end_with_answer"
    if len(block_types) < 5:
        if block_types == ["think", "tool_call", "tool_response", "answer"]:
            return False, "missing_post_tool_think"
        return False, "missing_required_tool_turn"

    idx = 1
    tool_turns = 0
    while idx < len(block_types) - 1:
        if idx + 2 >= len(block_types):
            return False, "truncated_before_answer"
        chunk = block_types[idx : idx + 3]
        if chunk == ["tool_call", "tool_response", "answer"]:
            return False, "missing_post_tool_think"
        if chunk != ["tool_call", "tool_response", "think"]:
            return False, "bad_block_order"
        tool_turns += 1
        idx += 3

    if idx != len(block_types) - 1:
        return False, "bad_block_order"
    if tool_turns < 1:
        return False, "missing_required_tool_turn"
    return True, "exact_required_sequence"


def validate_reasoning_search_format(text: str) -> tuple[bool, str]:
    stripped = (text or "").strip()
    blocks, stray_text = parse_blocks(stripped)
    block_types = [b["type"] for b in blocks]

    exact_ok, exact_reason = exact_sequence_reason(block_types, stray_text)
    if not exact_ok:
        return False, exact_reason

    if stripped.count("<think>") != stripped.count("</think>") or stripped.count("<think>") == 0:
        return False, "<think> </think> not paired"
    if stripped.count("<answer>") != 1 or stripped.count("</answer>") != 1:
        return False, "<answer> or </answer> not found"

    for block in blocks:
        if block["type"] == "tool_call":
            valid_payload, payload_reason, _ = tool_payload_valid(block["content"])
            if not valid_payload:
                return False, payload_reason

    answer_blocks = [b["content"] for b in blocks if b["type"] == "answer"]
    answer_text = answer_blocks[-1] if answer_blocks else ""
    if not has_well_formed_boxed(answer_text):
        return False, "answer is missing \\boxed{} format"

    return True, "format is correct"


def score_reasoning_length(think_blocks: list[str]) -> float:
    if not think_blocks:
        return 0.0
    scores = []
    for block in think_blocks:
        words = count_words(block)
        if words <= 1:
            score = 0.1
        elif words <= 4:
            score = 0.55
        elif words <= 120:
            score = 1.0
        elif words <= 220:
            score = 0.9
        elif words <= 320:
            score = 0.72
        else:
            score = 0.45
        scores.append(score)
    return round(sum(scores) / len(scores), 6)


def score_repetition(think_blocks: list[str]) -> float:
    if not think_blocks:
        return 0.0
    normalized = [normalize_ws(t) for t in think_blocks if normalize_ws(t)]
    if not normalized:
        return 0.0
    unique = len(set(normalized))
    return round(unique / len(normalized), 6)


def score_tool_efficiency(tool_turns: int, queries: list[str], think_blocks: list[str]) -> float:
    if tool_turns <= 0:
        return 0.0
    if tool_turns == 1:
        base = 0.9
    elif tool_turns == 2:
        base = 1.0
    elif tool_turns == 3:
        base = 0.98
    elif tool_turns <= 5:
        base = 0.9
    else:
        base = 0.8

    normalized_queries = [normalize_ws(q) for q in queries if normalize_ws(q)]
    query_unique_ratio = len(set(normalized_queries)) / len(normalized_queries) if normalized_queries else 0.0
    think_unique_ratio = len(set(normalize_ws(t) for t in think_blocks if normalize_ws(t))) / len(think_blocks) if think_blocks else 0.0

    if tool_turns >= 6 and query_unique_ratio < 0.5:
        base -= 0.25
    elif tool_turns >= 4 and query_unique_ratio < 0.67:
        base -= 0.1

    if tool_turns >= 6 and think_unique_ratio < 0.5:
        base -= 0.15

    return round(clamp01(base), 6)


def score_reasoning_cycles(block_types: list[str]) -> float:
    if len(block_types) < 2:
        return 0.0

    score = 0.0
    max_score = 0.0
    for i in range(len(block_types) - 1):
        cur = block_types[i]
        nxt = block_types[i + 1]
        max_score += 1.0
        if cur == "think" and nxt in {"tool_call", "answer"}:
            score += 1.0
        elif cur == "tool_call" and nxt == "tool_response":
            score += 1.0
        elif cur == "tool_response" and nxt == "think":
            score += 1.2
        elif cur == "tool_response" and nxt == "answer":
            score -= 1.0
        elif cur == "tool_call" and nxt == "answer":
            score -= 1.0
        elif cur == "tool_call" and nxt == "tool_call":
            score -= 1.0
        elif cur == "tool_response" and nxt == "tool_response":
            score -= 1.0
        elif cur == "answer":
            score -= 0.5

    return round(clamp01(score / max_score), 6) if max_score else 0.0


def score_exploration_quality(queries: list[str], think_blocks: list[str]) -> float:
    if not queries:
        return 0.0

    normalized_queries = [normalize_ws(q) for q in queries if normalize_ws(q)]
    unique_query_ratio = len(set(normalized_queries)) / len(normalized_queries) if normalized_queries else 0.0

    repeated_query_penalty = 0.0
    if normalized_queries:
        query_counts = Counter(normalized_queries)
        repeated_query_penalty = sum(count - 1 for count in query_counts.values() if count > 1) / len(normalized_queries)

    normalized_thinks = [normalize_ws(t) for t in think_blocks if normalize_ws(t)]
    repeated_think_penalty = 0.0
    if normalized_thinks:
        think_counts = Counter(normalized_thinks)
        repeated_think_penalty = sum(count - 1 for count in think_counts.values() if count > 1) / len(normalized_thinks)

    score = 0.65 + 0.35 * unique_query_ratio - 0.35 * repeated_query_penalty - 0.2 * repeated_think_penalty
    return round(clamp01(score), 6)


def score_query_quality(queries: list[str], tool_json_valid_ratio: float) -> float:
    if not queries:
        return 0.0
    scores = []
    for query in queries:
        words = count_words(query)
        score = 1.0
        if "\n" in query or "<" in query or ">" in query:
            score -= 0.35
        if words < 2:
            score -= 0.4
        elif words > 12:
            score -= min(0.5, (words - 12) * 0.03)
        scores.append(clamp01(score))
    return round(clamp01((sum(scores) / len(scores)) * tool_json_valid_ratio), 6)


def score_answer_closure(answer_text: str, answer_at_end: bool, stray_text: bool) -> float:
    if not answer_text:
        return 0.0
    score = 1.0 if answer_at_end else 0.5
    lowered = answer_text.lower()
    for phrase in ("i think", "let me", "i should", "i need to"):
        if phrase in lowered:
            score -= 0.15
    if "<think>" in answer_text or "<tool_call>" in answer_text or "<tool_response>" in answer_text:
        score -= 0.4
    if stray_text:
        score -= 0.2
    return round(clamp01(score), 6)


def summarize_failure_labels(score: TraceScore) -> list[str]:
    labels = []
    if not score.exact_valid:
        labels.append(score.exact_reason)
    if not score.validator_valid and score.validator_reason != score.exact_reason:
        labels.append(score.validator_reason)
    if score.component_scores["tool_json_valid_ratio"] < 1.0:
        labels.append("tool_json_issues")
    if score.component_scores["reasoning_length"] < 0.4:
        labels.append("weak_or_rambling_think")
    if score.component_scores["tool_efficiency"] < 0.5:
        labels.append("tool_inefficiency_or_missing_tool")
    if score.component_scores["reasoning_cycles"] < 0.6:
        labels.append("bad_reasoning_cycle")
    if score.component_scores["exploration_quality"] < 0.6:
        labels.append("low_query_diversity_or_repeat_loop")
    if score.component_scores["repetition"] < 1.0:
        labels.append("repetition")
    return labels


def score_trace(text: str) -> TraceScore:
    raw = text or ""
    stripped = raw.strip()
    blocks, stray_text = parse_blocks(stripped)
    block_types = [b["type"] for b in blocks]
    think_blocks = [b["content"] for b in blocks if b["type"] == "think"]
    answer_blocks = [b["content"] for b in blocks if b["type"] == "answer"]
    tool_blocks = [b["content"] for b in blocks if b["type"] == "tool_call"]
    tool_responses = [b["content"] for b in blocks if b["type"] == "tool_response"]
    answer_text = answer_blocks[-1] if answer_blocks else ""
    answer_at_end = bool(blocks) and blocks[-1]["type"] == "answer"

    exact_seq_ok, exact_reason = exact_sequence_reason(block_types, stray_text)
    validator_valid, validator_reason = validate_reasoning_search_format(stripped)

    valid_queries = []
    tool_json_valid = 0
    for payload in tool_blocks:
        ok, _, query = tool_payload_valid(payload)
        if ok:
            tool_json_valid += 1
            valid_queries.append(query)

    starts_with_think = 1.0 if stripped.startswith("<think>") else 0.0
    no_stray_text = 1.0 if not stray_text else 0.0
    paired_thinks = 1.0 if raw.count("<think>") == raw.count("</think>") and raw.count("<think>") > 0 else 0.0
    one_answer = 1.0 if raw.count("<answer>") == 1 and raw.count("</answer>") == 1 else 0.0
    answer_has_boxed = 1.0 if has_well_formed_boxed(answer_text) else 0.0
    at_least_one_tool_turn = 1.0 if len(tool_blocks) >= 1 and len(tool_responses) >= 1 else 0.0
    tool_json_valid_ratio = tool_json_valid / len(tool_blocks) if tool_blocks else 0.0

    seq_score = score_reasoning_cycles(block_types)

    partial_score = (
        0.08 * starts_with_think
        + 0.08 * paired_thinks
        + 0.15 * one_answer
        + 0.08 * answer_has_boxed
        + 0.08 * no_stray_text
        + 0.15 * at_least_one_tool_turn
        + 0.15 * tool_json_valid_ratio
        + 0.23 * seq_score
    )
    partial_score = round(clamp01(partial_score), 6)

    reasoning_length = score_reasoning_length(think_blocks)
    repetition = score_repetition(think_blocks)
    tool_efficiency = score_tool_efficiency(min(len(tool_blocks), len(tool_responses)), valid_queries, think_blocks)
    query_quality = score_query_quality(valid_queries, tool_json_valid_ratio)
    answer_closure = score_answer_closure(answer_text, answer_at_end, stray_text)
    exploration_quality = score_exploration_quality(valid_queries, think_blocks)

    inline_score = (
        0.18 * reasoning_length
        + 0.17 * repetition
        + 0.20 * tool_efficiency
        + 0.15 * query_quality
        + 0.12 * answer_closure
        + 0.18 * seq_score
        + 0.20 * exploration_quality
    )
    inline_score = round(clamp01(inline_score), 6)

    rule_score = 1.0 if exact_seq_ok and validator_valid else min(0.75 * partial_score + 0.25 * seq_score, 0.95)
    hybrid_score = round(clamp01(0.9 * rule_score + 0.1 * inline_score), 6)

    component_scores = {
        "starts_with_think": starts_with_think,
        "paired_thinks": paired_thinks,
        "one_answer": one_answer,
        "answer_has_boxed": answer_has_boxed,
        "no_stray_text": no_stray_text,
        "has_tool_turn": at_least_one_tool_turn,
        "tool_json_valid_ratio": round(tool_json_valid_ratio, 6),
        "reasoning_cycles": seq_score,
        "reasoning_length": reasoning_length,
        "repetition": repetition,
        "tool_efficiency": tool_efficiency,
        "query_quality": query_quality,
        "answer_closure": answer_closure,
        "exploration_quality": exploration_quality,
    }

    stats = {
        "num_blocks": len(blocks),
        "num_think": len(think_blocks),
        "num_tool_call": len(tool_blocks),
        "num_tool_response": len(tool_responses),
        "num_answer": len(answer_blocks),
        "avg_think_words": round(sum(count_words(t) for t in think_blocks) / len(think_blocks), 3) if think_blocks else 0.0,
        "max_think_words": max((count_words(t) for t in think_blocks), default=0),
    }

    score = TraceScore(
        exact_valid=bool(exact_seq_ok and validator_valid),
        exact_reason=exact_reason if not (exact_seq_ok and validator_valid) else "exact_valid",
        validator_valid=bool(validator_valid),
        validator_reason=validator_reason,
        partial_score=partial_score,
        inline_score=inline_score,
        hybrid_score=hybrid_score,
        block_types=block_types,
        failure_labels=[],
        component_scores=component_scores,
        stats=stats,
    )
    score.failure_labels = summarize_failure_labels(score)
    return score


def newest_run_dir(results_root: Path) -> Path:
    dirs = [p for p in results_root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"no run directories found under {results_root}")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def read_intermediate_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "intermediate_data.json"
    if not path.exists():
        raise FileNotFoundError(f"missing intermediate_data.json in {run_dir}")
    data = load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"expected list in {path}")
    return data


def is_completed_row(row: dict[str, Any]) -> bool:
    output = row.get("output")
    if not isinstance(output, dict):
        return False
    final_response = output.get("final_response")
    if not isinstance(final_response, str) or not final_response.strip():
        return False
    stop_reason = output.get("stop_reason")
    if stop_reason is not None and not isinstance(stop_reason, str):
        return False
    return True


def ensure_prompts_entry(
    prompts_jsonl: Path,
    candidate: str,
    prompt_text: str,
    prompt_key: str,
    description: str,
    status: str,
    commit: str,
    scoring: dict[str, Any] | None = None,
    run_dir: str = "",
) -> dict[str, Any]:
    rows = load_jsonl(prompts_jsonl)
    matching = [idx for idx, row in enumerate(rows) if row.get("candidate") == candidate]
    record = {
        "candidate": candidate,
        "status": status,
        "good_traces": scoring.get("good_traces", "") if scoring else "",
        "run_dir": run_dir,
        "comment": description,
        "prompt_key": prompt_key,
        "commit": commit,
        "prompt_sha256": sha256_text(prompt_text),
        "updated_at": utc_now(),
        "prompt": prompt_text,
    }
    if scoring:
        record["scoring"] = scoring
    if matching:
        existing = rows[matching[-1]]
        existing.update(record)
    else:
        record["created_at"] = utc_now()
        rows.append(record)
    write_jsonl(prompts_jsonl, rows)
    return record


def upsert_results_row(
    results_tsv: Path,
    commit: str,
    candidate: str,
    status: str,
    exact_score: str,
    partial_avg: float,
    inline_avg: float,
    hybrid_avg: float,
    dominant_failure: str,
    run_dir: str,
    description: str,
) -> None:
    safe_desc = description.replace("\t", " ").replace("\n", " ").strip()
    safe_failure = dominant_failure.replace("\t", " ").replace("\n", " ").strip()
    new_line = (
        f"{commit}\t{candidate}\t{status}\t{exact_score}\t"
        f"{partial_avg:.3f}\t{inline_avg:.3f}\t{hybrid_avg:.3f}\t"
        f"{safe_failure}\t{run_dir}\t{safe_desc}\n"
    )

    if not results_tsv.exists():
        with results_tsv.open("w") as f:
            f.write(RESULTS_HEADER)
            f.write(new_line)
        return

    with results_tsv.open() as f:
        lines = f.readlines()

    if not lines:
        lines = [RESULTS_HEADER]
    elif not lines[0].startswith("commit\tcandidate\tstatus\t"):
        lines.insert(0, RESULTS_HEADER)
    else:
        lines[0] = RESULTS_HEADER

    updated = False
    out_lines = [lines[0]]
    for line in lines[1:]:
        parts = line.rstrip("\n").split("\t", 5)
        if len(parts) >= 2 and parts[1] == candidate:
            out_lines.append(new_line)
            updated = True
        else:
            out_lines.append(line if line.endswith("\n") else line + "\n")

    if not updated:
        out_lines.append(new_line)

    with results_tsv.open("w") as f:
        f.writelines(out_lines)


def write_score_summary(run_dir: Path, payload: dict[str, Any]) -> Path:
    out_path = run_dir / "autoresearch_score.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")
    return out_path


def _parse_exact_score(exact_score: str) -> float:
    if not exact_score:
        return -1.0
    match = re.search(r"=\s*([0-9.]+)\s*$", exact_score)
    if match:
        return float(match.group(1))
    return -1.0


def _is_better_candidate(candidate_payload: dict[str, Any], current_payload: dict[str, Any] | None) -> bool:
    if not current_payload:
        return True

    cand_exact = _parse_exact_score(candidate_payload.get("exact_score", ""))
    curr_exact = _parse_exact_score(current_payload.get("exact_score", ""))
    if cand_exact != curr_exact:
        return cand_exact > curr_exact

    cand_hybrid = float(candidate_payload.get("hybrid_avg", 0.0) or 0.0)
    curr_hybrid = float(current_payload.get("hybrid_avg", 0.0) or 0.0)
    if cand_hybrid != curr_hybrid:
        return cand_hybrid > curr_hybrid

    cand_partial = float(candidate_payload.get("partial_avg", 0.0) or 0.0)
    curr_partial = float(current_payload.get("partial_avg", 0.0) or 0.0)
    if cand_partial != curr_partial:
        return cand_partial > curr_partial

    cand_prompt_len = int(candidate_payload.get("prompt_char_len", 10**18) or 10**18)
    curr_prompt_len = int(current_payload.get("prompt_char_len", 10**18) or 10**18)
    return cand_prompt_len < curr_prompt_len


def update_current_best(path: Path, candidate_payload: dict[str, Any]) -> bool:
    current = None
    if path.exists():
        with path.open() as f:
            current = json.load(f)

    if not _is_better_candidate(candidate_payload, current):
        return False

    with path.open("w") as f:
        json.dump(candidate_payload, f, indent=2, ensure_ascii=True)
        f.write("\n")
    return True


def aggregate_scores(rows: list[dict[str, Any]], max_rows: int = 0) -> dict[str, Any]:
    completed_rows = [row for row in rows if is_completed_row(row)]
    if max_rows and max_rows > 0:
        completed_rows = completed_rows[:max_rows]
    exact_count = 0
    partial_scores = []
    inline_scores = []
    hybrid_scores = []
    validator_count = 0
    exact_failures = Counter()
    failure_labels = Counter()
    sample_failures = []

    for idx, row in enumerate(completed_rows):
        text = row.get("output", {}).get("final_response", "")
        score = score_trace(text)
        if score.exact_valid:
            exact_count += 1
        if score.validator_valid:
            validator_count += 1
        partial_scores.append(score.partial_score)
        inline_scores.append(score.inline_score)
        hybrid_scores.append(score.hybrid_score)
        exact_failures[score.exact_reason] += 0 if score.exact_valid else 1
        for label in score.failure_labels:
            failure_labels[label] += 1
        if not score.exact_valid and len(sample_failures) < 5:
            sample_failures.append(
                {
                    "index": idx,
                    "question": row.get("question", ""),
                    "exact_reason": score.exact_reason,
                    "validator_reason": score.validator_reason,
                    "partial_score": score.partial_score,
                    "inline_score": score.inline_score,
                    "hybrid_score": score.hybrid_score,
                    "snippet": text[:1000],
                }
            )

    total = len(completed_rows)
    exact_rate = exact_count / total if total else 0.0
    validator_rate = validator_count / total if total else 0.0
    top_exact_failures = exact_failures.most_common(10)
    dominant_failure = top_exact_failures[0][0] if top_exact_failures else ""
    return {
        "total_traces": total,
        "raw_rows": len(rows),
        "completed_rows": total,
        "exact_valid_traces": exact_count,
        "validator_valid_traces": validator_count,
        "good_traces": f"{exact_count}/{total} = {exact_rate:.3f}" if total else "0/0 = 0.000",
        "exact_rate": round(exact_rate, 6),
        "validator_rate": round(validator_rate, 6),
        "partial_avg": round(sum(partial_scores) / total, 6) if total else 0.0,
        "inline_avg": round(sum(inline_scores) / total, 6) if total else 0.0,
        "hybrid_avg": round(sum(hybrid_scores) / total, 6) if total else 0.0,
        "dominant_failure": dominant_failure,
        "top_exact_failures": top_exact_failures,
        "top_failure_labels": failure_labels.most_common(10),
        "sample_failures": sample_failures,
    }


def cmd_snapshot_prompt(args: argparse.Namespace) -> None:
    prompt_text = load_prompt(args.prompt_key)
    commit = git_commit_short()
    record = ensure_prompts_entry(
        prompts_jsonl=Path(args.prompts_jsonl),
        candidate=args.candidate,
        prompt_text=prompt_text,
        prompt_key=args.prompt_key,
        description=args.description,
        status=args.status,
        commit=commit,
        scoring=None,
        run_dir="",
    )
    print(json.dumps(record, indent=2, ensure_ascii=True))


def cmd_score_run(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root)
    run_dir = Path(args.run_dir) if args.run_dir else newest_run_dir(results_root)
    rows = read_intermediate_rows(run_dir)
    prompt_text = load_prompt(args.prompt_key)
    commit = git_commit_short()
    scoring = aggregate_scores(rows, max_rows=args.max_rows)
    scoring["results_root"] = str(results_root)
    scoring["run_dir"] = str(run_dir)
    scoring["prompt_key"] = args.prompt_key
    scoring["scored_at"] = utc_now()

    payload = {
        "candidate": args.candidate,
        "status": args.status,
        "description": args.description,
        "commit": commit,
        "prompt_sha256": sha256_text(prompt_text),
        "prompt_char_len": len(prompt_text),
        "scoring": scoring,
    }

    summary_path = None
    best_updated = False
    if not args.dry_run:
        summary_path = write_score_summary(run_dir, payload)
        ensure_prompts_entry(
            prompts_jsonl=Path(args.prompts_jsonl),
            candidate=args.candidate,
            prompt_text=prompt_text,
            prompt_key=args.prompt_key,
            description=args.description,
            status=args.status,
            commit=commit,
            scoring=scoring,
            run_dir=str(run_dir),
        )
        upsert_results_row(
            results_tsv=Path(args.results_tsv),
            commit=commit,
            candidate=args.candidate,
            status=args.status,
            exact_score=scoring["good_traces"],
            partial_avg=scoring["partial_avg"],
            inline_avg=scoring["inline_avg"],
            hybrid_avg=scoring["hybrid_avg"],
            dominant_failure=scoring["dominant_failure"],
            run_dir=str(run_dir),
            description=args.description,
        )
        if args.status == "keep":
            best_updated = update_current_best(
                CURRENT_BEST_JSON,
                {
                    "candidate": args.candidate,
                    "status": args.status,
                    "exact_score": scoring["good_traces"],
                    "partial_avg": scoring["partial_avg"],
                    "inline_avg": scoring["inline_avg"],
                    "hybrid_avg": scoring["hybrid_avg"],
                    "dominant_failure": scoring.get("dominant_failure", ""),
                    "run_dir": str(run_dir),
                    "description": args.description,
                    "prompt_key": args.prompt_key,
                    "prompt": prompt_text,
                    "prompt_sha256": sha256_text(prompt_text),
                    "prompt_char_len": len(prompt_text),
                    "updated_at": utc_now(),
                },
            )

    printable = {
        "candidate": args.candidate,
        "status": args.status,
        "run_dir": str(run_dir),
        "good_traces": scoring["good_traces"],
        "partial_avg": scoring["partial_avg"],
        "inline_avg": scoring["inline_avg"],
        "hybrid_avg": scoring["hybrid_avg"],
        "dominant_failure": scoring["dominant_failure"],
        "top_exact_failures": scoring["top_exact_failures"][:5],
        "top_failure_labels": scoring["top_failure_labels"][:5],
        "summary_path": str(summary_path) if summary_path else "",
        "current_best_updated": best_updated,
    }
    print(json.dumps(printable, indent=2, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AutoResearch helper for selecting the prompt that best reproduces the required ReSearch format"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot-prompt", help="record the current prompt in prompts.jsonl")
    snapshot.add_argument("--candidate", required=True)
    snapshot.add_argument("--description", required=True)
    snapshot.add_argument("--status", default="draft_not_evaluated")
    snapshot.add_argument("--prompt-key", default=PROMPT_KEY)
    snapshot.add_argument("--prompts-jsonl", default=str(PROMPTS_JSONL))
    snapshot.set_defaults(func=cmd_snapshot_prompt)

    score = subparsers.add_parser("score-run", help="score a run and append to results")
    score.add_argument("--candidate", required=True)
    score.add_argument("--description", required=True)
    score.add_argument("--status", default="iterate")
    score.add_argument("--run-dir", default="")
    score.add_argument("--results-root", default=str(RESULTS_ROOT))
    score.add_argument("--results-tsv", default=str(RESULTS_TSV))
    score.add_argument("--prompts-jsonl", default=str(PROMPTS_JSONL))
    score.add_argument("--prompt-key", default=PROMPT_KEY)
    score.add_argument("--max-rows", type=int, default=0)
    score.add_argument("--dry-run", action="store_true")
    score.set_defaults(func=cmd_score_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
