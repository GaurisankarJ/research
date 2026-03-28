from openai import OpenAI
import logging
import sys
import argparse
import json
import jsonlines
import os
import re
from urllib.parse import urlparse

from utils import retry, execute, init_logger

LOG_NAME = "llm_judge"


def openai_base_url(url: str) -> str:
    """Ollama exposes an OpenAI-compatible API at …/v1; accept either http://host:11434 or …/v1."""
    u = url.strip().rstrip("/")
    if u.endswith("/v1"):
        return u
    parsed = urlparse(u if "://" in u else f"http://{u}")
    path = parsed.path or ""
    if path in ("", "/"):
        return f"{parsed.scheme}://{parsed.netloc}/v1"
    return u


def parse_judge_json(content: str) -> dict:
    """Parse model output into the judge object. Ollama often wraps JSON in markdown fences."""
    raw = (content or "").strip()
    if not raw:
        raise ValueError("empty model content")
    if "```" in raw:
        for block in raw.split("```"):
            block = block.strip()
            if block.lower().startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                raw = block
                break
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise
        return json.loads(m.group(0))


llm_judge_prompt = """You will be given a question and its ground truth answer list where each item can be a ground truth answer. \
Provided a pred_answer, you need to judge if the pred_answer correctly answers the question based on the ground truth answer list. \
You should first give your rationale for the judgement, and then give your judgement result (i.e., correct or incorrect).

Here is the criteria for the judgement:
1. The pred_answer doesn't need to be exactly the same as any of the ground truth answers, but should be semantically same for the question.
2. Each item in the ground truth answer list can be viewed as a ground truth answer for the question, and the pred_answer should be semantically same to at least one of them.

question: {question}
ground truth answers: {gt_answer}
pred_answer: {pred_answer}

The output should in the following json format:
```json 
{{
    "rationale": "your rationale for the judgement, as a text",
    "judgement": "your judgement result, can only be 'correct' or 'incorrect'"
}}
```

Your output:"""

def read_lines(args):
    lines = json.load(open(os.path.join(args.input_dir, 'intermediate_data.json')))

    return lines

def cal_llm_judge_metric(args, logger: logging.Logger) -> None:
    if not os.path.isfile(args.output_path) or os.path.getsize(args.output_path) == 0:
        logger.warning("No llm_judge output lines; skipping metric (check errors above).")
        return
    num_correct = 0
    num_total = 0
    with jsonlines.open(args.output_path) as reader:
        for line in reader:
            if line["llm_judge"]["judgement"] == "correct":
                num_correct += 1
            num_total += 1
    if num_total == 0:
        logger.warning("llm_judge.jsonl is empty; skipping metric.")
        return
    score = num_correct / num_total
    with open(args.metric_path, "w") as f:
        f.write(f"llm_judge_metric: {score}\n")

def run(args):
    base = openai_base_url(args.base_url)
    client = OpenAI(base_url=base, api_key=args.api_key)
    logger = logging.getLogger(LOG_NAME)
    logger.info(f"OpenAI client base_url (resolved): {base}")

    lines = read_lines(args)
    logger.info(f"Read {len(lines)} lines from {args.input_dir}")

    @retry(max=5, sleep=1, logger=logger)
    def run_judge(line):
        prompt = llm_judge_prompt.format(
            question=line['question'], 
            gt_answer=str(line['golden_answers']), 
            pred_answer=line['output']['pred']
        )

        # Do not use response_format=json_object: Ollama's /v1 compatibility often ignores or mishandles it
        # (hangs or errors). OpenAI-style servers still follow the prompt below.
        response = client.chat.completions.create(
            model=args.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1024,
            timeout=args.request_timeout,
        )

        content = response.choices[0].message.content
        try:
            res = parse_judge_json(content)
            assert "judgement" in res and "rationale" in res
        except Exception as e:
            logger.info(f"Error parsing response: {e}")
            logger.info(f"Response: {content!r}")
            raise e

        line['llm_judge'] = res
        return line

    execute(run_judge, lines, args.output_path, args.max_workers, logger)

    cal_llm_judge_metric(args, logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", 
        type=str,
        help="The directory of the evaluation output of FlashRAG"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=os.environ.get("OLLAMA_MODEL", "qwen3.5:9b"),
        help="Ollama model tag (same as `ollama run`). Override with OLLAMA_MODEL.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434"),
        help="Ollama HTTP root (default) or full OpenAI-compatible base ending in /v1. Override with OLLAMA_BASE.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "ollama"),
        help="Unused by Ollama; any non-empty string is fine. Or set OPENAI_API_KEY for other servers.",
    )
    parser.add_argument(
        "--log_path", 
        type=str, 
        default="llm_judge.log",
        help="The path of the log file"
    )
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=10,
        help="The maximum number of workers for the parallel execution"
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=float(os.environ.get("LLM_JUDGE_TIMEOUT", "600")),
        help="Per-request HTTP timeout in seconds (local Ollama can be slow).",
    )
    args = parser.parse_args()
    
    args.output_path = os.path.join(args.input_dir, 'llm_judge.jsonl')
    args.metric_path = os.path.join(args.input_dir, 'llm_judge_metric.txt')

    logger = init_logger(args.log_path, LOG_NAME)
    if sys.stderr.isatty():
        _h = logging.StreamHandler(sys.stderr)
        _h.setLevel(logging.INFO)
        _h.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(_h)
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")

    run(args)