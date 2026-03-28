# Client-side parallelism for remote LLM / retriever HTTP calls.
# Single-GPU setups (e.g. one L4 or A100) cannot sustain hundreds of concurrent
# decode/KV-cache slots; keep this modest to avoid KV thrashing and queue blowups.
INFERENCE_MAX_WORKERS = 8
