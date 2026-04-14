from fastapi import FastAPI, HTTPException
import argparse
import logging
from pydantic import BaseModel
from typing import List, Tuple, Union
import asyncio
from collections import deque
import os
import time

from flashrag.config import Config
from flashrag.utils import get_retriever

app = FastAPI()
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("RETRIEVER_SERVING_LOG_LEVEL", "INFO"))

retriever_list = []
available_retrievers = deque()
retriever_semaphore = None


def _preview_text(text: str, limit: int = 120) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."

def init_retriever(args):
    global retriever_semaphore
    config = Config(args.config)
    for i in range(args.num_retriever):
        print(f"Initializing retriever {i+1}/{args.num_retriever}")
        retriever = get_retriever(config)
        retriever_list.append(retriever)
        available_retrievers.append(i)
    # create a semaphore to limit the number of retrievers that can be used concurrently
    retriever_semaphore = asyncio.Semaphore(args.num_retriever)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "retrievers": {
            "total": len(retriever_list),
            "available": len(available_retrievers)
        }
    }

class QueryRequest(BaseModel):
    query: str
    top_n: int = 10
    return_score: bool = False

class BatchQueryRequest(BaseModel):
    query: List[str]
    top_n: int = 10
    return_score: bool = False

class Document(BaseModel):
    id: str
    contents: str

@app.post("/search", response_model=Union[Tuple[List[Document], List[float]], List[Document]])
async def search(request: QueryRequest):
    query = request.query
    top_n = request.top_n
    return_score = request.return_score
    request_start = time.perf_counter()

    if not query or not query.strip():
        print(f"Query content cannot be empty: {query}")
        raise HTTPException(
            status_code=400,
            detail="Query content cannot be empty"
        )

    semaphore_wait_start = time.perf_counter()
    async with retriever_semaphore:
        queue_wait_s = time.perf_counter() - semaphore_wait_start
        retriever_idx = available_retrievers.popleft()
        try:
            retrieval_start = time.perf_counter()
            if return_score:
                results, scores = retriever_list[retriever_idx].search(query, top_n, return_score)
                retrieval_s = time.perf_counter() - retrieval_start
                total_request_s = time.perf_counter() - request_start
                logger.info(
                    "search ok retriever_idx=%s queue_wait_s=%.3f retrieval_s=%.3f total_request_s=%.3f "
                    "top_n=%s query_chars=%s return_score=%s preview=%r",
                    retriever_idx,
                    queue_wait_s,
                    retrieval_s,
                    total_request_s,
                    top_n,
                    len(query),
                    return_score,
                    _preview_text(query),
                )
                return [Document(id=result['id'], contents=result['contents']) for result in results], scores
            else:
                results = retriever_list[retriever_idx].search(query, top_n, return_score)
                retrieval_s = time.perf_counter() - retrieval_start
                total_request_s = time.perf_counter() - request_start
                logger.info(
                    "search ok retriever_idx=%s queue_wait_s=%.3f retrieval_s=%.3f total_request_s=%.3f "
                    "top_n=%s query_chars=%s return_score=%s preview=%r",
                    retriever_idx,
                    queue_wait_s,
                    retrieval_s,
                    total_request_s,
                    top_n,
                    len(query),
                    return_score,
                    _preview_text(query),
                )
                return [Document(id=result['id'], contents=result['contents']) for result in results]
        except Exception:
            total_request_s = time.perf_counter() - request_start
            logger.exception(
                "search failed retriever_idx=%s queue_wait_s=%.3f total_request_s=%.3f top_n=%s "
                "query_chars=%s return_score=%s preview=%r",
                retriever_idx,
                queue_wait_s,
                total_request_s,
                top_n,
                len(query),
                return_score,
                _preview_text(query),
            )
            raise
        finally:
            available_retrievers.append(retriever_idx)

@app.post("/batch_search", response_model=Union[List[List[Document]], Tuple[List[List[Document]], List[List[float]]]])
async def batch_search(request: BatchQueryRequest):
    query = request.query
    top_n = request.top_n
    return_score = request.return_score
    request_start = time.perf_counter()

    semaphore_wait_start = time.perf_counter()
    async with retriever_semaphore:
        queue_wait_s = time.perf_counter() - semaphore_wait_start
        retriever_idx = available_retrievers.popleft()
        try:
            retrieval_start = time.perf_counter()
            if return_score:
                results, scores = retriever_list[retriever_idx].batch_search(query, top_n, return_score)
                retrieval_s = time.perf_counter() - retrieval_start
                total_request_s = time.perf_counter() - request_start
                logger.info(
                    "batch_search ok retriever_idx=%s queue_wait_s=%.3f retrieval_s=%.3f total_request_s=%.3f "
                    "num_queries=%s query_chars=%s top_n=%s return_score=%s preview=%r",
                    retriever_idx,
                    queue_wait_s,
                    retrieval_s,
                    total_request_s,
                    len(query),
                    sum(len(q) for q in query),
                    top_n,
                    return_score,
                    _preview_text(query[0]) if query else "",
                )
                return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))], scores
            else:
                results = retriever_list[retriever_idx].batch_search(query, top_n, return_score)
                retrieval_s = time.perf_counter() - retrieval_start
                total_request_s = time.perf_counter() - request_start
                logger.info(
                    "batch_search ok retriever_idx=%s queue_wait_s=%.3f retrieval_s=%.3f total_request_s=%.3f "
                    "num_queries=%s query_chars=%s top_n=%s return_score=%s preview=%r",
                    retriever_idx,
                    queue_wait_s,
                    retrieval_s,
                    total_request_s,
                    len(query),
                    sum(len(q) for q in query),
                    top_n,
                    return_score,
                    _preview_text(query[0]) if query else "",
                )
                return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))]
        except Exception:
            total_request_s = time.perf_counter() - request_start
            logger.exception(
                "batch_search failed retriever_idx=%s queue_wait_s=%.3f total_request_s=%.3f "
                "num_queries=%s query_chars=%s top_n=%s return_score=%s preview=%r",
                retriever_idx,
                queue_wait_s,
                total_request_s,
                len(query),
                sum(len(q) for q in query),
                top_n,
                return_score,
                _preview_text(query[0]) if query else "",
            )
            raise
        finally:
            available_retrievers.append(retriever_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./retriever_config.yaml")
    parser.add_argument("--num_retriever", type=int, default=1)
    parser.add_argument("--port", type=int, default=80)
    args = parser.parse_args()
    
    init_retriever(args)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

