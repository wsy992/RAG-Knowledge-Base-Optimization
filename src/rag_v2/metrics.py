"""Pure retrieval metrics used by the benchmark."""

from __future__ import annotations

import math

from .schemas import RetrievedChunk


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")


def recall_at_k(results: list[RetrievedChunk], gold_doc_ids: set[str], k: int) -> float:
    _validate_k(k)
    if not gold_doc_ids:
        return 0.0
    found = {item.chunk.doc_id for item in results[:k] if item.chunk.doc_id in gold_doc_ids}
    return len(found) / len(gold_doc_ids)


def mean_reciprocal_rank(results: list[RetrievedChunk], gold_doc_ids: set[str], k: int) -> float:
    _validate_k(k)
    if not gold_doc_ids:
        return 0.0
    for rank, item in enumerate(results[:k], start=1):
        if item.chunk.doc_id in gold_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[RetrievedChunk], gold_doc_ids: set[str], k: int) -> float:
    _validate_k(k)
    if not gold_doc_ids:
        return 0.0
    dcg = sum(
        (1.0 / math.log2(rank + 1))
        for rank, item in enumerate(results[:k], start=1)
        if item.chunk.doc_id in gold_doc_ids
    )
    ideal_relevant = min(len(gold_doc_ids), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0
