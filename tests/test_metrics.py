import pytest

from rag_v2.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from rag_v2.schemas import DocumentChunk, RetrievedChunk


def result(doc_id: str, rank: int) -> RetrievedChunk:
    chunk = DocumentChunk(doc_id, doc_id, f"{doc_id}.md", "section", f"{doc_id}::1", doc_id)
    return RetrievedChunk(chunk, score=1.0 / rank, rank=rank, retriever="test")


def test_recall_mrr_and_ndcg_use_gold_document_ids():
    ranked = [result("doc-a", 1), result("doc-b", 2), result("doc-c", 3)]
    gold = {"doc-a", "doc-c"}

    assert recall_at_k(ranked, gold, 1) == pytest.approx(0.5)
    assert recall_at_k(ranked, gold, 3) == pytest.approx(1.0)
    assert mean_reciprocal_rank(ranked, gold, 3) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, gold, 3) == pytest.approx(1.5 / (1.0 + 1.0 / 1.5849625007))


def test_empty_results_and_unknown_gold_are_safe():
    assert recall_at_k([], {"doc-a"}, 5) == 0.0
    assert mean_reciprocal_rank([], {"doc-a"}, 5) == 0.0
    assert ndcg_at_k([], {"doc-a"}, 5) == 0.0
    assert recall_at_k([], set(), 5) == 0.0
