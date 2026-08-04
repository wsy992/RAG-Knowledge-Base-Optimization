import json
from pathlib import Path

import pytest

from rag_v2.retrieval import Bm25Index, DenseIndex, HybridRetriever, Retriever
from rag_v2.schemas import DocumentChunk


class FakeEmbeddingProvider:
    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping[text] for text in texts]


def load_chunks() -> list[DocumentChunk]:
    payload = json.loads(Path("tests/fixtures/retrieval_chunks.json").read_text(encoding="utf-8"))
    return [DocumentChunk(**item) for item in payload]


def build_retriever() -> Retriever:
    chunks = load_chunks()
    provider = FakeEmbeddingProvider(
        {
            chunks[0].text: [1.0, 0.0],
            chunks[1].text: [0.0, 1.0],
            chunks[2].text: [0.7, 0.7],
            chunks[3].text: [0.2, 0.98],
            "预测模型": [1.0, 0.0],
            "关系数据库": [0.0, 1.0],
        }
    )
    bm25 = Bm25Index()
    bm25.build(chunks)
    dense = DenseIndex()
    dense.build(chunks, provider)
    return Retriever(chunks=chunks, bm25=bm25, dense=dense, embeddings=provider)


def test_bm25_retrieval_finds_lexical_match():
    results = build_retriever().retrieve("机器学习", mode="bm25", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.doc_id == "ml-course"
    assert results[0].retriever == "bm25"


def test_dense_retrieval_finds_paraphrase_by_embedding():
    results = build_retriever().retrieve("预测模型", mode="dense", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.doc_id == "ml-course"
    assert results[0].retriever == "dense"


def test_hybrid_fusion_deduplicates_chunk_ids_and_keeps_scores():
    retriever = build_retriever()
    lexical = retriever.retrieve("机器学习", mode="bm25", top_k=3)
    dense = retriever.retrieve("预测模型", mode="dense", top_k=3)
    results = HybridRetriever.fuse(lexical, dense, top_k=3)

    assert len(results) == len({item.chunk.chunk_id for item in results})
    assert len(results) == 3
    assert all(item.retriever == "hybrid" for item in results)
    assert all("rrf_score" in item.metadata for item in results)


def test_retrieve_rejects_non_positive_top_k():
    with pytest.raises(ValueError, match="top_k"):
        build_retriever().retrieve("机器学习", mode="bm25", top_k=0)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        build_retriever().retrieve("机器学习", mode="unknown", top_k=3)

def test_dense_index_can_load_persisted_vectors():
    chunks = load_chunks()
    provider = FakeEmbeddingProvider({"query": [1.0, 0.0]})
    dense = DenseIndex()
    dense.build_from_vectors(chunks, [[1.0, 0.0] for _ in chunks])

    result = dense.search("query", provider, top_k=1)

    assert result[0].chunk.chunk_id == min(chunk.chunk_id for chunk in chunks)
