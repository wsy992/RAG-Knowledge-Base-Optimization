from pathlib import Path

import pytest

from rag_v2.schemas import (
    BenchmarkCase,
    DocumentChunk,
    PipelineConfig,
    RetrievedChunk,
)


def test_document_chunk_strips_text_and_preserves_identifiers():
    chunk = DocumentChunk(
        doc_id="doc-1",
        title="课程介绍",
        source="course.md",
        section="核心课程",
        chunk_id="doc-1::chunk-0001",
        text="  数据结构与算法  ",
    )

    assert chunk.text == "数据结构与算法"
    assert chunk.chunk_id == "doc-1::chunk-0001"


def test_document_chunk_rejects_missing_identity_or_content():
    with pytest.raises(ValueError, match="chunk_id"):
        DocumentChunk("doc-1", "title", "source", "section", "", "content")

    with pytest.raises(ValueError, match="text"):
        DocumentChunk("doc-1", "title", "source", "section", "chunk-1", " ")


def test_benchmark_case_requires_gold_documents_for_answerable_questions():
    with pytest.raises(ValueError, match="gold_doc_ids"):
        BenchmarkCase(
            case_id="case-1",
            question="课程有哪些？",
            category="direct",
            gold_doc_ids=[],
            reference_answer="数据结构与算法",
            answer_points=["数据结构与算法"],
            should_abstain=False,
        )

    unknown = BenchmarkCase(
        case_id="case-unknown",
        question="宿舍怎么样？",
        category="unknown",
        gold_doc_ids=[],
        reference_answer="",
        answer_points=[],
        should_abstain=True,
    )
    assert unknown.should_abstain is True


def test_pipeline_config_resolves_paths_and_validates_top_k(tmp_path: Path):
    config = PipelineConfig(
        corpus_dir=tmp_path / "corpus",
        artifact_dir=tmp_path / "artifacts",
        embedding_model="nomic-embed-text",
        reranker_model="BAAI/bge-reranker-v2-m3",
        top_k=5,
        rerank_top_k=3,
    )

    assert config.corpus_dir == tmp_path / "corpus"
    assert config.top_k == 5

    with pytest.raises(ValueError, match="top_k"):
        PipelineConfig(
            corpus_dir=tmp_path,
            artifact_dir=tmp_path,
            embedding_model="embedding",
            reranker_model="reranker",
            top_k=0,
            rerank_top_k=1,
        )


def test_retrieved_chunk_exposes_score_and_source_metadata():
    chunk = DocumentChunk("doc-1", "title", "source", "section", "chunk-1", "content")
    result = RetrievedChunk(chunk=chunk, score=0.8, rank=1, retriever="bm25")

    assert result.chunk.doc_id == "doc-1"
    assert result.score == pytest.approx(0.8)
    assert result.metadata == {}
