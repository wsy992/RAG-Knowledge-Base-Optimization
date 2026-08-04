from rag_v2.pipeline import RagPipeline
from rag_v2.schemas import (
    DocumentChunk,
    GenerationResult,
    PipelineConfig,
    RetrievedChunk,
    RewriteResult,
)


def candidate() -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk("doc-1", "课程", "course.md", "核心课程", "doc-1::1", "机器学习"),
        score=0.8,
        rank=1,
        retriever="hybrid",
    )


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, mode, top_k):
        self.calls.append((query, mode, top_k))
        return [candidate()]


class FakeRewriter:
    def rewrite(self, question, history):
        return RewriteResult(question, "智能科技专业 核心课程", "补全主题", False)


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, top_k):
        self.calls.append((query, candidates, top_k))
        return candidates[:top_k]


class FakeGenerator:
    def answer(self, question, contexts):
        return GenerationResult("答案 [source:doc-1::1]", ["doc-1::1"], False, 1.0)


def test_pipeline_uses_rewritten_query_and_records_each_stage():
    retriever = FakeRetriever()
    reranker = FakeReranker()
    pipeline = RagPipeline(retriever, FakeRewriter(), reranker, FakeGenerator())
    config = PipelineConfig(
        corpus_dir="data/demo_corpus",
        artifact_dir="artifacts",
        embedding_model="embedding",
        reranker_model="reranker",
        mode="hybrid",
        top_k=5,
        rerank_top_k=2,
        rewrite_enabled=True,
        rerank_enabled=True,
        generation_enabled=True,
    )

    response = pipeline.run("学费多少？", [{"role": "user", "content": "有哪些课？"}], config)

    assert retriever.calls == [("智能科技专业 核心课程", "hybrid", 5)]
    assert reranker.calls[0][0] == "智能科技专业 核心课程"
    assert response.generation.cited_chunk_ids == ["doc-1::1"]
    assert [event["stage"] for event in response.trace] == [
        "rewrite",
        "retrieve",
        "rerank",
        "generation",
    ]


def test_pipeline_can_disable_generation_for_retrieval_only_benchmark():
    pipeline = RagPipeline(FakeRetriever(), FakeRewriter(), FakeReranker(), FakeGenerator())
    config = PipelineConfig(
        corpus_dir="data/demo_corpus",
        artifact_dir="artifacts",
        embedding_model="embedding",
        reranker_model="reranker",
        mode="bm25",
        rewrite_enabled=False,
        rerank_enabled=False,
        generation_enabled=False,
    )

    response = pipeline.run("课程", [], config)

    assert response.generation.answer == ""
    assert response.generation.abstained is False
    assert [event["stage"] for event in response.trace] == ["rewrite", "retrieve", "generation"]
