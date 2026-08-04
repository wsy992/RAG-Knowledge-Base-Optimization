from pathlib import Path

from demo.app import build_view_model
from rag_v2.embeddings import StaticEmbeddingProvider
from rag_v2.pipeline import RagPipeline
from rag_v2.runtime import build_pipeline, build_runtime
from rag_v2.schemas import PipelineConfig


class FakeChatClient:
    def complete(self, messages, *, temperature=0.1, response_format=None):
        if response_format:
            return '{"rewritten_query":"机器学习课程","reason":"补全主题"}'
        return '{"answer":"课程包含机器学习 [source:course::chunk-0001]"}'


class FakeReranker:
    def rerank(self, query, candidates, top_k):
        return candidates[:top_k]


def test_pipeline_factory_and_demo_view_model_expose_trace_and_citations(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "course.md").write_text("# 课程\n\n机器学习课程介绍监督学习。", encoding="utf-8")
    config = PipelineConfig(
        corpus_dir=corpus,
        artifact_dir=tmp_path / "artifacts",
        embedding_model="embedding",
        reranker_model="reranker",
        mode="bm25",
        top_k=3,
        rerank_top_k=1,
        rewrite_enabled=True,
        rerank_enabled=True,
        generation_enabled=True,
    )
    chunks_text = "机器学习课程介绍监督学习。"
    embeddings = StaticEmbeddingProvider({chunks_text: [1.0, 0.0], "机器学习课程": [1.0, 0.0]})

    pipeline = build_pipeline(
        config,
        embedding_provider=embeddings,
        chat_client=FakeChatClient(),
        reranker=FakeReranker(),
    )
    response = pipeline.run("学什么？", [], config)
    view = build_view_model(response)

    assert view["rewritten_query"] == "机器学习课程"
    assert view["retrieved"][0]["chunk_id"] == "course::chunk-0001"
    assert view["citations"] == ["course::chunk-0001"]
    assert {event["stage"] for event in view["trace"]} == {"rewrite", "retrieve", "rerank", "generation"}


def test_runtime_bundle_builds_dense_index_once_and_can_run_multiple_configs(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "course.md").write_text("# 课程\n\n机器学习课程介绍监督学习。", encoding="utf-8")
    config = PipelineConfig(
        corpus_dir=corpus,
        artifact_dir=tmp_path / "artifacts",
        embedding_model="embedding",
        reranker_model="reranker",
        mode="bm25",
        top_k=3,
        rerank_top_k=1,
        rewrite_enabled=False,
        rerank_enabled=False,
        generation_enabled=False,
    )
    text = "机器学习课程介绍监督学习。"
    embeddings = StaticEmbeddingProvider({text: [1.0, 0.0], "机器学习": [1.0, 0.0]})

    bundle = build_runtime(
        config,
        embedding_provider=embeddings,
        chat_client=FakeChatClient(),
        reranker=FakeReranker(),
    )

    first = bundle.pipeline.run("机器学习", [], config)
    second = bundle.pipeline.run("机器学习", [], config)

    assert bundle.chunks
    assert first.retrieved[0].chunk.chunk_id == second.retrieved[0].chunk.chunk_id
