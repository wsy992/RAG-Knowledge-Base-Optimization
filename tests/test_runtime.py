from pathlib import Path

from rag_v2.config import V2Config
from rag_v2.corpus import CorpusLoader
from rag_v2.indexing import corpus_hash, save_dense_artifact
from rag_v2.runtime import build_runtime
from rag_v2.schemas import PipelineConfig


class CountingEmbeddingProvider:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


class UnusedChatClient:
    def complete(self, *args, **kwargs):
        raise AssertionError("the offline runtime test must not call chat")


class NoopReranker:
    def rerank(self, query, candidates, top_k):
        return candidates[:top_k]


def test_runtime_reuses_a_matching_dense_artifact(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "course.md").write_text("# 课程\n\n机器学习课程。", encoding="utf-8")
    chunks = CorpusLoader().load(corpus)
    artifact_dir = tmp_path / "artifacts"
    save_dense_artifact(
        artifact_dir,
        chunks=chunks,
        vectors=[[1.0, 0.0] for _ in chunks],
        corpus_digest=corpus_hash(corpus),
        embedding_model="test-embedding",
    )
    config = V2Config(
        config_path=tmp_path / "v2.yaml",
        corpus_dir=corpus,
        artifact_dir=artifact_dir,
        benchmark_path=tmp_path / "dataset.jsonl",
        embedding_model="test-embedding",
        reranker_model="test-reranker",
        ollama_base_url="http://127.0.0.1:11434",
        deepseek_base_url="https://api.deepseek.com/v1",
        deepseek_model="deepseek-chat",
    )
    provider = CountingEmbeddingProvider()

    bundle = build_runtime(
        config,
        embedding_provider=provider,
        chat_client=UnusedChatClient(),
        reranker=NoopReranker(),
    )

    response = bundle.pipeline.run(
        "机器学习",
        [],
        PipelineConfig(
            corpus_dir=corpus,
            artifact_dir=artifact_dir,
            embedding_model="test-embedding",
            reranker_model="test-reranker",
            mode="bm25",
            rewrite_enabled=False,
            rerank_enabled=False,
            generation_enabled=False,
        ),
    )
    assert provider.calls == 0
    assert response.retrieved
