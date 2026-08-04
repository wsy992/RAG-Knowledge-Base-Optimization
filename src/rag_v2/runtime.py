"""Runtime factory connecting configuration to the composable v2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import V2Config
from .corpus import CorpusLoader
from .embeddings import EmbeddingProvider, OllamaEmbeddingProvider
from .generation import GroundedGenerator
from .indexing import corpus_hash, load_dense_artifact
from .pipeline import RagPipeline
from .providers import ChatClient, ChatProviderError, DeepSeekChatClient
from .rerank import CrossEncoderReranker
from .retrieval import Bm25Index, DenseIndex, Retriever
from .rewrite import DeepSeekQueryRewriter
from .schemas import DocumentChunk, PipelineConfig


class UnavailableChatClient:
    def complete(self, messages, *, temperature=0.1, response_format=None):
        raise ChatProviderError("DeepSeek generation is disabled or DEEPSEEK_API_KEY is missing")


@dataclass(frozen=True)
class RuntimeBundle:
    """Built indexes and pipeline components reusable across benchmark rows."""

    pipeline: RagPipeline
    chunks: list[DocumentChunk]
    embedding_provider: EmbeddingProvider
    chat_client: ChatClient
    reranker: Any


def build_runtime(
    config: V2Config | PipelineConfig,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    chat_client: ChatClient | None = None,
    reranker: Any | None = None,
) -> RuntimeBundle:
    if isinstance(config, V2Config):
        pipeline_config = PipelineConfig(
            corpus_dir=config.corpus_dir,
            artifact_dir=config.artifact_dir,
            embedding_model=config.embedding_model,
            reranker_model=config.reranker_model,
            top_k=config.top_k,
            rerank_top_k=config.rerank_top_k,
        )
        loader = CorpusLoader(max_chars=config.max_chars, overlap=config.overlap)
        if embedding_provider is None:
            embedding_provider = OllamaEmbeddingProvider(
                base_url=config.ollama_base_url,
                model=config.embedding_model,
            )
        if chat_client is None:
            chat_client = (
                DeepSeekChatClient(
                    api_key=config.deepseek_api_key,
                    model=config.deepseek_model,
                    base_url=config.deepseek_base_url,
                )
                if config.deepseek_api_key
                else UnavailableChatClient()
            )
        if reranker is None:
            reranker = CrossEncoderReranker(config.reranker_model)
    else:
        pipeline_config = config
        loader = CorpusLoader()
        if embedding_provider is None:
            raise ValueError("embedding_provider is required with PipelineConfig")
        if chat_client is None:
            chat_client = UnavailableChatClient()
        if reranker is None:
            reranker = CrossEncoderReranker(config.reranker_model)

    chunks = loader.load(pipeline_config.corpus_dir)
    bm25 = Bm25Index()
    bm25.build(chunks)
    dense = DenseIndex()
    persisted_vectors = load_dense_artifact(
        pipeline_config.artifact_dir,
        chunks=chunks,
        corpus_digest=corpus_hash(pipeline_config.corpus_dir),
        embedding_model=pipeline_config.embedding_model,
    )
    if persisted_vectors is None:
        dense.build(chunks, embedding_provider)
    else:
        dense.build_from_vectors(chunks, persisted_vectors)
    retriever = Retriever(chunks=chunks, bm25=bm25, dense=dense, embeddings=embedding_provider)
    pipeline = RagPipeline(
        retriever=retriever,
        rewriter=DeepSeekQueryRewriter(chat_client),
        reranker=reranker,
        generator=GroundedGenerator(chat_client),
    )
    return RuntimeBundle(
        pipeline=pipeline,
        chunks=chunks,
        embedding_provider=embedding_provider,
        chat_client=chat_client,
        reranker=reranker,
    )


def build_pipeline(
    config: V2Config | PipelineConfig,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    chat_client: ChatClient | None = None,
    reranker: Any | None = None,
) -> RagPipeline:
    return build_runtime(
        config,
        embedding_provider=embedding_provider,
        chat_client=chat_client,
        reranker=reranker,
    ).pipeline
