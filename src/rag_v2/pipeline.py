"""Composable RAG execution pipeline with per-stage traces."""

from __future__ import annotations

import time
from typing import Any, Protocol

from .generation import GroundedGenerator
from .rerank import CrossEncoderReranker
from .rewrite import DeepSeekQueryRewriter
from .schemas import GenerationResult, PipelineConfig, RagResponse, RewriteResult, RetrievedChunk


class RetrieverProtocol(Protocol):
    def retrieve(self, query: str, mode: str, top_k: int) -> list[RetrievedChunk]:
        """Retrieve candidate chunks."""


class RewriterProtocol(Protocol):
    def rewrite(self, question: str, history: list[dict[str, str]]) -> RewriteResult:
        """Rewrite a user question."""


class RerankerProtocol(Protocol):
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Rerank candidate chunks."""


class GeneratorProtocol(Protocol):
    def answer(self, question: str, contexts: list[RetrievedChunk]) -> GenerationResult:
        """Generate an answer from retrieved contexts."""


class RagPipeline:
    def __init__(
        self,
        retriever: RetrieverProtocol,
        rewriter: RewriterProtocol,
        reranker: RerankerProtocol,
        generator: GeneratorProtocol,
    ) -> None:
        self.retriever = retriever
        self.rewriter = rewriter
        self.reranker = reranker
        self.generator = generator

    def run(
        self,
        question: str,
        history: list[dict[str, str]],
        config: PipelineConfig,
    ) -> RagResponse:
        if not question.strip():
            raise ValueError("question must not be empty")
        trace: list[dict[str, Any]] = []

        started = time.perf_counter()
        if config.rewrite_enabled:
            rewrite = self.rewriter.rewrite(question, history)
        else:
            rewrite = RewriteResult(question.strip(), question.strip(), "rewrite disabled", False)
        self._add_trace(trace, "rewrite", started, {"used_fallback": rewrite.used_fallback})

        started = time.perf_counter()
        retrieved = self.retriever.retrieve(rewrite.rewritten_query, config.mode, config.top_k)
        self._add_trace(trace, "retrieve", started, {"count": len(retrieved), "mode": config.mode})

        if config.rerank_enabled and retrieved:
            started = time.perf_counter()
            retrieved = self.reranker.rerank(rewrite.rewritten_query, retrieved, config.rerank_top_k)
            self._add_trace(trace, "rerank", started, {"count": len(retrieved)})

        started = time.perf_counter()
        if config.generation_enabled:
            generation = self.generator.answer(question, retrieved)
        else:
            generation = GenerationResult("", [], False, 0.0)
        self._add_trace(trace, "generation", started, {"abstained": generation.abstained})

        return RagResponse(rewrite=rewrite, retrieved=retrieved, generation=generation, trace=trace)

    @staticmethod
    def _add_trace(trace: list[dict[str, Any]], stage: str, started: float, details: dict[str, Any]) -> None:
        trace.append(
            {
                "stage": stage,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                **details,
            }
        )
