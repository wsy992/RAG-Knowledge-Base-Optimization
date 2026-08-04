"""Lexical, dense, and hybrid retrieval implementations."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import EmbeddingProvider
from .schemas import DocumentChunk, RetrievedChunk

RetrievalMode = Literal["bm25", "dense", "hybrid"]


def _tokens(text: str) -> list[str]:
    tokens = [token.strip().lower() for token in jieba.lcut(text) if token.strip()]
    return tokens or list(text.strip().lower())


def _validate_top_k(top_k: int) -> None:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")


class Bm25Index:
    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self._index: BM25Okapi | None = None

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self._index = BM25Okapi([_tokens(chunk.text) for chunk in self.chunks]) if self.chunks else None

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        _validate_top_k(top_k)
        if self._index is None:
            return []

        scores = self._index.get_scores(_tokens(query))
        ranked = sorted(
            range(len(self.chunks)),
            key=lambda index: (-float(scores[index]), self.chunks[index].chunk_id),
        )[:top_k]
        return [
            RetrievedChunk(
                chunk=self.chunks[index],
                score=float(scores[index]),
                rank=rank,
                retriever="bm25",
            )
            for rank, index in enumerate(ranked, start=1)
        ]


class DenseIndex:
    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self._matrix: np.ndarray | None = None

    def build(self, chunks: list[DocumentChunk], embeddings: EmbeddingProvider) -> None:
        chunks = list(chunks)
        vectors = embeddings.embed([chunk.text for chunk in chunks])
        self.build_from_vectors(chunks, vectors)

    def build_from_vectors(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]] | np.ndarray,
    ) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            self._matrix = None
            return

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(self.chunks):
            raise ValueError("embedding provider must return one 2-D vector per chunk")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("embedding vectors must have non-zero norm")
        self._matrix = vectors / norms

    def search(self, query: str, embeddings: EmbeddingProvider, top_k: int) -> list[RetrievedChunk]:
        _validate_top_k(top_k)
        if self._matrix is None:
            return []
        query_vector = np.asarray(embeddings.embed([query]), dtype=np.float32)
        if query_vector.shape != (1, self._matrix.shape[1]):
            raise ValueError("query embedding dimension does not match the dense index")
        norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        if np.any(norm == 0):
            raise ValueError("query embedding must have non-zero norm")
        query_vector = query_vector / norm
        scores = (self._matrix @ query_vector[0]).tolist()
        ranked = sorted(
            range(len(self.chunks)),
            key=lambda index: (-float(scores[index]), self.chunks[index].chunk_id),
        )[:top_k]
        return [
            RetrievedChunk(
                chunk=self.chunks[index],
                score=float(scores[index]),
                rank=rank,
                retriever="dense",
            )
            for rank, index in enumerate(ranked, start=1)
        ]


class HybridRetriever:
    RRF_K = 60

    @classmethod
    def fuse(
        cls,
        bm25_results: list[RetrievedChunk],
        dense_results: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        _validate_top_k(top_k)
        scores: dict[str, float] = defaultdict(float)
        chunks: dict[str, DocumentChunk] = {}
        sources: dict[str, set[str]] = defaultdict(set)
        for result in [*bm25_results, *dense_results]:
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] += 1.0 / (cls.RRF_K + result.rank)
            chunks[chunk_id] = result.chunk
            sources[chunk_id].add(result.retriever)

        ranked_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
        return [
            RetrievedChunk(
                chunk=chunks[chunk_id],
                score=scores[chunk_id],
                rank=rank,
                retriever="hybrid",
                metadata={
                    "rrf_score": scores[chunk_id],
                    "sources": sorted(sources[chunk_id]),
                },
            )
            for rank, chunk_id in enumerate(ranked_ids, start=1)
        ]


class Retriever:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        bm25: Bm25Index,
        dense: DenseIndex,
        embeddings: EmbeddingProvider,
    ) -> None:
        self.chunks = list(chunks)
        self.bm25 = bm25
        self.dense = dense
        self.embeddings = embeddings

    def retrieve(self, query: str, mode: RetrievalMode, top_k: int) -> list[RetrievedChunk]:
        _validate_top_k(top_k)
        if mode == "bm25":
            return self.bm25.search(query, top_k)
        if mode == "dense":
            return self.dense.search(query, self.embeddings, top_k)
        if mode == "hybrid":
            bm25 = self.bm25.search(query, top_k)
            dense = self.dense.search(query, self.embeddings, top_k)
            return HybridRetriever.fuse(bm25, dense, top_k)
        raise ValueError(f"Unsupported retrieval mode: {mode}")
