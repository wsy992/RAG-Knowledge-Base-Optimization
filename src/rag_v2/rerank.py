"""Cross-Encoder reranking with a deterministic fallback."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

from .schemas import RetrievedChunk


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        score_fn: Callable[[str, Sequence[str]], Sequence[float]] | None = None,
    ) -> None:
        self.model_name = model_name
        self._score_fn = score_fn
        self._model = None
        self._load_error: Exception | None = None

    def _get_score_fn(self) -> Callable[[str, Sequence[str]], Sequence[float]]:
        if self._score_fn is not None:
            return self._score_fn
        if self._load_error is not None:
            raise RuntimeError("reranker model is unavailable") from self._load_error
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                self._load_error = exc
                raise

        def score(query: str, texts: Sequence[str]) -> Sequence[float]:
            pairs = [(query, text) for text in texts]
            return self._model.predict(pairs).tolist()

        return score

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not candidates:
            return []
        try:
            scores = list(self._get_score_fn()(query, [item.chunk.text for item in candidates]))
            if len(scores) != len(candidates):
                raise ValueError("reranker returned an unexpected score count")
            ranked = sorted(
                zip(candidates, scores),
                key=lambda item: (-float(item[1]), item[0].chunk.chunk_id),
            )[:top_k]
            return [
                replace(
                    candidate,
                    score=float(score),
                    rank=rank,
                    retriever="rerank",
                    metadata={**candidate.metadata, "rerank_score": float(score)},
                )
                for rank, (candidate, score) in enumerate(ranked, start=1)
            ]
        except Exception:
            return [
                replace(
                    candidate,
                    rank=rank,
                    metadata={**candidate.metadata, "rerank_fallback": True},
                )
                for rank, candidate in enumerate(candidates[:top_k], start=1)
            ]
