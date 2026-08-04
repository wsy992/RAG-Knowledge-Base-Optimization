"""Typed contracts shared by the v2 pipeline and evaluation code."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _required(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


@dataclass(frozen=True)
class DocumentChunk:
    doc_id: str
    title: str
    source: str
    section: str
    chunk_id: str
    text: str

    def __post_init__(self) -> None:
        for name in ("doc_id", "title", "source", "chunk_id"):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        object.__setattr__(self, "section", self.section.strip())
        object.__setattr__(self, "text", _required(self.text, "text"))


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    rank: int
    retriever: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if not self.retriever.strip():
            raise ValueError("retriever must not be empty")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    question: str
    category: str
    gold_doc_ids: list[str]
    reference_answer: str
    answer_points: list[str]
    should_abstain: bool
    history: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("case_id", "question", "category"):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        object.__setattr__(self, "gold_doc_ids", [item.strip() for item in self.gold_doc_ids if item.strip()])
        object.__setattr__(self, "answer_points", [item.strip() for item in self.answer_points if item.strip()])
        object.__setattr__(self, "reference_answer", self.reference_answer.strip())
        if not self.should_abstain and not self.gold_doc_ids:
            raise ValueError("gold_doc_ids must not be empty for answerable cases")
        if not self.should_abstain and not self.reference_answer:
            raise ValueError("reference_answer must not be empty for answerable cases")


@dataclass(frozen=True)
class PipelineConfig:
    corpus_dir: Path
    artifact_dir: Path
    embedding_model: str
    reranker_model: str
    top_k: int = 5
    rerank_top_k: int = 3
    mode: str = "hybrid"
    rewrite_enabled: bool = True
    rerank_enabled: bool = True
    generation_enabled: bool = True

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.rerank_top_k < 1:
            raise ValueError("rerank_top_k must be >= 1")
        if self.rerank_top_k > self.top_k:
            raise ValueError("rerank_top_k must be <= top_k")
        if self.mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("mode must be bm25, dense, or hybrid")
        for name in ("embedding_model", "reranker_model"):
            _required(getattr(self, name), name)


@dataclass(frozen=True)
class RewriteResult:
    original_query: str
    rewritten_query: str
    reason: str
    used_fallback: bool = False


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    cited_chunk_ids: list[str]
    abstained: bool
    latency_ms: float = 0.0


@dataclass(frozen=True)
class RagResponse:
    rewrite: RewriteResult
    retrieved: list[RetrievedChunk]
    generation: GenerationResult
    trace: list[dict[str, Any]] = field(default_factory=list)
