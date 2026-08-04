import json
import sys
import types

import pytest

from rag_v2.generation import ABSTENTION_ANSWER, GroundedGenerator
from rag_v2.rerank import CrossEncoderReranker
from rag_v2.rewrite import DeepSeekQueryRewriter
from rag_v2.providers import ChatProviderError
from rag_v2.schemas import DocumentChunk, RetrievedChunk


def make_candidates() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            DocumentChunk("doc-a", "A", "a.md", "A", "a::1", "甲内容"),
            score=0.4,
            rank=1,
            retriever="dense",
        ),
        RetrievedChunk(
            DocumentChunk("doc-b", "B", "b.md", "B", "b::1", "乙内容"),
            score=0.3,
            rank=2,
            retriever="dense",
        ),
    ]


def test_cross_encoder_reranker_reorders_candidates_by_score():
    reranker = CrossEncoderReranker(score_fn=lambda query, texts: [0.1, 0.9])

    results = reranker.rerank("问题", make_candidates(), top_k=2)

    assert [result.chunk.doc_id for result in results] == ["doc-b", "doc-a"]
    assert results[0].metadata["rerank_score"] == pytest.approx(0.9)
    assert results[0].rank == 1


def test_cross_encoder_reranker_falls_back_to_candidates_on_provider_failure():
    def fail_score(query, texts):
        raise RuntimeError("model unavailable")

    original = make_candidates()
    results = CrossEncoderReranker(score_fn=fail_score).rerank("问题", original, top_k=2)

    assert [result.chunk.chunk_id for result in results] == ["a::1", "b::1"]
    assert results[0].metadata["rerank_fallback"] is True


def test_cross_encoder_reranker_caches_model_load_failure(monkeypatch):
    calls = []

    class FailingCrossEncoder:
        def __init__(self, model_name):
            calls.append(model_name)
            raise RuntimeError("model unavailable")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FailingCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    reranker = CrossEncoderReranker("test-reranker")

    reranker.rerank("query", make_candidates(), top_k=2)
    reranker.rerank("query", make_candidates(), top_k=2)

    assert calls == ["test-reranker"]


class FailingChatClient:
    def complete(self, messages, *, temperature=0.1, response_format=None):
        raise ChatProviderError("provider unavailable")


class JsonChatClient:
    def __init__(self, payload):
        self.payload = payload
        self.messages = []

    def complete(self, messages, *, temperature=0.1, response_format=None):
        self.messages.append(messages)
        return json.dumps(self.payload, ensure_ascii=False)


def test_query_rewriter_falls_back_to_original_question_on_provider_failure():
    result = DeepSeekQueryRewriter(FailingChatClient()).rewrite(
        "学费多少？", [{"role": "user", "content": "智能科技专业有哪些课？"}]
    )

    assert result.original_query == "学费多少？"
    assert result.rewritten_query == "学费多少？"
    assert result.used_fallback is True


def test_query_rewriter_parses_json_and_includes_recent_history():
    client = JsonChatClient({"rewritten_query": "智能科技专业 学费", "reason": "补全主题"})
    result = DeepSeekQueryRewriter(client).rewrite(
        "学费多少？", [{"role": "user", "content": "智能科技专业有哪些课？"}]
    )

    assert result.rewritten_query == "智能科技专业 学费"
    assert result.used_fallback is False
    assert "智能科技专业有哪些课？" in client.messages[0][1]["content"]


def test_grounded_generator_abstains_without_context():
    result = GroundedGenerator(FailingChatClient()).answer("未知问题", [])

    assert result.abstained is True
    assert result.answer == ABSTENTION_ANSWER
    assert result.cited_chunk_ids == []


def test_grounded_generator_extracts_source_citations():
    client = JsonChatClient({"answer": "机器学习课程见 [source:a::1]"})
    contexts = make_candidates()[:1]

    result = GroundedGenerator(client).answer("有哪些课程？", contexts)

    assert result.abstained is False
    assert result.cited_chunk_ids == ["a::1"]
    assert "[source:a::1]" in result.answer
