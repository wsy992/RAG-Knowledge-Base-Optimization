"""Evidence-grounded answer generation and source citation parsing."""

from __future__ import annotations

import json
import re
import time

from .providers import ChatClient, ChatProviderError
from .schemas import GenerationResult, RetrievedChunk

ABSTENTION_ANSWER = "无法根据知识库证据回答该问题。"
_CITATION_PATTERN = re.compile(r"\[source:([^\]]+)\]")


class GroundedGenerator:
    def __init__(self, client: ChatClient) -> None:
        self.client = client

    def answer(self, question: str, contexts: list[RetrievedChunk]) -> GenerationResult:
        started = time.perf_counter()
        if not contexts:
            return GenerationResult(ABSTENTION_ANSWER, [], True, self._elapsed_ms(started))

        allowed_ids = {item.chunk.chunk_id for item in contexts}
        context_text = "\n\n".join(
            f"[source:{item.chunk.chunk_id}] {item.chunk.text}" for item in contexts
        )
        messages = [
            {
                "role": "system",
                "content": "你是严格基于证据回答的知识库助手。只能使用给定上下文，事实性内容必须引用 [source:chunk_id]。证据不足时只回答无法根据知识库证据回答该问题。",
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n上下文：\n{context_text}\n\n请给出简洁答案并保留来源引用。",
            },
        ]
        try:
            raw = self.client.complete(messages, temperature=0.1)
            answer = self._extract_answer(raw)
        except (ChatProviderError, ValueError, TypeError, json.JSONDecodeError):
            return GenerationResult(ABSTENTION_ANSWER, [], True, self._elapsed_ms(started))

        cited = [citation for citation in _CITATION_PATTERN.findall(answer) if citation in allowed_ids]
        if "无法根据知识库证据回答" in answer:
            return GenerationResult(ABSTENTION_ANSWER, [], True, self._elapsed_ms(started))
        return GenerationResult(answer.strip(), list(dict.fromkeys(cited)), False, self._elapsed_ms(started))

    @staticmethod
    def _extract_answer(raw: str) -> str:
        content = raw.strip()
        if content.startswith("{"):
            payload = json.loads(content)
            content = str(payload.get("answer", "")).strip()
        if not content:
            raise ValueError("empty generated answer")
        return content

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 3)
