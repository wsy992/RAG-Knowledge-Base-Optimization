"""Query rewriting with history-aware prompts and safe fallback."""

from __future__ import annotations

import json
import re

from .providers import ChatClient, ChatProviderError
from .schemas import RewriteResult


class DeepSeekQueryRewriter:
    def __init__(self, client: ChatClient, max_history_turns: int = 4) -> None:
        self.client = client
        self.max_history_turns = max_history_turns

    def rewrite(self, question: str, history: list[dict[str, str]]) -> RewriteResult:
        original = question.strip()
        if not original:
            raise ValueError("question must not be empty")
        recent_history = history[-self.max_history_turns :]
        history_text = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '').strip()}"
            for item in recent_history
            if item.get("content", "").strip()
        ) or "无历史对话"
        messages = [
            {
                "role": "system",
                "content": "你是检索查询改写器，只返回 JSON：{\"rewritten_query\":\"...\",\"reason\":\"...\"}。不要回答问题。",
            },
            {
                "role": "user",
                "content": f"历史对话：\n{history_text}\n\n当前问题：{original}\n请补全指代并改写为适合知识库检索的查询。",
            },
        ]
        try:
            raw = self.client.complete(messages, temperature=0.1, response_format={"type": "json_object"})
            payload = json.loads(self._strip_code_fence(raw))
            rewritten = str(payload.get("rewritten_query", "")).strip()
            reason = str(payload.get("reason", "")).strip()
            if not rewritten:
                raise ValueError("rewritten_query is empty")
            return RewriteResult(original, rewritten, reason or "model rewrite", False)
        except (ChatProviderError, ValueError, TypeError, json.JSONDecodeError, AttributeError):
            return RewriteResult(original, original, "rewrite provider unavailable; used original query", True)

    @staticmethod
    def _strip_code_fence(content: str) -> str:
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
