"""External chat provider interfaces used by rewriting and generation."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import requests


class ChatProviderError(RuntimeError):
    """Raised when a chat provider cannot return a usable response."""


class ChatClient(Protocol):
    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.1,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Return the assistant text for a chat request."""


class DeepSeekChatClient:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        timeout: float = 60.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.1,
        response_format: dict[str, str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ChatProviderError(f"chat request failed: {exc}") from exc
        if response.status_code != 200:
            raise ChatProviderError(f"chat request returned HTTP {response.status_code}")
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ChatProviderError("chat response did not contain assistant content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ChatProviderError("chat response contained empty assistant content")
        return content.strip()
