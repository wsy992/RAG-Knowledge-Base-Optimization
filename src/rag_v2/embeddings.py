"""Embedding provider protocols and deterministic local helpers."""

from __future__ import annotations

from typing import Protocol, Sequence

import requests


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector for each input text."""


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider cannot return valid vectors."""


class StaticEmbeddingProvider:
    """Small provider useful for tests and offline smoke runs."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            return [self.vectors[text] for text in texts]
        except KeyError as exc:
            raise KeyError(f"No static embedding configured for text: {exc.args[0]}") from exc


class OllamaEmbeddingProvider:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "nomic-embed-text",
        timeout: float = 60.0,
        api_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_token = api_token

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        headers = {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": values},
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise EmbeddingProviderError(f"Ollama embedding request failed: {exc}") from exc
        if response.status_code != 200:
            raise EmbeddingProviderError(f"Ollama embedding request returned HTTP {response.status_code}")
        try:
            payload = response.json()
            vectors = payload["embeddings"]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingProviderError("Ollama response did not contain embeddings") from exc
        if not isinstance(vectors, list) or len(vectors) != len(values):
            raise EmbeddingProviderError("Ollama returned an unexpected embedding count")
        if any(not isinstance(vector, list) or not vector for vector in vectors):
            raise EmbeddingProviderError("Ollama returned an invalid embedding vector")
        return vectors


class OllamaLangchainEmbeddings:
    """Small LangChain-compatible adapter used by the RAGAS evaluator."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.provider.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self.provider.embed([text])
        if len(vectors) != 1:
            raise EmbeddingProviderError("embedding provider returned an unexpected query vector count")
        return vectors[0]
