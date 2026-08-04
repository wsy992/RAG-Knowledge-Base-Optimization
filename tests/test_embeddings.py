import pytest

from rag_v2.embeddings import EmbeddingProviderError, OllamaEmbeddingProvider


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_ollama_provider_returns_one_vector_per_text(monkeypatch):
    calls = []

    def fake_post(url, *, json, timeout, headers=None):
        calls.append((url, json, timeout))
        return FakeResponse(200, {"embeddings": [[1.0, 0.0], [0.0, 1.0]]})

    monkeypatch.setattr("rag_v2.embeddings.requests.post", fake_post)
    provider = OllamaEmbeddingProvider(base_url="http://localhost:11434", model="nomic-embed-text")

    result = provider.embed(["第一段", "第二段"])

    assert result == [[1.0, 0.0], [0.0, 1.0]]
    assert calls[0][0] == "http://localhost:11434/api/embed"
    assert calls[0][1]["model"] == "nomic-embed-text"
    assert calls[0][1]["input"] == ["第一段", "第二段"]


def test_ollama_provider_rejects_bad_response_without_leaking_credentials(monkeypatch):
    def fake_post(url, *, json, timeout, headers=None):
        return FakeResponse(503, {"error": "service unavailable"})

    monkeypatch.setattr("rag_v2.embeddings.requests.post", fake_post)
    provider = OllamaEmbeddingProvider(api_token="secret-token")

    with pytest.raises(EmbeddingProviderError) as exc_info:
        provider.embed(["text"])

    assert "secret-token" not in str(exc_info.value)
    assert "503" in str(exc_info.value)


def test_ollama_provider_returns_empty_list_without_network(monkeypatch):
    def fail_post(*args, **kwargs):
        raise AssertionError("empty input must not call Ollama")

    monkeypatch.setattr("rag_v2.embeddings.requests.post", fail_post)

    assert OllamaEmbeddingProvider().embed([]) == []
