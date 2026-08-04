from pathlib import Path
import sys

from scripts.run_benchmark import _p95, _resolve_generation_configs
import scripts.run_benchmark as benchmark
from rag_v2.config import V2Config
from rag_v2.embeddings import EmbeddingProviderError


def make_config(api_key: str = "") -> V2Config:
    return V2Config(
        config_path=Path("v2.yaml"),
        corpus_dir=Path("corpus"),
        artifact_dir=Path("artifacts"),
        benchmark_path=Path("dataset.jsonl"),
        embedding_model="embedding",
        reranker_model="reranker",
        ollama_base_url="http://127.0.0.1:11434",
        deepseek_base_url="https://api.deepseek.com/v1",
        deepseek_model="deepseek-chat",
        deepseek_api_key=api_key,
    )


def test_auto_generation_is_disabled_without_a_key_and_optimized_with_one():
    assert _resolve_generation_configs("auto", make_config()) == set()
    assert _resolve_generation_configs("auto", make_config("key")) == {"hybrid_rerank_rewrite"}


def test_p95_is_deterministic_for_small_latency_samples():
    assert _p95([3.0, 1.0, 2.0, 8.0]) == 8.0
    assert _p95([]) == 0.0


def test_main_converts_embedding_provider_failure_to_setup_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(benchmark, "build_runtime", lambda config: (_ for _ in ()).throw(EmbeddingProviderError("ollama unavailable")))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_benchmark.py", "--config", "config/v2.yaml", "--generation-config", "none", "--max-cases", "1", "--output", str(tmp_path / "report")],
    )

    assert benchmark.main() == 2
    assert "Benchmark setup failed" in capsys.readouterr().err
