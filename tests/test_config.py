from pathlib import Path

import pytest

from rag_v2.config import ConfigError, load_config


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config" / "v2.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        "corpus_dir: data/demo_corpus\n"
        "artifact_dir: artifacts\n"
        "benchmark_path: eval/dataset.jsonl\n"
        "embedding_model: nomic-embed-text\n"
        "reranker_model: BAAI/bge-reranker-v2-m3\n"
        "ollama_base_url: http://127.0.0.1:11434\n"
        "deepseek_base_url: https://api.deepseek.com/v1\n"
        "deepseek_model: deepseek-chat\n",
        encoding="utf-8",
    )
    return config_path


def test_load_config_resolves_relative_paths_and_hides_secret(monkeypatch, tmp_path: Path):
    config_path = write_config(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "hidden-secret")

    config = load_config(config_path)

    assert config.corpus_dir == (tmp_path / "config" / "data/demo_corpus").resolve()
    assert config.benchmark_path == (tmp_path / "config" / "eval/dataset.jsonl").resolve()
    assert "hidden-secret" not in repr(config)
    assert config.deepseek_api_key == "hidden-secret"


def test_generation_credentials_are_required_only_for_generation(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    config = load_config(write_config(tmp_path))

    config.validate(requires_generation=False)
    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
        config.validate(requires_generation=True)


def test_invalid_yaml_value_has_actionable_error(tmp_path: Path):
    config_path = tmp_path / "v2.yaml"
    config_path.write_text("top_k: 0\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="top_k"):
        load_config(config_path)


def test_load_config_reads_repo_dotenv_without_overriding_process_env(monkeypatch, tmp_path: Path):
    config_path = write_config(tmp_path)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    config = load_config(config_path)

    assert config.deepseek_api_key == "from-dotenv"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-process")
    assert load_config(config_path).deepseek_api_key == "from-process"
