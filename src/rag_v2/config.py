"""Configuration loading without leaking secrets into logs or repr output."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when v2 configuration is incomplete or invalid."""


def _read_dotenv(config_path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE files without changing the process environment."""

    candidates = [config_path.parent / ".env", config_path.parent.parent / ".env", Path.cwd() / ".env"]
    values: dict[str, str] = {}
    for dotenv_path in dict.fromkeys(candidate.resolve() for candidate in candidates):
        if not dotenv_path.is_file():
            continue
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in values:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            values[key] = value
    return values


@dataclass(frozen=True)
class V2Config:
    config_path: Path
    corpus_dir: Path
    artifact_dir: Path
    benchmark_path: Path
    embedding_model: str
    reranker_model: str
    ollama_base_url: str
    deepseek_base_url: str
    deepseek_model: str
    deepseek_api_key: str = field(default="", repr=False)
    top_k: int = 5
    rerank_top_k: int = 3
    max_chars: int = 500
    overlap: int = 80

    def validate(self, requires_generation: bool = False) -> None:
        if self.top_k < 1:
            raise ConfigError("top_k must be >= 1")
        if self.rerank_top_k < 1 or self.rerank_top_k > self.top_k:
            raise ConfigError("rerank_top_k must be between 1 and top_k")
        if self.max_chars < 1:
            raise ConfigError("max_chars must be >= 1")
        if self.overlap < 0 or self.overlap >= self.max_chars:
            raise ConfigError("overlap must be >= 0 and less than max_chars")
        if requires_generation and not self.deepseek_api_key.strip():
            raise ConfigError("DEEPSEEK_API_KEY is required for generation")


def load_config(path: Path) -> V2Config:
    config_path = Path(path).resolve()
    dotenv_values = _read_dotenv(config_path)
    try:
        raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config: {config_path}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping")

    def path_value(name: str, default: str) -> Path:
        value = raw.get(name, default)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{name} must be a non-empty path")
        candidate = Path(value)
        return (config_path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    try:
        config = V2Config(
            config_path=config_path,
            corpus_dir=path_value("corpus_dir", "data/demo_corpus"),
            artifact_dir=path_value("artifact_dir", "artifacts"),
            benchmark_path=path_value("benchmark_path", "eval/dataset.jsonl"),
            embedding_model=str(raw.get("embedding_model", "nomic-embed-text")),
            reranker_model=str(raw.get("reranker_model", "BAAI/bge-reranker-v2-m3")),
            ollama_base_url=str(raw.get("ollama_base_url", "http://127.0.0.1:11434")),
            deepseek_base_url=str(raw.get("deepseek_base_url", "https://api.deepseek.com/v1")),
            deepseek_model=str(raw.get("deepseek_model", "deepseek-chat")),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", dotenv_values.get("DEEPSEEK_API_KEY", "")).strip(),
            top_k=int(raw.get("top_k", 5)),
            rerank_top_k=int(raw.get("rerank_top_k", 3)),
            max_chars=int(raw.get("max_chars", 500)),
            overlap=int(raw.get("overlap", 80)),
        )
        config.validate()
        return config
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(str(exc)) from exc
