"""Build the reusable Ollama dense-index artifact for the v2 pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag_v2.config import ConfigError, load_config  # noqa: E402
from rag_v2.corpus import CorpusLoader  # noqa: E402
from rag_v2.embeddings import EmbeddingProviderError, OllamaEmbeddingProvider  # noqa: E402
from rag_v2.indexing import corpus_hash, save_dense_artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/v2.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        loader = CorpusLoader(max_chars=config.max_chars, overlap=config.overlap)
        chunks = loader.load(config.corpus_dir)
        if not chunks:
            raise ConfigError(f"No supported documents found in {config.corpus_dir}")
        provider = OllamaEmbeddingProvider(
            base_url=config.ollama_base_url,
            model=config.embedding_model,
        )
        vectors = provider.embed([chunk.text for chunk in chunks])
        manifest_path = save_dense_artifact(
            config.artifact_dir,
            chunks=chunks,
            vectors=vectors,
            corpus_digest=corpus_hash(config.corpus_dir),
            embedding_model=config.embedding_model,
        )
    except (ConfigError, FileNotFoundError, EmbeddingProviderError, ValueError) as exc:
        print(f"Index build failed: {exc}", file=sys.stderr)
        print("Check that Ollama is running and the configured embedding model is available.", file=sys.stderr)
        return 2

    summary = {
        "manifest": str(manifest_path),
        "chunk_count": len(chunks),
        "embedding_model": config.embedding_model,
        "corpus_sha256": corpus_hash(config.corpus_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
