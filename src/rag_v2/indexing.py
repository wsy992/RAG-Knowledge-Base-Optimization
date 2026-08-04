"""Persistent dense-index artifacts with corpus/model validation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .corpus import SUPPORTED_SUFFIXES
from .schemas import DocumentChunk

MANIFEST_NAME = "index_manifest.json"
DENSE_ARTIFACT_NAME = "dense_embeddings.npz"


def corpus_hash(root: Path) -> str:
    """Return a stable digest for supported corpus files and their relative paths."""

    root = Path(root).resolve()
    digest = hashlib.sha256()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def save_dense_artifact(
    artifact_dir: Path,
    *,
    chunks: Sequence[DocumentChunk],
    vectors: Sequence[Sequence[float]],
    corpus_digest: str,
    embedding_model: str,
) -> Path:
    """Persist vectors plus enough metadata to reject stale indexes."""

    if len(chunks) != len(vectors):
        raise ValueError("vectors must contain one row per chunk")
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks) or matrix.shape[1] < 1:
        raise ValueError("vectors must be a non-empty 2-D matrix")
    if not embedding_model.strip():
        raise ValueError("embedding_model must not be empty")

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_dir / DENSE_ARTIFACT_NAME,
        chunk_ids=np.asarray([chunk.chunk_id for chunk in chunks], dtype=str),
        vectors=matrix,
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": corpus_digest,
        "embedding_model": embedding_model,
        "chunk_count": len(chunks),
        "vector_dimension": int(matrix.shape[1]),
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "dense_artifact": DENSE_ARTIFACT_NAME,
    }
    manifest_path = artifact_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def load_dense_artifact(
    artifact_dir: Path,
    *,
    chunks: Sequence[DocumentChunk],
    corpus_digest: str,
    embedding_model: str,
) -> list[list[float]] | None:
    """Load an artifact only when all corpus/model/chunk identity checks pass."""

    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / MANIFEST_NAME
    dense_path = artifact_dir / DENSE_ARTIFACT_NAME
    if not manifest_path.exists() or not dense_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if (
            manifest.get("schema_version") != 1
            or manifest.get("corpus_sha256") != corpus_digest
            or manifest.get("embedding_model") != embedding_model
            or manifest.get("chunk_ids") != chunk_ids
        ):
            return None
        with np.load(dense_path, allow_pickle=False) as data:
            stored_ids = data["chunk_ids"].astype(str).tolist()
            matrix = np.asarray(data["vectors"], dtype=np.float32)
        if stored_ids != chunk_ids or matrix.ndim != 2 or matrix.shape != (len(chunks), int(manifest["vector_dimension"])):
            return None
        if np.any(~np.isfinite(matrix)):
            return None
        return matrix.tolist()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
