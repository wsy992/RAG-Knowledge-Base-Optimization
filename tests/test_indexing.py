from pathlib import Path

from rag_v2.corpus import CorpusLoader
from rag_v2.indexing import corpus_hash, load_dense_artifact, save_dense_artifact


def test_dense_artifact_round_trip_is_tied_to_corpus_hash(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "course.md"
    source.write_text("# 课程\n\n机器学习课程。", encoding="utf-8")
    chunks = CorpusLoader(max_chars=100, overlap=10).load(corpus)
    vectors = [[1.0, 0.0] for _ in chunks]
    artifact_dir = tmp_path / "artifacts"
    digest = corpus_hash(corpus)

    save_dense_artifact(
        artifact_dir,
        chunks=chunks,
        vectors=vectors,
        corpus_digest=digest,
        embedding_model="test-embedding",
    )

    loaded = load_dense_artifact(
        artifact_dir,
        chunks=chunks,
        corpus_digest=digest,
        embedding_model="test-embedding",
    )
    assert loaded == vectors

    source.write_text("# 课程\n\n机器学习课程发生变化。", encoding="utf-8")
    assert (
        load_dense_artifact(
            artifact_dir,
            chunks=CorpusLoader(max_chars=100, overlap=10).load(corpus),
            corpus_digest=corpus_hash(corpus),
            embedding_model="test-embedding",
        )
        is None
    )
