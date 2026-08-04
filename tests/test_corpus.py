from pathlib import Path

from rag_v2.corpus import CorpusLoader, load_benchmark_cases


def test_markdown_loader_preserves_sections_and_deterministic_chunk_ids(tmp_path: Path):
    (tmp_path / "course.md").write_text(
        "# 智能科技\n\n## 核心课程\n\n数据结构与算法、机器学习和数据库系统。\n\n"
        "## 培养目标\n\n培养学生运用人工智能解决实际问题。",
        encoding="utf-8",
    )

    chunks = CorpusLoader(max_chars=80, overlap=10).load(tmp_path)

    assert [chunk.chunk_id for chunk in chunks] == [
        "course::chunk-0001",
        "course::chunk-0002",
    ]
    assert chunks[0].section == "核心课程"
    assert "机器学习" in chunks[0].text
    assert chunks[1].section == "培养目标"


def test_loader_supports_txt_and_ignores_unsupported_extensions(tmp_path: Path):
    (tmp_path / "major.txt").write_text("就业方向包括 AI 算法工程师和数据分析师。", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("not,corpus", encoding="utf-8")

    chunks = CorpusLoader(max_chars=100, overlap=0).load(tmp_path)

    assert len(chunks) == 1
    assert chunks[0].doc_id == "major"
    assert "AI 算法工程师" in chunks[0].text


def test_loader_rejects_invalid_chunk_parameters(tmp_path: Path):
    try:
        CorpusLoader(max_chars=0)
    except ValueError as exc:
        assert "max_chars" in str(exc)
    else:
        raise AssertionError("max_chars=0 should fail")


def test_benchmark_loader_reads_jsonl_and_keeps_categories(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"case_id":"direct-1","question":"课程有哪些？","category":"direct",'
        '"gold_doc_ids":["course"],"reference_answer":"数据结构与算法",'
        '"answer_points":["数据结构与算法"],"should_abstain":false}\n'
        '{"case_id":"unknown-1","question":"宿舍怎么样？","category":"unknown",'
        '"gold_doc_ids":[],"reference_answer":"","answer_points":[],"should_abstain":true}\n',
        encoding="utf-8",
    )

    cases = load_benchmark_cases(dataset)

    assert [case.category for case in cases] == ["direct", "unknown"]
    assert cases[1].should_abstain is True
