from collections import Counter
from pathlib import Path

from rag_v2.corpus import load_benchmark_cases


def test_demo_dataset_has_declared_split_counts_and_valid_document_ids():
    cases = load_benchmark_cases(Path("eval/dataset.jsonl"))
    counts = Counter(case.category for case in cases)
    valid_doc_ids = {
        "cityu_intelligent_technology",
        "distributed_training",
        "llm_application_principles",
        "llm_stack_practice",
        "llm_stack_algorithms",
        "instruction_alignment",
        "inference_optimization",
    }

    assert len(cases) == 60
    assert counts == {
        "direct": 15,
        "fuzzy": 15,
        "cross_document": 12,
        "multi_turn": 10,
        "unknown": 8,
    }
    assert all(set(case.gold_doc_ids) <= valid_doc_ids for case in cases)
    assert all(case.history for case in cases if case.category == "multi_turn")
