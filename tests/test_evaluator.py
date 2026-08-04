from rag_v2.evaluator import GenerationReport, RetrievalReport, evaluate_generation, evaluate_retrieval
from rag_v2.schemas import BenchmarkCase, DocumentChunk, RetrievedChunk


def make_result(doc_id: str, rank: int = 1) -> RetrievedChunk:
    chunk = DocumentChunk(doc_id, doc_id, f"{doc_id}.md", "section", f"{doc_id}::1", doc_id)
    return RetrievedChunk(chunk, score=1.0 / rank, rank=rank, retriever="test")


def test_evaluate_retrieval_excludes_unknown_cases_from_retrieval_denominator():
    cases = [
        BenchmarkCase("direct-1", "课程？", "direct", ["course"], "课程", ["课程"], False),
        BenchmarkCase("unknown-1", "宿舍？", "unknown", [], "", [], True),
    ]

    def runner(case):
        return [make_result("course")] if case.case_id == "direct-1" else []

    report = evaluate_retrieval(cases, runner)

    assert isinstance(report, RetrievalReport)
    assert report.rows[0]["recall@1"] == 1.0
    assert report.rows[0]["evaluated_cases"] == 1
    assert report.per_case[1]["retrieval_evaluated"] is False


class FakeRagasRunner:
    def __init__(self):
        self.received = None

    def evaluate(self, samples):
        self.received = samples
        return GenerationReport(
            metrics={"faithfulness": 0.8, "answer_relevancy": 0.7},
            status="ok",
            per_case=[{"case_id": "case-1", "faithfulness": 0.8}],
        )


def test_evaluate_generation_passes_actual_context_samples_to_ragas_runner():
    runner = FakeRagasRunner()
    samples = [
        {
            "case_id": "case-1",
            "user_input": "课程有哪些？",
            "response": "机器学习 [source:course::1]",
            "retrieved_contexts": ["机器学习"],
            "reference": "机器学习",
        }
    ]

    report = evaluate_generation(samples, runner)

    assert isinstance(report, GenerationReport)
    assert runner.received == samples
    assert report.metrics["faithfulness"] == 0.8
