"""Benchmark orchestration and real RAGAS integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from .schemas import BenchmarkCase, RetrievedChunk


@dataclass
class RetrievalReport:
    rows: list[dict[str, float | int]]
    per_case: list[dict[str, Any]]


@dataclass
class GenerationReport:
    metrics: dict[str, float]
    status: str
    per_case: list[dict[str, Any]]
    error: str | None = None


class RagasRunnerProtocol(Protocol):
    def evaluate(self, samples: list[dict[str, Any]]) -> GenerationReport:
        """Evaluate answer/context samples with a real generation evaluator."""


def evaluate_retrieval(
    cases: list[BenchmarkCase],
    runner: Callable[[BenchmarkCase], list[RetrievedChunk]],
) -> RetrievalReport:
    per_case: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = {
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "mrr@5": [],
        "ndcg@5": [],
    }
    for case in cases:
        results = runner(case)
        evaluated = not case.should_abstain
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "category": case.category,
            "retrieved_count": len(results),
            "retrieval_evaluated": evaluated,
        }
        if evaluated:
            gold = set(case.gold_doc_ids)
            row.update(
                {
                    "recall@1": recall_at_k(results, gold, 1),
                    "recall@3": recall_at_k(results, gold, 3),
                    "recall@5": recall_at_k(results, gold, 5),
                    "mrr@5": mean_reciprocal_rank(results, gold, 5),
                    "ndcg@5": ndcg_at_k(results, gold, 5),
                }
            )
            for name in metric_values:
                metric_values[name].append(float(row[name]))
        per_case.append(row)

    evaluated_cases = len(metric_values["recall@1"])
    summary: dict[str, float | int] = {
        name: (sum(values) / len(values) if values else 0.0)
        for name, values in metric_values.items()
    }
    summary["evaluated_cases"] = evaluated_cases
    return RetrievalReport(rows=[summary], per_case=per_case)


def evaluate_generation(
    samples: list[dict[str, Any]],
    ragas_runner: RagasRunnerProtocol,
) -> GenerationReport:
    return ragas_runner.evaluate(samples)


class RagasRunner:
    """Adapter for RAGAS 0.4+ using actual retrieved contexts."""

    def __init__(self, llm: Any, embeddings: Any) -> None:
        self.llm = llm
        self.embeddings = embeddings

    def evaluate(self, samples: list[dict[str, Any]]) -> GenerationReport:
        if not samples:
            return GenerationReport(metrics={}, status="not_run", per_case=[])
        try:
            from ragas import EvaluationDataset, evaluate
            from ragas.dataset_schema import SingleTurnSample
            from ragas.metrics.collections import AnswerRelevancy, Faithfulness

            dataset = EvaluationDataset(
                samples=[
                    SingleTurnSample(
                        user_input=sample["user_input"],
                        response=sample["response"],
                        retrieved_contexts=sample["retrieved_contexts"],
                        reference=sample.get("reference"),
                    )
                    for sample in samples
                ]
            )
            result = evaluate(
                dataset,
                metrics=[
                    Faithfulness(llm=self.llm),
                    AnswerRelevancy(llm=self.llm, embeddings=self.embeddings),
                ],
                raise_exceptions=False,
                show_progress=False,
            )
            frame = result.to_pandas()
            metrics: dict[str, float] = {}
            for name in ("faithfulness", "answer_relevancy"):
                if name in frame.columns:
                    values = frame[name].dropna()
                    if not values.empty:
                        metrics[name] = float(values.mean())
            per_case = []
            rows = frame.to_dict(orient="records")
            for index, row in enumerate(rows):
                per_case.append({"case_id": samples[index].get("case_id", str(index)), **row})
            status = "ok" if metrics else "error"
            return GenerationReport(metrics=metrics, status=status, per_case=per_case)
        except Exception as exc:  # provider/version errors are reported, not converted to fake scores
            return GenerationReport(metrics={}, status="error", per_case=[], error=str(exc))
