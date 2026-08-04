"""Run the five v2 retrieval configurations and write JSON/Markdown reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag_v2.config import ConfigError, V2Config, load_config  # noqa: E402
from rag_v2.corpus import CorpusLoader, load_benchmark_cases  # noqa: E402
from rag_v2.embeddings import EmbeddingProviderError, OllamaLangchainEmbeddings  # noqa: E402
from rag_v2.evaluator import GenerationReport, RagasRunner, evaluate_generation, evaluate_retrieval  # noqa: E402
from rag_v2.indexing import corpus_hash, load_dense_artifact  # noqa: E402
from rag_v2.runtime import RuntimeBundle, build_runtime  # noqa: E402
from rag_v2.schemas import BenchmarkCase, PipelineConfig, RagResponse  # noqa: E402


EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {"name": "bm25", "mode": "bm25", "rerank": False, "rewrite": False},
    {"name": "dense", "mode": "dense", "rerank": False, "rewrite": False},
    {"name": "dense_rerank", "mode": "dense", "rerank": True, "rewrite": False},
    {"name": "hybrid_rerank", "mode": "hybrid", "rerank": True, "rewrite": False},
    {"name": "hybrid_rerank_rewrite", "mode": "hybrid", "rerank": True, "rewrite": True},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/v2.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark"))
    parser.add_argument(
        "--generation-config",
        choices=("auto", "none", "optimized", "all"),
        default="auto",
        help="Run real DeepSeek/RAGAS on none, the optimized row, or all rows.",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Optional small smoke-run limit.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return round(ordered[index], 3)


def _pipeline_config(config: V2Config, experiment: dict[str, Any], generation_enabled: bool) -> PipelineConfig:
    return PipelineConfig(
        corpus_dir=config.corpus_dir,
        artifact_dir=config.artifact_dir,
        embedding_model=config.embedding_model,
        reranker_model=config.reranker_model,
        top_k=config.top_k,
        rerank_top_k=config.rerank_top_k,
        mode=experiment["mode"],
        rewrite_enabled=bool(experiment["rewrite"]),
        rerank_enabled=bool(experiment["rerank"]),
        generation_enabled=generation_enabled,
    )


def _resolve_generation_configs(requested: str, config: V2Config) -> set[str]:
    if requested == "none":
        return set()
    if requested == "optimized":
        return {"hybrid_rerank_rewrite"}
    if requested == "all":
        return {experiment["name"] for experiment in EXPERIMENTS}
    return {"hybrid_rerank_rewrite"} if config.deepseek_api_key else set()


def _category_summary(per_case: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_case:
        if row.get("retrieval_evaluated"):
            grouped.setdefault(str(row["category"]), []).append(row)
    result: dict[str, dict[str, float | int]] = {}
    for category, rows in sorted(grouped.items()):
        result[category] = {
            "cases": len(rows),
            "recall@5": round(sum(float(row["recall@5"]) for row in rows) / len(rows), 6),
            "mrr@5": round(sum(float(row["mrr@5"]) for row in rows) / len(rows), 6),
            "ndcg@5": round(sum(float(row["ndcg@5"]) for row in rows) / len(rows), 6),
        }
    return result


def _build_ragas_runner(bundle: RuntimeBundle, config: V2Config) -> RagasRunner:
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=config.deepseek_model,
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(OllamaLangchainEmbeddings(bundle.embedding_provider))
    return RagasRunner(llm=llm, embeddings=embeddings)


def _run_ragas(
    responses: dict[str, RagResponse | None],
    cases: list[BenchmarkCase],
    runner: RagasRunner,
) -> GenerationReport:
    samples: list[dict[str, Any]] = []
    for case in cases:
        response = responses.get(case.case_id)
        if response is None or case.should_abstain:
            continue
        samples.append(
            {
                "case_id": case.case_id,
                "user_input": case.question,
                "response": response.generation.answer,
                "retrieved_contexts": [item.chunk.text for item in response.retrieved],
                "reference": case.reference_answer,
            }
        )
    return evaluate_generation(samples, runner)


def _safe_error(error: str | None, config: V2Config) -> str | None:
    if not error:
        return None
    return error.replace(config.deepseek_api_key, "***") if config.deepseek_api_key else error


def _run_experiment(
    bundle: RuntimeBundle,
    config: V2Config,
    cases: list[BenchmarkCase],
    experiment: dict[str, Any],
    generation_enabled: bool,
    ragas_runner: RagasRunner | None,
) -> dict[str, Any]:
    pipeline_config = _pipeline_config(config, experiment, generation_enabled)
    responses: dict[str, RagResponse | None] = {}
    latencies: list[float] = []
    failures: list[dict[str, str]] = []
    for case in cases:
        started = time.perf_counter()
        try:
            response = bundle.pipeline.run(case.question, case.history, pipeline_config)
            responses[case.case_id] = response
        except Exception as exc:  # keep the report complete and count provider failures explicitly
            responses[case.case_id] = None
            failures.append({"case_id": case.case_id, "error": _safe_error(str(exc), config) or "unknown error"})
        latencies.append((time.perf_counter() - started) * 1000)

    retrieval_report = evaluate_retrieval(
        cases,
        lambda case: responses[case.case_id].retrieved if responses[case.case_id] is not None else [],
    )
    retrieval_summary = dict(retrieval_report.rows[0])
    retrieval_summary.update(
        {
            "mean_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "p95_latency_ms": _p95(latencies),
            "failure_count": len(failures),
        }
    )

    answerable_responses = [
        responses[case.case_id]
        for case in cases
        if not case.should_abstain and responses[case.case_id] is not None
    ]
    generation_summary: dict[str, Any] = {
        "status": "not_run",
        "metrics": {},
        "evaluated_cases": 0,
        "abstention_rate": round(
            sum(1 for response in answerable_responses if response.generation.abstained) / len(answerable_responses),
            6,
        )
        if answerable_responses
        else 0.0,
        "citation_rate": round(
            sum(1 for response in answerable_responses if response.generation.cited_chunk_ids) / len(answerable_responses),
            6,
        )
        if answerable_responses
        else 0.0,
    }
    if generation_enabled and ragas_runner is not None:
        generation_report = _run_ragas(responses, cases, ragas_runner)
        generation_summary.update(
            {
                "status": generation_report.status,
                "metrics": generation_report.metrics,
                "evaluated_cases": len(generation_report.per_case),
                "error": _safe_error(generation_report.error, config),
            }
        )

    per_case = []
    for row in retrieval_report.per_case:
        case_id = str(row["case_id"])
        response = responses[case_id]
        per_case.append(
            {
                **row,
                "abstained": response.generation.abstained if response is not None else None,
                "citations": response.generation.cited_chunk_ids if response is not None else [],
            }
        )
    return {
        "name": experiment["name"],
        "retrieval": retrieval_summary,
        "category_summary": _category_summary(retrieval_report.per_case),
        "generation": generation_summary,
        "failures": failures,
        "per_case": per_case,
    }


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# RAG v2 Benchmark",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        f"Cases: `{report['dataset']['case_count']}`",
        f"Corpus SHA-256: `{report['corpus']['sha256']}`",
        f"Dataset SHA-256: `{report['dataset']['sha256']}`",
        "",
        "Generation/RAGAS is marked `not_run` when no DeepSeek key is configured. Retrieval metrics remain fully offline after the dense artifact is built.",
        "",
        "## Summary",
        "",
        "| configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 | nDCG@5 | mean ms | p95 ms | RAGAS | Faithfulness | Answer relevancy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for experiment in report["experiments"]:
        retrieval = experiment["retrieval"]
        generation = experiment["generation"]
        metrics = generation.get("metrics", {})
        lines.append(
            "| {name} | {r1} | {r3} | {r5} | {mrr} | {ndcg} | {mean} | {p95} | {status} | {faith} | {relevancy} |".format(
                name=experiment["name"],
                r1=_fmt(retrieval.get("recall@1")),
                r3=_fmt(retrieval.get("recall@3")),
                r5=_fmt(retrieval.get("recall@5")),
                mrr=_fmt(retrieval.get("mrr@5")),
                ndcg=_fmt(retrieval.get("ndcg@5")),
                mean=_fmt(retrieval.get("mean_latency_ms")),
                p95=_fmt(retrieval.get("p95_latency_ms")),
                status=generation.get("status", "not_run"),
                faith=_fmt(metrics.get("faithfulness")),
                relevancy=_fmt(metrics.get("answer_relevancy")),
            )
        )
    lines.extend(["", "## Category breakdown", ""])
    for experiment in report["experiments"]:
        lines.extend([f"### {experiment['name']}", "", "| category | cases | Recall@5 | MRR@5 | nDCG@5 |", "|---|---:|---:|---:|---:|"])
        for category, values in experiment["category_summary"].items():
            lines.append(
                f"| {category} | {values['cases']} | {_fmt(values['recall@5'])} | {_fmt(values['mrr@5'])} | {_fmt(values['ndcg@5'])} |"
            )
        if experiment["failures"]:
            lines.extend(["", f"Failures: `{len(experiment['failures'])}`"])
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        cases = load_benchmark_cases(config.benchmark_path)
        if args.max_cases:
            if args.max_cases < 1:
                raise ConfigError("--max-cases must be >= 1")
            cases = cases[: args.max_cases]
        generation_configs = _resolve_generation_configs(args.generation_config, config)
        config.validate(requires_generation=bool(generation_configs))

        loader = CorpusLoader(max_chars=config.max_chars, overlap=config.overlap)
        chunks = loader.load(config.corpus_dir)
        artifact_loaded = load_dense_artifact(
            config.artifact_dir,
            chunks=chunks,
            corpus_digest=corpus_hash(config.corpus_dir),
            embedding_model=config.embedding_model,
        )
        bundle = build_runtime(config)
        ragas_runner = _build_ragas_runner(bundle, config) if generation_configs else None
    except (ConfigError, EmbeddingProviderError, FileNotFoundError, ValueError, ImportError) as exc:
        print(f"Benchmark setup failed: {exc}", file=sys.stderr)
        if "DEEPSEEK_API_KEY" in str(exc):
            print("Set DEEPSEEK_API_KEY or use --generation-config none.", file=sys.stderr)
        return 2

    results = []
    for experiment in EXPERIMENTS:
        generation_enabled = experiment["name"] in generation_configs
        print(f"Running {experiment['name']} ({len(cases)} cases)...")
        results.append(
            _run_experiment(
                bundle,
                config,
                cases,
                experiment,
                generation_enabled,
                ragas_runner if generation_enabled else None,
            )
        )

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "embedding_model": config.embedding_model,
            "reranker_model": config.reranker_model,
            "top_k": config.top_k,
            "rerank_top_k": config.rerank_top_k,
            "generation_configs": sorted(generation_configs),
        },
        "provider_status": {
            "ollama_embeddings": "configured",
            "dense_artifact": "reused" if artifact_loaded is not None else "rebuilt_in_memory",
            "deepseek": "configured" if config.deepseek_api_key else "not_configured",
            "ragas": "requested" if generation_configs else "not_run",
        },
        "corpus": {"directory": config.corpus_dir.name, "sha256": corpus_hash(config.corpus_dir), "chunk_count": len(bundle.chunks)},
        "dataset": {"path": config.benchmark_path.name, "sha256": _sha256(config.benchmark_path), "case_count": len(cases)},
        "experiments": results,
    }
    output_base = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_base.with_suffix(".json")
    markdown_path = output_base.with_suffix(".md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, markdown_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
