# RAG v2 Benchmark and Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-controlled RAG v2 pipeline, a multi-document benchmark with real retrieval and generation metrics, and a local Streamlit demo while preserving the existing Chatchat-based project as a baseline.

**Architecture:** The v2 code lives under `src/rag_v2/` and exposes explicit corpus, retrieval, reranking, rewrite, generation, pipeline, and evaluation interfaces. The existing `enhancements/` modules remain as v1 experiments and are connected only through explicit adapters or separate experiment flags. A fixed corpus and JSONL benchmark drive both CLI reports and the demo.

**Tech Stack:** Python 3.11, dataclasses, PyYAML, NumPy/FAISS-compatible dense retrieval, `rank-bm25`, `sentence-transformers` Cross-Encoder, Ollama embeddings, DeepSeek-compatible chat API, RAGAS, pytest, Streamlit.

## Global Constraints

- All code must run on Windows with Python 3.11 and PowerShell commands documented in `README.md`.
- No API key, model output containing private data, or absolute path such as `D:\chatchat-project` may be committed.
- Unit tests must run without network access, Ollama, Chatchat, or a DeepSeek key.
- Real generation and RAGAS metrics must be marked unavailable when their provider is unavailable; heuristic faithfulness must not be used as a substitute.
- Every retrieval experiment must use the same corpus, benchmark cases, and generation configuration.
- Existing Chatchat behavior is not rewritten wholesale; v2 is additive and independently testable.
- Every new production function is introduced through a failing test first, then a minimal implementation, then a complete test run.

---

### Task 1: Create an isolated implementation workspace and record the baseline

**Files:**
- Create: no repository files; create a linked worktree at `.worktrees/rag-v2-benchmark-demo` on branch `codex/rag-v2-benchmark-demo`.
- Inspect: `README.md`, `requirements.txt`, `enhancements/*.py`, `eval/ragas_eval.py`, `docs/CHANGELOG.md`.

**Interfaces:**
- Produces an isolated checkout from commit `4fdde5e` with the design and plan documents already present.
- Produces a baseline report containing Python version, installed package versions, `pytest` availability, and the result of `python -m compileall enhancements eval`.

- [ ] **Step 1: Verify repository isolation state**

Run:

```powershell
git -C D:\chatchat-project\github-repo rev-parse --git-dir
git -C D:\chatchat-project\github-repo rev-parse --git-common-dir
git -C D:\chatchat-project\github-repo branch --show-current
git -C D:\chatchat-project\github-repo status --short --branch
```

Expected: the repository is clean on `master`; it is not already a linked worktree.

- [ ] **Step 2: Verify the worktree directory is safe**

Run:

```powershell
git -C D:\chatchat-project\github-repo check-ignore -q .worktrees
```

If this fails, add `.worktrees/` to the repository `.gitignore`, commit that isolated-directory rule, and rerun the check before creating the worktree.

- [ ] **Step 3: Create the branch and worktree**

Run:

```powershell
git -C D:\chatchat-project\github-repo worktree add D:\chatchat-project\github-repo\.worktrees\rag-v2-benchmark-demo -b codex/rag-v2-benchmark-demo master
```

- [ ] **Step 4: Run the baseline checks**

Run from the worktree:

```powershell
python --version
python -m compileall enhancements eval
python -m pytest --collect-only -q
```

Expected: compilation succeeds; if no tests exist, pytest reports zero collected tests without hiding import errors. Record any environment failure before implementation.

---

### Task 2: Add package configuration, typed data contracts, and corpus loading

**Files:**
- Create: `pyproject.toml`
- Create: `src/rag_v2/__init__.py`
- Create: `src/rag_v2/schemas.py`
- Create: `src/rag_v2/corpus.py`
- Create: `tests/test_schemas.py`
- Create: `tests/test_corpus.py`
- Create: `data/demo_corpus/manifest.json`
- Create: `data/demo_corpus/*.md` and `data/demo_corpus/*.txt`

**Interfaces:**
- `DocumentChunk(doc_id: str, title: str, source: str, section: str, chunk_id: str, text: str)`
- `RetrievedChunk(chunk: DocumentChunk, score: float, rank: int, retriever: str, metadata: dict[str, object])`
- `BenchmarkCase(case_id: str, question: str, category: str, gold_doc_ids: list[str], reference_answer: str, answer_points: list[str], should_abstain: bool)`
- `V2Config(corpus_dir: Path, artifact_dir: Path, embedding_model: str, reranker_model: str, top_k: int, rerank_top_k: int)`
- `CorpusLoader.load(root: Path) -> list[DocumentChunk]`
- `CorpusLoader.chunk_document(text: str, doc_id: str, title: str, source: str, max_chars: int = 500, overlap: int = 80) -> list[DocumentChunk]`
- `load_benchmark_cases(path: Path) -> list[BenchmarkCase]`
- `PipelineConfig(mode: str, top_k: int, rerank_top_k: int, rewrite_enabled: bool, generation_enabled: bool)`

- [ ] **Step 1: Write failing schema tests**

Add tests asserting that a valid chunk preserves all identifiers, a blank `chunk_id` raises `ValueError`, and a benchmark case with no gold document IDs is allowed only when `should_abstain` is true.

- [ ] **Step 2: Run the schema tests and verify the expected failure**

Run:

```powershell
python -m pytest tests/test_schemas.py -q
```

Expected: FAIL because `src/rag_v2/schemas.py` does not exist.

- [ ] **Step 3: Implement the minimal dataclasses and validation**

Use frozen dataclasses, strip text at construction, and raise clear `ValueError` messages for missing IDs, empty text, or an invalid abstention/gold-document combination.

- [ ] **Step 4: Run the schema tests and verify they pass**

Run the same command; expected: PASS.

- [ ] **Step 5: Write failing corpus tests**

Create temporary Markdown and TXT files in the test fixture. Assert that headings become `section`, chunks have deterministic IDs, overlap does not duplicate the first character range, and unsupported extensions are ignored with a warning.

- [ ] **Step 6: Implement corpus loading and chunking**

Support Markdown, TXT, and text-extractable PDF through PyMuPDF when installed. Keep source metadata on every chunk and expose a deterministic sort order by `doc_id` and `chunk_id`.

- [ ] **Step 7: Add the demo corpus and manifest**

Copy the existing six Chatchat sample Markdown documents from `D:\chatchat-project\Langchain-Chatchat\.venv\Lib\site-packages\chatchat\data\knowledge_base\samples\content` and the existing `D:\test-rag\澳门城市大学智能科技专业介绍.txt` into `data/demo_corpus/` with stable ASCII filenames. Record their local source paths and source titles in `manifest.json`; keep the corpus limited to text content needed for the demo.

- [ ] **Step 8: Add package metadata and run all corpus tests**

Configure `pyproject.toml` with `src` as the package root and pytest's `pythonpath`. Run:

```powershell
python -m pytest tests/test_schemas.py tests/test_corpus.py -q
```

Expected: PASS with no network calls.

- [ ] **Step 9: Commit the self-contained corpus layer**

```powershell
git add pyproject.toml src/rag_v2 tests/test_schemas.py tests/test_corpus.py data/demo_corpus
git commit -m "feat: add v2 corpus and evaluation data contracts"
```

---

### Task 3: Implement deterministic BM25, dense, and hybrid retrieval

**Files:**
- Create: `src/rag_v2/embeddings.py`
- Create: `src/rag_v2/index.py`
- Create: `src/rag_v2/retrieval.py`
- Create: `tests/test_retrieval.py`
- Create: `tests/fixtures/retrieval_chunks.json`
- Modify: `.gitignore` to exclude `.env`, `artifacts/`, `.pytest_cache/`, and Python cache directories.
- Modify: `requirements.txt` to include the direct v2 dependencies already used by code.

**Interfaces:**
- `EmbeddingProvider.embed(texts: list[str]) -> list[list[float]]`
- `Bm25Index.build(chunks: list[DocumentChunk]) -> None`
- `DenseIndex.build(chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None`
- `Retriever.retrieve(query: str, mode: Literal["bm25", "dense", "hybrid"], top_k: int) -> list[RetrievedChunk]`
- `HybridRetriever.fuse(bm25: list[RetrievedChunk], dense: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]`

- [ ] **Step 1: Write failing retrieval tests**

Use four short fixture chunks and a deterministic `FakeEmbeddingProvider`. Assert that BM25 finds an exact lexical match, dense retrieval finds a paraphrase, hybrid removes duplicate chunk IDs, and requesting `top_k=0` raises `ValueError`.

- [ ] **Step 2: Run retrieval tests and verify the expected failure**

```powershell
python -m pytest tests/test_retrieval.py -q
```

Expected: FAIL because the v2 index and retrieval modules are missing.

- [ ] **Step 3: Implement BM25 and dense indexes**

Use `jieba` tokenization for Chinese BM25 text and cosine similarity for dense vectors. Store serialized indexes under a configured artifact directory; use NumPy for the deterministic core and an optional FAISS adapter only when FAISS is available.

- [ ] **Step 4: Implement hybrid fusion**

Use reciprocal rank fusion with `k=60`, deduplicate by `chunk_id`, preserve each component's score in `RetrievedChunk.metadata`, and return a deterministic tie-break order.

- [ ] **Step 5: Run retrieval tests and refactor only after green**

Run:

```powershell
python -m pytest tests/test_retrieval.py -q
```

Expected: PASS. Then run the full test suite and keep it green.

- [ ] **Step 6: Add Ollama embedding provider tests**

Test that a fake HTTP response is converted into a float vector and that non-200 responses raise a provider-specific error without exposing the API URL or credentials in the exception text.

- [ ] **Step 7: Implement the Ollama provider and index build CLI**

Create `scripts/build_v2_index.py` to load `data/demo_corpus`, call Ollama only when building a dense index, and write a manifest containing corpus hash, embedding model, chunk count, and build timestamp.

- [ ] **Step 8: Commit the retrieval layer**

```powershell
git add requirements.txt src/rag_v2/embeddings.py src/rag_v2/index.py src/rag_v2/retrieval.py tests/test_retrieval.py tests/fixtures
git commit -m "feat: add v2 lexical dense and hybrid retrieval"
```

---

### Task 4: Add reranking, query rewriting, grounded generation, and trace records

**Files:**
- Create: `src/rag_v2/providers.py`
- Create: `src/rag_v2/rerank.py`
- Create: `src/rag_v2/rewrite.py`
- Create: `src/rag_v2/generation.py`
- Create: `src/rag_v2/pipeline.py`
- Create: `tests/test_rerank_rewrite.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- `Reranker.rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]`
- `QueryRewriter.rewrite(question: str, history: list[dict[str, str]]) -> RewriteResult`
- `Generator.answer(question: str, contexts: list[RetrievedChunk]) -> GenerationResult`
- `RewriteResult(original_query: str, rewritten_query: str, reason: str, used_fallback: bool)`
- `GenerationResult(answer: str, cited_chunk_ids: list[str], abstained: bool, latency_ms: float)`
- `RagResponse(rewrite: RewriteResult, retrieved: list[RetrievedChunk], generation: GenerationResult, trace: list[dict[str, object]])`
- `RagPipeline.run(question: str, history: list[dict[str, str]], config: PipelineConfig) -> RagResponse`

- [ ] **Step 1: Write failing fallback tests**

Test that a reranker preserves candidate order when the model provider fails, the query rewriter returns the original question on API failure, and the generator returns an abstention response when there are no contexts.

- [ ] **Step 2: Run the tests and verify they fail for missing implementations**

```powershell
python -m pytest tests/test_rerank_rewrite.py tests/test_pipeline.py -q
```

- [ ] **Step 3: Implement provider protocols and deterministic fallbacks**

Create provider interfaces so unit tests use `FakeReranker`, `FakeRewriter`, and `FakeGenerator`. The real providers read only from `V2_CONFIG_PATH` and environment variables.

- [ ] **Step 4: Implement Cross-Encoder reranking**

Load `BAAI/bge-reranker-v2-m3` lazily through `sentence-transformers`, score query/chunk pairs, sort by score, and record reranker latency in the trace. On model load or inference failure, return the original candidates and a trace warning.

- [ ] **Step 5: Implement DeepSeek query rewriting**

Send the current question and up to the last four history turns to the chat API. Require a JSON response with `rewritten_query` and `reason`; reject malformed output and fall back to the original question. Never send API keys in request logs.

- [ ] **Step 6: Implement grounded generation and citations**

Prompt the generator to use only the supplied contexts, cite each factual paragraph with `[source:chunk_id]`, and answer with a fixed abstention phrase when evidence is insufficient. Parse the response into `GenerationResult` without removing source citations.

- [ ] **Step 7: Run the tests and commit**

```powershell
python -m pytest tests/test_rerank_rewrite.py tests/test_pipeline.py -q
python -m pytest -q
git add src/rag_v2/providers.py src/rag_v2/rerank.py src/rag_v2/rewrite.py src/rag_v2/generation.py src/rag_v2/pipeline.py tests/test_rerank_rewrite.py tests/test_pipeline.py
git commit -m "feat: add grounded v2 RAG pipeline"
```

Expected: all offline tests pass; no test calls a real provider.

---

### Task 5: Replace heuristic evaluation with benchmark metrics and actual RAGAS integration

**Files:**
- Create: `src/rag_v2/metrics.py`
- Create: `src/rag_v2/evaluator.py`
- Create: `eval/dataset.jsonl`
- Create: `tests/test_metrics.py`
- Create: `tests/test_evaluator.py`

**Interfaces:**
- `recall_at_k(results: list[RetrievedChunk], gold_doc_ids: set[str], k: int) -> float`
- `mean_reciprocal_rank(results: list[RetrievedChunk], gold_doc_ids: set[str], k: int) -> float`
- `ndcg_at_k(results: list[RetrievedChunk], gold_doc_ids: set[str], k: int) -> float`
- `RetrievalReport(rows: list[dict[str, float]], per_case: list[dict[str, object]])`
- `GenerationReport(metrics: dict[str, float], status: str, per_case: list[dict[str, object]])`
- `evaluate_retrieval(cases: list[BenchmarkCase], runner: Callable) -> RetrievalReport`
- `evaluate_generation(samples: list[dict], ragas_runner: RagasRunner) -> GenerationReport`
- `RagasRunner.evaluate(samples: list[dict]) -> GenerationReport`

- [ ] **Step 1: Write failing metric tests**

Use a fixed three-result ranking and gold IDs to assert exact Recall@1, Recall@3, MRR@3, and nDCG@3 values. Add tests for an unknown case and an empty result list.

- [ ] **Step 2: Implement pure retrieval metrics**

Keep the functions side-effect free and make the denominator explicit for abstention cases. Run:

```powershell
python -m pytest tests/test_metrics.py -q
```

Expected: PASS after implementation.

- [ ] **Step 3: Write evaluator tests with a fake pipeline**

Assert that the evaluator runs every case with the same configuration, stores per-case traces, aggregates category-specific rates, and writes no report until all cases have been processed.

- [ ] **Step 4: Implement benchmark orchestration**

Implement `eval/benchmark.py` to run the five configurations `bm25`, `dense`, `dense_rerank`, `hybrid_rerank`, and `hybrid_rerank_rewrite`. Output `reports/benchmark.json` and `reports/benchmark.md` with corpus hash, dataset hash, configuration, metrics, latency, failure count, and provider status.

- [ ] **Step 5: Implement actual RAGAS adapter**

Create an adapter that passes `user_input`, `response`, `retrieved_contexts`, and `reference` into the installed RAGAS version. Add a provider status field and an explicit `not_run` state when DeepSeek or RAGAS prerequisites are absent. Do not call the old `evaluate_faithfulness` heuristic from `eval/ragas_eval.py`.

- [ ] **Step 6: Author and validate the benchmark dataset**

Create at least 60 manually checked cases over the seven demo documents: 15 direct, 15 fuzzy, 12 cross-document, 10 multi-turn, and 8 unknown. Add a dataset validation test requiring unique `case_id`, valid `gold_doc_ids`, non-empty reference answers for non-abstention cases, and category counts matching the declared split.

- [ ] **Step 7: Run offline evaluator tests and commit**

```powershell
python -m pytest tests/test_metrics.py tests/test_evaluator.py -q
python -m pytest -q
git add src/rag_v2/metrics.py src/rag_v2/evaluator.py eval/dataset.jsonl tests/test_metrics.py tests/test_evaluator.py
git commit -m "feat: add reproducible retrieval benchmark and RAGAS adapter"
```

---

### Task 6: Add configuration, CLI commands, and the local demo

**Files:**
- Create: `.env.example`
- Create: `config/v2.yaml`
- Create: `scripts/run_benchmark.py`
- Create: `scripts/run_demo.ps1`
- Create: `demo/app.py`
- Create: `tests/test_config.py`
- Create: `tests/test_demo.py`
- Modify: `README.md`

**Interfaces:**
- `load_config(path: Path) -> V2Config`
- `python scripts/build_v2_index.py --config config/v2.yaml`
- `python scripts/run_benchmark.py --config config/v2.yaml --output reports/benchmark`
- `powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1 -Config config/v2.yaml`

- [ ] **Step 1: Write failing configuration tests**

Assert that relative corpus/artifact paths resolve from the repository root, missing `DEEPSEEK_API_KEY` is reported only when generation is requested, and secrets never appear in `repr(config)` or error messages.

- [ ] **Step 2: Implement configuration loading and environment validation**

Use a small dependency-free parser for simple `.env` `KEY=VALUE` lines. Keep API base, model names, top-k values, and feature flags in `config/v2.yaml`.

- [ ] **Step 3: Write the demo smoke test**

Test the Streamlit app's pipeline factory with fake providers and assert that a response includes `rewritten_query`, non-empty retrieved chunks, citations, and timing fields. The test must not launch a browser or call Streamlit's server.

- [ ] **Step 4: Implement the Streamlit demo**

Provide a configuration selector for baseline versus optimized retrieval, a text input, expandable panels for rewrite/recall/rerank, a cited answer panel, and a latency summary. Display a clear provider error with setup instructions instead of a traceback.

- [ ] **Step 5: Implement the PowerShell launcher and README commands**

The launcher checks Python 3.11, `.env`, Ollama availability, and the built index before running Streamlit. README must include:

```powershell
Copy-Item .env.example .env
ollama pull nomic-embed-text
python scripts/build_v2_index.py --config config/v2.yaml
python scripts/run_benchmark.py --config config/v2.yaml --output reports/benchmark
powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1 -Config config/v2.yaml
```

- [ ] **Step 6: Run configuration and demo tests**

```powershell
python -m pytest tests/test_config.py tests/test_demo.py -q
python -m pytest -q
```

- [ ] **Step 7: Commit the local demo and configuration**

```powershell
git add .env.example config/v2.yaml scripts/run_benchmark.py scripts/run_demo.ps1 demo/app.py tests/test_config.py tests/test_demo.py README.md
git commit -m "feat: add local v2 benchmark and Streamlit demo"
```

---

### Task 7: Make the legacy modules honest and remove hard-coded runtime paths

**Files:**
- Modify: `enhancements/knowledge_graph.py`
- Modify: `enhancements/hard_case_mining.py`
- Modify: `enhancements/federated_search.py`
- Modify: `eval/ragas_eval.py`
- Create: `tests/test_legacy_cli_config.py`
- Modify: `README.md`

**Interfaces:**
- Existing module APIs remain import-compatible.
- CLI defaults become repository-relative and accept explicit `--api-base`, `--data-dir`, `--report`, and `--case-dir` options.

- [ ] **Step 1: Write failing path/configuration tests**

Assert that each CLI parser's defaults contain no `D:\` path, that a caller can pass a temporary data directory, and that `federated_search.search_all` does not issue a second search for every knowledge base merely to build `source_breakdown`.

- [ ] **Step 2: Implement repository-relative defaults and the source breakdown fix**

Use `Path(__file__).resolve().parents[1]` as the repository root, pass API base through constructors, and reuse already collected result counts. Do not alter the legacy module's public class names.

- [ ] **Step 3: Correct README and legacy evaluation wording**

Describe `eval/ragas_eval.py` as a legacy keyword/heuristic report unless it is replaced by the v2 evaluator. Remove claims of actual RAGAS faithfulness, active learning, or parallel reranking that are not backed by the v2 code and report.

- [ ] **Step 4: Run legacy tests and commit**

```powershell
python -m pytest tests/test_legacy_cli_config.py -q
python -m pytest -q
git add enhancements eval/ragas_eval.py README.md tests/test_legacy_cli_config.py
git commit -m "fix: align legacy module claims and runtime configuration"
```

---

### Task 8: Run real experiments, verify the report, and prepare the portfolio handoff

**Files:**
- Create: `reports/benchmark.json`
- Create: `reports/benchmark.md`
- Modify: `README.md`
- Modify: `docs/CHANGELOG.md`

- [ ] **Step 1: Build the local indexes**

With Ollama running and `nomic-embed-text` available, run:

```powershell
python scripts/build_v2_index.py --config config/v2.yaml
```

Verify that the artifact manifest contains the seven document sources, a non-zero chunk count, an embedding model name, and a corpus hash.

- [ ] **Step 2: Run the retrieval benchmark without generation**

```powershell
python scripts/run_benchmark.py --config config/v2.yaml --output reports/benchmark --retrieval-only
```

Verify that all five experiment rows are present and that Recall@k, MRR, nDCG, latency, and failure count are numeric.

- [ ] **Step 3: Run the full generation and RAGAS benchmark**

With `DEEPSEEK_API_KEY` configured, run:

```powershell
python scripts/run_benchmark.py --config config/v2.yaml --output reports/benchmark
```

Verify that faithfulness and answer relevancy are calculated from retrieved contexts, unknown questions have an explicit abstention score, and the report records provider/model configuration without secrets.

- [ ] **Step 4: Perform a manual demo smoke test**

Start the demo, ask one direct question, one fuzzy question, one follow-up question, and one unknown question. Check that every answer displays its source chunk IDs and that the UI exposes baseline versus optimized traces.

- [ ] **Step 5: Update the README with measured results**

Replace all unsupported metric claims with the generated report values. Include the experiment table, benchmark split counts, setup commands, limitations, and the honest project positioning as a Chatchat-based RAG optimization system with an independently implemented v2 benchmark/demo layer.

- [ ] **Step 6: Run final verification**

```powershell
python -m pytest -q
python -m compileall src demo scripts enhancements eval
git diff --check
git status --short
```

Expected: all tests pass, compilation succeeds, diff check is clean, and only intentional files are modified.

- [ ] **Step 7: Commit and push the feature branch**

```powershell
git add reports README.md docs/CHANGELOG.md
git commit -m "feat: deliver reproducible RAG v2 benchmark and demo"
git push -u origin codex/rag-v2-benchmark-demo
```

Do not force-push or modify the existing remote history. Report the pushed branch and the exact commands needed to run the demo from a clean clone.
