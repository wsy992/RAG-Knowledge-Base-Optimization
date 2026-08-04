# RAG 检索优化与评测系统 v2 设计

> 状态：待用户审阅

## 目标

在现有 LangChain-Chatchat 二次开发项目的基础上，新增一套可独立控制、可重复运行的 RAG benchmark 和本地演示层。v2 不把项目包装成从零实现的 RAG 框架，而是明确区分：

1. Chatchat 原始流程作为 baseline；
2. 仓库中独立实现的检索、重排、查询改写和评测组件作为优化方案；
3. 固定的多文档语料和人工标注问题集作为实验依据。

最终产出应能同时支持实习面试演示和授课型硕士申请材料：面试者可以看到每一步检索结果，README 可以展示真实的对照实验表格，代码和数据可以在本地复现。

## 非目标

- 不重写 LangChain-Chatchat 的全部源码，也不训练新的语言模型。
- 不把知识图谱、难例挖掘或联邦检索仅凭模块名称包装成已验证的研究成果。
- 不在第一版 v2 中加入 OCR、VLM、Agent 等与核心评测无关的功能。
- 不把 DeepSeek API key、Ollama 私有配置或本地绝对路径提交到 GitHub。

## 运行约束

- 运行方式采用用户确认的 A 方案：本地运行，允许使用 DeepSeek API 和 Ollama。
- Ollama 默认提供 `nomic-embed-text` embedding 服务；生成和 query rewrite 默认使用 DeepSeek 兼容 API。
- Reranker 默认使用 `BAAI/bge-reranker-v2-m3`，模型首次运行时下载并缓存。
- 所有外部服务都通过环境变量或 YAML 配置注入；没有外部服务时，单元测试和检索指标仍可运行。

## 数据设计

### 演示语料

仓库新增 `data/demo_corpus/`，包含 5–10 份具有明确来源的课程、培养方案、专业特色、技术原理或就业方向文档。每份文档通过稳定的 `doc_id` 标识，并在 `data/demo_corpus/manifest.json` 中记录标题、来源、来源日期和许可说明。

文档解析阶段统一生成以下结构：

```json
{
  "doc_id": "course-overview",
  "title": "课程介绍",
  "source": "demo_corpus/course_overview.md",
  "section": "核心课程",
  "chunk_id": "course-overview::chunk-0001",
  "text": "..."
}
```

首个版本支持 Markdown、TXT 和可选 PDF 文本提取；图片 OCR 不属于本次验收范围。切分后的 chunk 必须保留 `doc_id`、`section` 和 `chunk_id`，以便计算召回指标并在演示页面展示引用。

### Benchmark 问题集

新增 `eval/dataset.jsonl`，最低 60 道人工检查的问题，覆盖：

- 15 道直接问题：问题与文档表达接近；
- 15 道模糊或口语化问题：测试 query rewrite；
- 12 道跨文档问题：答案需要组合两个及以上文档片段；
- 10 道多轮指代问题：测试历史上下文补全；
- 8 道未知问题：知识库没有证据，系统应拒答。

每道题包含 `case_id`、`question`、`category`、`gold_doc_ids`、`reference_answer`、`answer_points` 和 `should_abstain`。gold 文档 ID 和答案要点由人工确认，不能由被评测的生成模型自动决定。

## v2 架构

新增独立的 `src/rag_v2/` 包，按职责拆分为：

- `config.py`：读取 YAML 和环境变量，校验必需配置；
- `corpus.py`：解析文档、切分 chunk、保存 corpus manifest；
- `index.py`：构建和加载 BM25、dense、hybrid 所需的索引；
- `retrieval.py`：提供统一的 `retrieve(query, mode, top_k)` 接口；
- `rerank.py`：调用 Cross-Encoder 对候选 chunk 重排，并记录耗时；
- `rewrite.py`：调用 DeepSeek 改写检索问题，失败时回退原问题；
- `generation.py`：基于上下文生成带 `[source:chunk_id]` 引用的答案，并在证据不足时拒答；
- `pipeline.py`：串联 query rewrite、召回、重排、生成和 trace 记录；
- `schemas.py`：定义文档、检索结果、评测样本和 trace 的数据结构。

现有 `enhancements/` 目录保留为 v1 优化实验：知识图谱模块可以作为可选召回增强，难例挖掘作为错误分析工具，联邦检索只有在真正完成并发和融合后才进入 v2 实验矩阵。Chatchat 仍作为历史 baseline 和兼容入口，不再作为 v2 所有组件的隐式依赖。

## 实验矩阵

所有实验使用同一语料、同一问题集、相同的 top-k 范围和相同的生成模型，固定配置并输出 JSON 报告。至少包含以下五行：

1. `bm25`：词法检索 baseline；
2. `dense`：Ollama embedding + dense 检索；
3. `dense_rerank`：dense 召回后使用 Cross-Encoder 重排；
4. `hybrid_rerank`：BM25 与 dense 结果融合后重排；
5. `hybrid_rerank_rewrite`：在上一行基础上加入 query rewrite。

知识图谱增强和多轮记忆作为独立开关运行，不与主实验结果混在一起。每行记录配置名称、问题数量、检索指标、生成指标、平均延迟、P95 延迟、失败数和运行时间。

## 评测指标

### 检索指标

- `Recall@1`、`Recall@3`、`Recall@5`：根据 `gold_doc_ids` 判断是否召回正确文档；
- `MRR@5`：第一个 gold 文档出现位置的倒数平均值；
- `nDCG@5`：根据 gold 文档相关性等级计算排序质量。

### 生成指标

使用真实 RAGAS 接口计算 `faithfulness` 和 `answer_relevancy`。faithfulness 必须同时传入问题、生成答案和实际检索 contexts；不能使用答案长度、关键词或“回答了就算忠实”的替代规则。对未知问题额外计算拒答准确率。

若外部 LLM 服务不可用，benchmark 仍输出检索指标，并明确把生成指标标记为未运行，不能伪造 0.94 等结果。

## 本地演示

新增 `demo/app.py`，使用 Streamlit 提供单页演示。每次提问展示：

1. 用户原问题；
2. query rewrite 结果及是否发生回退；
3. BM25、dense 或 hybrid 的候选结果；
4. Rerank 分数和最终上下文；
5. 带来源引用的答案；
6. 当前请求的检索耗时、生成耗时和总耗时。

演示页面支持切换 baseline 和 optimized 配置，让面试者可以用同一个问题直观看到召回结果和答案来源的变化。

## 可复现入口

新增以下入口和配置：

- `.env.example`：API key、API base、Ollama 地址和模型名示例；
- `config/v2.yaml`：语料目录、索引目录、top-k、reranker 和实验配置；
- `scripts/build_v2_index.py`：构建语料和索引；
- `scripts/run_benchmark.py`：运行指定实验矩阵并生成 JSON/Markdown 报告；
- `scripts/run_demo.ps1`：检查配置、启动 Streamlit 演示；
- `reports/`：保存本地生成的结果摘要，但不提交包含 API 返回内容的私密运行日志。

README 必须给出 Windows PowerShell 和通用 Python 两种运行方式，并说明首次运行需要启动 Ollama、拉取 embedding 模型和配置 DeepSeek API。

## 测试策略

新增 `tests/`，不依赖网络和真实模型即可运行：

- corpus 解析保留 doc ID、section 和 chunk ID；
- BM25/dense 结果可以正确合并、去重和排序；
- MRR、Recall、nDCG 在固定小样例上的数值正确；
- unknown 问题的拒答判断不会把正常答案误判为拒答；
- query rewrite API 失败时回退原问题；
- trace 能完整记录每个阶段的输入、输出和耗时；
- 配置缺少 API key 时给出明确错误，不打印敏感值。

真实 Ollama、DeepSeek 和 RAGAS 调用作为可选 smoke test，不在普通单元测试中强制执行。

## 验收标准

v2 完成时必须满足：

1. `pytest` 在无网络环境下通过；
2. benchmark 可以在无生成模型时完成检索指标评测；
3. 使用 Ollama 和 DeepSeek 后能完成完整生成评测；
4. 报告同时包含至少五种检索方案，且每种方案使用同一问题集；
5. RAGAS 指标来自实际 contexts，不使用当前脚本中的启发式 faithfulness；
6. demo 能显示检索证据、引用来源和耗时；
7. README 的命令在干净环境下可复现，代码中不存在 `D:\chatchat-project` 等绝对路径；
8. 简历表述与代码、数据和报告完全一致。

## 交付顺序

先完成数据契约、离线 corpus/index、检索指标和 baseline；再加入 rerank、rewrite、生成评测和 demo；最后处理知识图谱、难例分析的 v2 接入、README 和简历可用结果。每个阶段都先写失败测试，再实现最小功能并运行完整测试。
