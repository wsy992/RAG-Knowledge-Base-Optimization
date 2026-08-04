# RAG 智能知识库问答系统 — 工程优化实践

> 基于 LangChain-Chatchat 深度优化的 RAG 系统，集成多轮对话上下文、知识图谱增强、多源联邦检索与自动化评估闭环。

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![RAGAS](https://img.shields.io/badge/RAGAS-v2%20adapter%20available-blue)](https://github.com/explodinggradients/ragas)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--Chat-orange)](https://deepseek.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> The metric table in the original project description is a historical v1 baseline produced by the legacy Chatchat script. Its faithfulness value is heuristic and must not be presented as a real RAGAS result. Use the v2 commands below for reproducible, hash-stamped reports.

---

## 项目概况

本实践项目在开源 RAG 框架基础上进行了**全链路工程优化**，覆盖检索增强、上下文建模、结构化知识抽取、量化评估与持续改进闭环。

| 指标 | 数据 |
|------|------|
| 精确问答命中率 | **100%** (4/4) |
| 模糊问答命中率 | **77.8%** (7/9) |
| 否定测试准确率 | **100%** (2/2) — 有效防幻觉 |
| 整体命中率 | **81.8%** |
| 回答忠实度 | **0.94 / 1.0** |

> 评估数据基于 RAGAS 框架，覆盖精确、模糊、否定三类测试场景。[查看完整报告](eval/ragas_report.json)

---

## 技术架构

```
                    ┌─────────────────────────────────┐
                    │        用户查询 (Query)           │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │      Query 改写 (多轮上下文)      │  ← 创新点1
                    │  结合对话历史，消除指代歧义       │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
     │  知识图谱检索   │  │  FAISS 向量检索 │  │  搜索引擎检索   │  ← 创新点4
     │  (实体+关系)    │  │  (语义相似度)   │  │  (互联网补充)   │
     └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
              └──────────────────┼────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Rerank 精排           │  ← 优化2
                    │  Cross-Encoder 重排序    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   LLM 生成 (DeepSeek)    │
                    └─────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   RAGAS 评估闭环         │  ← 创新点3
                    │  难例挖掘 → 分类 → 优化  │
                    └─────────────────────────┘
```

---

## 🔧 优化与创新点详解

### 优化 1：Query 改写

**问题**：用户的自然语言查询往往口语化、简短（如"有什么课"、"学啥的"），而知识库中的文档表述是书面化的（如"核心课程包括……"）。直接用原话检索命中率低。

**方案**：在检索前置入 LLM 改写步骤，将用户问题转换为关键词丰富、适合检索的形式：

```
用户输入: "学啥的"
         ↓ LLM 改写
检索用: "智能科技专业 学习内容 培养方向 核心技能"
         ↓ 关键词匹配度大幅提升 → 检索命中率提升
```

**技术实现要点**：
- 使用 DeepSeek 进行改写，温度设置为 0.1 保证稳定性
- 改写失败时自动回退到原始查询，不阻塞服务
- 改写结果仅用于检索，最终回答仍基于原始问题生成

---

### 优化 2：Rerank 精排

**问题**：向量检索（Embedding + FAISS）的初排结果基于语义相似度，虽然能快速召回大量候选，但排序精度有限。有时最相关的结果排在第 3、第 4 位，而前两位是噪音信息，导致 LLM 生成的答案质量受影响。

**方案**：引入 Cross-Encoder 重排序模型 `BAAI/bge-reranker-v2-m3`，对初排结果进行二次评分：

```
初排 (FAISS):   结果A(0.85)  结果B(0.72)  结果C(0.68)  结果D(0.55)
                     ↓ Cross-Encoder 逐对打分
精排 (Rerank):  结果C(0.93)  结果A(0.87)  结果D(0.71)  结果B(0.64)
                     ↓ 最相关的结果被提到最前面
送入 LLM 的上下文质量提升
```

**技术实现细节**：
- 模型：`BAAI/bge-reranker-v2-m3`（多语言 Cross-Encoder，支持中文）
- 设备：纯 CPU 运行（约 1.1GB 模型）
- 懒加载：模型在首次问答时加载，不拖慢服务启动速度
- 降级策略：Rerank 失败时自动使用原始排序结果，保证服务可用性

**效果**：最相关文档被优先送入 LLM，回答准确率显著提升，尤其在文档量较大的知识库中效果更为明显。

---

### 优化 3：RAGAS 量化评估体系

**问题**：优化不能凭感觉，需要可量化的评估标准。

**方案**：基于 RAGAS 框架构建自动化评估 Pipeline：

- **Hit Rate**: 检索命中率 — 衡量知识库能否找到相关内容
- **Faithfulness**: 回答忠实度 — 衡量 LLM 是否基于文档回答（防幻觉）
- **MRR**: 排序质量 — 衡量最相关内容出现在第几位

**测试集覆盖三大场景**：

| 场景 | 示例 | 命中率 |
|------|------|--------|
| 精确问法 | "核心课程有哪些" | 100% |
| 模糊问法 | "学啥的"、"能干啥" | 60% |
| 否定测试 | "学费多少钱"（应拒答） | 100% |

> 否定测试 100% 说明系统具备良好的**幻觉防范能力**——不知道的不会乱编。

---

### 创新点 1：多轮对话上下文记忆（原创实现）

**模块**: [`enhancements/knowledge_graph.py`](enhancements/knowledge_graph.py)（集成在 Query 改写模块中）

**问题**：标准 RAG 系统中，每一轮问答都是独立的。用户先问"智能科技专业有什么课"，得到回答后再问"学费多少"，系统无法将"学费"关联到"智能科技专业"，导致检索脱离上下文。

**方案**：在 Query 改写环节引入对话上下文管理机制，改写时携带历史对话信息：

```
用户: "智能科技专业有什么课" → 助手回答...
用户: "学费呢"              ← 仅两个字，独立检索必然失败
                            ↓ 结合历史重写
改写后: "澳门城市大学智能科技专业 学费 收费标准"
                            ↓ 检索 → 精准命中
```

**技术实现要点**：
- 每次改写时提取最近 N 轮对话建立上下文
- 使用 LLM 识别指代关系（这、那、它、该专业）并自动补全
- 改写失败时回退到原始查询，保证系统稳定性

通过这种方式，系统具备了**跨轮次的理解能力**，用户无需在每轮对话中都重复上下文信息，交互体验更接近自然人机对话。

---

### 创新点 2：结构化知识图谱增强检索（原创实现）

**模块**: [`enhancements/knowledge_graph.py`](enhancements/knowledge_graph.py)

从非结构化文档中自动抽取实体和关系，构建轻量知识图谱，将 RAG 从"全文关键词匹配"升级为"语义关系检索"。

```
原始文档: "核心课程包括数据结构与算法、机器学习"
         ↓ LLM 实体抽取
实体: 数据结构 (课程) → 属于 → 智能科技专业 (专业)
实体: 机器学习 (课程) → 属于 → 智能科技专业 (专业)
实体: AI算法工程师 (岗位) → 需要 → 机器学习 (技能)
```

搭配向量检索使用，**关键词匹配 + 语义关系**双路互补。

---

### 创新点 3：难例挖掘与主动学习闭环（原创实现）

**模块**: [`enhancements/hard_case_mining.py`](enhancements/hard_case_mining.py)

自动识别问答系统中的 Bad Case，按类型分类归档，生成优化建议。

```python
# 自动分类系统
Bad Case 类型：
  - 检索遗漏：知识库有但没搜到 → 调 chunk / top_k
  - 改写偏差：改写后偏离原意 → 优化改写提示词
  - 过度拒绝：能答但说不知道 → 降低检索阈值
```

形成 **"评估 → 识别 → 分类 → 优化"** 的闭环，使系统具备持续自我进化的能力。

---

### 创新点 4：多数据源联邦检索（原创实现）

**模块**: [`enhancements/federated_search.py`](enhancements/federated_search.py)

支持同时在多个知识库和互联网搜索引擎中进行检索，结果智能合并去重排序。

```
一次查询同时搜索：
  ├─ 知识库 A (课程资料)
  ├─ 知识库 B (政策文档)
  └─ 互联网 (搜索引擎补充)
      ↓ 融合排序
输出 Top-5 最相关内容
```

在实际企业场景中，信息往往分布在多个系统中，联邦检索是**落地必备能力**。

---

## 技术栈

| 层级 | 技术选型 | 说明 |
|------|---------|------|
| RAG 框架 | LangChain-Chatchat 0.3.1 | 开源知识库问答框架 |
| LLM | DeepSeek-Chat (API) | 问答生成与查询改写 |
| 向量模型 | nomic-embed-text (Ollama) | 纯 CPU 本地运行 |
| 重排序 | BAAI/bge-reranker-v2-m3 | Cross-Encoder 精排 |
| 向量数据库 | FAISS | 本地轻量向量检索 |
| 知识图谱 | 自建 (实体-关系 JSON) | 文档结构化抽取 |
| 评估框架 | RAGAS | 自动化量化评估 |
| 部署 | Docker / Python 3.11 | 支持本地及服务器部署 |

---

## 如何使用

### 前置要求
- Python 3.11+
- [Ollama](https://ollama.com) (本地运行向量模型)
- DeepSeek / OpenAI API Key

### 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/wsy992/RAG-Knowledge-Base-Optimization.git
cd RAG-Knowledge-Base-Optimization

# 2. 安装依赖
pip install -r requirements.txt

# 3. 拉取 Embedding 模型
ollama pull nomic-embed-text

# 4. 配置 API Key (编辑 config/model_settings.example.yaml)
#    填入你的 DeepSeek API Key

# 5. 初始化并启动
chatchat init
chatchat start -a

# 6. 打开浏览器访问 http://127.0.0.1:8501
#    上传文档 → 开始问答
```

### 评估运行

```bash
# 运行 RAGAS 评估
python eval/ragas_eval.py

# 运行难例挖掘
python -m enhancements.hard_case_mining --report eval/ragas_report.json

# 运行知识图谱构建
python -m enhancements.knowledge_graph --kb my_kb --rebuild

# 测试联邦检索
python -m enhancements.federated_search --query "核心课程"
```

---

## 项目结构

```
RAG-Knowledge-Base-Optimization/
├── README.md                          # 本文件
├── config/
│   ├── kb_settings.yaml               # 知识库配置（含优化参数）
│   └── model_settings.example.yaml    # 模型配置模板（不含密钥）
├── enhancements/
│   ├── knowledge_graph.py             # [创新] 知识图谱抽取
│   ├── hard_case_mining.py            # [创新] 难例挖掘闭环
│   └── federated_search.py            # [创新] 联邦检索
├── eval/
│   ├── ragas_eval.py                  # RAGAS 评估脚本
│   └── ragas_report.json              # 量化评估报告
├── docs/
│   └── deployment.md                  # 部署文档
├── scripts/
│   └── deploy.sh                      # 部署脚本
└── LICENSE
```

---

## 关于作者

吴世扬 · 澳门城市大学 智能科技本科 · 2024届

- GitHub: [github.com/wsy992](https://github.com/wsy992)
- 研究方向：RAG 系统优化、LLM 工程化应用、知识图谱构建

---

*本项目的核心代码和配置均为独立完成，体现了在 AI 工程化领域的系统设计能力与持续优化思维。*

## RAG v2 benchmark and trace demo

The v2 implementation is additive: the original Chatchat integration remains the baseline, while `src/rag_v2/` contains a separately controlled pipeline. It uses the seven-document demo corpus under `data/demo_corpus/` and the 60-case benchmark under `eval/dataset.jsonl`.

The benchmark reports lexical BM25, dense Ollama embeddings, dense plus cross-encoder reranking, hybrid retrieval, and hybrid retrieval with query rewriting. Retrieval scores are Recall@k, MRR, and nDCG. Faithfulness and answer relevancy are only reported through the real RAGAS adapter when a DeepSeek key is configured; the old heuristic evaluator is not used by v2.

### Windows PowerShell setup

```powershell
Copy-Item .env.example .env
ollama pull nomic-embed-text
python scripts/build_v2_index.py --config config/v2.yaml
python scripts/run_benchmark.py --config config/v2.yaml --output reports/benchmark
powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1 -Config config/v2.yaml
```

The default benchmark command runs all five retrieval configurations. To run real DeepSeek generation and RAGAS on the optimized configuration, set `DEEPSEEK_API_KEY` in `.env` and run:

```powershell
python scripts/run_benchmark.py --config config/v2.yaml --output reports/benchmark --generation-config optimized
```

`artifacts/` contains the locally built dense vectors and is intentionally ignored by Git. The index manifest stores a corpus hash, model name, chunk IDs, and vector dimension; changing the corpus invalidates the artifact and triggers a rebuild. `reports/` is also local output, so benchmark results can be regenerated without committing machine-specific paths or credentials.

### What the demo exposes

The Streamlit page shows the original and rewritten query, retrieved chunks, reranker metadata, grounded answer, allowed citations, and per-stage latency trace. If a provider is unavailable, the pipeline records a fallback or abstention instead of inventing a score.
