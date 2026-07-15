# 优化日志

## v1.0 — 基准版本
- 部署 LangChain-Chatchat 0.3.1
- 接入 DeepSeek API + Ollama (nomic-embed-text)
- 默认配置：chunk 750 | overlap 150 | top_k 3 | score_threshold 2.0

## v1.1 — 基础调优
| 参数 | 改前 | 改后 | 理由 |
|------|------|------|------|
| CHUNK_SIZE | 750 | 500 | 更细粒度切分，提升精准度 |
| OVERLAP_SIZE | 150 | 100 | 减少冗余 |
| VECTOR_SEARCH_TOP_K | 3 | 5 | 扩大召回，给 LLM 更多上下文 |
| SCORE_THRESHOLD | 2.0 | 0.5 | 过滤低质量结果 |

## v1.2 — Query 改写 + Rerank
- 新增 Query 改写模块：口语化问题改写为检索式表达
- 启用 bge-reranker-v2-m3 Cross-Encoder 重排序
- 加载优化：改为懒加载，避免启动时加载 1.1GB 模型

## v1.3 — RAGAS 评估体系
- 构建 11 条测试用例，覆盖精确/模糊/否定三类场景
- 首次评估结果：命中率 81.8% | 忠实度 0.94

## v1.4 — 四大创新
1. **多轮对话上下文记忆**：改写时携带历史，消除指代歧义
2. **结构化知识图谱**：LLM 自动抽取实体关系，增强检索语义
3. **难例挖掘闭环**：自动识别 Bad Case 并分类归档
4. **联邦检索**：多知识库 + 互联网并行搜索
