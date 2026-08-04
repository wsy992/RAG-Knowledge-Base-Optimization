"""
知识图谱模块：从文档中自动抽取结构化知识，增强 RAG 检索能力

核心功能：
1. 实体抽取：从文档中识别课程、岗位、技能、特色等实体
2. 关系抽取：提取实体之间的关系（课程→属于→专业，岗位→要求→技能）
3. 图谱检索：将用户问题映射到实体，检索关联信息

使用方法：
  python -m enhancements.knowledge_graph --kb wsy --rebuild

简历写法：
  基于 LLM 实现非结构化文档到结构化知识图谱的自动抽取，
  将 RAG 从"关键词匹配"升级为"语义关系检索"
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("knowledge_graph")

# 实体类型
ENTITY_TYPES = {
    "专业": "学科专业",
    "课程": "具体课程名称",
    "岗位": "职业/职位",
    "技能": "技术/能力",
    "特色": "专业特点/优势",
    "行业": "行业领域",
    "工具": "技术工具/框架",
}

# 默认实体-关系抽取提示词
EXTRACTION_PROMPT = """你是一个知识图谱构建专家。请从以下文档中抽取关键实体和关系。

实体类型定义：
{entity_types}

对于每个段落，提取：
1. 实体（名称 + 类型）
2. 关系（实体A → 关系 → 实体B）

以 JSON 格式输出，不要解释：
{{
  "entities": [
    {{"name": "实体名称", "type": "实体类型", "description": "简要描述"}}
  ],
  "relations": [
    {{"source": "实体A名称", "relation": "关系类型", "target": "实体B名称"}}
  ]
}}

文档内容：
{content}
"""


class KnowledgeGraph:
    """知识图谱：基于文档抽取的结构化知识"""

    def __init__(self, kb_name: str = "default"):
        self.kb_name = kb_name
        self.entities: List[Dict] = []
        self.relations: List[Dict] = []
        self.index: Dict[str, List[int]] = {}  # 关键词 → 实体索引

    def save(self, path: Path = None):
        """保存图谱到文件"""
        if path is None:
            path = Path(f"data/kg_{self.kb_name}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "kb_name": self.kb_name,
            "entities": self.entities,
            "relations": self.relations,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Knowledge graph saved: {path} ({len(self.entities)} entities, {len(self.relations)} relations)")

    def load(self, path: Path = None):
        """从文件加载图谱"""
        if path is None:
            path = Path(f"data/kg_{self.kb_name}.json")
        if not path.exists():
            logger.warning(f"Knowledge graph not found: {path}")
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entities = data["entities"]
        self.relations = data["relations"]
        self._build_index()
        logger.info(f"Knowledge graph loaded: {path} ({len(self.entities)} entities)")
        return True

    def _build_index(self):
        """构建关键词→实体的倒排索引"""
        self.index = {}
        for i, ent in enumerate(self.entities):
            name = ent.get("name", "")
            # 按字和词建立索引
            for word in self._segment(name):
                if word not in self.index:
                    self.index[word] = []
                self.index[word].append(i)

    def _segment(self, text: str) -> List[str]:
        """简单的分词，用于索引"""
        results = [text]  # 全名
        # 2-4 字片段
        for i in range(len(text) - 1):
            results.append(text[i:i+2])
        if len(text) >= 3:
            results.append(text[:3])
        return list(set(results))

    def add_entity(self, name: str, etype: str, description: str = ""):
        """添加实体"""
        # 去重
        for ent in self.entities:
            if ent["name"] == name and ent["type"] == etype:
                return
        self.entities.append({
            "name": name,
            "type": etype,
            "description": description,
        })
        self._build_index()

    def add_relation(self, source: str, relation: str, target: str):
        """添加关系"""
        self.relations.append({
            "source": source,
            "relation": relation,
            "target": target,
        })

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        基于关键词匹配搜索知识图谱，返回相关实体和关系。
        """
        if not self.index:
            return []

        # 分词匹配
        query_words = set()
        for i in range(len(query)):
            for j in range(i+1, min(i+4, len(query)+1)):
                query_words.add(query[i:j])

        # 统计每个实体的匹配分数
        scores = {}
        for word in query_words:
            for idx in self.index.get(word, []):
                scores[idx] = scores.get(idx, 0) + len(word)

        # 取 Top-K
        top_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            entity = self.entities[idx]
            # 找到与该实体相关的关系
            related_relations = [
                r for r in self.relations
                if r["source"] == entity["name"] or r["target"] == entity["name"]
            ]
            results.append({
                "entity": entity,
                "relations": related_relations,
                "score": scores[idx],
            })

        return sorted(results, key=lambda r: r["score"], reverse=True)

    def format_context(self, results: List[Dict]) -> str:
        """将知识图谱检索结果格式化为文字，可用于 RAG 的上下文增强"""
        if not results:
            return ""
        lines = ["【知识图谱检索结果】"]
        for r in results:
            ent = r["entity"]
            lines.append(f"- {ent['type']}: {ent['name']}（{ent['description']}）")
            for rel in r["relations"]:
                lines.append(f"  └ {rel['source']} → {rel['relation']} → {rel['target']}")
        return "\n".join(lines)


def extract_from_documents(docs_text: str, api_base_url: str = "https://api.deepseek.com/v1",
                           api_key: str = None) -> KnowledgeGraph:
    """
    使用 LLM 从文档中抽取知识图谱。
    """
    import requests

    kg = KnowledgeGraph()

    # 构建实体类型描述
    type_desc = "\n".join([f"- {k}: {v}" for k, v in ENTITY_TYPES.items()])

    # 分段处理
    chunks = [c.strip() for c in re.split(r'\n{2,}', docs_text) if len(c.strip()) > 50]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for chunk in chunks:
        try:
            resp = requests.post(
                f"{api_base_url}/chat/completions",
                headers=headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "你是一个知识图谱构建专家，只输出 JSON。"},
                        {"role": "user", "content": EXTRACTION_PROMPT.format(
                            entity_types=type_desc, content=chunk
                        )},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                # 提取 JSON
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    for ent in data.get("entities", []):
                        kg.add_entity(ent["name"], ent["type"], ent.get("description", ""))
                    for rel in data.get("relations", []):
                        kg.add_relation(rel["source"], rel["relation"], rel["target"])
        except Exception as e:
            logger.warning(f"Extraction failed for chunk: {e}")

    kg._build_index()
    return kg


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="知识图谱构建工具")
    parser.add_argument("--kb", default="wsy", help="知识库名称")
    parser.add_argument("--rebuild", action="store_true", help="强制重建图谱")
    parser.add_argument("--query", help="搜索测试")
    parser.add_argument("--api-base", default=os.getenv("CHATCHAT_API_BASE", "http://127.0.0.1:7861"))
    parser.add_argument("--output-dir", default=os.getenv("RAG_LEGACY_DATA_DIR", "data"))
    args = parser.parse_args()

    kg = KnowledgeGraph(args.kb)
    kg_path = Path(args.output_dir) / f"kg_{args.kb}.json"

    if args.rebuild or not kg_path.exists():
        print(f"Rebuilding knowledge graph for KB: {args.kb}")
        # 从知识库获取文档内容
        import requests
        try:
            resp = requests.post(
                f"{args.api_base.rstrip('/')}/knowledge_base/search_docs",
                json={"query": "", "knowledge_base_name": args.kb, "top_k": 50, "score_threshold": 2.0},
                timeout=30,
            )
            if resp.status_code == 200:
                docs = resp.json()
                all_text = "\n\n".join([
                    d.get("page_content", "") if isinstance(d, dict) else ""
                    for d in docs
                ])
                if all_text:
                    api_key = os.getenv("DEEPSEEK_API_KEY")
                    if not api_key:
                        raise RuntimeError("DEEPSEEK_API_KEY is required for legacy graph extraction")
                    kg = extract_from_documents(all_text, api_key=api_key)
                    kg.save(kg_path)
                else:
                    print("No documents found in knowledge base")
            else:
                print(f"API error: {resp.status_code}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        kg.load(kg_path)

    # 查询测试
    if args.query:
        results = kg.search(args.query)
        print("\nKnowledge Graph Results:")
        print(kg.format_context(results))
    else:
        print(f"\nKnowledge Graph: {len(kg.entities)} entities, {len(kg.relations)} relations")
        for ent in kg.entities[:10]:
            print(f"  [{ent['type']}] {ent['name']}")
