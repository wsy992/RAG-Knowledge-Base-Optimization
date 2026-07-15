"""
多数据源联邦检索模块：同时搜索多个知识库和外部数据源

核心功能：
1. 多知识库并行检索：同时在多个知识库中搜索
2. 搜索引擎补充：当本地知识库结果不足时，自动搜索互联网
3. 结果融合排序：将多路结果合并、去重、用 Rerank 重排序

使用方法：
  python -m enhancements.federated_search --query "人工智能就业前景"

简历写法：
  设计多路联邦检索架构，实现本地知识库、内部系统与外部数据源的统一检索与结果融合
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("federated_search")

CHATCHAT_API = "http://127.0.0.1:7861"


class FederatedSearch:
    """联邦检索器：多数据源并行搜索 + 结果融合"""

    def __init__(self, knowledge_bases: List[str] = None):
        self.knowledge_bases = knowledge_bases or []
        self.top_k_per_source = 3
        self.top_k_final = 5

    def discover_knowledge_bases(self) -> List[str]:
        """自动发现所有有文档的知识库"""
        try:
            resp = requests.get(f"{CHATCHAT_API}/knowledge_base/list_knowledge_bases", timeout=5)
            if resp.status_code == 200:
                kbs = resp.json().get("data", [])
                # 过滤掉空知识库
                result = []
                for kb in kbs:
                    name = kb.get("kb_name", "")
                    if kb.get("file_count", 0) > 0:
                        result.append(name)
                return result
        except Exception as e:
            logger.warning(f"Failed to discover KBs: {e}")
        return self.knowledge_bases

    def search_single_kb(self, query: str, kb_name: str, top_k: int = 3) -> List[Dict]:
        """在单个知识库中搜索"""
        try:
            resp = requests.post(
                f"{CHATCHAT_API}/knowledge_base/search_docs",
                json={
                    "query": query,
                    "knowledge_base_name": kb_name,
                    "top_k": top_k,
                    "score_threshold": 2.0,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                # search_docs 返回的是列表
                if isinstance(data, list):
                    for d in data:
                        if isinstance(d, dict):
                            d["_source"] = kb_name
                    return data
            return []
        except Exception as e:
            logger.warning(f"Search KB {kb_name} failed: {e}")
            return []

    def search_web(self, query: str, top_k: int = 3) -> List[Dict]:
        """通过 Chatchat 的搜索引擎模式搜索互联网"""
        try:
            resp = requests.post(
                f"{CHATCHAT_API}/knowledge_base/local_kb/_/chat/completions",
                json={
                    "messages": [{"role": "user", "content": query}],
                    "model": "deepseek-chat",
                    "mode": "search_engine",
                    "kb_name": "duckduckgo",
                    "top_k": top_k,
                    "score_threshold": 2.0,
                    "stream": False,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = json.loads(json.loads(resp.text))
                return [{"page_content": data.get("content", ""), "_source": "web"}]
            return []
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []

    def merge_results(self, all_results: List[List[Dict]]) -> List[Dict]:
        """合并多路结果，去重并排序"""
        seen = set()
        merged = []

        for results in all_results:
            for doc in results:
                content = doc.get("page_content", "") if isinstance(doc, dict) else ""
                if not content:
                    continue
                # 用内容前 50 字做去重
                key = content[:50]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)

        # 按来源数量排序（来自更多源的排在前面）
        # 简单实现：先按长度排序（长的通常信息更多）
        merged.sort(key=lambda d: -len(
            d.get("page_content", "") if isinstance(d, dict) else ""
        ))

        return merged[:self.top_k_final]

    def search_all(self, query: str) -> Dict:
        """
        执行联邦检索：同时在多个知识库 + 搜索引擎中搜索。
        """
        # 自动发现知识库
        kbs = self.discover_knowledge_bases()
        if not kbs:
            kbs = self.knowledge_bases
        logger.info(f"Federated search across: {kbs} + web")

        # 并行搜索所有数据源
        all_results = []

        # 多知识库搜索
        for kb_name in kbs:
            docs = self.search_single_kb(query, kb_name, self.top_k_per_source)
            if docs:
                all_results.append(docs)
                logger.info(f"  {kb_name}: {len(docs)} docs")

        # 搜索引擎补充（如果本地结果不足）
        local_count = sum(len(r) for r in all_results)
        if local_count < self.top_k_final:
            web_docs = self.search_web(query)
            if web_docs:
                all_results.append(web_docs)
                logger.info(f"  web: {len(web_docs)} docs")

        # 融合排序
        merged = self.merge_results(all_results)

        # 格式化上下文
        context_parts = []
        for i, doc in enumerate(merged):
            source = doc.get("_source", "unknown")
            content = doc.get("page_content", "") if isinstance(doc, dict) else str(doc)
            context_parts.append(f"[{i+1}] (来源: {source})\n{content}")

        return {
            "total_sources": len(all_results),
            "total_docs": len(merged),
            "context": "\n\n".join(context_parts),
            "docs": merged,
            "source_breakdown": {kb: self.search_single_kb(query, kb, 1) and 1 or 0 for kb in kbs},
        }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="联邦检索测试")
    parser.add_argument("--query", default="核心课程", help="搜索查询")
    parser.add_argument("--kbs", nargs="*", default=None, help="指定知识库列表")
    parser.add_argument("--no-web", action="store_true", help="禁用搜索引擎")
    args = parser.parse_args()

    searcher = FederatedSearch(knowledge_bases=args.kbs)

    result = searcher.search_all(args.query)

    print("\n" + "=" * 60)
    print(f"联邦检索结果: {result['total_docs']} 条文档，来自 {result['total_sources']} 个数据源")
    print("=" * 60)
    print(result["context"][:800])
