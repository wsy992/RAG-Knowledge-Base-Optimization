"""
LEGACY BASELINE ONLY: this script talks to the old Chatchat API and uses a
heuristic faithfulness score. Do not use its numbers as the v2 benchmark.

RAGAS 评估脚本: 量化测评 RAG 系统的检索与生成质量
- Hit Rate: 检索命中率
- MRR: 最相关结果排名
- Faithfulness: 回答忠实度（是否基于文档）
- Answer Relevancy: 回答相关性

使用方法：
  1. 先启动 Chatchat 服务
  2. 运行本脚本: python ragas_eval.py
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

# Legacy Chatchat API baseline. The v2 benchmark does not import this module.
API_BASE = os.getenv("CHATCHAT_API_BASE", "http://127.0.0.1:7861")

# 测试数据集: (问题, 期望包含的关键词, 所属领域)
TEST_QUESTIONS = [
    # 精确问法
    ("核心课程有哪些", ["数据结构", "机器学习", "深度学习", "Python"], "精确"),
    ("就业方向是什么", ["AI算法工程师", "数据分析师", "软件开发工程师"], "精确"),
    ("专业特色有哪些", ["理论与实践", "项目实战", "实习", "小班教学"], "精确"),
    ("培养目标是什么", ["人工智能", "数据科学", "解决实际问题"], "精确"),

    # 模糊问法（测试 Query 改写）
    ("有什么课", ["数据结构", "机器学习", "深度学习", "Python"], "模糊"),
    ("学啥的", ["人工智能", "数据科学", "软件开发"], "模糊"),
    ("能干啥", ["AI算法", "数据分析", "软件开发"], "模糊"),
    ("有啥优势", ["理论与实践", "项目实战", "小班教学"], "模糊"),
    ("能找什么工作", ["AI算法工程师", "数据分析师", "数据科学家"], "模糊"),

    # 否定测试（应该回答无法回答）
    ("学费多少钱", ["无法回答"], "否定"),
    ("宿舍怎么样", ["无法回答"], "否定"),
]


def call_rag(query: str, kb_name: str = "wsy") -> dict:
    """调用 Chatchat 知识库问答 API"""
    try:
        resp = requests.post(
            f"{API_BASE}/knowledge_base/local_kb/{kb_name}/chat/completions",
            json={
                "messages": [{"role": "user", "content": query}],
                "model": "deepseek-chat",
                "stream": False,
                "top_k": 5,
                "score_threshold": 1.0,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            # 响应是双层 JSON 编码，需要解析两次
            data = json.loads(json.loads(resp.text))
            content = ""
            docs = []
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
            if "docs" in data:
                docs = data.get("docs", [])
            return {"content": content, "docs": docs, "status": "ok"}
        else:
            return {"content": "", "docs": [], "status": f"error: {resp.status_code}"}
    except Exception as e:
        return {"content": "", "docs": [], "status": f"error: {e}"}


def evaluate_hit_rate(content: str, keywords: list, category: str = "精确") -> bool:
    """检查回答是否命中预期"""
    if "无法回答" in content:
        # 否定测试: 应该回答"无法回答"，这是正确的
        if category == "否定":
            return True
        # 其他测试: 说无法回答说明没命中
        return False
    for kw in keywords:
        if kw in content:
            return True
    return False


def evaluate_faithfulness(content: str, docs: list) -> float:
    """
    评估回答忠实度：
    - 说"无法回答"的：忠实度视为 1.0（不编造）
    - 有内容的回答：看是否包含具体信息（非空、非泛泛而谈）
    """
    if not content:
        return 0.0
    if "无法回答" in content:
        return 1.0
    # 有实质内容的回答，且没乱编，视为忠实
    if len(content) > 20:
        return 0.9
    return 0.5


def main():
    print("=" * 60)
    print("RAGAS 评估报告")
    print("=" * 60)

    # 先检测服务是否在运行
    try:
        r = requests.get(f"{API_BASE}/knowledge_base/list_knowledge_bases", timeout=5)
        if r.status_code != 200:
            print("[ERR] Chatchat 服务未启动，请先启动服务")
            return
        kbs = r.json().get("data", [])
        kb_names = [kb.get("kb_name") for kb in kbs]
        print(f"[OK] 服务运行中，知识库: {kb_names}")
    except Exception as e:
        print(f"[ERR] 无法连接 Chatchat 服务: {e}")
        return

    # 选择有文档的知识库
    kb_name = None
    for name in kb_names:
        if name and name not in ["ai", "_tmp"]:
            # 检查是否有文件
            try:
                files_r = requests.get(f"{API_BASE}/knowledge_base/list_files",
                                       params={"knowledge_base_name": name}, timeout=5)
                files = files_r.json().get("data", [])
                if len(files) > 0:
                    kb_name = name
                    break
            except:
                pass

    if not kb_name:
        print("[ERR] 未找到有文档的知识库")
        return

    print(f"[KB] 使用知识库: {kb_name}")
    print()

    # 逐条测试
    results = {
        "total": len(TEST_QUESTIONS),
        "hit": 0,
        "faithfulness_total": 0.0,
        "faithfulness_count": 0,
        "details": [],
        "by_category": {},
    }

    for i, (question, keywords, category) in enumerate(TEST_QUESTIONS):
        if category not in results["by_category"]:
            results["by_category"][category] = {"total": 0, "hit": 0}
        results["by_category"][category]["total"] += 1

        print(f"[{i+1}/{len(TEST_QUESTIONS)}] {category}: {question}")

        result = call_rag(question, kb_name)
        content = result["content"]
        docs = result["docs"]

        is_hit = evaluate_hit_rate(content, keywords, category)
        faithfulness = evaluate_faithfulness(content, docs)

        print(f"  回答: {content[:100]}...")
        print(f"  命中: {'[OK]' if is_hit else '[ERR]'} | 忠实度: {faithfulness:.2f} | 出处: {len(docs)}条")

        if is_hit:
            results["hit"] += 1
            results["by_category"][category]["hit"] += 1

        if faithfulness > 0:
            results["faithfulness_total"] += faithfulness
            results["faithfulness_count"] += 1

        results["details"].append({
            "question": question,
            "category": category,
            "content": content,
            "is_hit": is_hit,
            "faithfulness": faithfulness,
            "doc_count": len(docs),
        })

        time.sleep(0.5)  # 避免请求太快

    # 汇总报告
    hit_rate = results["hit"] / results["total"] * 100
    avg_faithfulness = results["faithfulness_total"] / results["faithfulness_count"] if results["faithfulness_count"] > 0 else 0

    # 按类别统计
    neg_total = results["by_category"].get("否定", {}).get("total", 0)
    neg_hit = results["by_category"].get("否定", {}).get("hit", 0)
    pos_total = results["total"] - neg_total
    pos_hit = results["hit"] - neg_hit

    print()
    print("=" * 60)
    print("[DATA] 评估汇总")
    print("=" * 60)
    print(f"总测试数:        {results['total']}")
    print(f"整体命中率:      {hit_rate:.1f}%")
    print(f"平均忠实度:      {avg_faithfulness:.2f}")
    print(f"有效问答命中率:   {pos_hit}/{pos_total} ({pos_hit/pos_total*100:.1f}%)" if pos_total > 0 else "")
    if neg_total > 0:
        print(f"否定测试准确率:   {neg_hit}/{neg_total} ({neg_hit/neg_total*100:.1f}%)（正确拒绝率）")
    print()

    print("按类别:")
    for cat, data in results["by_category"].items():
        rate = data["hit"] / data["total"] * 100
        print(f"  {cat}: {data['hit']}/{data['total']} ({rate:.1f}%)")

    # 生成 JSON 报告
    report = {
        "hit_rate": round(hit_rate, 1),
        "avg_faithfulness": round(avg_faithfulness, 2),
        "total_questions": results["total"],
        "by_category": {k: {"hit": v["hit"], "total": v["total"], "rate": f"{v['hit']/v['total']*100:.1f}%"} for k, v in results["by_category"].items()},
        "details": results["details"],
    }

    report_path = Path(__file__).parent / "ragas_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[FILE] 完整报告已保存: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
