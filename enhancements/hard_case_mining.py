"""
难例挖掘模块：自动识别和分类 RAG 系统的失败案例

核心功能：
1. 从 RAGAS 评估结果中识别 Bad Case
2. 按错误类型自动分类（检索遗漏、改写偏差、忠实度低等）
3. 每次问答后自动记录日志，可增量分析
4. 生成优化建议报告

使用方法：
  python -m enhancements.hard_case_mining --report ragas_report.json

简历写法：
  搭建 Bad Case 采集与分析 Pipeline，自动识别低质量回答并触发针对性优化
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("hard_case_mining")

# Bad Case 类型定义
CASE_TYPES = {
    "检索遗漏": "知识库中有相关内容但检索未命中",
    "改写偏差": "Query 改写导致偏离原意",
    "忠实度低": "回答内容与文档不一致",
    "过度拒绝": "本可回答但误判为无法回答",
    "理解错误": "对问题的意图理解有偏差",
}

# 优化建议模板
OPTIMIZATION_TIPS = {
    "检索遗漏": "建议优化 chunk 切分策略或调整 top_k 参数",
    "改写偏差": "建议优化 query 改写提示词或禁用特定问题的改写",
    "忠实度低": "建议检查 rerank 模型效果或调整 score_threshold",
    "过度拒绝": "建议降低 score_threshold 阈值或增加检索量",
    "理解错误": "建议在 query 改写中加入意图识别前置步骤",
}

class HardCaseMiner:
    """难例挖掘器"""

    def __init__(self, case_dir: str = "data/hard_cases"):
        self.case_dir = Path(case_dir)
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.cases: List[Dict] = []
        self.load()

    def load(self):
        """加载已有难例"""
        case_file = self.case_dir / "cases.json"
        if case_file.exists():
            with open(case_file, "r", encoding="utf-8") as f:
                self.cases = json.load(f)

    def save(self):
        """保存难例"""
        case_file = self.case_dir / "cases.json"
        with open(case_file, "w", encoding="utf-8") as f:
            json.dump(self.cases, f, ensure_ascii=False, indent=2)

    def analyze_eval_report(self, report_path: str) -> Dict:
        """分析 RAGAS 评估报告，提取 Bad Case 并分类"""
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        # 统计概况
        summary = {
            "总测试数": report.get("total_questions", 0),
            "命中率": report.get("hit_rate", 0),
            "平均忠实度": report.get("avg_faithfulness", 0),
            "bad_cases": [],
            "category_stats": {},
        }

        for detail in report.get("details", []):
            is_hit = detail.get("is_hit", False)
            faithfulness = detail.get("faithfulness", 1.0)
            category = detail.get("category", "")
            question = detail.get("question", "")
            content = detail.get("content", "")

            # 判断是否是 Bad Case
            case_type = None
            if not is_hit and category != "否定":
                case_type = self._classify_bad_case(question, content)
            elif faithfulness < 0.5:
                case_type = "忠实度低"
            elif not is_hit and category == "否定" and "无法回答" not in content:
                case_type = "理解错误"

            if case_type:
                case = {
                    "question": question,
                    "category": category,
                    "case_type": case_type,
                    "faithfulness": faithfulness,
                    "content": content[:200],
                    "suggestion": OPTIMIZATION_TIPS.get(case_type, "需要人工分析"),
                    "timestamp": datetime.now().isoformat(),
                }
                self.cases.append(case)
                summary["bad_cases"].append(case)

                if case_type not in summary["category_stats"]:
                    summary["category_stats"][case_type] = 0
                summary["category_stats"][case_type] += 1

        self.save()
        return summary

    def _classify_bad_case(self, question: str, content: str) -> str:
        """根据问题和回答判断 Bad Case 类型"""
        if "无法回答" in content:
            # 应该能回答但没答上
            # 如果问题很短（可能改写失败）
            if len(question) <= 6:
                return "改写偏差"
            return "检索遗漏"
        elif len(content) < 10:
            return "检索遗漏"
        # 检查回答是否偏离问题（简单关键词检查）
        q_words = set(question)
        c_words = set(content)
        overlap = len(q_words & c_words) / max(len(q_words), 1)
        if overlap < 0.1 and len(content) > 20:
            return "理解错误"
        return "检索遗漏"

    def log_interaction(self, question: str, answer: str, faithfulness: float,
                       query_rewritten: str = ""):
        """记录单次问答交互，用于增量分析"""
        entry = {
            "question": question,
            "answer": answer[:200],
            "query_rewritten": query_rewritten,
            "faithfulness": faithfulness,
            "timestamp": datetime.now().isoformat(),
        }
        # 追加到交互日志
        log_file = self.case_dir / "interactions.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def generate_report(self) -> str:
        """生成优化建议报告"""
        if not self.cases:
            return "尚无 Bad Case 记录"

        lines = []
        lines.append("=" * 60)
        lines.append("Bad Case 分析报告")
        lines.append("=" * 60)
        lines.append(f"总难例数: {len(self.cases)}")
        lines.append("")

        # 按类型统计
        type_stats = {}
        for case in self.cases:
            ct = case["case_type"]
            type_stats[ct] = type_stats.get(ct, 0) + 1

        lines.append("错误类型分布:")
        for ct, count in sorted(type_stats.items(), key=lambda x: -x[1]):
            pct = count / len(self.cases) * 100
            lines.append(f"  {ct}: {count} 例 ({pct:.0f}%)")
            lines.append(f"    建议: {OPTIMIZATION_TIPS.get(ct, '')}")

        lines.append("")
        lines.append("详细列表:")
        for i, case in enumerate(self.cases[:20], 1):
            lines.append(f"\n[{i}] [{case['case_type']}] {case['question']}")
            lines.append(f"    回答: {case['content'][:100]}...")
            lines.append(f"    建议: {case['suggestion']}")

        if len(self.cases) > 20:
            lines.append(f"\n... 还有 {len(self.cases) - 20} 条未显示")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="难例挖掘工具")
    parser.add_argument("--report", default="reports/legacy_ragas_report.json", help="RAGAS 评估报告路径")
    parser.add_argument("--interaction", nargs=3, metavar=("Q", "A", "F"), help="记录单次交互")
    parser.add_argument("--case-dir", default="data/hard_cases", help="难例数据目录")
    args = parser.parse_args()

    miner = HardCaseMiner(case_dir=args.case_dir)

    if args.interaction:
        q, a, f = args.interaction
        miner.log_interaction(q, a, float(f))
        print(f"Interaction logged: {q}")
    elif args.report:
        summary = miner.analyze_eval_report(args.report)
        print(miner.generate_report())
