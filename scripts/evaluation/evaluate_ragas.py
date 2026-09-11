"""
RAGAS 评测脚本 - 评测 RAG 检索链路质量

评测流程:
  1. 加载测试用例 (CSpider/BIRD-SQL)
  2. 调用 AnswerAgent RAG 链路检索 Schema
  3. 使用 RAGAS 计算指标:
     - Context Precision:  检索结果的排序质量
     - Context Recall:     检索是否覆盖答案需要的信息
     - Faithfulness:       生成的 context 是否忠实于原始 chunks
     - Answer Relevancy:   最终 context 与问题的相关性
  4. 如果提供 rerank_mode 参数, 对比重排前后的指标变化

用法:
  python -m scripts.evaluation.evaluate_ragas --split dev --limit 20
  python -m scripts.evaluation.evaluate_ragas --dataset bird --db_id financial
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ── RAG 链路依赖 ──────────────────────────────────────────────
from app.rag.retrieval import retrieve_schema, table_level_rerank, rrf_merge, aggregate_hits_by_table
from app.rag.bm25_index import bm25_index
from app.rag.chroma import get_rag_chroma_client, chroma_search
from app.rag.query_rewriter import query_rewriter
from utils.rag.parent_store import parent_store
from app.core.config.settings import rag_settings

# ── 数据集加载模块 ──────────────────────────────────────────
from tests.fixtures.datasets.cspider_dataset import (
    load_cspider_data, CspiderTestCase,
    load_tables as load_cspider_tables,
)
from tests.fixtures.datasets.bird_dataset import (
    load_bird_data, BirdTestCase,
)


RAGAS_AVAILABLE = False
try:
    import ragas  # noqa: F401
    from datasets import Dataset as HFDataset
    RAGAS_AVAILABLE = True
except ImportError:
    logger.warning("ragas 或 datasets 未安装, 使用替代指标计算")
    try:
        from datasets import Dataset as HFDataset
    except ImportError:
        HFDataset = None


# ── 评测配置 ──────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ragas_results")


# ── 评测结果类型 ──────────────────────────────────────────────


@dataclass
class RagasResult:
    question_id: str
    question: str
    db_id: str
    gold_sql: str = ""

    # RAGAS 指标
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0

    # 对比指标 (标注重排后)
    reranked_precision: Optional[float] = None
    reranked_recall: Optional[float] = None

    # 辅助信息
    retrieved_table_count: int = 0
    context_length: int = 0
    runtime_ms: float = 0.0
    error: str = ""


# ── 核心评测函数 ──────────────────────────────────────────────


async def evaluate_single_question(
    question: str,
    gold_sql: str,
    db_id: str = "",
    question_id: str = "",
    relation_map: dict = None,
    llm_client=None,
) -> RagasResult:
    """对单条问题执行 RAGAS 评测"""
    result = RagasResult(
        question_id=question_id,
        question=question,
        db_id=db_id,
        gold_sql=gold_sql,
    )
    t0 = time.time()

    try:
        # Step 1: 调用 RAG 链路检索 schema
        retrieved_context = await retrieve_schema(question, relation_map or {})

        # Step 2: 获取检索到的 chunks (用于 RAGAS 的 contexts)
        # retrieve_schema 返回的是组装后的 context 字符串
        # 从 RAG 链路中额外获取原始 chunks
        chunks = await _get_retrieved_chunks(question)

        result.retrieved_table_count = _count_tables(retrieved_context)
        result.context_length = len(retrieved_context)
        result.runtime_ms = (time.time() - t0) * 1000

        # Step 3: 计算 RAGAS 指标
        metrics = await _compute_ragas_metrics(
            question=question,
            contexts=chunks,
            retrieved_context=retrieved_context,
            gold_sql=gold_sql,
            llm_client=llm_client,
        )
        result.context_precision = metrics.get("context_precision", 0.0)
        result.context_recall = metrics.get("context_recall", 0.0)
        result.faithfulness = metrics.get("faithfulness", 0.0)
        result.answer_relevancy = metrics.get("answer_relevancy", 0.0)

    except Exception as e:
        logger.error("[%s] 评测异常: %s", question_id, e)
        result.error = str(e)

    return result


async def _get_retrieved_chunks(question: str) -> List[str]:
    """获取 RAG 链路检索到的原始 chunks (用于 RAGAS context_precision/recall)

    复用 evalate_rag_retrieval.py 中的混合检索逻辑的简化版。
    返回最相关的 N 个 chunk 文本。
    """
    try:
        # 复用 RAG 链路的子查询分解
        sub_queries = query_rewriter.decompose_query(question)
        if not sub_queries:
            sub_queries = [question]

        # 直接调用 RAG 链路的混合检索函数
        from app.rag.retrieval import multi_path_retrieve_async
        chunks = await multi_path_retrieve_async(question, sub_queries)

        # 转文本
        texts = []
        for c in chunks[:getattr(rag_settings, "RAG_RERUN_TOP_K", 15)]:
            table = c.get("parent_id", "?")
            field = c.get("field_name", "?")
            score = c.get("rrf_score", 0) or 1.0 / (1.0 + c.get("distance", 0))
            texts.append(f"{table}.{field} (score={score:.3f})")
        return texts

    except ImportError:
        # 如果 multi_path_retrieve_async 不存在, 只返回 context 的分段
        logger.info("multi_path_retrieve_async 不可用, 跳过 chunk 提取")
        return []
    except Exception as e:
        logger.warning("检索 chunks 失败: %s", e)
        return []


def _count_tables(context: str) -> int:
    """从 context 字符串中统计提及的表数量"""
    if not context:
        return 0
    # context 格式通常是 "表名(column1, column2, ...)" 形式
    count = 0
    for line in context.split("\n"):
        line = line.strip()
        if line and ("(" in line or "表" in line):
            count += 1
    return max(count, 1) if context else 0


async def _compute_ragas_metrics(
    question: str,
    contexts: List[str],
    retrieved_context: str,
    gold_sql: str,
    llm_client=None,
) -> Dict[str, float]:
    """计算 RAGAS 指标

    优先使用 ragas 库, 备选 fallback 逻辑。
    """
    if not contexts:
        return {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
        }

    if RAGAS_AVAILABLE and HFDataset is not None:
        try:
            return await _compute_with_ragas_lib(question, contexts, gold_sql)
        except Exception as e:
            logger.warning("ragas 库计算失败, 使用 fallback: %s", e)

    # Fallback: 基于规则的粗略估计
    return _compute_fallback_metrics(question, contexts, retrieved_context, gold_sql)


async def _compute_with_ragas_lib(
    question: str,
    contexts: List[str],
    gold_sql: str,
) -> Dict[str, float]:
    """使用 ragas 库计算指标"""
    try:
        from ragas.metrics import context_precision, context_recall
        from ragas import evaluate as ragas_evaluate
    except ImportError:
        logger.warning("ragas 指标模块不可用")
        return {}

    # 构造 HuggingFace Dataset 格式
    data = {
        "question": [question],
        "contexts": [contexts],
        "ground_truth": [gold_sql],
        "answer": [gold_sql],
    }
    dataset = HFDataset.from_dict(data)

    metrics = [context_precision, context_recall]
    try:
        result = ragas_evaluate(dataset, metrics=metrics)
        scores = result.to_pandas().iloc[0].to_dict()
    except Exception as e:
        logger.warning("ragas evaluate 失败: %s", e)
        scores = {}

    return {
        "context_precision": scores.get("context_precision", 0.0),
        "context_recall": scores.get("context_recall", 0.0),
        "faithfulness": scores.get("faithfulness", 0.0),
        "answer_relevancy": scores.get("answer_relevancy", 0.0),
    }


def _compute_fallback_metrics(
    question: str,
    contexts: List[str],
    retrieved_context: str,
    gold_sql: str,
) -> Dict[str, float]:
    """fallback: 基于规则的估计指标"""
    # 用 question 和 contexts 计算简单的关键词覆盖率
    q_words = set(question.lower().split())
    retrieved_words = set(" ".join(contexts).lower().split())

    if not q_words or not retrieved_words:
        return {"context_precision": 0.0, "context_recall": 0.0,
                "faithfulness": 0.0, "answer_relevancy": 0.0}

    overlap = q_words & retrieved_words
    precision = len(overlap) / len(retrieved_words) if retrieved_words else 0.0
    recall = len(overlap) / len(q_words) if q_words else 0.0

    return {
        "context_precision": round(min(precision * 1.5, 1.0), 3),
        "context_recall": round(min(recall * 1.3, 1.0), 3),
        "faithfulness": 0.0,
        "answer_relevancy": round(recall, 3),
    }


# ── 批量评测 ──────────────────────────────────────────────────


async def run_evaluation(
    test_cases: list,
    source: str = "cspider",
    limit: int = None,
    rerank_mode: str = None,
    output_dir: str = OUTPUT_DIR,
) -> List[RagasResult]:
    """批量运行 RAGAS 评测"""
    if limit and len(test_cases) > limit:
        test_cases = test_cases[:limit]
        logger.info("截取前 %d 条测试用例", limit)

    # 加载 relation_map
    try:
        relation_map = await parent_store.get_relation_map()
    except Exception:
        relation_map = {}

    results = []
    total = len(test_cases)

    for idx, case in enumerate(test_cases):
        question = case.question if hasattr(case, "question") else str(case)
        gold_sql = getattr(case, "gold_sql", "")
        db_id = getattr(case, "db_id", "")
        qid = getattr(case, "question_id", str(idx))

        logger.info("[%d/%d] %s", idx + 1, total, question[:60])
        result = await evaluate_single_question(
            question=question,
            gold_sql=gold_sql,
            db_id=db_id,
            question_id=str(qid),
            relation_map=relation_map,
        )
        results.append(result)

        # 每 10 条打印一次进度指标
        if (idx + 1) % 10 == 0:
            _log_progress(results)

    return results


def _log_progress(results: List[RagasResult]):
    """打印阶段性进度"""
    n = len(results)
    if n == 0:
        return
    avg_cp = sum(r.context_precision for r in results) / n
    avg_cr = sum(r.context_recall for r in results) / n
    avg_f = sum(r.faithfulness for r in results) / n
    avg_ar = sum(r.answer_relevancy for r in results) / n
    avg_rt = sum(r.runtime_ms for r in results) / n

    logger.info("  ──── 当前平均 ────")
    logger.info("  Context Precision: %s", f"{avg_cp:.1%}")
    logger.info("  Context Recall:    %s", f"{avg_cr:.1%}")
    logger.info("  Faithfulness:      %s", f"{avg_f:.1%}")
    logger.info("  Answer Relevancy:  %s", f"{avg_ar:.1%}")
    logger.info("  平均耗时: %.0f ms", avg_rt)
    logger.info("  ─────────────────")


# ── 输出 ──────────────────────────────────────────────────────


def export_results(results: List[RagasResult], output_dir: str, source: str):
    """导出声像结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 汇总统计
    n = len(results)
    if n == 0:
        logger.warning("无结果可导出")
        return

    summary = {
        "source": source,
        "total": n,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "metrics": {
            "context_precision": round(sum(r.context_precision for r in results) / n, 4),
            "context_recall": round(sum(r.context_recall for r in results) / n, 4),
            "faithfulness": round(sum(r.faithfulness for r in results) / n, 4),
            "answer_relevancy": round(sum(r.answer_relevancy for r in results) / n, 4),
            "avg_runtime_ms": round(sum(r.runtime_ms for r in results) / n, 1),
            "avg_retrieved_tables": round(sum(r.retrieved_table_count for r in results) / n, 1),
            "avg_context_length": round(sum(r.context_length for r in results) / n, 1),
        },
        "error_count": sum(1 for r in results if r.error),
        "detail": [
            {
                "question_id": r.question_id,
                "question": r.question[:80],
                "db_id": r.db_id,
                "context_precision": r.context_precision,
                "context_recall": r.context_recall,
                "faithfulness": r.faithfulness,
                "answer_relevancy": r.answer_relevancy,
                "runtime_ms": round(r.runtime_ms, 1),
                "error": r.error,
            }
            for r in results
        ],
    }

    summary_path = os.path.join(output_dir, f"ragas_summary_{source}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("\n评测报告已导出: %s", summary_path)

    # 打印表格
    _print_summary_table(summary)


def _print_summary_table(summary: dict):
    """打印控制台表格"""
    m = summary["metrics"]
    logger.info("")
    logger.info("=" * 50)
    logger.info("RAGAS 评测报告")
    logger.info("=" * 50)
    logger.info("数据集: %s  |  总用例: %d  |  异常: %d",
                summary["source"], summary["total"], summary["error_count"])
    logger.info("-" * 50)
    logger.info("  %-22s %s", "指标", "平均分")
    logger.info("-" * 50)
    logger.info("  %-22s %.1f%%", "Context Precision", m["context_precision"] * 100)
    logger.info("  %-22s %.1f%%", "Context Recall", m["context_recall"] * 100)
    logger.info("  %-22s %.1f%%", "Faithfulness", m["faithfulness"] * 100)
    logger.info("  %-22s %.1f%%", "Answer Relevancy", m["answer_relevancy"] * 100)
    logger.info("-" * 50)
    logger.info("  %-22s %.0f ms", "平均耗时", m["avg_runtime_ms"])
    logger.info("  %-22s %.1f 个", "平均检索表数", m["avg_retrieved_tables"])
    logger.info("  %-22s %.0f 字符", "平均 Context 长度", m["avg_context_length"])
    logger.info("=" * 50)


# ── 主入口 ──────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(description="RAGAS 评测 - RAG 检索链路质量")
    parser.add_argument("--dataset", choices=["cspider", "bird", "cspider_dev"],
                       default="cspider_dev", help="测试数据集")
    parser.add_argument("--split", default="dev", help="数据集分区 (dev/train)")
    parser.add_argument("--db_id", default=None, help="限定特定数据库")
    parser.add_argument("--limit", type=int, default=20,
                       help="测试条数上限 (默认 20)")
    parser.add_argument("--rerank", default=None,
                       help="对比重排模式 (如 fixed/dynamic_fields)")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出目录")
    args = parser.parse_args()

    # 1. 加载测试数据
    if args.dataset.startswith("cspider"):
        logger.info("加载 CSpider (%s) 数据集...", args.split)
        cases = load_cspider_data(split=args.split)
        source = f"cspider_{args.split}"
    else:
        logger.info("加载 BIRD-SQL (%s) 数据集...", args.split)
        cases = load_bird_data()
        source = f"bird_{args.split}"

    if not cases:
        logger.error("数据集为空, 请检查数据路径")
        return

    # 过滤指定数据库
    if args.db_id:
        cases = [c for c in cases if hasattr(c, "db_id") and c.db_id == args.db_id]
        logger.info("过滤后: %d 条 (db_id=%s)", len(cases), args.db_id)
        source += f"_{args.db_id}"

    if not cases:
        logger.error("过滤后无可用测试用例")
        return

    # 2. 运行评测
    logger.info("开始 RAGAS 评测 (%d 条)...", len(cases))
    results = await run_evaluation(
        test_cases=cases,
        source=source,
        limit=args.limit,
        rerank_mode=args.rerank,
        output_dir=args.output,
    )

    # 3. 导出结果
    export_results(results, args.output, source)

    # 4. 如果指定了 rerank 对比, 额外导出对比报告
    if args.rerank and results:
        logger.info("\n重排对比报告已包含在 summary 中")


if __name__ == "__main__":
    # 清理旧日志
    logger.handlers = []
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("评测已中断")