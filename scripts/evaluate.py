"""
评测脚本 - Text-to-SQL Agent 评测框架

核心指标：
- Execution Accuracy (EX): 生成的SQL执行结果是否与标准答案一致
- Intent Accuracy: 意图识别是否正确
- Retry Rate: 平均重试次数

用法：
  python scripts/evaluate.py
  python scripts/evaluate.py --test-file scripts/test_cases.json
  python scripts/evaluate.py --user-id test_user --output results.json
"""
import asyncio
import json
import sys
import os
import time
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("evaluate")


async def execute_sql_safely(sql: str) -> list:
    """安全执行SQL并返回结果"""
    from sqlalchemy import text
    from config.db_config import AsyncSessionLocal

    if not sql or not sql.strip().upper().startswith("SELECT"):
        return []

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(text(sql))
            rows = result.mappings().all()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.debug(f"SQL执行失败: {e}")
        return []


def compare_results(pred_result: list, gold_result: list) -> bool:
    """比较两个查询结果是否一致（忽略顺序）"""
    if not pred_result and not gold_result:
        return True
    if not pred_result or not gold_result:
        return False

    try:
        pred_set = set()
        for row in pred_result:
            pred_set.add(tuple(sorted(str(v) for v in row.values())))

        gold_set = set()
        for row in gold_result:
            gold_set.add(tuple(sorted(str(v) for v in row.values())))

        return pred_set == gold_set
    except Exception:
        return False


async def run_single_test(test_case: dict, user_id: str) -> dict:
    """运行单条测试用例"""
    from app.agent.workflow import workflow

    question = test_case["question"]
    gold_sql = test_case.get("gold_sql", "")
    difficulty = test_case.get("difficulty", "medium")
    category = test_case.get("category", "")

    start_time = time.time()
    state = await workflow.execute(question, user_id=user_id, session_id=f"eval-{test_case['id']}")
    elapsed = time.time() - start_time

    pred_sql = state.sql_query or ""
    intent = state.intent or ""
    skill = state.skill_name or ""
    retry_count = state.retry_count

    if difficulty == "chat":
        intent_correct = intent == "daily_chat"
        ex_correct = intent_correct
    else:
        intent_correct = intent == "database_query"

        if pred_sql and gold_sql:
            pred_result = await execute_sql_safely(pred_sql)
            gold_result = await execute_sql_safely(gold_sql)
            ex_correct = compare_results(pred_result, gold_result)
        else:
            ex_correct = False

    return {
        "id": test_case["id"],
        "question": question,
        "difficulty": difficulty,
        "category": category,
        "pred_sql": pred_sql,
        "gold_sql": gold_sql,
        "intent": intent,
        "skill": skill,
        "intent_correct": intent_correct,
        "ex_correct": ex_correct,
        "retry_count": retry_count,
        "elapsed": round(elapsed, 2),
        "response": state.response[:200] if state.response else ""
    }


async def run_evaluation(test_file: str, user_id: str, output_file: str = None):
    """运行完整评测"""
    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"\n{'='*60}")
    print(f"🧪 Text-to-SQL Agent 评测")
    print(f"{'='*60}")
    print(f"测试用例: {len(test_cases)} 条")
    print(f"用户ID: {user_id}")
    print(f"{'='*60}\n")

    results = []
    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc['question'][:40]}... ", end="", flush=True)
        result = await run_single_test(tc, user_id)
        results.append(result)

        status = "✅" if result["ex_correct"] else "❌"
        print(f"{status} intent={result['intent']} skill={result['skill']} retries={result['retry_count']} ({result['elapsed']}s)")

    db_cases = [r for r in results if r["difficulty"] != "chat"]
    chat_cases = [r for r in results if r["difficulty"] == "chat"]

    total = len(results)
    total_ex = sum(1 for r in results if r["ex_correct"])
    total_intent = sum(1 for r in results if r["intent_correct"])
    avg_retry = sum(r["retry_count"] for r in db_cases) / max(len(db_cases), 1)
    avg_time = sum(r["elapsed"] for r in results) / max(len(results), 1)

    simple_cases = [r for r in db_cases if r["difficulty"] == "simple"]
    medium_cases = [r for r in db_cases if r["difficulty"] == "medium"]
    hard_cases = [r for r in db_cases if r["difficulty"] == "hard"]

    simple_acc = sum(1 for r in simple_cases if r["ex_correct"]) / max(len(simple_cases), 1)
    medium_acc = sum(1 for r in medium_cases if r["ex_correct"]) / max(len(medium_cases), 1)
    hard_acc = sum(1 for r in hard_cases if r["ex_correct"]) / max(len(hard_cases), 1)
    db_acc = sum(1 for r in db_cases if r["ex_correct"]) / max(len(db_cases), 1)
    chat_acc = sum(1 for r in chat_cases if r["ex_correct"]) / max(len(chat_cases), 1)

    report = {
        "timestamp": datetime.now().isoformat(),
        "test_file": test_file,
        "total_cases": total,
        "overall_accuracy": round(total_ex / total, 4),
        "intent_accuracy": round(total_intent / total, 4),
        "db_accuracy": round(db_acc, 4),
        "chat_accuracy": round(chat_acc, 4),
        "simple_accuracy": round(simple_acc, 4),
        "medium_accuracy": round(medium_acc, 4),
        "hard_accuracy": round(hard_acc, 4),
        "avg_retry_count": round(avg_retry, 2),
        "avg_elapsed": round(avg_time, 2),
        "results": results
    }

    print(f"\n{'='*60}")
    print(f"📊 评测报告")
    print(f"{'='*60}")
    print(f"总用例数:     {total}")
    print(f"总体准确率:   {report['overall_accuracy']*100:.1f}%")
    print(f"意图准确率:   {report['intent_accuracy']*100:.1f}%")
    print(f"数据库查询:   {report['db_accuracy']*100:.1f}% ({len(db_cases)}条)")
    print(f"日常聊天:     {report['chat_accuracy']*100:.1f}% ({len(chat_cases)}条)")
    print(f"  - 简单查询: {report['simple_accuracy']*100:.1f}% ({len(simple_cases)}条)")
    print(f"  - 中等查询: {report['medium_accuracy']*100:.1f}% ({len(medium_cases)}条)")
    print(f"  - 复杂查询: {report['hard_accuracy']*100:.1f}% ({len(hard_cases)}条)")
    print(f"平均重试次数: {report['avg_retry_count']}")
    print(f"平均耗时:     {report['avg_elapsed']}s")
    print(f"{'='*60}")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📁 详细结果已保存到: {output_file}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Text-to-SQL Agent 评测")
    parser.add_argument("--test-file", default="scripts/test_cases.json", help="测试用例文件路径")
    parser.add_argument("--user-id", default="eval_user", help="评测用户ID")
    parser.add_argument("--output", default=None, help="输出结果文件路径")
    args = parser.parse_args()

    output = args.output or f"scripts/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    asyncio.run(run_evaluation(args.test_file, args.user_id, output))
