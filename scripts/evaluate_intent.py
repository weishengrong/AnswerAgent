"""
意图识别 + 路由评测脚本

用法：
  python scripts/evaluate_intent.py
  python scripts/evaluate_intent.py --test-file scripts/intent_test_cases.json
  python scripts/evaluate_intent.py --output scripts/intent_results.json
"""
import asyncio
import json
import sys
import os
import time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


EVAL_FIELDS = ["intent", "need_memory", "needs_react", "skill"]


def derive_skill(intent: str, need_memory: bool, needs_react: bool) -> str:
    """根据 selector 的真值表反推应该选中的 skill"""
    if intent == "daily_chat":
        return "data_chat"
    if needs_react or need_memory:
        return "complex_query"
    return "simple_query"


async def run_single_test(test_case: dict) -> dict:
    from app.agent.intent import intent_agent

    user_input = test_case["user_input"]
    prompt = f"用户输入：{user_input}"

    start_time = time.time()
    try:
        result = await intent_agent.run(prompt)
        intent_data = result.output
        elapsed = time.time() - start_time

        pred = {
            "intent": intent_data.intent.value,
            "reason": intent_data.reason,
            "need_memory": intent_data.need_memory,
            "memory_reason": intent_data.memory_reason,
            "needs_react": intent_data.needs_react and intent_data.intent.value == "database_query",
            "react_reason": intent_data.react_reason,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        pred = {
            "intent": "daily_chat",
            "reason": f"识别失败: {str(e)}",
            "need_memory": False,
            "memory_reason": "",
            "needs_react": False,
            "react_reason": "",
        }

    pred["skill"] = derive_skill(pred["intent"], pred["need_memory"], pred["needs_react"])

    expected = {
        "intent": test_case["expected_intent"],
        "need_memory": test_case["expected_need_memory"],
        "needs_react": test_case["expected_needs_react"],
        "skill": test_case["expected_skill"],
    }

    field_results = {field: pred[field] == expected[field] for field in EVAL_FIELDS}
    all_correct = all(field_results.values())

    return {
        "id": test_case["id"],
        "user_input": user_input,
        "category": test_case["category"],
        "expected": expected,
        "predicted": pred,
        "field_results": field_results,
        "all_correct": all_correct,
        "elapsed": round(elapsed, 2),
    }


async def run_evaluation(test_file: str, output_file: str = None):
    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"\n{'='*70}")
    print(f"🧪 意图识别 + 路由评测")
    print(f"{'='*70}")
    print(f"测试用例: {len(test_cases)} 条")
    print(f"评测字段: {EVAL_FIELDS}")
    print(f"{'='*70}\n")

    results = []
    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc['user_input'][:30]:<30} ", end="", flush=True)
        result = await run_single_test(tc)
        results.append(result)

        status = "✅" if result["all_correct"] else "❌"
        wrong_fields = [k for k, v in result["field_results"].items() if not v]
        detail = f" 误判: {wrong_fields}" if wrong_fields else ""
        print(
            f"{status} pred={result['predicted']['intent']} "
            f"react={result['predicted']['needs_react']} "
            f"skill={result['predicted']['skill']}{detail}"
        )

    total = len(results)
    all_correct_count = sum(1 for r in results if r["all_correct"])

    field_stats = {}
    for field in EVAL_FIELDS:
        correct = sum(1 for r in results if r["field_results"][field])
        field_stats[field] = {"correct": correct, "total": total, "accuracy": round(correct / total, 4)}

    category_stats = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})
    for r in results:
        cat = r["category"]
        category_stats[cat]["total"] += 1
        if r["all_correct"]:
            category_stats[cat]["correct"] += 1
        else:
            wrong_fields = [k for k, v in r["field_results"].items() if not v]
            category_stats[cat]["errors"].append({
                "id": r["id"],
                "user_input": r["user_input"],
                "expected": r["expected"],
                "predicted": r["predicted"],
                "wrong_fields": wrong_fields,
            })

    skill_confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        skill_confusion[r["expected"]["skill"]][r["predicted"]["skill"]] += 1

    avg_elapsed = sum(r["elapsed"] for r in results) / total

    report = {
        "timestamp": datetime.now().isoformat(),
        "test_file": test_file,
        "total_cases": total,
        "all_correct_count": all_correct_count,
        "all_correct_rate": round(all_correct_count / total, 4),
        "field_stats": field_stats,
        "category_stats": {k: {
            "correct": v["correct"],
            "total": v["total"],
            "accuracy": round(v["correct"] / v["total"], 4),
            "errors": v["errors"],
        } for k, v in category_stats.items()},
        "skill_confusion_matrix": {
            expected: dict(predicted) for expected, predicted in skill_confusion.items()
        },
        "avg_elapsed": round(avg_elapsed, 2),
        "results": results,
    }

    print(f"\n{'='*70}")
    print(f"📊 评测报告")
    print(f"{'='*70}")
    print(f"总用例数:       {total}")
    print(f"完全匹配率:     {report['all_correct_rate']*100:.1f}% ({all_correct_count}/{total})")
    print(f"")
    print(f"--- 各字段准确率 ---")
    for field, stats in field_stats.items():
        print(f"  {field:<16} {stats['accuracy']*100:.1f}% ({stats['correct']}/{stats['total']})")
    print(f"")
    print(f"--- 各类别准确率 ---")
    for cat, stats in sorted(category_stats.items(), key=lambda x: x[1]["correct"] / max(x[1]["total"], 1)):
        acc = round(stats["correct"] / stats["total"], 4)
        print(f"  {cat:<20} {acc*100:.1f}% ({stats['correct']}/{stats['total']})")
    print(f"")
    print(f"--- 技能路由混淆矩阵 (rows=expected, cols=predicted) ---")
    skills = sorted({s for s in skill_confusion} | {p for row in skill_confusion.values() for p in row})
    header = " " * 16 + "".join(f"{s:>16}" for s in skills)
    print(header)
    for exp_skill in skills:
        row = skill_confusion.get(exp_skill, {})
        cells = "".join(f"{row.get(p, 0):>16}" for p in skills)
        print(f"  {exp_skill:<14}{cells}")
    print(f"")
    print(f"平均耗时:       {avg_elapsed:.2f}s")
    print(f"{'='*70}")

    print(f"\n--- 错误明细 ---")
    has_error = False
    for cat, stats in category_stats.items():
        for err in stats["errors"]:
            has_error = True
            print(f"  [{err['id']}] \"{err['user_input']}\" ({cat})")
            print(f"       误判字段: {err['wrong_fields']}")
            print(f"       期望: {err['expected']}")
            print(
                f"       实际: intent={err['predicted']['intent']}, "
                f"need_memory={err['predicted']['need_memory']}, "
                f"needs_react={err['predicted']['needs_react']}, "
                f"skill={err['predicted']['skill']}"
            )
            print(f"       理由: {err['predicted'].get('reason', '')}")
            print()
    if not has_error:
        print("  🎉 全部正确！")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"📁 详细结果已保存到: {output_file}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="意图识别 + 路由评测")
    parser.add_argument("--test-file", default="scripts/intent_test_cases.json", help="测试用例文件路径")
    parser.add_argument("--output", default=None, help="输出结果文件路径")
    args = parser.parse_args()

    output = args.output or f"scripts/intent_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    asyncio.run(run_evaluation(args.test_file, output))
