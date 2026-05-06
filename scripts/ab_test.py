"""
对比实验脚本 - A/B 测试不同配置的效果

用法：
  python scripts/ab_test.py
  python scripts/ab_test.py --test-file scripts/test_cases.json
"""
import asyncio
import json
import sys
import os
import copy
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.evaluate import run_evaluation, execute_sql_safely, compare_results
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("ab_test")


async def run_with_config(test_file: str, user_id: str, config_name: str, config_overrides: dict = None):
    """在指定配置下运行评测"""
    if config_overrides:
        for key, value in config_overrides.items():
            parts = key.split(".")
            obj = __import__("app.agent.state", fromlist=["AgentState"])
            if len(parts) == 1:
                setattr(obj.AgentState.model_fields[parts[0]], "default", value)

    result = await run_evaluation(test_file, user_id, output_file=None)
    return config_name, result


async def run_ab_test(test_file: str, user_id: str):
    """运行 A/B 对比实验"""

    print(f"\n{'#'*60}")
    print(f"🔬 A/B 对比实验")
    print(f"{'#'*60}\n")

    experiments = [
        ("baseline", "基线（当前配置）"),
    ]

    results = {}

    for exp_name, exp_desc in experiments:
        print(f"\n{'='*60}")
        print(f"🧪 实验: {exp_name} - {exp_desc}")
        print(f"{'='*60}\n")

        _, result = await run_with_config(test_file, f"{user_id}_{exp_name}", exp_name)
        results[exp_name] = result

    print(f"\n{'#'*60}")
    print(f"📊 对比实验总结")
    print(f"{'#'*60}\n")

    print(f"{'指标':<20} | {'基线':>10}")
    print(f"{'-'*20}-+-{'-'*10}")

    baseline = results.get("baseline", {})

    metrics = [
        ("总体准确率", "overall_accuracy"),
        ("意图准确率", "intent_accuracy"),
        ("数据库查询", "db_accuracy"),
        ("简单查询", "simple_accuracy"),
        ("中等查询", "medium_accuracy"),
        ("复杂查询", "hard_accuracy"),
        ("平均重试次数", "avg_retry_count"),
        ("平均耗时(s)", "avg_elapsed"),
    ]

    for name, key in metrics:
        val = baseline.get(key, 0)
        if "准确率" in name or "查询" in name:
            print(f"{name:<20} | {val*100:>9.1f}%")
        else:
            print(f"{name:<20} | {val:>10}")

    output_file = f"scripts/ab_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📁 详细结果已保存到: {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A/B 对比实验")
    parser.add_argument("--test-file", default="scripts/test_cases.json", help="测试用例文件路径")
    parser.add_argument("--user-id", default="ab_test_user", help="评测用户ID")
    args = parser.parse_args()

    asyncio.run(run_ab_test(args.test_file, args.user_id))
