"""
DeepEval Agent 功能正确性评测 - Text-to-SQL 端到端

评测流程:
  1. 加载 BIRD-SQL financial 数据库的中文测试用例 (30 条)
  2. 通过 HTTP 调用 AnswerAgent API (POST /ask)
  3. 执行生成 SQL 和 Gold SQL 在 MySQL 上
  4. 计算 Execution Accuracy (EX)，即结果集是否等价
  5. 统计意图识别准确率、响应延迟

用法:
  # 先启动 AnswerAgent 服务 (uvicorn app.main:app), 再运行:
  python tests/test_agent_functional.py                      # 用中文数据集测
  python tests/test_agent_functional.py --lang en            # 用英文原版测
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from types import SimpleNamespace

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import httpx

# ── 数据集加载 ──────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load_financial_cases(use_zh: bool = True) -> List:
    """加载 financial 测试用例

    Args:
        use_zh: True 用中文问题, False 用英文原版
    """
    zh_path = os.path.join(DATA_DIR, "bird_financial_zh.jsonl")
    if use_zh and os.path.exists(zh_path):
        # 从翻译后的 JSONL 加载
        cases = []
        with open(zh_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                case = SimpleNamespace(
                    question_id=r["question_id"],
                    question=r["question_zh"],  # 中文问题
                    gold_sql=r["gold_sql"],
                    db_id=r["db_id"],
                    difficulty=r["difficulty"],
                    evidence=r.get("evidence", ""),
                )
                cases.append(case)
        logger.info("中文数据集加载完成: %d 条 (from bird_financial_zh.jsonl)", len(cases))
        return cases
    else:
        # 从原始 BIRD-SQL 加载（英文）
        from tests.fixtures.datasets.bird_dataset import load_bird_data
        cases = load_bird_data()
        financial_cases = [c for c in cases if c.db_id == "financial"]
        # 按 question_id 去重
        seen = set()
        unique = []
        for c in financial_cases:
            if c.question_id not in seen:
                seen.add(c.question_id)
                unique.append(c)
        logger.info("英文数据集加载完成: %d 条", len(unique))
        return unique


# 全局数据集（文件级别加载一次）
_USE_ZH = True
if "--lang" in sys.argv:
    idx = sys.argv.index("--lang")
    if idx + 1 < len(sys.argv) and sys.argv[idx + 1] == "en":
        _USE_ZH = False
FINANCIAL_CASES = _load_financial_cases(use_zh=_USE_ZH)

# ── MySQL 连接 (从 .env 读取) ────────────────────────────
from dotenv import load_dotenv
load_dotenv()
import re

_db_url = os.getenv("DATABASE_URL", "")
_db_match = re.match(r"mysql\+[^:]+://([^:]+):([^@]+)@([^:]+):(\d+)/(\w+)", _db_url)
if _db_match:
    MYSQL_CONFIG = {
        "host": _db_match.group(3),
        "port": int(_db_match.group(4)),
        "user": _db_match.group(1),
        "password": _db_match.group(2),
        "database": _db_match.group(5),
    }
else:
    MYSQL_CONFIG = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": "financial",
    }

# ── Agent API ──────────────────────────────────────────────
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8000")

# ── 输出目录 ──────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "deepeval_results")


# ══════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════


def _normalize_sql(sql: str) -> str:
    """标准化 SQL, 便于比较"""
    import re
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.upper()
    return sql


def _exec_sql(sql: str, timeout: int = 10) -> Tuple[bool, List[dict], str]:
    """在 MySQL 上执行 SQL, 返回 (是否成功, 结果行列表, 错误信息)

    使用 pymysql 连接 MySQL 并执行。
    """
    try:
        import pymysql
        conn = pymysql.connect(
            host=MYSQL_CONFIG["host"],
            port=MYSQL_CONFIG["port"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
            database=MYSQL_CONFIG["database"],
            connect_timeout=5,
            read_timeout=timeout,
        )
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        conn.close()
        return True, rows, ""
    except Exception as e:
        return False, [], str(e)


def _compare_results(gold_rows: List[dict], pred_rows: List[dict]) -> bool:
    """比较两个结果集是否等价 (无视行序, 只比较集合)"""
    if len(gold_rows) != len(pred_rows):
        return False

    if not gold_rows and not pred_rows:
        return True

    # 将所有行转为排序后的元组列表, 比较集合
    def _row_set(rows):
        if not rows:
            return {()}
        result = set()
        for row in rows:
            result.add(tuple(str(v) for v in row.values()))
        return result

    return _row_set(gold_rows) == _row_set(pred_rows)


# ══════════════════════════════════════════════════════════════
# 测试数据
# ══════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def bird_financial_cases() -> List:
    """返回全局数据集"""
    return FINANCIAL_CASES


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ══════════════════════════════════════════════════════════════
# 核心测试
# ══════════════════════════════════════════════════════════════


class TestAgentFunctional:
    """Agent 功能正确性测试"""

    # 统计用类变量
    results = []

    @pytest.mark.parametrize("case_idx", range(len(FINANCIAL_CASES)))  # financial 条数
    def test_financial_question(self, case_idx: int, bird_financial_cases: List):
        """测试 financial 库的每条问题"""
        if case_idx >= len(bird_financial_cases):
            pytest.skip(f"用例索引 {case_idx} 超出范围 (共 {len(bird_financial_cases)} 条)")

        case = bird_financial_cases[case_idx]
        result = self._run_single_test(case)
        self.results.append(result)

        # 断言
        assert result["sql_exec_ok"], (
            f"[{case.question_id}] SQL 执行失败\n"
            f"  问题: {case.question[:60]}\n"
            f"  预期: {case.gold_sql[:80]}\n"
            f"  生成: {result.get('pred_sql', '')[:80]}\n"
            f"  错误: {result.get('error', '')}"
        )
        assert result["execution_accuracy"], (
            f"[{case.question_id}] 结果集不匹配\n"
            f"  问题: {case.question[:60]}\n"
            f"  Gold 行数: {result.get('gold_row_count', 0)}\n"
            f"  生成行数: {result.get('pred_row_count', 0)}"
        )

    def _run_single_test(self, case: BirdTestCase) -> dict:
        """单条测试执行"""
        result = {
            "question_id": case.question_id,
            "question": case.question,
            "gold_sql": case.gold_sql,
            "db_id": case.db_id,
            "difficulty": case.difficulty,
            "sql_exec_ok": False,
            "execution_accuracy": False,
            "intent_correct": False,
            "latency_ms": 0,
            "error": "",
        }
        t0 = time.time()

        try:
            # Step 1: 调用 Agent API
            response = self._call_agent(case.question)
            result["latency_ms"] = (time.time() - t0) * 1000
            result["agent_response"] = response.get("response", "")
            result["agent_intent"] = response.get("intent", "")
            result["agent_skill"] = response.get("skill", "")

            # Step 2: 执行 Gold SQL
            gold_ok, gold_rows, gold_err = _exec_sql(case.gold_sql)
            result["gold_sql_ok"] = gold_ok
            result["gold_row_count"] = len(gold_rows) if gold_ok else 0

            if not gold_ok:
                result["error"] = f"Gold SQL 执行失败: {gold_err}"
                return result

            result["gold_rows"] = gold_rows

            # Step 3: 从 Agent 响应中提取 SQL
            pred_sql = self._extract_sql_from_response(response, case.question)
            result["pred_sql"] = pred_sql

            if not pred_sql:
                result["error"] = "未能从 Agent 响应中提取 SQL"
                return result

            # Step 4: 执行生成 SQL
            pred_ok, pred_rows, pred_err = _exec_sql(pred_sql)
            result["pred_sql_ok"] = pred_ok
            result["pred_row_count"] = len(pred_rows) if pred_ok else 0

            if not pred_ok:
                result["error"] = f"生成 SQL 执行失败: {pred_err}"
                return result

            result["pred_rows"] = pred_rows

            # Step 5: 比较结果集
            result["execution_accuracy"] = _compare_results(gold_rows, pred_rows)
            result["sql_exec_ok"] = True

            # Step 6: 意图正确性 (简单判断)
            result["intent_correct"] = response.get("intent") == "database_query"

        except Exception as e:
            result["error"] = str(e)
            result["latency_ms"] = (time.time() - t0) * 1000

        return result

    def _call_agent(self, question: str) -> dict:
        """调用 AnswerAgent API"""
        try:
            resp = httpx.post(
                f"{AGENT_URL}/ask",
                params={"question": question},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.RequestError as e:
            raise RuntimeError(f"Agent API 调用失败: {e}")

    def _extract_sql_from_response(self, response: dict, question: str) -> Optional[str]:
        """从 Agent 响应中提取 SQL 语句

        优先从 react_trace 中提取, 其次从 response 文本中提取。
        """
        import re

        # 方式 1: 从 react_trace 中提取
        react_trace = response.get("react_trace")
        if react_trace:
            for step in react_trace:
                if isinstance(step, dict):
                    sql = step.get("sql") or step.get("query") or step.get("result", "")
                    if sql and re.search(r'(SELECT|INSERT|UPDATE|DELETE|WITH)\s', sql, re.I):
                        return sql

        # 方式 2: 从 response 文本中用正则提取
        resp_text = response.get("response", "")
        # 匹配 SQL 代码块
        sql_block = re.search(r'```sql\s*(.*?)\s*```', resp_text, re.I | re.DOTALL)
        if sql_block:
            return sql_block.group(1).strip()

        # 匹配 SELECT 开头的 SQL
        sql_match = re.search(
            r'(SELECT\s+.*?FROM\s+.*?(?:WHERE|GROUP|ORDER|HAVING|LIMIT|;|$))',
            resp_text, re.I | re.DOTALL
        )
        if sql_match:
            return sql_match.group(1).strip()

        return None


# ══════════════════════════════════════════════════════════════
# 汇总报告 (pytest_sessionfinish)
# ══════════════════════════════════════════════════════════════


def pytest_sessionfinish(session):
    """测试结束后生成汇总报告"""
    results = TestAgentFunctional.results
    if not results:
        return

    n = len(results)
    ex_ok = sum(1 for r in results if r.get("execution_accuracy"))
    sql_ok = sum(1 for r in results if r.get("sql_exec_ok"))
    gold_ok = sum(1 for r in results if r.get("gold_sql_ok"))
    intent_ok = sum(1 for r in results if r.get("intent_correct"))
    avg_latency = sum(r.get("latency_ms", 0) for r in results) / n

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = {
        "source": "bird_financial",
        "total": n,
        "execution_accuracy": round(ex_ok / n, 4),
        "sql_execution_success_rate": round(sql_ok / n, 4),
        "gold_sql_execution_rate": round(gold_ok / n, 4),
        "intent_accuracy": round(intent_ok / n, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "details": [
            {
                "question_id": r["question_id"],
                "question": r["question"][:80],
                "difficulty": r.get("difficulty", ""),
                "execution_accuracy": r.get("execution_accuracy", False),
                "intent_correct": r.get("intent_correct", False),
                "latency_ms": round(r.get("latency_ms", 0), 1),
                "error": r.get("error", "")[:100],
            }
            for r in results
        ],
    }

    summary_path = os.path.join(OUTPUT_DIR, "agent_functional_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印表格
    _print_table(summary)


def _print_table(summary: dict):
    """打印汇总表格"""
    m = summary
    logger.info("\n" + "=" * 55)
    logger.info("DeepEval Agent 功能正确性评测报告")
    logger.info("=" * 55)
    logger.info("数据集: %s  |  总用例: %d", m["source"], m["total"])
    logger.info("-" * 55)
    logger.info("  %-30s %s", "指标", "得分")
    logger.info("-" * 55)
    logger.info("  %-30s %.1f%%", "Execution Accuracy (EX)", m["execution_accuracy"] * 100)
    logger.info("  %-30s %.1f%%", "SQL 生成执行成功率", m["sql_execution_success_rate"] * 100)
    logger.info("  %-30s %.1f%%", "Gold SQL 执行成功率", m["gold_sql_execution_rate"] * 100)
    logger.info("  %-30s %.1f%%", "意图识别准确率", m["intent_accuracy"] * 100)
    logger.info("-" * 55)
    logger.info("  %-30s %.0f ms", "平均响应延迟", m["avg_latency_ms"])
    logger.info("=" * 55)
    logger.info("\n详情已导出: data/deepeval_results/agent_functional_summary.json")


# ══════════════════════════════════════════════════════════════
# 快捷入口
# ══════════════════════════════════════════════════════════════


if __name__ == "__main__":
    # 直接运行模式 (不走 pytest, 方便调试)
    logging.basicConfig(level=logging.INFO)

    cases = FINANCIAL_CASES

    if not cases:
        logger.error("未找到 financial 测试用例")
        sys.exit(1)

    logger.info("直接模式: %d 条 financial 测试用例", len(cases))

    tester = TestAgentFunctional()
    results = []
    for i, case in enumerate(cases):
        logger.info("[%d/%d] %s", i + 1, len(cases), case.question[:60])
        result = tester._run_single_test(case)
        results.append(result)
        status = "✓" if result["execution_accuracy"] else "✗"
        logger.info("  %s EX=%s (%.0fms) %s",
                     status, result["execution_accuracy"],
                     result.get("latency_ms", 0),
                     result.get("error", "")[:60])

    TestAgentFunctional.results = results
    _print_table({
        "source": "bird_financial",
        "total": len(results),
        "execution_accuracy": sum(1 for r in results if r["execution_accuracy"]) / len(results),
        "sql_execution_success_rate": sum(1 for r in results if r["sql_exec_ok"]) / len(results),
        "gold_sql_execution_rate": sum(1 for r in results if r.get("gold_sql_ok", False)) / len(results),
        "intent_accuracy": sum(1 for r in results if r.get("intent_correct", False)) / len(results),
        "avg_latency_ms": sum(r.get("latency_ms", 0) for r in results) / len(results),
    })