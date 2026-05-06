"""
ReAct + Reflect 端到端测评脚本（L2 + L3）

依赖：MySQL（已 source init_test_data.sql）+ Milvus（已 rebuild_rag_index）+ Redis + LLM API。

用法：
  python scripts/evaluate_react.py
  python scripts/evaluate_react.py --only l2
  python scripts/evaluate_react.py --only l3
  python scripts/evaluate_react.py --output scripts/react_results.json

测评维度：
  L2 - ReAct 健壮性（针对 P1-A：structured output 改造）
       关注：_think 是否稳定输出合法 ReactStep、参数错误兜底是否触发、最终状态
  L3 - 反思路径（针对 P1-B：reflect_error 结构化 + 主流程分支消费）
       关注：next_action 分类是否正确、主流程是否走对分支
"""
import asyncio
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("anyio").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ====================================================================
# 测试 case 设计
# ====================================================================

# L2：会触发 ComplexQuerySkill ReAct 循环的复杂查询
# 数据基于 init_test_data.sql 设计，确保查询有非空结果
L2_CASES: List[Dict[str, Any]] = [
    {
        "id": "L2-01",
        "input": "对比研发部和市场部的考勤情况",
        "expected_skill": "complex_query",
        "expected_status": ["success"],
        "note": "多对象对比，需要 ReAct 拆解为分别查再对比",
    },
    {
        "id": "L2-02",
        "input": "分析最近5天每个部门的迟到次数趋势",
        "expected_skill": "complex_query",
        "expected_status": ["success"],
        "note": "趋势分析 + 多步聚合",
    },
    {
        "id": "L2-03",
        "input": "查询打卡时间晚于9点的所有员工和他们所在部门",
        "expected_skill": "complex_query",
        "expected_status": ["success"],
        "note": "条件筛选 + 跨表 JOIN",
    },
    {
        "id": "L2-04",
        "input": "找出最近5天内既迟到又有加班记录的员工",
        "expected_skill": "complex_query",
        "expected_status": ["success"],
        "note": "嵌套筛选（既...又），需要交集",
    },
    {
        "id": "L2-05",
        "input": "查询每个部门使用各设备的打卡次数排名",
        "expected_skill": "complex_query",
        "expected_status": ["success"],
        "note": "多维聚合 + 排序",
    },
    {
        "id": "L2-06",
        "input": "对比研发部和运营部的人员构成",
        "expected_skill": "complex_query",
        "expected_status": ["success"],
        "note": "多部门对比",
    },
    {
        "id": "L2-07",
        "input": "查询2026-05-04到2026-05-05两天内打卡次数最多的前5名员工",
        "expected_skill": "complex_query",
        "expected_status": ["success"],
        "note": "时间范围 + 排名",
    },
]

# L3：故意制造错误，验证 reflect_error 的 next_action 分类 + 主流程消费
L3_CASES: List[Dict[str, Any]] = [
    {
        "id": "L3-01",
        "input": "查询 fake_attendance 表的最近5条记录",
        "trigger_type": "table_not_found",
        "expected_next_action": ["need_more_schema", "regenerate_sql"],
        "expected_log_pattern": ["🔍 反思建议补搜 schema", "📚 反思建议补搜历史示例"],
        "expected_log_optional": True,
        "note": "引用不存在的表，期望 need_more_schema 信号",
    },
    {
        "id": "L3-02",
        "input": "查一下数据",
        "trigger_type": "ambiguous_question",
        "expected_next_action": ["ask_user"],
        "expected_log_pattern": ["🤔 反思建议向用户澄清"],
        "expected_log_optional": True,
        "expected_clarification_needed": True,
        "note": "极度模糊的问题，期望 ask_user 提前中止",
    },
    {
        "id": "L3-03",
        "input": "查询所有员工的 fake_column 字段",
        "trigger_type": "column_not_found",
        "expected_next_action": ["need_more_schema", "regenerate_sql"],
        "expected_log_pattern": ["🔍 反思建议补搜 schema"],
        "expected_log_optional": True,
        "note": "引用不存在的列，期望 need_more_schema 或 regenerate_sql",
    },
    # unrecoverable 需要破坏环境（断 DB 连接），不适合自动化，跳过
]


# ====================================================================
# 日志捕获工具
# ====================================================================

class LogCapture:
    """临时把 root logger 的输出收集到内存 buffer，用于检查关键信号是否打印"""

    def __init__(self, level: int = logging.DEBUG):
        self.buffer = StringIO()
        self.handler = logging.StreamHandler(self.buffer)
        self.handler.setLevel(level)
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.level = level
        self._old_level: Optional[int] = None

    def __enter__(self):
        root = logging.getLogger()
        self._old_level = root.level
        root.setLevel(self.level)
        root.addHandler(self.handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        root = logging.getLogger()
        root.removeHandler(self.handler)
        if self._old_level is not None:
            root.setLevel(self._old_level)

    @property
    def text(self) -> str:
        return self.buffer.getvalue()

    def contains_any(self, patterns: List[str]) -> Dict[str, bool]:
        text = self.text
        return {p: (p in text) for p in patterns}


# ====================================================================
# L2 评测：跑一条 case 收集 ReAct 健壮性指标
# ====================================================================

def _build_skill_with_no_mcp(skill_cls):
    """构造一个剥离 MCP 工具的 Skill 实例。

    评测脚本场景下，MCP stdio 子进程 + pydantic-ai run_stream 的 anyio
    TaskGroup 两层嵌套会触发 'cancel scope in different task' 兼容性错误，
    生产 fastapi 长 event loop 不会有此问题。这里在评测时把 mcp_* 工具
    剥离，让 LLM 选不到 MCP 工具，从而隔离掉这个无关 bug。
    """
    skill = skill_cls()
    skill.tools = [t for t in skill.tools if not t.name.startswith("mcp_")]
    return skill


async def run_l2_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """跑一条 L2 case，输出健壮性指标"""
    from app.agent.intent import IntentNode
    from app.agent.skills.simple_query import SimpleQuerySkill
    from app.agent.skills.complex_query import ComplexQuerySkill
    from app.agent.skills.data_chat import DataChatSkill
    from app.agent.skills.selector import SkillSelector
    from app.agent.state import AgentState

    state = AgentState(user_input=case["input"], user_id="eval_l2")
    intent_node = IntentNode()
    selector = SkillSelector([
        _build_skill_with_no_mcp(SimpleQuerySkill),
        _build_skill_with_no_mcp(ComplexQuerySkill),
        DataChatSkill(),
    ])

    start = time.time()
    error_msg: Optional[str] = None

    with LogCapture() as cap:
        try:
            state = await intent_node.execute(state)
            skill = await selector.select(state)
            actual_skill = skill.name if skill else None

            if actual_skill == "complex_query":
                state = await skill.execute(state)
            else:
                # 没路由到 complex_query，直接收集，后面分析
                pass
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"

    elapsed = time.time() - start

    # 提取 ReAct trace 指标
    trace = state.react_trace or []
    empty_thought = sum(1 for s in trace if not (s.get("thought") or "").strip())
    empty_params_with_action = sum(
        1 for s in trace
        if s.get("action") not in ("done", "ask_user", "")
        and not s.get("parameters")
    )
    tool_not_found = sum(
        1 for s in trace
        if "工具" in (s.get("observation") or "") and "不存在" in (s.get("observation") or "")
    )
    param_error = sum(
        1 for s in trace if "参数错误" in (s.get("observation") or "")
    )

    # think_failed 通过日志匹配（_think 失败时会打印这条）
    log_text = cap.text
    think_failed = log_text.count("❌ Think 推理失败")

    return {
        "id": case["id"],
        "input": case["input"],
        "expected_skill": case["expected_skill"],
        "actual_skill": actual_skill,
        "expected_status": case["expected_status"],
        "actual_status": state.query_status,
        "react_steps": len(trace),
        "metrics": {
            "think_failed": think_failed,
            "empty_thought": empty_thought,
            "empty_params_with_action": empty_params_with_action,
            "tool_not_found": tool_not_found,
            "param_error": param_error,
        },
        "skill_correct": actual_skill == case["expected_skill"],
        "status_acceptable": state.query_status in case["expected_status"],
        "passed": (
            actual_skill == case["expected_skill"]
            and state.query_status in case["expected_status"]
            and think_failed == 0
            and empty_thought == 0
            and empty_params_with_action == 0
        ),
        "elapsed_sec": round(elapsed, 2),
        "response_preview": (state.response or "")[:200],
        "error": error_msg,
    }


# ====================================================================
# L3 评测：跑一条 case 收集反思路径指标
# ====================================================================

async def run_l3_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """跑一条 L3 case，输出反思分类 + 主流程消费指标"""
    from app.agent.intent import IntentNode
    from app.agent.skills.simple_query import SimpleQuerySkill
    from app.agent.skills.complex_query import ComplexQuerySkill
    from app.agent.skills.data_chat import DataChatSkill
    from app.agent.skills.selector import SkillSelector
    from app.agent.state import AgentState

    state = AgentState(user_input=case["input"], user_id="eval_l3")
    intent_node = IntentNode()
    selector = SkillSelector([
        _build_skill_with_no_mcp(SimpleQuerySkill),
        _build_skill_with_no_mcp(ComplexQuerySkill),
        DataChatSkill(),
    ])

    start = time.time()
    error_msg: Optional[str] = None

    with LogCapture() as cap:
        try:
            state = await intent_node.execute(state)
            skill = await selector.select(state)
            actual_skill = skill.name if skill else None
            if actual_skill in ("simple_query", "complex_query"):
                state = await skill.execute(state)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"

    elapsed = time.time() - start

    # 提取反思结果
    reflection = state.reflection if isinstance(state.reflection, dict) else {}
    actual_next_action = reflection.get("next_action")
    next_action_correct = (
        actual_next_action in case["expected_next_action"]
        if actual_next_action else False
    )

    # 检查关键日志是否打印（主流程消费的关键标志）
    log_hits = cap.contains_any(case.get("expected_log_pattern", []))
    log_consumed = any(log_hits.values()) if log_hits else None

    # 检查 clarification 状态（如果 case 期望走 ask_user 分支）
    clarification_correct: Optional[bool] = None
    if "expected_clarification_needed" in case:
        clarification_correct = state.clarification_needed == case["expected_clarification_needed"]

    return {
        "id": case["id"],
        "input": case["input"],
        "trigger_type": case["trigger_type"],
        "expected_next_action": case["expected_next_action"],
        "actual_next_action": actual_next_action,
        "next_action_correct": next_action_correct,
        "expected_log_patterns": case.get("expected_log_pattern", []),
        "log_hits": log_hits,
        "log_consumed": log_consumed,
        "clarification_correct": clarification_correct,
        "final_status": state.query_status,
        "clarification_needed": state.clarification_needed,
        "reflection": {
            "root_cause": reflection.get("root_cause"),
            "next_action": actual_next_action,
            "fix_hint": (reflection.get("fix_hint") or "")[:150],
            "schema_keywords": reflection.get("schema_keywords"),
        } if reflection else None,
        "passed": next_action_correct and (log_consumed is None or log_consumed) and (
            clarification_correct is None or clarification_correct
        ),
        "elapsed_sec": round(elapsed, 2),
        "response_preview": (state.response or "")[:200],
        "error": error_msg,
    }


# ====================================================================
# 报告生成
# ====================================================================

def print_l2_report(results: List[Dict[str, Any]]):
    print("\n" + "=" * 70)
    print("L2 ReAct 健壮性测评（针对 P1-A: structured output 改造）")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    skill_correct = sum(1 for r in results if r["skill_correct"])
    status_ok = sum(1 for r in results if r["status_acceptable"])

    total_steps = sum(r["react_steps"] for r in results)
    total_think_failed = sum(r["metrics"]["think_failed"] for r in results)
    total_empty_thought = sum(r["metrics"]["empty_thought"] for r in results)
    total_empty_params = sum(r["metrics"]["empty_params_with_action"] for r in results)
    total_tool_nf = sum(r["metrics"]["tool_not_found"] for r in results)
    total_param_err = sum(r["metrics"]["param_error"] for r in results)

    print(f"\n[总体]")
    print(f"  通过: {passed}/{total}")
    print(f"  路由正确: {skill_correct}/{total}")
    print(f"  最终状态可接受: {status_ok}/{total}")

    print(f"\n[ReAct 健壮性核心指标]")
    print(f"  累计 ReAct 步数: {total_steps}")
    print(f"  ❌ Think 失败次数: {total_think_failed}     (P1-A 核心指标，期望 0)")
    print(f"  ❌ thought 为空步数: {total_empty_thought}      (期望 0)")
    print(f"  ❌ action 非 done 但参数为空: {total_empty_params}    (期望 0)")
    print(f"  ⚠️  工具不存在次数: {total_tool_nf}          (兜底机制可触发，<=1/case 可接受)")
    print(f"  ⚠️  参数错误兜底次数: {total_param_err}       (兜底机制可触发，<=1/case 可接受)")

    print(f"\n[详细]")
    for r in results:
        status_icon = "✅" if r["passed"] else "❌"
        print(f"  {status_icon} {r['id']} | steps={r['react_steps']:>2} | "
              f"think_fail={r['metrics']['think_failed']} | "
              f"empty_thought={r['metrics']['empty_thought']} | "
              f"final={r['actual_status']} | {r['elapsed_sec']}s")
        if not r["passed"]:
            print(f"      input: {r['input']}")
            if r["error"]:
                print(f"      error: {r['error']}")


def print_l3_report(results: List[Dict[str, Any]]):
    print("\n" + "=" * 70)
    print("L3 反思路径测评（针对 P1-B: reflect_error 结构化 + 主流程消费）")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    next_action_ok = sum(1 for r in results if r["next_action_correct"])

    print(f"\n[总体]")
    print(f"  通过: {passed}/{total}")
    print(f"  next_action 分类正确: {next_action_ok}/{total}")

    print(f"\n[详细]")
    for r in results:
        status_icon = "✅" if r["passed"] else "❌"
        print(f"  {status_icon} {r['id']} | trigger={r['trigger_type']}")
        print(f"      input: {r['input']}")
        print(f"      expected next_action: {r['expected_next_action']}")
        print(f"      actual next_action:   {r['actual_next_action']}")
        if r["log_hits"]:
            for pattern, hit in r["log_hits"].items():
                icon = "✓" if hit else "✗"
                print(f"      [{icon}] log: '{pattern}'")
        if r["clarification_correct"] is not None:
            icon = "✓" if r["clarification_correct"] else "✗"
            print(f"      [{icon}] clarification_needed={r['clarification_needed']}")
        if r["reflection"]:
            print(f"      reflection.root_cause: {r['reflection']['root_cause']}")
            print(f"      reflection.fix_hint:   {r['reflection']['fix_hint']}")
            if r["reflection"]["schema_keywords"]:
                print(f"      reflection.schema_keywords: {r['reflection']['schema_keywords']}")
        if r["error"]:
            print(f"      ⚠️ error: {r['error']}")
        print()


# ====================================================================
# 主流程
# ====================================================================

async def main():
    parser = argparse.ArgumentParser(description="ReAct + Reflect 测评")
    parser.add_argument("--only", choices=["l2", "l3"], help="只跑某一层（默认全跑）")
    parser.add_argument("--output", default=None, help="结果 JSON 输出路径")
    args = parser.parse_args()

    # 配置 logger，让 LogCapture 能捕获到
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    l2_results: List[Dict[str, Any]] = []
    l3_results: List[Dict[str, Any]] = []

    if args.only != "l3":
        print("\n" + "=" * 70)
        print(f"开始 L2 测评：{len(L2_CASES)} 条 case")
        print("=" * 70)
        for i, case in enumerate(L2_CASES, 1):
            print(f"\n[L2 {i}/{len(L2_CASES)}] {case['id']}: {case['input']}")
            try:
                result = await run_l2_case(case)
                l2_results.append(result)
                icon = "✅" if result["passed"] else "❌"
                print(f"  {icon} steps={result['react_steps']} | "
                      f"final={result['actual_status']} | {result['elapsed_sec']}s")
            except Exception as e:
                print(f"  ❌ 跑测异常：{type(e).__name__}: {e}")
                l2_results.append({"id": case["id"], "input": case["input"],
                                   "error": str(e), "passed": False})

    if args.only != "l2":
        print("\n" + "=" * 70)
        print(f"开始 L3 测评：{len(L3_CASES)} 条 case")
        print("=" * 70)
        for i, case in enumerate(L3_CASES, 1):
            print(f"\n[L3 {i}/{len(L3_CASES)}] {case['id']}: {case['input']}")
            try:
                result = await run_l3_case(case)
                l3_results.append(result)
                icon = "✅" if result["passed"] else "❌"
                print(f"  {icon} next_action={result['actual_next_action']} | "
                      f"final={result['final_status']} | {result['elapsed_sec']}s")
            except Exception as e:
                print(f"  ❌ 跑测异常：{type(e).__name__}: {e}")
                l3_results.append({"id": case["id"], "input": case["input"],
                                   "error": str(e), "passed": False})

    # 报告
    if l2_results:
        print_l2_report(l2_results)
    if l3_results:
        print_l3_report(l3_results)

    # 保存 JSON
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"scripts/react_results_{ts}.json"

    report = {
        "timestamp": datetime.now().isoformat(),
        "l2": {
            "total": len(l2_results),
            "passed": sum(1 for r in l2_results if r.get("passed")),
            "results": l2_results,
        },
        "l3": {
            "total": len(l3_results),
            "passed": sum(1 for r in l3_results if r.get("passed")),
            "results": l3_results,
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📄 详细报告已保存：{args.output}")


if __name__ == "__main__":
    asyncio.run(main())
