"""
SQLAgent —— SQL 生成与执行 Agent

架构（Prompt-based ReAct 版本）：
    START → sql_react(纯文本 LLM 调用) → _parse_tool_call(解析文本中的JSON) → has_tool_call?
                                          ↓ yes                              ↓ no
                                    _exec_tools_node(手动执行工具)      output_node → END
                                          ↓
                                      sql_react (循环)

内部 ReAct 循环，工具集：generate_sql + validate_sql + execute_sql + reflect_error
最大步数：8

注意：本模块使用 Prompt-based ReAct 模式，不依赖 LLM 的 function calling / bind_tools 能力。
"""

import json
import logging
import re
import inspect
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END, START, StateGraph

from app.core.llm import get_llm
from app.agent.state import AgentState
from app.agent.tools.langchain_tools import (
    generate_sql,
    validate_sql,
    execute_sql,
    reflect_error,
)
from prompts.agents.sql import build_sql_agent_message

logger = logging.getLogger(__name__)

SQL_AGENT_TOOLS = [generate_sql, validate_sql, execute_sql, reflect_error]
MAX_SQL_STEPS = 15

# 工具名称 → 函数映射（用于 prompt-based ReAct 模式下的手动工具执行）
TOOL_MAP = {
    "generate_sql": generate_sql,
    "validate_sql": validate_sql,
    "execute_sql": execute_sql,
    "reflect_error": reflect_error,
}


def _build_tools_description() -> str:
    """生成工具描述文本"""
    lines = []
    for t in SQL_AGENT_TOOLS:
        desc = t.description.split("\n")[0] if t.description else ""
        lines.append(f"- {t.name}: {desc}")
    return "\n".join(lines)


async def _run_tool(tool_func, args: dict):
    """Run LangChain StructuredTool or a plain async/sync callable."""
    if hasattr(tool_func, "ainvoke"):
        return await tool_func.ainvoke(args)
    result = tool_func(**args)
    if inspect.isawaitable(result):
        return await result
    return result


def _parse_tool_call(text: str) -> dict | None:
    """
    从 LLM 文本输出中解析工具调用。

    支持两种格式：
    1. 纯 JSON: {"tool": "generate_sql", "args": {"question": "..."}}
    2. markdown 代码块: ```json\n{...}\n```

    Returns:
        {"name": str, "args": dict} 或 None（表示无工具调用，是最终回答）
    """
    text = text.strip()

    # 尝试直接解析 JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "tool" in obj:
            return {"name": obj["tool"], "args": obj.get("args", {})}
    except (json.JSONDecodeError, TypeError):
        pass

    # 尝试提取 markdown json 代码块
    json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if json_block_match:
        try:
            obj = json.loads(json_block_match.group(1).strip())
            if isinstance(obj, dict) and "tool" in obj:
                return {"name": obj["tool"], "args": obj.get("args", {})}
        except (json.JSONDecodeError, TypeError):
            pass

    # 尝试匹配行内 JSON 模式
    inline_match = re.search(r'\{[^{}]*"tool"\s*:\s*"[^"]+"[^{}]*\}', text)
    if inline_match:
        try:
            obj = json.loads(inline_match.group(0))
            if isinstance(obj, dict) and "tool" in obj:
                return {"name": obj["tool"], "args": obj.get("args", {})}
        except (json.JSONDecodeError, TypeError):
            pass

    return None  # 无工具调用


async def sql_react(state: AgentState) -> dict:
    """SQLAgent ReAct 推理节点（Prompt-based，不依赖 function calling）"""
    messages = list(state.get("messages", []))
    user_input = state.get("user_input", "")

    # 从 agent_results 获取 SchemaAgent 输出的表结构
    agent_results = state.get("agent_results", {})
    schema_context = agent_results.get("schema", "")

    # 构建 system message（已更新为 prompt-based 格式）
    tools_desc = _build_tools_description()
    system_content = build_sql_agent_message(
        tools_description=tools_desc,
        max_steps=MAX_SQL_STEPS,
        user_input=user_input,
        schema_context=schema_context or "（未提供表结构信息）",
    )

    # 组装消息（不使用 bind_tools）
    full_messages = [SystemMessage(content=system_content)]
    if messages:
        for msg in messages:
            if not isinstance(msg, SystemMessage):
                full_messages.append(msg)
    else:
        if user_input:
            full_messages.append(HumanMessage(content=user_input))

    # 纯文本调用 LLM（不绑定工具）
    llm = get_llm(timeout_key='sql_generation')
    response = await llm.ainvoke(full_messages)
    ai_content = response.content

    step_count = state.get("react_step_count", 0) + 1

    # 解析是否有工具调用
    tool_call = _parse_tool_call(ai_content)

    pending_tool_call = None
    if tool_call:
        # 有工具调用 → 只记录到内部状态，不使用 LangChain tool_calls 协议
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        pending_tool_call = {
            "id": f"call_{step_count}_{tool_name}",
            "name": tool_name,
            "args": tool_args,
        }

        logger.info(f"🔧 SQLAgent 解析到工具调用: {tool_name}({list(tool_args.keys())})")

        ai_message = AIMessage(content=ai_content)
    else:
        # 无工具调用 → 最终回答
        ai_message = AIMessage(content=ai_content)

    return {
        "messages": messages + [ai_message],
        "react_step_count": step_count,
        "pending_tool_call": pending_tool_call,
    }


def should_continue(state: AgentState) -> Literal["tools", "output"]:
    """条件边：继续调用工具还是结束"""
    messages = state.get("messages", [])
    step_count = state.get("react_step_count", 0)

    if step_count >= MAX_SQL_STEPS:
        logger.info(f"SQLAgent 达到最大步数 {MAX_SQL_STEPS}，终止循环")
        return "output"

    if not messages:
        return "output"

    if state.get("pending_tool_call"):
        return "tools"

    # AIMessage 无 tool_calls → 结束
    return "output"


async def output_node(state: AgentState) -> dict:
    """输出节点：从 messages 提取 SQL 文本和执行结果"""
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")

    executed_sql = ""
    query_data = ""
    error_info = ""

    for result in state.get("tool_results", []):
        name = result.get("name", "")
        args = result.get("args", {})
        content = result.get("content", "")
        if name == "execute_sql" and args.get("sql"):
            executed_sql = args["sql"]
        if name == "generate_sql" and not executed_sql:
            executed_sql = content.strip()
        if name == "execute_sql":
            query_data = content
        if name == "reflect_error":
            error_info = content

    # 写入 agent_results
    agent_results = dict(state.get("agent_results", {}))
    agent_results["sql"] = {
        "executed_sql": executed_sql,
        "query_data": query_data,
        "error": error_info,
    }

    logger.info(f"🔧 SQLAgent 输出：SQL={len(executed_sql)} 字符, 数据={len(query_data)} 字符")

    return {
        "agent_results": agent_results,
        "sql_query": executed_sql,
        "current_agent": "sql_agent_done",
    }


async def _exec_tools_node(state: AgentState) -> dict:
    """
    手动执行工具节点（替代 LangChain 的 ToolNode）

    从内部状态读取待执行工具，执行后用普通文本 Observation 回灌，
    避免产生 tool_calls 字段或 tool role 消息。
    """
    messages = list(state.get("messages", []))
    tool_call = state.get("pending_tool_call")

    if not tool_call:
        return {"messages": messages, "pending_tool_call": None}

    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})

    logger.info(f"🔨 执行工具: {tool_name}")

    args = tool_args if isinstance(tool_args, dict) else {}
    tool_func = TOOL_MAP.get(tool_name)

    if tool_func:
        try:
            result = await _run_tool(tool_func, args)
            content = result if isinstance(result, str) else str(result)
        except Exception as e:
            logger.error(f"❌ 工具 {tool_name} 执行失败: {e}")
            content = f"工具执行失败: {e}"
        logger.info(f"✅ 工具 {tool_name} 完成: {len(content)} 字符")
    else:
        logger.error(f"❌ 未知工具: {tool_name}")
        content = f'错误：未知工具 "{tool_name}"'

    observation = HumanMessage(
        content=(
            f"Observation:\n"
            f"工具: {tool_name}\n"
            f"参数: {json.dumps(args, ensure_ascii=False)}\n"
            f"结果:\n{content}"
        )
    )
    tool_results = list(state.get("tool_results", []))
    tool_results.append({
        "id": tool_call.get("id", ""),
        "name": tool_name,
        "args": args,
        "content": content,
    })

    return {
        "messages": messages + [observation],
        "pending_tool_call": None,
        "tool_results": tool_results,
    }


def build_sql_agent_graph():
    """构建 SQLAgent 子图（Prompt-based ReAct 版本）"""
    graph = StateGraph(AgentState)

    graph.add_node("sql_react", sql_react)
    graph.add_node("tools", _exec_tools_node)   # ← 用自定义节点替代 ToolNode
    graph.add_node("output", output_node)

    graph.add_edge(START, "sql_react")
    graph.add_conditional_edges("sql_react", should_continue)
    graph.add_edge("tools", "sql_react")
    graph.add_edge("output", END)

    return graph.compile()


_sql_agent_graph = None


def get_sql_agent_graph():
    """获取 SQLAgent 子图单例"""
    global _sql_agent_graph
    if _sql_agent_graph is None:
        _sql_agent_graph = build_sql_agent_graph()
    return _sql_agent_graph
