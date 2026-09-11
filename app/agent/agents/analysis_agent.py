"""
AnalysisAgent —— 结果格式化 Agent

架构（Prompt-based ReAct 版本）：
    START → analysis_react(纯文本 LLM 调用) → _parse_tool_call(解析文本中的JSON) → has_tool_call?
                                          ↓ yes                              ↓ no
                                    _exec_tools_node(手动执行工具)      output_node → END
                                          ↓
                                      analysis_react (循环)

内部 ReAct 循环，工具集：format_result + get_date_range + get_workdays + get_current_time
最大步数：5

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
    format_result,
    get_date_range,
    get_workdays,
    get_current_time,
)
from prompts.agents.analysis import build_analysis_agent_message

logger = logging.getLogger(__name__)

ANALYSIS_AGENT_TOOLS = [format_result, get_date_range, get_workdays, get_current_time]
MAX_ANALYSIS_STEPS = 5

# 工具名称 → 函数映射（用于 prompt-based ReAct 模式下的手动工具执行）
TOOL_MAP = {
    "format_result": format_result,
    "get_date_range": get_date_range,
    "get_workdays": get_workdays,
    "get_current_time": get_current_time,
}


def _build_tools_description() -> str:
    """生成工具描述文本"""
    lines = []
    for t in ANALYSIS_AGENT_TOOLS:
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
    1. 纯 JSON: {"tool": "format_result", "args": {"data": "..."}}
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


def _strip_internal_process(text: str) -> str:
    """Remove leaked ReAct/planning content from a model fallback answer."""
    if not text:
        return ""

    stop_markers = (
        "为了回答您的问题",
        "我需要执行以下步骤",
        "根据规则",
        "实际执行的 SQL",
        "修正理解",
        "策略调整",
        "Thought:",
        "Action:",
        "Observation:",
        "```json",
        '{"tool"',
    )
    if any(marker in text for marker in stop_markers):
        return ""
    return text.strip()


async def analysis_react(state: AgentState) -> dict:
    """AnalysisAgent ReAct 推理节点（Prompt-based，不依赖 function calling）"""
    messages = list(state.get("messages", []))
    user_input = state.get("user_input", "")

    # 从 agent_results 获取 SQLAgent 输出
    agent_results = state.get("agent_results", {})
    sql_result_data = agent_results.get("sql", {})

    if isinstance(sql_result_data, dict):
        sql_result = sql_result_data.get("query_data", "")
        executed_sql = sql_result_data.get("executed_sql", "")
    else:
        sql_result = str(sql_result_data)
        executed_sql = ""

    # 构建 system message
    tools_desc = _build_tools_description()
    system_content = build_analysis_agent_message(
        tools_description=tools_desc,
        max_steps=MAX_ANALYSIS_STEPS,
        user_input=user_input,
        sql_result=sql_result or "（无查询结果）",
        executed_sql=executed_sql or "（未执行SQL）",
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
    llm = get_llm(timeout_key='chat')
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

        logger.info(f"🔧 AnalysisAgent 解析到工具调用: {tool_name}({list(tool_args.keys())})")

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

    if step_count >= MAX_ANALYSIS_STEPS:
        logger.info(f"AnalysisAgent 达到最大步数 {MAX_ANALYSIS_STEPS}，终止循环")
        return "output"

    if not messages:
        return "output"

    if state.get("pending_tool_call"):
        return "tools"

    return "output"


async def output_node(state: AgentState) -> dict:
    """输出节点：从 messages 提取最终格式化回答"""
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")
    agent_results = state.get("agent_results", {})
    sql_result_data = agent_results.get("sql", {})
    query_data = ""

    if isinstance(sql_result_data, dict):
        query_data = sql_result_data.get("query_data", "") or ""
    elif sql_result_data:
        query_data = str(sql_result_data)

    # 检查是否有 format_result 工具调用的结果
    for result in reversed(state.get("tool_results", [])):
        if result.get("name") == "format_result":
            formatted = result.get("content", "")
            if formatted:
                logger.info(f"📊 AnalysisAgent 输出（format_result）：{len(formatted)} 字符")
                return {
                    "response": formatted.strip(),
                    "skill_name": "sql_react",
                    "query_status": "success",
                    "current_agent": "analysis_agent_done",
                }

    # 防止模型把思考过程当最终回答：有查询结果时直接调用格式化工具。
    if query_data and query_data != "（无查询结果）":
        try:
            formatted = await _run_tool(format_result, {
                "question": user_input,
                "query_result": query_data,
            })
            response = formatted if isinstance(formatted, str) else str(formatted)
            if response.strip():
                logger.info(f"📊 AnalysisAgent 输出（fallback format_result）：{len(response)} 字符")
                return {
                    "response": response.strip(),
                    "skill_name": "sql_react",
                    "query_status": "success",
                    "current_agent": "analysis_agent_done",
                }
        except Exception as e:
            logger.error(f"❌ AnalysisAgent fallback format_result 失败: {e}")

    # 从最后一条 AIMessage 获取回答
    last_ai_content = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            last_ai_content = msg.content
            break

    if last_ai_content:
        response = _strip_internal_process(last_ai_content)
    else:
        response = ""

    if not response:
        response = "查询无数据" if not query_data else "抱歉，无法完成结果格式化。"

    logger.info(f"📊 AnalysisAgent 输出：{len(response)} 字符")

    return {
        "response": response,
        "skill_name": "sql_react",
        "query_status": "success",
        "current_agent": "analysis_agent_done",
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


def build_analysis_agent_graph():
    """构建 AnalysisAgent 子图（Prompt-based ReAct 版本）"""
    graph = StateGraph(AgentState)

    graph.add_node("analysis_react", analysis_react)
    graph.add_node("tools", _exec_tools_node)   # ← 用自定义节点替代 ToolNode
    graph.add_node("output", output_node)

    graph.add_edge(START, "analysis_react")
    graph.add_conditional_edges("analysis_react", should_continue)
    graph.add_edge("tools", "analysis_react")
    graph.add_edge("output", END)

    return graph.compile()


_analysis_agent_graph = None


def get_analysis_agent_graph():
    """获取 AnalysisAgent 子图单例"""
    global _analysis_agent_graph
    if _analysis_agent_graph is None:
        _analysis_agent_graph = build_analysis_agent_graph()
    return _analysis_agent_graph
