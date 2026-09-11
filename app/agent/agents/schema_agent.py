"""
SchemaAgent —— 表结构检索 Agent

架构（Prompt-based ReAct 版本）：
    START → schema_react(纯文本 LLM 调用) → _parse_tool_call(解析文本中的JSON) → has_tool_call?
                                          ↓ yes                              ↓ no
                                    _exec_tools_node(手动执行工具)      output_node → END
                                          ↓
                                      schema_react (循环)

内部 ReAct 循环，工具集：search_schema + search_examples + search_memory
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
    search_schema,
    search_examples,
    search_memory,
)
from prompts.agents.schema import build_schema_agent_message

logger = logging.getLogger(__name__)

SCHEMA_AGENT_TOOLS = [search_schema, search_examples, search_memory]
MAX_SCHEMA_STEPS = 5

# 工具名称 → 函数映射（用于 prompt-based ReAct 模式下的手动工具执行）
TOOL_MAP = {
    "search_schema": search_schema,
    "search_examples": search_examples,
    "search_memory": search_memory,
}


def _build_tools_description() -> str:
    """生成工具描述文本"""
    lines = []
    for t in SCHEMA_AGENT_TOOLS:
        desc = t.description.split("\n")[0] if t.description else ""
        lines.append(f"- {t.name}: {desc}")
    return "\n".join(lines)


async def _run_tool(tool_func, args: dict):
    """运行 LangChain 的结构化工具（StructuredTool）或普通的异步/同步可调用对象"""
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
    1. 纯 JSON: {"tool": "search_schema", "args": {"question": "..."}}
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


async def schema_react(state: AgentState) -> dict:
    """SchemaAgent ReAct 推理节点（Prompt-based，不依赖 function calling）"""
    messages = list(state.get("messages", []))
    user_input = state.get("user_input", "")

    # 从 agent_results 获取上游传递的上下文（如有）
    agent_results = state.get("agent_results", {})

    # 收集记忆
    memory_texts_str = ""
    memories = state.get("memories")
    if memories and isinstance(memories, dict) and memories.get("results"):
        mem_list = []
        for mem in memories["results"][:3]:
            text = mem.get("memory", "") if isinstance(mem, dict) else ""
            if text:
                mem_list.append(text)
        if mem_list:
            memory_texts_str = "\n".join(mem_list)

    # 构建 system message
    tools_desc = _build_tools_description()
    system_content = build_schema_agent_message(
        tools_description=tools_desc,
        max_steps=MAX_SCHEMA_STEPS,
        user_input=user_input,
        memory_texts=memory_texts_str,
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

        logger.info(f"🔧 SchemaAgent 解析到工具调用: {tool_name}({list(tool_args.keys())})")

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
    
    if step_count >= MAX_SCHEMA_STEPS:
        logger.info(f"SchemaAgent 达到最大步数 {MAX_SCHEMA_STEPS}，终止循环")
        return "output"
    
    if not messages:
        return "output"
    
    if state.get("pending_tool_call"):
        return "tools"
    
    return "output"


async def output_node(state: AgentState) -> dict:
    """输出节点：从 messages 提取最终表结构文本"""
    messages = state.get("messages", [])
    
    # 从纯文本工具执行结果中收集所有检索结果
    schema_results = []
    for result in state.get("tool_results", []):
        name = result.get("name", "")
        if name in ("search_schema", "search_examples", "search_memory"):
            content = result.get("content", "")
            if content and "未找到" not in content and "失败" not in content:
                schema_results.append(content)
    
    # 从最后一条 AIMessage 获取总结
    last_ai_content = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            last_ai_content = msg.content
            break
    
    # 优先使用工具结果，其次使用 AI 总结
    if schema_results:
        schema_text = "\n\n".join(schema_results)
    elif last_ai_content:
        schema_text = last_ai_content
    else:
        schema_text = "未找到相关表结构信息"
    
    logger.info(f"📋 SchemaAgent 输出：{len(schema_text)} 字符")
    
    # 写入 agent_results
    agent_results = dict(state.get("agent_results", {}))
    agent_results["schema"] = schema_text
    
    return {
        "agent_results": agent_results,
        "current_agent": "schema_agent_done",
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


def build_schema_agent_graph():
    """构建 SchemaAgent 子图（Prompt-based ReAct 版本）"""
    graph = StateGraph(AgentState)

    graph.add_node("schema_react", schema_react)
    graph.add_node("tools", _exec_tools_node)   # ← 用自定义节点替代 ToolNode
    graph.add_node("output", output_node)

    graph.add_edge(START, "schema_react")
    graph.add_conditional_edges("schema_react", should_continue)
    graph.add_edge("tools", "schema_react")
    graph.add_edge("output", END)

    return graph.compile()


_schema_agent_graph = None


def get_schema_agent_graph():
    """获取 SchemaAgent 子图单例"""
    global _schema_agent_graph
    if _schema_agent_graph is None:
        _schema_agent_graph = build_schema_agent_graph()
    return _schema_agent_graph
