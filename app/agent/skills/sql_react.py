"""
SQL ReAct 子图 —— 多 Agent 架构

架构：
    START → coordinator_node → dispatch_node → should_continue_dispatch
                                    ↓ dispatch                ↓ end
                              dispatch_node                  END

Coordinator 拆解任务后，按序调度 SchemaAgent → SQLAgent → AnalysisAgent
各子 Agent 内部有独立 ReAct 循环。
"""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph

from app.core.llm import get_llm, get_structured_llm
from app.agent.state import AgentState
from app.agent.agents.schema_agent import get_schema_agent_graph
from app.agent.agents.sql_agent import get_sql_agent_graph
from app.agent.agents.analysis_agent import get_analysis_agent_graph
from app.events import EventType
from prompts.coordinator import build_coordinator_message, SubTaskList

logger = logging.getLogger(__name__)


# ============================================================
# Coordinator —— 多 Agent 调度器
# ============================================================

async def coordinator_node(state: AgentState) -> dict:
    """Coordinator 节点：任务拆解"""
    user_input = state.get("user_input", "")

    try:
        structured_llm = get_structured_llm(SubTaskList, tags=["skip_stream"])
        prompt = build_coordinator_message(user_input)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"请分析这个查询请求：{user_input}"),
        ]
        result = await structured_llm.ainvoke(messages)

        sub_tasks = []
        if result.needs_decomposition and result.sub_tasks:
            for st in result.sub_tasks:
                sub_tasks.append({
                    "step": st.step,
                    "description": st.description,
                    "depends_on": st.depends_on,
                })
            logger.info(f"📋 任务拆解为 {len(sub_tasks)} 个子任务：{[t['description'] for t in sub_tasks]}")
        else:
            logger.info(f"📋 无需拆解：{result.reason}")

        return {
            "sub_tasks": sub_tasks,
            "current_agent": "coordinator_done",
        }
    except Exception as e:
        logger.error(f"❌ 任务拆解失败：{e}，降级为不拆解")
        return {
            "sub_tasks": [],
            "current_agent": "coordinator_done",
        }


async def dispatch_node(state: AgentState) -> dict:
    """调度节点：按序执行 SchemaAgent → SQLAgent → AnalysisAgent"""
    user_input = state.get("user_input", "")
    agent_results = state.get("agent_results", {})
    current_agent = state.get("current_agent", "")

    config = get_config()
    emitter = config.get("configurable", {}).get("emitter")

    schema_done = "schema" in agent_results
    sql_done = "sql" in agent_results

    if not schema_done:
        # === 执行 SchemaAgent ===
        if emitter:
            emitter.emit(EventType.THINKING, {
                "phase": "schema_agent",
                "message": "检索表结构...",
            })
        logger.info("🔀 调度 SchemaAgent")

        schema_graph = get_schema_agent_graph()
        sub_state = dict(state)
        sub_state["react_step_count"] = 0
        sub_state["messages"] = []

        result = None
        async for event in schema_graph.astream_events(sub_state, config=config, version="v2"):
            kind = event.get("event")
            if kind == "on_chain_end":
                output = event.get("data", {}).get("output")
                if isinstance(output, dict) and "agent_results" in output:
                    result = output
            elif kind == "on_chat_model_stream":
                event_tags = event.get("tags", []) or []
                if "skip_stream" not in event_tags:
                    chunk = event.get("data", {}).get("chunk")
                    token = ""
                    if chunk and hasattr(chunk, "content"):
                        token = chunk.content if isinstance(chunk.content, str) else str(chunk.content or "")
                    if token and emitter:
                        emitter.emit_thinking_token(token)
        result = result or sub_state

        new_agent_results = dict(agent_results)
        new_agent_results.update(result.get("agent_results", {}))

        return {
            "agent_results": new_agent_results,
            "current_agent": "schema_agent",
            "react_step_count": result.get("react_step_count", 0),
        }

    elif not sql_done:
        # === 执行 SQLAgent ===
        if emitter:
            emitter.emit(EventType.THINKING, {
                "phase": "sql_agent",
                "message": "生成SQL并执行...",
            })
        logger.info("🔀 调度 SQLAgent")

        sql_graph = get_sql_agent_graph()
        sub_state = dict(state)
        sub_state["react_step_count"] = 0
        sub_state["messages"] = []

        result = None
        async for event in sql_graph.astream_events(sub_state, config=config, version="v2"):
            kind = event.get("event")
            if kind == "on_chain_end":
                output = event.get("data", {}).get("output")
                if isinstance(output, dict) and "agent_results" in output:
                    result = output
            elif kind == "on_chat_model_stream":
                event_tags = event.get("tags", []) or []
                if "skip_stream" not in event_tags:
                    chunk = event.get("data", {}).get("chunk")
                    token = ""
                    if chunk and hasattr(chunk, "content"):
                        token = chunk.content if isinstance(chunk.content, str) else str(chunk.content or "")
                    if token and emitter:
                        emitter.emit_thinking_token(token)
        result = result or sub_state

        new_agent_results = dict(agent_results)
        new_agent_results.update(result.get("agent_results", {}))

        return {
            "agent_results": new_agent_results,
            "sql_query": result.get("sql_query", ""),
            "current_agent": "sql_agent",
            "react_step_count": result.get("react_step_count", 0),
        }

    else:
        # === 执行 AnalysisAgent ===
        if emitter:
            emitter.emit(EventType.THINKING, {
                "phase": "analysis_agent",
                "message": "整理结果...",
            })
        logger.info("🔀 调度 AnalysisAgent")

        analysis_graph = get_analysis_agent_graph()
        sub_state = dict(state)
        sub_state["react_step_count"] = 0
        sub_state["messages"] = []

        result = None
        async for event in analysis_graph.astream_events(sub_state, config=config, version="v2"):
            kind = event.get("event")
            if kind == "on_chain_end":
                output = event.get("data", {}).get("output")
                if isinstance(output, dict) and "agent_results" in output:
                    result = output
            elif kind == "on_chat_model_stream":
                event_tags = event.get("tags", []) or []
                if "skip_stream" not in event_tags:
                    chunk = event.get("data", {}).get("chunk")
                    token = ""
                    if chunk and hasattr(chunk, "content"):
                        token = chunk.content if isinstance(chunk.content, str) else str(chunk.content or "")
                    if token and emitter:
                        emitter.emit_thinking_token(token)
        result = result or sub_state

        new_agent_results = dict(agent_results)
        new_agent_results.update(result.get("agent_results", {}))

        return {
            "agent_results": new_agent_results,
            "response": result.get("response", ""),
            "query_status": result.get("query_status", "success"),
            "skill_name": result.get("skill_name", "sql_react"),
            "current_agent": "analysis_agent",
            "react_step_count": result.get("react_step_count", 0),
        }


def should_continue_dispatch(state: AgentState) -> Literal["dispatch", "end"]:
    """条件边：判断是否还有 Agent 需要调度"""
    agent_results = state.get("agent_results", {})
    current_agent = state.get("current_agent", "")

    if current_agent == "analysis_agent":
        return "end"

    return "dispatch"


# ============================================================
# SQL ReAct Graph 构建
# ============================================================

def build_sql_react_graph():
    """构建 SQL ReAct 多 Agent 子图"""
    graph = StateGraph(AgentState)

    graph.add_node("coordinator", coordinator_node)
    graph.add_node("dispatch", dispatch_node)

    graph.add_edge(START, "coordinator")
    graph.add_edge("coordinator", "dispatch")
    graph.add_conditional_edges("dispatch", should_continue_dispatch, {
        "dispatch": "dispatch",
        "end": END,
    })

    return graph.compile()


_react_graph = None


def get_sql_react_graph():
    """获取 SQL ReAct 子图单例（懒加载）"""
    global _react_graph
    if _react_graph is None:
        _react_graph = build_sql_react_graph()
    return _react_graph
