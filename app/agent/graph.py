"""
LangGraph 主图 —— 意图路由 + 聊天节点

构建 StateGraph：
  START → intent_router → route_after_intent → (sql_simple | sql_react | chat) → END
"""

import logging
import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_config

from app.agent.state import AgentState
from app.core.llm import get_llm, get_structured_llm
from app.agent.skills.sql_simple import build_sql_simple_graph
from app.agent.skills.sql_react import build_sql_react_graph
from app.events import EventType
from app.memory.session import smart_session_memory
from app.memory.management import memory_manager
from app.memory.filter import memory_filter
from app.rag.retrieval import retrieve_schema, retrieve_table_info_simple
from prompts.intent import INTENT_PROMPT
from prompts.chat import build_chat_messages

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────
# 1. IntentResult —— 结构化输出模型
# ────────────────────────────────────────────

class IntentResult(BaseModel):
    intent: Literal["database_query", "daily_chat"] = Field(description="意图类型")
    need_memory: bool = Field(description="是否需要检索长期记忆")
    memory_reason: str = Field(default="", description="需要检索记忆的理由")
    needs_react: bool = Field(default=False, description="是否需要ReAct多步推理")
    react_reason: str = Field(default="", description="需要ReAct的理由")
    reasoning: str = Field(description="判断理由")


# ────────────────────────────────────────────
# 2. intent_router 节点
# ────────────────────────────────────────────


async def intent_router(state: AgentState) -> dict:
    """意图识别节点：判断用户意图、是否需要记忆、是否需要 ReAct"""
    user_input = state.get("user_input", "")
    user_id = state.get("user_id", "default")
    logger.info(f"开始意图识别：{user_input}")

    _t0 = time.perf_counter()

    config = get_config()
    _t1 = time.perf_counter()
    emitter = config.get("configurable", {}).get("emitter")

    try:
        _t2 = time.perf_counter()
        # ✨ 使用场景化超时配置（意图识别 60s）
        from app.core.llm import get_structured_llm, LLMError, LLMTimeoutError, LLMClientError
        structured_llm = get_structured_llm(IntentResult, timeout_key='intent_recognition', tags=["skip_stream"])
        _t3 = time.perf_counter()

        _t4 = time.perf_counter()
        messages = INTENT_PROMPT.invoke({"user_input": user_input})
        _t5 = time.perf_counter()

        _t6 = time.perf_counter()
        result = await structured_llm.ainvoke(messages.to_messages())
        _t7 = time.perf_counter()

        # 分段计时日志
        logger.info(
            f"⏱️ 意图识别耗时分解（总 {(time.perf_counter()-_t0)*1000:.0f}ms）："
            f" get_config={(_t1-_t0)*1000:.0f}ms"
            f" structured_llm={(_t3-_t2)*1000:.0f}ms"
            f" prompt_invoke={(_t5-_t4)*1000:.0f}ms"
            f" llm_ainvoke={(_t7-_t6)*1000:.0f}ms"
            f" (prompt_size={len(str(messages))} 字符)"
        )

        intent_data: IntentResult = result

        need_memory = intent_data.need_memory
        memories = {}

        if need_memory:
            if emitter:
                emitter.emit(EventType.THINKING, {
                    "phase": "memory_retrieval",
                    "message": "正在检索长期记忆..."
                })
            optimized_query = await memory_filter.extract_search_query(user_input)
            memories = memory_manager.search_memory(
                query=optimized_query,
                user_id=user_id
            )
            if memories and memories.get("results"):
                logger.info(f"🧠 检索到 {len(memories['results'])} 条长期记忆 (query: '{user_input}' → '{optimized_query}')")
            else:
                logger.info("🧠 未检索到长期记忆")
                memories = {}
            logger.info(f"📋 需要检索长期记忆：{intent_data.memory_reason}")
        else:
            logger.info(f"📋 无需检索长期记忆：{intent_data.memory_reason}")

        # needs_react 仅在 database_query 时生效
        needs_react = intent_data.needs_react and intent_data.intent == "database_query"

        if emitter:
            emitter.emit(EventType.THINKING, {
                "phase": "intent_result",
                "message": f"意图：{intent_data.intent}",
                "intent": intent_data.intent,
                "need_memory": intent_data.need_memory,
                "needs_react": needs_react
            })

        logger.info(f"意图识别结果：{intent_data.intent}")
        logger.info(f"判断理由：{intent_data.reasoning}")
        logger.info(f"需要记忆：{intent_data.need_memory} - {intent_data.memory_reason}")
        logger.info(f"需要多步推理：{needs_react} - {intent_data.react_reason}")

        return {
            "intent": intent_data.intent,
            "intent_reason": intent_data.reasoning,
            "need_memory": intent_data.need_memory,
            "memory_reason": intent_data.memory_reason,
            "needs_react": needs_react,
            "react_reason": intent_data.react_reason,
            "memories": memories,
        }

    except LLMTimeoutError as e:
        logger.error(f"❌ 意图识别超时: {e}")
        if emitter:
            emitter.emit(EventType.THINKING, {
                "phase": "intent_error",
                "message": "意图识别超时，将作为日常聊天处理",
                "error_type": "timeout"
            })
        # 超时时降级为 daily_chat（保守策略）
        return {
            "intent": "daily_chat",
            "intent_reason": f"意图识别超时，降级处理：{str(e)}",
            "need_memory": False,
            "memory_reason": "超时无法判断",
            "needs_react": False,
            "react_reason": "超时无法判断",
            "memories": {},
        }

    except LLMClientError as e:
        logger.error(f"❌ 意图识别客户端错误: {e}")
        if emitter:
            emitter.emit(EventType.THINKING, {
                "phase": "intent_error",
                "message": f"意图识别失败（{e.status_code}），将作为日常聊天处理",
                "error_type": "client_error",
                "status_code": e.status_code
            })
        return {
            "intent": "daily_chat",
            "intent_reason": f"意图识别失败（HTTP {e.status_code}），降级处理",
            "need_memory": False,
            "memory_reason": "错误无法判断",
            "needs_react": False,
            "react_reason": "错误无法判断",
            "memories": {},
        }

    except LLMError as e:
        logger.error(f"❌ 意图识别 LLM 错误: {e}")
        if emitter:
            emitter.emit(EventType.THINKING, {
                "phase": "intent_error",
                "message": "意图识别失败，将作为日常聊天处理",
                "error_type": e.error_type,
                "retryable": e.retryable
            })
        return {
            "intent": "daily_chat",
            "intent_reason": f"LLM 错误降级：{str(e)[:100]}",
            "need_memory": False,
            "memory_reason": "错误无法判断",
            "needs_react": False,
            "react_reason": "错误无法判断",
            "memories": {},
        }

    except Exception as e:
        logger.error(f"❌ 意图识别未知异常: {type(e).__name__}: {e}")
        if emitter:
            emitter.emit(EventType.THINKING, {
                "phase": "intent_error",
                "message": "意图识别遇到意外错误",
                "error_type": "unknown"
            })
        return {
            "intent": "daily_chat",
            "intent_reason": f"未知异常降级：{type(e).__name__}: {str(e)[:100]}",
            "need_memory": False,
            "memory_reason": "异常无法判断",
            "needs_react": False,
            "react_reason": "异常无法判断",
            "memories": {},
        }


# ────────────────────────────────────────────
# 3. route_after_intent 条件边
# ────────────────────────────────────────────

def route_after_intent(state: AgentState) -> str:
    """根据意图路由到对应节点"""
    intent = state.get("intent", "daily_chat")
    if intent == "daily_chat":
        return "chat"
    if intent == "database_query":
        if state.get("needs_react"):
            return "sql_react"
        return "sql_simple"
    return "chat"


# ────────────────────────────────────────────
# 4. chat 节点
# ────────────────────────────────────────────

_SCHEMA_KEYWORDS = ["有哪些表", "什么数据", "能查什么", "数据库里有什么", "有哪些信息", "能查哪些", "表结构"]


def _is_schema_question(question: str) -> bool:
    return any(kw in question for kw in _SCHEMA_KEYWORDS)


async def _search_schema(question: str) -> str:
    """检索数据库表结构信息"""
    try:
        from app.core.config.redis import get_redis_client
        import json

        redis_client = await get_redis_client()
        relation_map = {}
        async for key in redis_client.scan_iter("rag:relation:*"):
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            table_name = key.replace("rag:relation:", "")
            data = await redis_client.get(key)
            if data:
                relation_map[table_name] = json.loads(data)

        if relation_map:
            metadata = await retrieve_schema(question, relation_map)
        else:
            metadata = await retrieve_table_info_simple(question)

        if not metadata or "未找到" in metadata:
            return ""
        return metadata
    except Exception as e:
        logger.error(f"❌ 表结构检索失败：{e}")
        return ""


async def chat(state: AgentState) -> dict:
    """日常聊天节点：复刻 DataChatSkill 逻辑"""
    user_input = state.get("user_input", "")
    session_id = state.get("session_id")
    logger.info(f"🔧 chat 节点执行：{user_input}")

    config = get_config()
    emitter = config.get("configurable", {}).get("emitter")

    # 短期记忆上下文
    ctx = ""
    if session_id:
        if emitter:
            emitter.emit(EventType.TOOL_CALL, {"tool": "build_context", "message": "正在构建会话上下文..."})
        ctx = await smart_session_memory.build_context(
            session_id=session_id, current_question=user_input
        )
        if ctx:
            logger.info(f"🧠 使用短期记忆上下文：{len(ctx)} 字符")
        if emitter:
            emitter.emit(EventType.TOOL_RESULT, {"tool": "build_context", "success": bool(ctx)})

    # 长期记忆上下文
    memories = state.get("memories", {})
    memory_texts = []
    if memories and memories.get("results"):
        for mem in memories["results"][:3]:
            text = mem.get("memory", "")
            if text:
                memory_texts.append(text)
        if memory_texts:
            logger.info("🧠 使用长期记忆上下文")

    # 表结构检索
    schema_data = ""
    if _is_schema_question(user_input):
        if emitter:
            emitter.emit(EventType.TOOL_CALL, {"tool": "search_schema", "message": "正在搜索表结构..."})
        schema_data = await _search_schema(user_input)
        if schema_data:
            logger.info("📋 使用表结构上下文")
        if emitter:
            emitter.emit(EventType.TOOL_RESULT, {"tool": "search_schema", "success": bool(schema_data)})

    if emitter:
        emitter.emit(EventType.THINKING, {"phase": "chat_generate", "message": "正在生成回复..."})

    try:
        # ✨ 使用场景化超时配置（聊天 30s）
        from app.core.llm import get_llm, LLMError, LLMTimeoutError
        llm = get_llm(timeout_key='chat')
        messages = build_chat_messages(
            user_input=user_input,
            short_term_context=ctx or "",
            long_term_context="\n".join(memory_texts) if memory_texts else "",
            schema_context=schema_data or "",
        )
        # 流式输出：逐 token 发射 ANSWER_STREAM 事件
        stream_id = f"chat_{uuid.uuid4().hex[:8]}"
        full_text = ""
        async for chunk in llm.astream(messages):
            token = chunk.content if isinstance(chunk.content, str) else str(chunk.content or "")
            if token:
                full_text += token
                if emitter:
                    emitter.emit(EventType.ANSWER_STREAM, {
                        "stream_id": stream_id,
                        "text": full_text
                    })
        response_text = full_text.strip()
        logger.info(f"✅ 聊天响应：{response_text[:100]}")
    except LLMTimeoutError as e:
        logger.error(f"❌ 聊天响应超时: {e}")
        if emitter:
            emitter.emit(EventType.THINKING, {
                "phase": "chat_error",
                "message": "聊天响应超时",
                "error_type": "timeout"
            })
        response_text = "抱歉，我响应超时了。请简化问题后重试。"
    except LLMError as e:
        logger.error(f"❌ 聊天 LLM 错误: {e}")
        if emitter:
            emitter.emit(EventType.THINKING, {
                "phase": "chat_error",
                "message": f"聊天失败（{e.error_type}）",
                "error_type": e.error_type,
                "retryable": e.retryable
            })
        response_text = f"抱歉，我在处理您的问题时遇到了错误：{str(e)[:100]}。请稍后重试。"
    except Exception as e:
        logger.error(f"❌ 聊天异常: {type(e).__name__}: {e}")
        if emitter:
            emitter.emit(EventType.THINKING, {
                "phase": "chat_error",
                "message": "聊天遇到意外错误",
                "error_type": "unknown"
            })
        response_text = "抱歉，系统出现了一些问题。请稍后重试。"

    return {
        "response": response_text,
        "skill_name": "chat",
    }


# ────────────────────────────────────────────
# 5. 子图节点：sql_simple / sql_react
# ────────────────────────────────────────────

async def sql_simple(state: AgentState) -> dict:
    """简单 SQL 查询子图（流式转发内部事件）"""
    try:
        graph = build_sql_simple_graph()
        config = get_config()
        emitter = config.get("configurable", {}).get("emitter")
        # 使用 astream_events 使子图内部 LLM token 事件可被捕获并转发
        final_result = None
        async for event in graph.astream_events(state, config=config, version="v2"):
            kind = event.get("event")
            if kind == "on_chain_end":
                output = event.get("data", {}).get("output")
                if isinstance(output, dict) and "response" in output:
                    final_result = output
            elif kind == "on_chat_model_stream":
                # 转发子图内部 LLM token 到外层 emitter（累积到同一个卡片）
                event_tags = event.get("tags", []) or []
                if "skip_stream" not in event_tags:
                    chunk = event.get("data", {}).get("chunk")
                    token = ""
                    if chunk and hasattr(chunk, "content"):
                        token = chunk.content if isinstance(chunk.content, str) else str(chunk.content or "")
                    if token and emitter:
                        emitter.emit_thinking_token(token)
        result = final_result or state
        # Extract only the fields we want to update
        return {k: v for k, v in result.items() if k in (
            "response", "sql_query", "query_status", "sql_valid", "sql_error",
            "retry_count", "clarification_needed", "clarification_question",
            "skill_name", "metadata_context", "examples_context"
        ) and v != state.get(k)}  # only return changed fields
    except Exception as e:
        logger.error(f"❌ SQL 简单查询子图执行失败: {type(e).__name__}: {e}")
        
        config = get_config()
        emitter = config.get("configurable", {}).get("emitter")
        if emitter:
            emitter.emit(EventType.THINKING, {
                "phase": "sql_simple_error",
                "message": f"SQL 查询执行失败：{str(e)[:200]}",
                "error_type": type(e).__name__
            })
        
        return {
            "response": f"查询执行失败：{str(e)[:200]}",
            "query_status": "error",
            "sql_valid": False,
            "sql_error": str(e),
            "skill_name": "sql_simple",
        }


async def sql_react(state: AgentState) -> dict:
    """ReAct 多步推理子图（流式转发内部事件）"""
    try:
        graph = build_sql_react_graph()
        config = get_config()
        emitter = config.get("configurable", {}).get("emitter")
        # 使用 astream_events 使子图内部 LLM token 事件可被捕获并转发
        final_result = None
        async for event in graph.astream_events(state, config=config, version="v2"):
            kind = event.get("event")
            if kind == "on_chain_end":
                output = event.get("data", {}).get("output")
                if isinstance(output, dict) and "response" in output:
                    final_result = output
            elif kind == "on_chat_model_stream":
                # 转发子图内部 LLM token 到外层 emitter（累积到同一个卡片）
                event_tags = event.get("tags", []) or []
                if "skip_stream" not in event_tags:
                    chunk = event.get("data", {}).get("chunk")
                    token = ""
                    if chunk and hasattr(chunk, "content"):
                        token = chunk.content if isinstance(chunk.content, str) else str(chunk.content or "")
                    if token and emitter:
                        emitter.emit_thinking_token(token)
        result = final_result or state
        return {k: v for k, v in result.items() if k in (
            "response", "sql_query", "query_status", "skill_name",
            "clarification_needed", "clarification_question",
            "react_trace", "react_step_count", "messages"
        ) and v != state.get(k)}
    except Exception as e:
        logger.error(f"❌ SQL ReAct 多步推理子图执行失败: {type(e).__name__}: {e}")
        
        config = get_config()
        emitter = config.get("configurable", {}).get("emitter")
        if emitter:
            emitter.emit(EventType.THINKING, {
                "phase": "sql_react_error",
                "message": f"SQL ReAct 推理执行失败：{str(e)[:200]}",
                "error_type": type(e).__name__
            })
        
        return {
            "response": f"多步推理查询失败：{str(e)[:200]}",
            "query_status": "error",
            "skill_name": "sql_react",
            "react_trace": [],
            "react_step_count": 0,
        }


# ────────────────────────────────────────────
# 6. build_graph + 单例
# ────────────────────────────────────────────

def build_graph():
    """构建并编译 LangGraph StateGraph"""
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("intent_router", intent_router)
    graph.add_node("sql_simple", sql_simple)
    graph.add_node("sql_react", sql_react)
    graph.add_node("chat", chat)

    # 添加边
    graph.add_edge(START, "intent_router")
    graph.add_conditional_edges("intent_router", route_after_intent, {
        "sql_simple": "sql_simple",
        "sql_react": "sql_react",
        "chat": "chat",
    })
    graph.add_edge("sql_simple", END)
    graph.add_edge("sql_react", END)
    graph.add_edge("chat", END)

    return graph.compile()


_graph = None


def get_graph():
    """获取编译后的图单例（懒加载）"""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
