"""问答路由"""

import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agent.graph import get_graph
from app.core.llm import LLMError, LLMTimeoutError, LLMClientError, LLMRateLimitError
from app.events import EventEmitter, EventType
from app.memory.session import smart_session_memory
from app.middleware import get_trace_id, service_health


router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("")
async def handle_ask(
    question: str,
    user_id: str = "default",
    session_id: str = None
):
    """
    处理用户请求

    流程：
    1. 使用前端传递的 session_id（或创建新的）
    2. 调用工作流引擎（意图识别 → Skill选择 → Skill执行）
    3. 保存到 Redis（短期记忆 - 自动触发长期记忆评估）
    4. 返回最终响应（支持澄清和正常回答两种类型）
    """
    if not session_id:
        session_id = f"session-{str(uuid.uuid4())[:8]}"

    if not service_health.chroma_healthy:
        service_health.try_recover_chroma()

    graph = get_graph()
    initial_state = {
        "user_input": question,
        "user_id": user_id,
        "session_id": session_id,
        "max_retries": 3,
        "retry_count": 0,
        "react_step_count": 0,
        "messages": [],
        "react_trace": [],
        "node_history": [],
        "query_result": [],
        "step_results": [],
        "plan_steps": [],
        "mcp_tools_used": [],
    }
    final_state = await graph.ainvoke(initial_state)

    if final_state.get("response") and not final_state.get("clarification_needed"):
        try:
            await smart_session_memory.add_conversation(
                session_id=session_id,
                user_message=question,
                assistant_message=final_state["response"],
                user_id=user_id
            )
        except Exception as e:
            logging.warning(f"⚠️ Redis 保存失败（降级）：{e}")
            service_health.mark_redis_down()

    if final_state.get("clarification_needed"):
        return {
            "response_type": "clarification",
            "clarification_question": final_state.get("clarification_question"),
            "session_id": session_id,
            "intent": final_state.get("intent"),
            "skill": final_state.get("skill_name"),
            "trace_id": get_trace_id()
        }

    return {
        "response_type": "answer",
        "response": final_state.get("response"),
        "intent": final_state.get("intent"),
        "session_id": session_id,
        "skill": final_state.get("skill_name"),
        "trace_id": get_trace_id(),
        "react_trace": final_state.get("react_trace") or None
    }


@router.post("/stream")
async def handle_ask_stream(
    question: str,
    user_id: str = "default",
    session_id: str = None
):
    if not session_id:
        session_id = f"session-{str(uuid.uuid4())[:8]}"

    if not service_health.chroma_healthy:
        service_health.try_recover_chroma()

    emitter = EventEmitter()
    graph = get_graph()
    initial_state = {
        "user_input": question,
        "user_id": user_id,
        "session_id": session_id,
        "max_retries": 3,
        "retry_count": 0,
        "react_step_count": 0,
        "messages": [],
        "react_trace": [],
        "node_history": [],
        "query_result": [],
        "step_results": [],
        "plan_steps": [],
        "mcp_tools_used": [],
    }

    async def run_and_save():
        """执行图并保存对话到会话记忆（带完整错误处理）"""
        import uuid as _uuid
        import traceback

        request_id = str(_uuid.uuid4())[:8]
        final_state = None

        emitter.init_thinking_stream(f"thinking_{request_id}")
        logging.info(f"🚀 开始执行图 (request_id={request_id}) | user_input={question[:50]}...")

        try:
            run_config = {"configurable": {"emitter": emitter}}
            async for event in graph.astream_events(initial_state, config=run_config, version="v2"):
                kind = event.get("event")

                if kind == "on_chain_start":
                    node_name = event.get("name", "")
                    _skip_prefixes = ("Graph", "LangGraph", "__root__")
                    if any(node_name.startswith(p) for p in _skip_prefixes):
                        continue
                    _node_labels = {
                        "intent_router": "意图识别",
                        "chat": "生成回答",
                        "sql_simple": "SQL 查询处理",
                        "sql_react": "多步推理查询",
                        "build_context": "构建会话上下文",
                        "search_schema": "检索表结构",
                        "search_examples": "检索相似示例",
                        "sql_loop": "生成 SQL",
                    }
                    emitter.emit(EventType.THINKING, {
                        "phase": "chain_start",
                        "name": node_name,
                        "message": _node_labels.get(node_name, node_name),
                    })

                elif kind == "on_chat_model_stream":
                    event_tags = event.get("tags", []) or []
                    if "skip_stream" not in event_tags:
                        chunk = event.get("data", {}).get("chunk")
                        token = ""
                        if chunk and hasattr(chunk, "content"):
                            token = chunk.content if isinstance(chunk.content, str) else str(chunk.content or "")
                        if token:
                            emitter.emit_thinking_token(token)

                elif kind == "on_tool_start":
                    emitter.emit(EventType.TOOL_CALL, {
                        "tool": event.get("name", ""),
                        "message": f"调用工具: {event.get('name', '')}",
                    })

                elif kind == "on_tool_end":
                    emitter.emit(EventType.TOOL_RESULT, {
                        "tool": event.get("name", ""),
                        "success": True,
                    })

                if kind == "on_chain_end":
                    output = event.get("data", {}).get("output")
                    if isinstance(output, dict) and "user_input" in output:
                        final_state = output

            logging.info(f"✅ 图执行成功 (request_id={request_id})")

            if final_state and final_state.get("response") and not final_state.get("clarification_needed"):
                try:
                    await smart_session_memory.add_conversation(
                        session_id=session_id,
                        user_message=question,
                        assistant_message=final_state["response"],
                        user_id=user_id
                    )
                except Exception as e:
                    logging.warning(f"⚠️ Redis 保存失败（降级）：{e}")
                    service_health.mark_redis_down()

            if final_state:
                if final_state.get("clarification_needed"):
                    emitter.emit_done(
                        response_type="clarification",
                        clarification_question=final_state.get("clarification_question"),
                        intent=final_state.get("intent"),
                        skill=final_state.get("skill_name"),
                    )
                else:
                    emitter.emit_done(
                        response_type="answer",
                        response=final_state.get("response"),
                        intent=final_state.get("intent"),
                        skill=final_state.get("skill_name"),
                        react_trace=final_state.get("react_trace"),
                    )
            else:
                emitter.emit_done(response_type="error", message="未获取到最终状态")

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            logging.error(
                f"❌ 图执行异常 (request_id={request_id}) | "
                f"type={error_type} | msg={error_msg[:200]}\n"
                f"{traceback.format_exc()}"
            )

            if isinstance(e, LLMTimeoutError):
                emitter.emit_error(
                    error_type="llm_timeout",
                    message="AI 模型响应超时，请稍后重试",
                    suggestion="可以尝试简化问题或稍后再试",
                    details={
                        "request_id": request_id,
                        "original_error": str(e),
                        "phase": "llm_call"
                    }
                )
            elif isinstance(e, LLMClientError):
                status_code = getattr(e, 'status_code', 'unknown')
                if status_code == 400:
                    emitter.emit_error(
                        error_type="client_error_400",
                        message="请求参数无效，可能是查询条件过于复杂",
                        suggestion="请简化问题后重试，或换个方式描述您的需求",
                        details={
                            "request_id": request_id,
                            "status_code": status_code,
                            "original_error": error_msg[:200]
                        }
                    )
                elif status_code == 401:
                    emitter.emit_error(
                        error_type="auth_error",
                        message="API 认证失败，请联系管理员检查配置",
                        details={
                            "request_id": request_id,
                            "status_code": status_code
                        }
                    )
                else:
                    emitter.emit_error(
                        error_type=f"client_error_{status_code}",
                        message=f"请求失败（{status_code}），请稍后重试",
                        suggestion="如果问题持续出现，请联系管理员",
                        details={
                            "request_id": request_id,
                            "status_code": status_code,
                            "original_error": error_msg[:200]
                        }
                    )
            elif isinstance(e, LLMRateLimitError):
                retry_after = getattr(e, 'retry_after', None) or "几秒"
                emitter.emit_error(
                    error_type="rate_limit",
                    message="API 调用频率过高，请稍后重试",
                    suggestion=f"建议等待 {retry_after} 后再试",
                    details={
                        "request_id": request_id,
                        "retry_after": retry_after
                    }
                )
            else:
                emitter.emit_error(
                    error_type="unknown_error",
                    message="系统内部错误，请稍后重试",
                    suggestion="如果问题持续出现，请联系管理员并提供此错误码",
                    details={
                        "request_id": request_id,
                        "error_class": error_type,
                        "error_message": error_msg[:300]
                    }
                )

            emitter.emit_done(
                response_type="error",
                message=error_msg[:200],
                error_type=error_type,
                request_id=request_id
            )

    async def event_stream():
        """SSE 事件流生成器"""
        task = None
        try:
            task = asyncio.create_task(run_and_save())
            async for chunk in emitter.stream():
                yield chunk

            if task and not task.done():
                await task

        except GeneratorExit:
            logging.warning(f"⚠️ 前端断开 SSE 连接，取消正在执行的任务")
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except asyncio.CancelledError:
            logging.warning(f"⚠️ SSE stream 被取消")
            if task and not task.done():
                task.cancel()
                raise

        except Exception as e:
            logging.error(f"❌ event_stream 异常: {type(e).__name__}: {e}")
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except:
                    pass
            try:
                error_data = json.dumps({
                    "type": EventType.ERROR,
                    "data": {
                        "error_type": "stream_error",
                        "message": "连接异常中断",
                        "suggestion": "请刷新页面重试"
                    },
                    "timestamp": time.time()
                }, ensure_ascii=False)
                yield f"data: {error_data}\n\n"
            except:
                pass
        finally:
            logging.debug(f"🔚 event_stream 已退出清理")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
