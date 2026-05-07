from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.agent.workflow import workflow
from app.memory.session import smart_session_memory
from app.memory.filter import memory_filter
from app.middleware import get_trace_id, service_health
from app.events import EventEmitter
from app.auth.api import router as auth_router
from app.auth.models import init_auth_db
from config.settings import chroma_settings
import asyncio
import logging
import uuid


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_auth_db()
    logging.info("✅ 用户认证数据库初始化完成")

    service_health.mark_milvus_up()
    logging.info("✅ Chroma 已初始化")

    if service_health.milvus_healthy:
        try:
            result = memory_filter.cleanup_expired_memories(user_id="default")
            if result["deleted"] > 0:
                logging.info(f"🧹 启动清理过期记忆：删除 {result['deleted']} 条，保留 {result['kept']} 条")
        except Exception as e:
            logging.warning(f"⚠️ 启动清理过期记忆失败（非致命）：{e}")
            if not service_health.milvus_healthy:
                service_health.try_recover_milvus()
    else:
        logging.info("⏭️ Chroma 不可用，跳过过期记忆清理")

    try:
        from app.rag.bm25_index import bm25_index
        loaded = await bm25_index.load_from_redis()
        if loaded:
            logging.info("✅ BM25 索引从 Redis 加载成功")
        else:
            logging.warning("⚠️ BM25 索引未找到，请运行 rebuild_rag_index.py 构建索引")
    except Exception as e:
        logging.warning(f"⚠️ BM25 索引加载失败（降级为纯向量检索）：{e}")

    yield

    try:
        from app.rag.chroma import ChromaSessionLocal
        ChromaSessionLocal().close()
        logging.info("✅ Chroma 连接已关闭")
    except Exception:
        pass


app = FastAPI(title="AnswerAgent", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "http://124.223.93.65", "http://124.223.93.65:3000", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s] %(message)s'
)


class TraceIdFilter(logging.Filter):
    def filter(self, record):
        record.trace_id = get_trace_id()
        return True


for handler in logging.root.handlers:
    handler.addFilter(TraceIdFilter())

@app.get("/health")
async def health_check():
    """服务健康检查"""
    return {
        "status": "healthy",
        "trace_id": get_trace_id(),
        "services": service_health.status()
    }


@app.post("/ask")
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

    if not service_health.milvus_healthy:
        service_health.try_recover_milvus()

    # /ask 不传 emitter，workflow 内部会使用 NullEmitter（空实现），
    # 工作流执行过程中的所有中间事件（THINKING/TOOL_CALL/ANSWER_STREAM 等）
    # 都会被 NullEmitter 静默吞掉，不产生任何开销，最终只返回完整结果。
    final_state = await workflow.execute(question, user_id, session_id=session_id)

    if final_state.response and not final_state.clarification_needed:
        try:
            await smart_session_memory.add_conversation(
                session_id=session_id,
                user_message=question,
                assistant_message=final_state.response,
                user_id=user_id
            )
        except Exception as e:
            logging.warning(f"⚠️ Redis 保存失败（降级）：{e}")
            service_health.mark_redis_down()

    if final_state.clarification_needed:
        return {
            "response_type": "clarification",
            "clarification_question": final_state.clarification_question,
            "session_id": session_id,
            "intent": final_state.intent,
            "skill": final_state.skill_name,
            "trace_id": get_trace_id()
        }

    return {
        "response_type": "answer",
        "response": final_state.response,
        "intent": final_state.intent,
        "session_id": session_id,
        "skill": final_state.skill_name,
        "trace_id": get_trace_id(),
        "react_trace": final_state.react_trace if final_state.react_trace else None
    }


@app.post("/ask/stream")
async def handle_ask_stream(
    question: str,
    user_id: str = "default",
    session_id: str = None
):
    if not session_id:
        session_id = f"session-{str(uuid.uuid4())[:8]}"

    if not service_health.milvus_healthy:
        service_health.try_recover_milvus()

    # 创建 EventEmitter 实例，作为工作流中间事件的推送通道。
    # 工作流执行中各节点调用 emitter.emit() 将事件推入 asyncio.Queue，
    # 下方 event_stream() 通过 emitter.stream() 消费队列，实时推送给前端。
    emitter = EventEmitter()

    async def run_and_save():
        # 将 emitter 传入工作流，使执行过程中的中间事件推入队列。
        # 这是 /ask/stream 与 /ask 的核心区别：/ask 不传 emitter，中间事件被 NullEmitter 吞掉。
        final_state = await workflow.execute(
            question, user_id, session_id=session_id, emitter=emitter
        )

        if final_state.response and not final_state.clarification_needed:
            try:
                await smart_session_memory.add_conversation(
                    session_id=session_id,
                    user_message=question,
                    assistant_message=final_state.response,
                    user_id=user_id
                )
            except Exception as e:
                logging.warning(f"⚠️ Redis 保存失败（降级）：{e}")
                service_health.mark_redis_down()

    async def event_stream():
        # 启动后台 Task 执行工作流（生产者：emit() 往队列放事件）
        task = asyncio.create_task(run_and_save())
        # 异步迭代消费队列（消费者：stream() 从队列取事件，格式化为 SSE 推送）
        async for chunk in emitter.stream():
            yield chunk
        # 确保工作流 Task 完成（如保存对话到 Redis）后再结束，避免资源泄漏
        await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.delete("/session/{session_id}")
async def end_session(session_id: str, user_id: str = "default"):
    try:
        await smart_session_memory.clear(session_id, user_id)
        return {"status": "success", "message": f"会话 {session_id} 已清理"}
    except Exception as e:
        logging.error(f"❌ 清理会话失败：{e}")
        return {"status": "error", "message": str(e)}


@app.get("/sessions")
async def list_sessions(user_id: str = "default"):
    try:
        sessions = await smart_session_memory.list_sessions(user_id)
        return {"sessions": sessions}
    except Exception as e:
        logging.error(f"❌ 获取会话列表失败：{e}")
        return {"sessions": []}


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    try:
        messages = await smart_session_memory.get_session_history(session_id)
        return {"session_id": session_id, "messages": messages}
    except Exception as e:
        logging.error(f"❌ 获取会话历史失败：{e}")
        return {"session_id": session_id, "messages": []}
