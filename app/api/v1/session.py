"""会话路由"""

import logging

from fastapi import APIRouter

from app.memory.session import smart_session_memory


router = APIRouter(prefix="/session", tags=["session"])


@router.delete("/{session_id}")
async def end_session(session_id: str, user_id: str = "default"):
    try:
        await smart_session_memory.clear(session_id, user_id)
        return {"status": "success", "message": f"会话 {session_id} 已清理"}
    except Exception as e:
        logging.error(f"❌ 清理会话失败：{e}")
        return {"status": "error", "message": str(e)}


@router.get("/sessions")
async def list_sessions(user_id: str = "default"):
    try:
        sessions = await smart_session_memory.list_sessions(user_id)
        return {"sessions": sessions}
    except Exception as e:
        logging.error(f"❌ 获取会话列表失败：{e}")
        return {"sessions": []}


@router.get("/{session_id}/history")
async def get_session_history(session_id: str):
    try:
        messages = await smart_session_memory.get_session_history(session_id)
        return {"session_id": session_id, "messages": messages}
    except Exception as e:
        logging.error(f"❌ 获取会话历史失败：{e}")
        return {"session_id": session_id, "messages": []}
