import logging
from typing import Optional

from app.agent.tools.base import BaseTool, ToolResult
from app.memory.session import smart_session_memory

logger = logging.getLogger(__name__)


class BuildContextTool(BaseTool):
    name = "build_context"
    description = "构建短期记忆上下文，获取当前会话的最近对话历史。用于理解用户的上下文语境。"
    parameters = {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "会话ID"
            },
            "current_question": {
                "type": "string",
                "description": "当前用户问题"
            }
        },
        "required": ["session_id", "current_question"]
    }

    async def execute(self, **kwargs) -> ToolResult:
        session_id = kwargs.get("session_id", "")
        current_question = kwargs.get("current_question", "")

        if not session_id:
            return ToolResult(success=True, data="")

        try:
            context = await smart_session_memory.build_context(
                session_id=session_id,
                current_question=current_question
            )
            return ToolResult(success=True, data=context)
        except Exception as e:
            logger.error(f"❌ 构建上下文失败：{e}")
            return ToolResult(success=True, data="")
