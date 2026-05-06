import logging
import uuid
from typing import Optional

from app.agent.skills.base import BaseSkill
from app.agent.state import AgentState
from app.agent.tools.schema_tool import SearchSchemaTool
from app.agent.tools.sql_tool import SearchMemoryTool
from app.agent.tools.memory_tool import BuildContextTool
from app.mcp.bridge import get_lazy_mcp_tools
from app.agent.llm_stream import stream_llm
from app.events import EventType, NullEmitter

logger = logging.getLogger(__name__)


class DataChatSkill(BaseSkill):
    name = "data_chat"
    description = "与数据相关的日常对话、咨询和解释。适合问候、能力询问、关于数据的一般性问题，如'你好'、'你能做什么'、'我们有哪些数据表'。"
    tools = [SearchMemoryTool(), SearchSchemaTool(), BuildContextTool(), *get_lazy_mcp_tools()]

    async def execute(self, state: AgentState, session_id: Optional[str] = None, emitter=None) -> AgentState:
        logger.info(f"🔧 DataChatSkill 执行：{state.user_input}")

        system_prompt = "你是一个友好的数据查询助手，正在与用户进行日常对话。请根据历史记忆（如果有）来回答用户，保持对话的连贯性。\n"

        if session_id:
            build_context = self.get_tool("build_context")
            if emitter:
                emitter.emit(EventType.TOOL_CALL, {"tool": "build_context", "message": "正在构建会话上下文..."})
            ctx_result = await build_context.execute(
                session_id=session_id, current_question=state.user_input
            )
            if ctx_result.success and ctx_result.data:
                system_prompt += f"\n\n【最近对话】\n{ctx_result.data}"
                logger.info(f"🧠 使用短期记忆上下文：{len(ctx_result.data)} 字符")
            if emitter:
                emitter.emit(EventType.TOOL_RESULT, {"tool": "build_context", "success": ctx_result.success})

        if state.memories and state.memories.get("results"):
            memory_texts = []
            for mem in state.memories["results"][:3]:
                text = mem.get("memory", "")
                if text:
                    memory_texts.append(text)
            if memory_texts:
                system_prompt += f"\n\n【长期记忆】\n" + "\n".join(memory_texts)
                logger.info(f"🧠 使用长期记忆上下文")

        if self._is_schema_question(state.user_input):
            search_schema = self.get_tool("search_schema")
            if emitter:
                emitter.emit(EventType.TOOL_CALL, {"tool": "search_schema", "message": "正在搜索表结构..."})
            schema_result = await search_schema.execute(question=state.user_input)
            if schema_result.success and schema_result.data:
                system_prompt += f"\n\n【数据库表结构概览】\n{schema_result.data}"
            if emitter:
                emitter.emit(EventType.TOOL_RESULT, {"tool": "search_schema", "success": schema_result.success})

        if emitter:
            emitter.emit(EventType.THINKING, {"phase": "chat_generate", "message": "正在生成回复..."})

        try:
            stream_id = f"chat_{uuid.uuid4().hex[:6]}"
            em = emitter or NullEmitter()

            response = await stream_llm(
                state.user_input, em, stream_id,
                event_type=EventType.ANSWER_STREAM,
                system_prompt=system_prompt
            )

            state.update(response=response.strip())
            logger.info(f"✅ 聊天响应：{response[:100]}")
        except Exception as e:
            logger.error(f"❌ 聊天失败：{e}")
            state.update(response=f"抱歉，我出现了一些问题：{str(e)}")

        state.add_node_history("data_chat_skill")
        return state

    def _is_schema_question(self, question: str) -> bool:
        keywords = ["有哪些表", "什么数据", "能查什么", "数据库里有什么", "有哪些信息", "能查哪些"]
        return any(kw in question for kw in keywords)

    def should_handle(self, state: AgentState) -> float:
        if state.intent == "daily_chat":
            return 0.9
        if not state.intent:
            return 0.5
        return 0.1
