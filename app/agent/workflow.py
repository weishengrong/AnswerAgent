import logging
import time
from typing import Optional

from app.agent.state import AgentState
from app.agent.intent import IntentNode
from app.agent.skills.simple_query import SimpleQuerySkill
from app.agent.skills.complex_query import ComplexQuerySkill
from app.agent.skills.data_chat import DataChatSkill
from app.agent.skills.selector import SkillSelector
from app.events import EventEmitter, NullEmitter, EventType
from app.middleware import new_trace_id, set_trace_id, get_trace_id, NodeTimer, TraceLogger

logger = logging.getLogger(__name__)
trace_logger = TraceLogger("workflow")


class AgentWorkflow:
    def __init__(self):
        self.intent_node = IntentNode()

        self.skills = [
            SimpleQuerySkill(),
            ComplexQuerySkill(),
            DataChatSkill(),
        ]
        self.skill_selector = SkillSelector(self.skills)

    async def execute(
        self,
        user_input: str,
        user_id: str = "default",
        session_id: Optional[str] = None,
        emitter: Optional[EventEmitter] = None
    ) -> AgentState:
        em = emitter or NullEmitter()

        tid = new_trace_id()
        set_trace_id(tid)

        trace_logger.info(f"🚀 开始工作流：user_input={user_input}, user_id={user_id}, session_id={session_id}")

        state = AgentState(
            user_input=user_input,
            user_id=user_id,
            session_id=session_id
        )

        start_time = time.time()

        try:
            em.emit(EventType.THINKING, {
                "phase": "intent",
                "message": "正在识别意图..."
            })

            async with NodeTimer("intent_node"):
                state = await self.intent_node.execute(state, session_id=session_id, emitter=em)

            em.emit(EventType.THINKING, {
                "phase": "skill_selection",
                "message": f"意图识别完成：{state.intent}，正在选择处理策略..."
            })

            skill = await self.skill_selector.select(state, emitter=em)
            state.update(skill_name=skill.name)
            trace_logger.info(f"🎯 选择技能：{skill.name}")

            em.emit(EventType.THINKING, {
                "phase": "skill_execute",
                "message": f"使用 {skill.name} 技能处理",
                "skill": skill.name
            })

            async with NodeTimer(f"skill_{skill.name}"):
                state = await skill.execute(state, session_id, emitter=em)

            elapsed = time.time() - start_time
            trace_logger.info(f"✅ 工作流完成 ({elapsed:.2f}s)：{state.response[:100] if state.response else 'No response'}...")

            if state.clarification_needed:
                em.emit(EventType.CLARIFICATION, {
                    "question": state.clarification_question,
                    "intent": state.intent,
                    "skill": state.skill_name,
                    "session_id": session_id
                })
            else:
                em.emit(EventType.ANSWER, {
                    "response": state.response,
                    "intent": state.intent,
                    "skill": state.skill_name,
                    "sql": state.sql_query,
                    "react_trace": state.react_trace if state.react_trace else None
                })

        except Exception as e:
            elapsed = time.time() - start_time
            trace_logger.error(f"❌ 工作流异常 ({elapsed:.2f}s)：{e}")
            state.update(response=f"抱歉，处理您的请求时出现异常：{str(e)}")
            em.emit(EventType.ERROR, {"message": str(e)})

        em.emit_done(
            session_id=session_id,
            intent=state.intent,
            skill=state.skill_name
        )

        return state


workflow = AgentWorkflow()
