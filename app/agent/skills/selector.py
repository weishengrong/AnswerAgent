import logging
from typing import List, Optional

from app.agent.skills.base import BaseSkill
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


class SkillSelector:
    """纯规则路由技能选择器

    根据意图识别结果做确定性路由：
    - daily_chat → DataChatSkill
    - database_query + need_memory → ComplexQuerySkill
    - database_query + !need_memory → SimpleQuerySkill
    """

    def __init__(self, skills: List[BaseSkill]):
        self.skills = {s.name: s for s in skills}

    async def select(self, state: AgentState, emitter=None) -> Optional[BaseSkill]:
        """基于 should_handle() 评分选择技能"""
        best_skill: Optional[BaseSkill] = None
        best_score = -1.0

        for skill in self.skills.values():
            score = skill.should_handle(state)
            logger.debug(f"📊 技能 {skill.name} 置信度：{score}")
            if score > best_score:
                best_score = score
                best_skill = skill

        if best_skill is None:
            best_skill = list(self.skills.values())[0]

        if emitter:
            from app.events import EventType
            emitter.emit(EventType.THINKING, {
                "phase": "skill_selected",
                "message": f"选择技能：{best_skill.name}",
                "skill": best_skill.name,
                "method": "rule"
            })

        logger.info(f"🎯 选择技能：{best_skill.name}（置信度：{best_score:.2f}）")
        return best_skill
