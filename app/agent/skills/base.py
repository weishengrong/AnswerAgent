from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from app.agent.state import AgentState
from app.agent.tools.base import BaseTool


class BaseSkill(ABC):
    name: str = ""
    description: str = ""
    tools: List[BaseTool] = []

    @abstractmethod
    async def execute(self, state: AgentState, session_id: Optional[str] = None, emitter=None) -> AgentState:
        ...

    def should_handle(self, state: AgentState) -> float:
        return 0.0

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def format_tools_for_prompt(self) -> str:
        lines = []
        for tool in self.tools:
            lines.append(tool.to_prompt_format())
        return "\n\n".join(lines)
