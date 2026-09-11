"""Prompt 模板包

统一 Prompt 模板管理模块，使用 LangChain ChatPromptTemplate 组件管理所有 Prompt。

使用方式：
    from prompts import INTENT_PROMPT, build_chat_messages
"""

from prompts.intent import INTENT_PROMPT
from prompts.chat import CHAT_PROMPT, build_chat_messages
from prompts.react import REACT_PROMPT, build_react_system_message
from prompts.coordinator import SubTask, SubTaskList, build_coordinator_message
from prompts.sql import build_sql_generate_prompt, FORMAT_RESULT_PROMPT, build_reflect_prompt
from prompts.agents import (
    build_schema_agent_message,
    build_sql_agent_message,
    build_analysis_agent_message,
)
