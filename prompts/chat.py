"""Chat 对话 Prompt"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


CHAT_SYSTEM_TEMPLATE = """你是一个友好的数据查询助手，正在与用户进行日常对话。请根据历史记忆（如果有）来回答用户，保持对话的连贯性。"""

CHAT_SHORT_MEMORY_TEMPLATE = """【最近对话】
{short_term_context}"""

CHAT_LONG_MEMORY_TEMPLATE = """【长期记忆】
{long_term_context}"""

CHAT_SCHEMA_TEMPLATE = """【数据库表结构概览】
{schema_context}"""

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CHAT_SYSTEM_TEMPLATE),
    MessagesPlaceholder("optional_context"),
    HumanMessagePromptTemplate.from_template("{user_input}"),
])


def build_chat_messages(
    user_input: str,
    short_term_context: str = "",
    long_term_context: str = "",
    schema_context: str = "",
) -> list:
    """构建 Chat 节点的消息列表（处理可选段落）"""
    optional_parts = []
    if short_term_context:
        optional_parts.append(CHAT_SHORT_MEMORY_TEMPLATE.format(short_term_context=short_term_context))
    if long_term_context:
        optional_parts.append(CHAT_LONG_MEMORY_TEMPLATE.format(long_term_context=long_term_context))
    if schema_context:
        optional_parts.append(CHAT_SCHEMA_TEMPLATE.format(schema_context=schema_context))

    context_text = "\n\n".join(optional_parts)
    return CHAT_PROMPT.format_messages(
        user_input=user_input,
        optional_context=[{"type": "system", "content": context_text}] if context_text else [],
    )
