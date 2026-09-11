"""ReAct 系统 Prompt"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)


REACT_SYSTEM_TEMPLATE = """你是一个数据查询 Agent，正在通过 Think-Action-Observation 循环解决用户的数据查询问题。

## 可用工具列表
{tools_description}

## 特殊动作
- 当你已经获得足够信息可以回答用户时，直接使用 `format_result` 工具格式化最终回答，不要再继续查询。
- 当你需要向用户澄清时，使用 `ask_user` 工具。

## 重要规则
1. 如果已经执行了 SQL 并获得了结果数据，请直接使用 `format_result` 工具格式化回答，不要重复查询。
2. 如果上下文中出现 "工具执行失败"、"参数错误" 或 "工具不存在"，请仔细看错误信息，调整下一步：换工具、换参数名、或调用 `reflect_error` 反思。
3. 工具参数的键名必须严格匹配工具描述里列出的参数名（例如 execute_sql 必须用 "sql" 而非 "query"）。
4. 每一步都要先思考当前已知信息和还缺什么，再选择合适的工具。
5. 推荐流程：search_schema → search_examples/search_memory → generate_sql → validate_sql → execute_sql → format_result
6. 如果 SQL 执行失败，先使用 reflect_error 分析原因，再根据反思结果决定下一步。
7. 如果 reflect_error 返回 next_action 为 ask_user 或 unrecoverable，请直接使用 format_result 或 ask_user 结束流程。
8. 最多进行 {max_steps} 步推理，请合理规划每一步。
"""

REACT_CONTEXT_TEMPLATE = """## 当前上下文
{context_info}"""

REACT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(REACT_SYSTEM_TEMPLATE),
    MessagesPlaceholder("context_messages"),
    MessagesPlaceholder("chat_history"),
])


def build_react_system_message(
    tools_description: str,
    max_steps: int,
    user_input: str = "",
    memory_texts: str = "",
    session_context: str = "",
    react_trace: str = "",
) -> str:
    """构建 ReAct 节点的完整 System Message"""
    system_content = REACT_SYSTEM_TEMPLATE.format(
        tools_description=tools_description,
        max_steps=max_steps,
    )

    context_parts = []
    if user_input:
        context_parts.append(f"用户问题：{user_input}")
    if memory_texts:
        context_parts.append(f"长期记忆：\n{memory_texts}")
    if session_context:
        context_parts.append(f"最近对话：\n{session_context}")
    if react_trace:
        context_parts.append(f"已有推理轨迹：\n{react_trace}")

    if context_parts:
        system_content += "\n\n" + REACT_CONTEXT_TEMPLATE.format(context_info="\n\n".join(context_parts))

    return system_content
