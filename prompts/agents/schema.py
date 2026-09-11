"""Schema Agent Prompt"""


SCHEMA_AGENT_SYSTEM_TEMPLATE = """你是一个数据库表结构检索专家。你的任务是根据用户问题，检索相关的数据库表结构信息。

## 可用工具
- search_schema: 根据问题检索相关表结构信息
- search_examples: 检索相似的历史查询示例
- search_memory: 检索长期记忆中的相关信息

## 工作流程
1. 先调用 search_schema 检索与问题相关的表结构
2. 如果需要参考示例，调用 search_examples
3. 如果需要历史上下文，调用 search_memory
4. 整合所有检索结果，输出完整的表结构描述

## 规则
1. 最多进行 {max_steps} 步推理
2. 每步必须调用一个工具，不要空转
3. 获得足够信息后，直接输出整合后的表结构描述，不要再调用工具
4. 如果检索不到相关表结构，输出"未找到相关表结构\""""


def build_schema_agent_message(
    tools_description: str,
    max_steps: int,
    user_input: str,
    memory_texts: str = "",
    session_context: str = "",
) -> str:
    """构建 SchemaAgent 的完整 System Message"""
    system_content = SCHEMA_AGENT_SYSTEM_TEMPLATE.format(max_steps=max_steps)

    context_parts = []
    context_parts.append(f"用户问题：{user_input}")
    if memory_texts:
        context_parts.append(f"长期记忆：\n{memory_texts}")
    if session_context:
        context_parts.append(f"最近对话：\n{session_context}")

    system_content += "\n\n" + "\n\n".join(context_parts)
    return system_content
