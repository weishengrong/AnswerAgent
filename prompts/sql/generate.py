"""SQL 生成 Prompt"""


SQL_ROLE_TEMPLATE = """角色定义：你是一个精通SQL的数据库专家助手。你的任务是根据提供的【数据库表结构信息】和用户的【自然语言问题】，编写准确、可执行的SQL查询语句。
核心指令：
严格基于上下文：只使用提供的【表结构信息】中的表名和列名。绝对不要臆造不存在的列或表。
理解用户意图：仔细分析用户的【自然语言问题】，识别出需要查询的字段、筛选条件、排序方式和聚合需求。
关联检索信息：用户的问题可能涉及多个表，你需要根据外键或字段名的语义关联来正确地连接（JOIN）表。
输出格式：只输出SQL代码块，不要包含任何解释性文字或寒暄。如果无法生成SQL，请输出 -- 无法根据提供的信息生成SQL。"""

SQL_SCHEMA_SECTION = """【相关表结构（来自知识库）】
{schema_context}"""

SQL_EXAMPLES_SECTION = """【相似查询示例】
{examples_context}"""

SQL_ERROR_SECTION = """【上次 SQL 错误】
{error_context}
请修正上述错误后重新生成 SQL。"""

SQL_FULL_SCHEMA_SECTION = """【数据库 Schema】
{full_schema}"""

SQL_FEW_SHOT_SECTION = """【Few-Shot 示例】
{few_shot_examples}"""

SQL_SHORT_MEMORY_SECTION = """【最近对话】
{short_term_context}"""

SQL_LONG_MEMORY_SECTION = """【长期记忆】
{memory_context}"""

SQL_GENERATE_HUMAN_TEMPLATE = """【任务】根据用户问题生成 SQL 查询语句

【用户问题分析】
{question}

{role_definition}

{optional_sections}"""


def build_sql_generate_prompt(
    question: str,
    role_definition: str = SQL_ROLE_TEMPLATE,
    schema_context: str = "",
    examples_context: str = "",
    error_context: str = "",
    full_schema: str = "",
    few_shot_examples: str = "",
    short_term_context: str = "",
    memory_context: str = "",
) -> str:
    """构建 SQL 生成工具的完整 Prompt 文本"""
    optional_sections = []

    if schema_context:
        optional_sections.append(SQL_SCHEMA_SECTION.format(schema_context=schema_context))

    if examples_context:
        optional_sections.append(SQL_EXAMPLES_SECTION.format(examples_context=examples_context))

    if error_context:
        optional_sections.append(SQL_ERROR_SECTION.format(error_context=error_context))

    if not schema_context:
        optional_sections.append(SQL_FULL_SCHEMA_SECTION.format(full_schema=full_schema))
    if not examples_context:
        optional_sections.append(SQL_FEW_SHOT_SECTION.format(few_shot_examples=few_shot_examples))

    prefix_parts = []
    if short_term_context:
        prefix_parts.append(SQL_SHORT_MEMORY_SECTION.format(short_term_context=short_term_context))
    if memory_context:
        prefix_parts.append(SQL_LONG_MEMORY_SECTION.format(memory_context=memory_context))

    body = SQL_GENERATE_HUMAN_TEMPLATE.format(
        question=question,
        role_definition=role_definition,
        optional_sections="\n\n".join(optional_sections),
    )

    if prefix_parts:
        body = "\n\n".join(prefix_parts) + "\n\n" + body

    return body
