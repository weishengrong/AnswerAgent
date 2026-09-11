"""SQL 错误反思 Prompt"""


REFLECT_INPUT_SECTION = """## 输入
- 用户问题：{user_question}
- 生成的 SQL：{sql}
- 错误信息：{error}"""

REFLECT_SCHEMA_SECTION = """- 相关表结构：
{schema_context}"""

REFLECT_OUTPUT_SECTION = """## 输出字段说明
- `root_cause`：错误根因，一句话说清楚（例如 "WHERE 子句把字符串字段当数字比较"）。
- `next_action`：必须从以下枚举中选一个：
  - `regenerate_sql`：SQL 本身的问题，重新生成即可（语法错、列名错、聚合维度错、JOIN 条件错等）。
  - `need_more_schema`：缺关键表结构信息，需要先补搜 schema 才能修。
  - `need_more_examples`：缺类似查询示例，需要补搜历史参考。
  - `ask_user`：用户问题本身有歧义或缺关键限定条件，应该向用户澄清而不是硬猜。
  - `unrecoverable`：数据库连接/权限/超时等环境问题，重试也没意义。
- `fix_hint`：给下一轮 SQL 生成的具体提示（一两句话，明确要改什么）。
- `schema_keywords`：仅当 `next_action=need_more_schema` 时填，列出要搜索的表名/字段名关键词；其他情况留空数组 []。

## 判断准则
1. 如果错误信息里出现 "Unknown column"、"no such column"、"table doesn't exist"，**优先判断为 `need_more_schema`**，并在 `schema_keywords` 里给出缺失的表/列名。
2. 如果是语法错（`syntax error near`）或聚合维度错，但表结构看起来够用，选 `regenerate_sql`。
3. 如果用户问题本身就模糊（如"查一下数据"），选 `ask_user`，`fix_hint` 写需要用户澄清的具体问题。
4. 如果错误是 "Connection refused"、"Access denied"、"timeout"，选 `unrecoverable`。

请严格按以下 JSON 格式输出，不要输出其他内容：
{{
    "root_cause": "错误根因",
    "next_action": "枚举值",
    "fix_hint": "修复建议",
    "schema_keywords": []
}}"""


def build_reflect_prompt(
    sql: str,
    error: str,
    user_question: str = "",
    schema_context: str = "",
) -> str:
    """构建 SQL 反思工具的完整 Prompt 文本"""
    header = f"""你是一个 SQL 调试专家。请分析以下 SQL 执行失败的原因，并给出结构化的修复建议。

{REFLECT_INPUT_SECTION.format(user_question=user_question or "(用户问题已在上下文中)", sql=sql, error=error)}"""

    if schema_context:
        header += "\n" + REFLECT_SCHEMA_SECTION.format(schema_context=schema_context)

    return header + "\n" + REFLECT_OUTPUT_SECTION
