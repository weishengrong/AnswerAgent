import logging
import re
import json
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from app.agent.tools.base import BaseTool, ToolResult
from app.memory.management import memory_manager
from app.memory.session import smart_session_memory
from config.settings import llm_settings
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from prompts.database_schema import DATABASE_SCHEMA, format_schema_for_prompt
from prompts.few_shot_examples import few_shot_examples

logger = logging.getLogger(__name__)


class FixActionType(str, Enum):
    """反思后建议的下一步动作枚举。

    主流程根据这个枚举决定走哪条修复路径，而不是把反思文本无差别拼进 prompt。
    """

    REGENERATE_SQL = "regenerate_sql"          # SQL 本身的问题（语法、列名、聚合错），重新生成即可
    NEED_MORE_SCHEMA = "need_more_schema"      # 缺关键表结构信息，需要补搜 schema 后再重生成
    NEED_MORE_EXAMPLES = "need_more_examples"  # 缺类似示例参考
    ASK_USER = "ask_user"                      # 用户问题本身有歧义，应中止重试并向用户澄清
    UNRECOVERABLE = "unrecoverable"            # 数据库连接/权限等环境问题，重试无意义


class ReflectionResult(BaseModel):
    """SQL 错误反思的结构化输出。

    把原本"一坨文字"的反思结论拆成可被主流程消费的字段：
    - next_action 是关键，主流程据此决定走哪条修复分支
    - schema_keywords 在 next_action=NEED_MORE_SCHEMA 时使用，主动补搜对应 schema
    """

    root_cause: str = Field(description="错误根因，一句话说清楚为什么会出错")
    next_action: FixActionType = Field(description="建议主流程执行的下一步动作")
    fix_hint: str = Field(description="给下一轮的具体修复建议，会被注入 SQL 生成的 error_context")
    schema_keywords: List[str] = Field(
        default_factory=list,
        description="当 next_action=need_more_schema 时，给出需要搜索的关键词列表（如表名、字段名）；其他情况留空"
    )


class SearchExamplesTool(BaseTool):
    name = "search_examples"
    description = "从长期记忆中检索与当前问题相似的历史查询示例。当需要参考类似的SQL写法时使用。"
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "用于检索示例的问题文本"
            },
            "user_id": {
                "type": "string",
                "description": "用户ID"
            }
        },
        "required": ["question"]
    }

    async def execute(self, **kwargs) -> ToolResult:
        question = kwargs.get("question", "")
        user_id = kwargs.get("user_id", "default")

        try:
            memories = memory_manager.search_memory(query=question, user_id=user_id)

            if memories and memories.get("results"):
                examples = []
                for mem in memories["results"]:
                    memory_text = mem.get("memory", "")
                    if memory_text:
                        examples.append(f"- {memory_text}")
                return ToolResult(success=True, data="\n".join(examples))

            return ToolResult(success=True, data="")

        except Exception as e:
            logger.error(f"❌ 示例检索失败：{e}")
            return ToolResult(success=False, error=str(e))


class SearchMemoryTool(BaseTool):
    name = "search_memory"
    description = "从长期记忆中检索与当前问题相关的记忆事实。用于指代消解、用户偏好查询等场景。会自动优化检索query，从用户消息中提取真正需要查找的信息。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索查询文本（可以是原始用户消息，会自动优化）"
            },
            "user_id": {
                "type": "string",
                "description": "用户ID"
            },
            "optimize_query": {
                "type": "boolean",
                "description": "是否优化检索query（默认True，LLM提取检索意图）"
            }
        },
        "required": ["query"]
    }

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        user_id = kwargs.get("user_id", "default")
        optimize_query = kwargs.get("optimize_query", True)

        try:
            if optimize_query:
                from app.memory.filter import memory_filter
                optimized_query = await memory_filter.extract_search_query(query)
                search_query = optimized_query
            else:
                search_query = query

            memories = memory_manager.search_memory(query=search_query, user_id=user_id)

            if memories and memories.get("results"):
                memory_texts = []
                for mem in memories["results"][:3]:
                    text = mem.get("memory", "")
                    if text:
                        memory_texts.append(text)
                return ToolResult(success=True, data="\n".join(memory_texts))

            return ToolResult(success=True, data="")

        except Exception as e:
            logger.error(f"❌ 记忆检索失败：{e}")
            return ToolResult(success=False, error=str(e))


class GenerateSQLTool(BaseTool):
    name = "generate_sql"
    description = "根据用户问题、表结构上下文生成SQL查询语句。返回生成的SQL文本。"
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "用户的自然语言问题"
            },
            "schema_context": {
                "type": "string",
                "description": "相关表结构上下文（可选）"
            },
            "examples_context": {
                "type": "string",
                "description": "相似查询示例（可选）"
            },
            "memory_context": {
                "type": "string",
                "description": "长期记忆上下文（可选）"
            },
            "short_term_context": {
                "type": "string",
                "description": "短期记忆上下文（可选）"
            },
            "error_context": {
                "type": "string",
                "description": "上次SQL错误信息，用于重试修正（可选）"
            }
        },
        "required": ["question"]
    }

    def _get_llm(self):
        llm_model = OpenAIChatModel(
            model_name=llm_settings.LLM_MODEL_NAME,
            provider=OpenAIProvider(
                base_url=llm_settings.LLM_BASE_URL,
                api_key=llm_settings.LLM_API_KEY
            ),
        )
        return Agent(llm_model, output_type=str)

    async def execute(self, **kwargs) -> ToolResult:
        question = kwargs.get("question", "")
        schema_context = kwargs.get("schema_context", "")
        examples_context = kwargs.get("examples_context", "")
        memory_context = kwargs.get("memory_context", "")
        short_term_context = kwargs.get("short_term_context", "")
        error_context = kwargs.get("error_context", "")

        if not question:
            return ToolResult(success=False, error="question 参数不能为空")

        try:
            prompt_parts = []
            prompt_parts.append("【任务】根据用户问题生成 SQL 查询语句\n\n")
            prompt_parts.append("【用户问题分析】\n")
            prompt_parts.append(f"{question}\n\n")

            prompt_parts.append(
                "角色定义：你是一个精通SQL的数据库专家助手。你的任务是根据提供的【数据库表结构信息】和用户的【自然语言问题】，编写准确、可执行的SQL查询语句。\n"
                "核心指令：\n"
                "严格基于上下文：只使用提供的【表结构信息】中的表名和列名。绝对不要臆造不存在的列或表。\n"
                "理解用户意图：仔细分析用户的【自然语言问题】，识别出需要查询的字段、筛选条件、排序方式和聚合需求。\n"
                "关联检索信息：用户的问题可能涉及多个表，你需要根据外键或字段名的语义关联来正确地连接（JOIN）表。\n"
                "输出格式：只输出SQL代码块，不要包含任何解释性文字或寒暄。如果无法生成SQL，请输出 -- 无法根据提供的信息生成SQL。\n"
            )

            if schema_context:
                prompt_parts.append("【相关表结构（来自知识库）】\n")
                prompt_parts.append(f"{schema_context}\n\n")

            if examples_context:
                prompt_parts.append("【相似查询示例】\n")
                prompt_parts.append(f"{examples_context}\n\n")

            if error_context:
                prompt_parts.append("【上次 SQL 错误】\n")
                prompt_parts.append(f"{error_context}\n")
                prompt_parts.append("请修正上述错误后重新生成 SQL。\n\n")

            prompt_parts.append("【数据库 Schema】\n")
            prompt_parts.append(format_schema_for_prompt(DATABASE_SCHEMA))
            prompt_parts.append("\n")

            prompt_parts.append("【Few-Shot 示例】\n")
            for ex in few_shot_examples:
                prompt_parts.append(f"问题：{ex['question']}\nSQL：{ex['sql']}\n\n")

            prompt = "".join(prompt_parts)

            if short_term_context:
                prompt = f"【最近对话】\n{short_term_context}\n\n{prompt}"

            if memory_context:
                prompt = f"【长期记忆】\n{memory_context}\n\n{prompt}"

            sql_agent = self._get_llm()
            result = await sql_agent.run(prompt)
            sql_query = result.output.strip()

            match = re.search(r"```sql\s*(.*?)\s*```", sql_query, re.DOTALL)
            if match:
                sql_query = match.group(1).strip()

            logger.info(f"✅ SQL 生成成功：{sql_query[:100]}...")
            return ToolResult(success=True, data=sql_query)

        except Exception as e:
            logger.error(f"❌ SQL 生成失败：{e}")
            return ToolResult(success=False, error=str(e))


class ValidateSQLTool(BaseTool):
    name = "validate_sql"
    description = "校验SQL语句的语法正确性和安全性（只允许SELECT语句）。返回校验结果和错误信息。"
    parameters = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "待校验的SQL语句"
            }
        },
        "required": ["sql"]
    }

    async def execute(self, **kwargs) -> ToolResult:
        import sqlparse

        sql = kwargs.get("sql", "")
        if not sql:
            return ToolResult(success=False, error="未提供SQL语句")

        sql = sql.strip()

        if not sql.upper().startswith("SELECT"):
            return ToolResult(
                success=True,
                data={"valid": False, "error": "SQL 不安全，只允许执行 SELECT 查询"}
            )

        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return ToolResult(
                    success=True,
                    data={"valid": False, "error": "SQL 语法错误：无法解析"}
                )

            for stmt in parsed:
                tokens = list(stmt.flatten())
                if not tokens:
                    return ToolResult(
                        success=True,
                        data={"valid": False, "error": "SQL 语法错误：空语句"}
                    )

            return ToolResult(success=True, data={"valid": True, "error": ""})

        except Exception as e:
            return ToolResult(
                success=True,
                data={"valid": False, "error": f"SQL 语法错误：{str(e)}"}
            )


class ExecuteSQLTool(BaseTool):
    name = "execute_sql"
    description = "执行SQL查询语句。会先进行EXPLAIN预检，预检通过后执行查询并返回结果数据。"
    parameters = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "要执行的SQL查询语句"
            }
        },
        "required": ["sql"]
    }

    async def execute(self, **kwargs) -> ToolResult:
        from sqlalchemy import text
        from config.db_config import AsyncSessionLocal

        sql = kwargs.get("sql", "")
        if not sql:
            return ToolResult(success=False, error="未提供SQL语句")

        sql = sql.strip()

        if not sql.upper().startswith("SELECT"):
            return ToolResult(success=False, error="只允许执行 SELECT 查询")

        try:
            async with AsyncSessionLocal() as db:
                explain_sql = f"EXPLAIN {sql}"
                explain_result = await db.execute(text(explain_sql))
                explain_output = explain_result.fetchone()
                logger.info(f"🔍 EXPLAIN 预检结果：{explain_output}")

            async with AsyncSessionLocal() as db:
                result = await db.execute(text(sql))
                raw_data = result.mappings().all()
                data = [dict(row) for row in raw_data]

            logger.info(f"✅ SQL 执行成功：{len(data)} 条记录")
            return ToolResult(success=True, data=data)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ SQL 执行失败：{error_msg}")
            return ToolResult(success=False, error=error_msg)


class FormatResultTool(BaseTool):
    name = "format_result"
    description = "将SQL查询结果转换为自然语言回答。当需要把数据结果组织成用户可读的回答时使用。"
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "用户的原始问题"
            },
            "query_result": {
                "type": "string",
                "description": "SQL查询结果的JSON字符串"
            }
        },
        "required": ["question", "query_result"]
    }

    def _get_llm(self):
        llm_model = OpenAIChatModel(
            model_name=llm_settings.LLM_MODEL_NAME,
            provider=OpenAIProvider(
                base_url=llm_settings.LLM_BASE_URL,
                api_key=llm_settings.LLM_API_KEY
            ),
        )
        return Agent(llm_model, output_type=str)

    async def execute(self, **kwargs) -> ToolResult:
        question = kwargs.get("question", "")
        query_result = kwargs.get("query_result", "")

        if not question or not query_result:
            return ToolResult(success=False, error="缺少必要参数")

        try:
            prompt = f"""用户问题：{question}

查询结果数据（JSON格式）：
{query_result}

请根据用户问题和查询结果数据，用自然语言回答用户。
要求：
1. 直接回答用户的问题
2. 如果是统计类问题（如有多少人、总数），先给出具体数字
3. 如果是列表类问题，列出所有关键数据
4. 保持准确，如果数据量超过一百条，就返回总结性的信息
5. 如果数据为空或异常，也请说明情况

请直接给出回答，不要说明你是什么模型或解释过程。
"""
            result_agent = self._get_llm()
            result = await result_agent.run(prompt)
            response = result.output.strip()

            return ToolResult(success=True, data=response)

        except Exception as e:
            logger.error(f"❌ 结果格式化失败：{e}")
            return ToolResult(success=True, data=f"查询结果：{query_result}")


class AskUserTool(BaseTool):
    name = "ask_user"
    description = "当用户问题存在歧义或信息不足时，向用户提问以获取澄清。返回提问内容，等待用户回答。"
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "向用户提出的澄清问题"
            }
        },
        "required": ["question"]
    }

    async def execute(self, **kwargs) -> ToolResult:
        question = kwargs.get("question", "")
        if not question:
            return ToolResult(success=False, error="question 参数不能为空")
        return ToolResult(success=True, data=question)


class ReflectErrorTool(BaseTool):
    name = "reflect_error"
    description = "分析SQL执行错误的原因并给出修正建议。当SQL生成或执行失败时，用于推理错误原因和修正策略。"
    parameters = {
        "type": "object",
        "properties": {
            "user_question": {
                "type": "string",
                "description": "用户的原始问题"
            },
            "sql": {
                "type": "string",
                "description": "生成但执行失败的SQL语句"
            },
            "error": {
                "type": "string",
                "description": "错误信息"
            },
            "schema_context": {
                "type": "string",
                "description": "相关表结构上下文（可选）"
            }
        },
        "required": ["user_question", "sql", "error"]
    }

    def _get_llm(self):
        llm_model = OpenAIChatModel(
            model_name=llm_settings.LLM_MODEL_NAME,
            provider=OpenAIProvider(
                base_url=llm_settings.LLM_BASE_URL,
                api_key=llm_settings.LLM_API_KEY
            ),
        )
        return Agent(
            llm_model,
            output_type=ReflectionResult,
            model_settings={"extra_body": {"options": {"num_ctx": 8192}}},
        )

    async def execute(self, **kwargs) -> ToolResult:
        user_question = kwargs.get("user_question", "")
        sql = kwargs.get("sql", "")
        error = kwargs.get("error", "")
        schema_context = kwargs.get("schema_context", "")

        if not user_question or not sql or not error:
            return ToolResult(success=False, error="缺少必要参数")

        try:
            prompt = f"""你是一个 SQL 调试专家。请分析以下 SQL 执行失败的原因，并给出结构化的修复建议。

## 输入
- 用户问题：{user_question}
- 生成的 SQL：{sql}
- 错误信息：{error}
"""
            if schema_context:
                prompt += f"- 相关表结构：\n{schema_context}\n"

            prompt += """
## 输出字段说明
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
"""

            reflect_agent = self._get_llm()
            result = await reflect_agent.run(prompt)
            reflection: ReflectionResult = result.output

            return ToolResult(success=True, data=reflection.model_dump())

        except Exception as e:
            logger.error(f"❌ 反思分析失败：{e}")
            return ToolResult(success=False, error=str(e))
