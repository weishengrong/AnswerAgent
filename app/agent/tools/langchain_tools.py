"""
LangChain @tool 适配层

将所有原有 BaseTool 工具迁移为 LangChain 原生 @tool 函数，
供 LangGraph / LangChain Agent 直接消费。
"""

import json
import logging
import re
import sys
import uuid
from enum import Enum
from typing import List, Optional

from langchain_core.tools import tool
from langgraph.config import get_config
from pydantic import BaseModel, Field

from app.events import EventType

logger = logging.getLogger(__name__)


# ──────────────────────────── Prompt 截断安全网 ────────────────────────────

def _truncate_prompt(prompt: str, max_chars: int) -> str:
    """截断 prompt 到指定字符数，优先保留关键段落（问题 + 角色 + Schema）

    策略：从后往前丢弃可选段落，保留核心的 question + role_definition + schema_context
    """
    if len(prompt) <= max_chars:
        return prompt

    # 按段落分割，识别核心段和可选段
    sections = []
    # 用段落标题作为分割点
    import re as _re
    pattern = _re.compile(r'(【[^】]+】)')
    parts = pattern.split(prompt)

    current_section = {"title": "", "content": "", "is_core": False}
    for i, part in enumerate(parts):
        if pattern.match(part):
            if current_section["content"] or current_section["title"]:
                sections.append(current_section)
            current_section = {
                "title": part,
                "content": "",
                "is_core": _is_core_section(part),
            }
        else:
            current_section["content"] += part
    if current_section["content"] or current_section["title"]:
        sections.append(current_section)

    # 先尝试只保留核心段
    core_text = "".join(s["title"] + s["content"] for s in sections if s["is_core"])
    if len(core_text) <= max_chars:
        return core_text

    # 核心段也超了，硬截断（保留头部）
    logger.warning(f"⚠️ 核心段仍超限（{len(core_text)} 字符），执行硬截断")
    return core_text[:max_chars - 200] + "\n\n...（内容已因长度限制截断）"


def _is_core_section(title: str) -> bool:
    """判断是否为核心段落（必须保留）"""
    core_keywords = [
        "任务", "用户问题分析", "角色定义",
        "相关表结构", "数据库 Schema",  # 至少保留一个 Schema 来源
    ]
    return any(kw in title for kw in core_keywords)


# ──────────────────────────── 结构化输出模型 ────────────────────────────

class FixActionType(str, Enum):
    """反思后建议的下一步动作枚举"""
    REGENERATE_SQL = "regenerate_sql"
    NEED_MORE_SCHEMA = "need_more_schema"
    NEED_MORE_EXAMPLES = "need_more_examples"
    ASK_USER = "ask_user"
    UNRECOVERABLE = "unrecoverable"


class ReflectionResult(BaseModel):
    """SQL 错误反思的结构化输出"""
    root_cause: str = Field(description="错误根因，一句话说清楚为什么会出错")
    next_action: FixActionType = Field(description="建议主流程执行的下一步动作")
    fix_hint: str = Field(description="给下一轮的具体修复建议，会被注入 SQL 生成的 error_context")
    schema_keywords: List[str] = Field(
        default_factory=list,
        description="当 next_action=need_more_schema 时，给出需要搜索的关键词列表；其他情况留空"
    )


# ──────────────────────────── 辅助：MCP 子进程调用 ────────────────────────────

async def _call_mcp_time_tool(tool_name: str, params: dict) -> str:
    """通过子进程调用 MCP time_server.py，返回文本结果"""
    import asyncio

    script_path = "mcp_servers/time_server.py"

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": params
        }
    }

    proc = await asyncio.create_subprocess_exec(
        sys.executable, script_path,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    proc.stdin.write((json.dumps(request) + "\n").encode())
    proc.stdin.write(b"\n")
    await proc.stdin.drain()

    try:
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=30.0)
        response = json.loads(line.decode())

        if "result" in response:
            content = response["result"].get("content", [])
            if content and isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "\n".join(texts)
            return json.dumps(response["result"], ensure_ascii=False)
        elif "error" in response:
            return json.dumps(response["error"], ensure_ascii=False)
        else:
            return json.dumps(response, ensure_ascii=False)
    except asyncio.TimeoutError:
        proc.kill()
        return json.dumps({"error": "MCP 调用超时"}, ensure_ascii=False)
    finally:
        try:
            proc.kill()
        except Exception:
            pass


# ──────────────────────────── 辅助：Redis 关系映射 ────────────────────────────

async def _get_relation_map() -> dict:
    """从 Redis 加载表关系映射"""
    from app.core.config.redis import get_redis_client

    redis_client = await get_redis_client()
    relation_map = {}
    async for key in redis_client.scan_iter("rag:relation:*"):
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        table_name = key.replace("rag:relation:", "")
        data = await redis_client.get(key)
        if data:
            relation_map[table_name] = json.loads(data)
    return relation_map


# ──────────────────────────── Schema 工具 ────────────────────────────

@tool
async def search_schema(question: str) -> str:
    """检索与用户问题相关的数据库表结构信息，包括字段详情、表关系等。当需要了解数据库有哪些表、某个表的字段结构时使用。

    Args:
        question: 用于检索表结构的问题文本
    """
    if not question:
        return "错误：question 参数不能为空"

    try:
        from app.rag.retrieval import retrieve_schema, retrieve_table_info_simple

        relation_map = await _get_relation_map()

        if relation_map:
            metadata = await retrieve_schema(question, relation_map)
        else:
            metadata = await retrieve_table_info_simple(question)

        if not metadata or "未找到" in metadata:
            return "未找到相关表结构信息"

        logger.info(f"📚 表结构检索成功：{len(metadata)} 字符")
        return metadata

    except Exception as e:
        logger.error(f"❌ 表结构检索失败：{e}")
        return f"表结构检索失败：{e}"


# ──────────────────────────── SQL 工具 ────────────────────────────

@tool
async def search_examples(question: str, user_id: str = "default") -> str:
    """从长期记忆中检索与当前问题相似的历史查询示例。当需要参考类似的SQL写法时使用。

    Args:
        question: 用于检索示例的问题文本
        user_id: 用户ID
    """
    try:
        from app.memory.management import memory_manager

        memories = memory_manager.search_memory(query=question, user_id=user_id)

        if memories and memories.get("results"):
            examples = []
            for mem in memories["results"]:
                memory_text = mem.get("memory", "")
                if memory_text:
                    examples.append(f"- {memory_text}")
            return "\n".join(examples)

        return ""

    except Exception as e:
        logger.error(f"❌ 示例检索失败：{e}")
        return f"示例检索失败：{e}"


@tool
async def search_memory(query: str, user_id: str = "default", optimize_query: bool = True) -> str:
    """从长期记忆中检索与当前问题相关的记忆事实。用于指代消解、用户偏好查询等场景。会自动优化检索query，从用户消息中提取真正需要查找的信息。

    Args:
        query: 检索查询文本（可以是原始用户消息，会自动优化）
        user_id: 用户ID
        optimize_query: 是否优化检索query（默认True，LLM提取检索意图）
    """
    try:
        from app.memory.management import memory_manager

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
            return "\n".join(memory_texts)

        return ""

    except Exception as e:
        logger.error(f"❌ 记忆检索失败：{e}")
        return f"记忆检索失败：{e}"


@tool
async def generate_sql(question: str, error_context: str = "") -> str:
    """根据用户问题生成SQL查询语句，返回生成的SQL文本。会自动检索表结构和历史示例。

    Args:
        question: 用户的自然语言问题
        error_context: 上次SQL错误信息（可选，用于重试修正）
    """
    if not question:
        return "错误：question 参数不能为空"

    try:
        from langchain_core.messages import HumanMessage
        from app.core.llm import get_llm, LLMClientError, LLMTimeoutError, LLMRateLimitError
        from prompts.database_schema import DATABASE_SCHEMA, format_schema_for_prompt
        from prompts.few_shot_examples import few_shot_examples
        from prompts.sql.generate import build_sql_generate_prompt

        # 准备 full_schema 和 few_shot_examples 文本（保持原有逻辑）
        full_schema_text = format_schema_for_prompt(DATABASE_SCHEMA)
        few_shot_text = "\n".join(
            f"问题：{ex['question']}\nSQL：{ex['sql']}" for ex in few_shot_examples
        )

        prompt = build_sql_generate_prompt(
            question=question,
            schema_context="",  # schema 由内部 DATABASE_SCHEMA 提供
            examples_context="",  # examples 由内部 few_shot_examples 提供
            error_context=error_context or "",
            full_schema=full_schema_text,
            few_shot_examples=few_shot_text,
            short_term_context="",
            memory_context="",
        )

        # ── Prompt 诊断日志：各 Section 字符数 / 预估 Token 数 ──
        _prompt_len = len(prompt)
        _est_tokens = _prompt_len // 4  # 粗略估算：中文约 4 字符/token
        logger.info(
            f"📏 Prompt 组装完成：{_prompt_len} 字符（约 {_est_tokens} tokens）| "
            f"question={len(question)} error={len(error_context or '')} "
            f"full_schema={len(full_schema_text)} few_shot={len(few_shot_text)}"
        )

        # ── Token 安全网：超过阈值时截断（留出空间给系统提示 + 响应）──
        MAX_SAFE_TOKENS = 200000  # 模型上限 262144，预留余量
        if _est_tokens > MAX_SAFE_TOKENS:
            logger.warning(
                f"⚠️ Prompt 预估 {_est_tokens} tokens 超过安全阈值 {MAX_SAFE_TOKENS}，"
                f"触发截断保护"
            )
            prompt = _truncate_prompt(prompt, max_chars=MAX_SAFE_TOKENS * 4)

        # ✨ 使用场景化超时配置（SQL生成允许更长超时）
        llm = get_llm(timeout_key='sql_generation')

        logger.info(f"🔄 开始调用 LLM 生成 SQL (timeout=120s)...")
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        sql_query = response.content.strip()

        logger.info(f"📥 LLM 原始返回：{response.content}")

        match = re.search(r"```sql\s*(.*?)\s*```", sql_query, re.DOTALL)
        if match:
            sql_query = match.group(1).strip()

        logger.info(f"✅ SQL 生成成功：{sql_query[:100]}...")
        return sql_query

    except LLMClientError as e:
        # 客户端错误（400 Bad Request 等），通常不可重试
        error_msg = f"SQL生成失败（客户端错误 {e.status_code}）"

        # 特殊处理：prompt 过长导致的 400 错误
        if e.status_code == 400:
            if len(prompt) > 50000:  # 超过 50KB 认为是 prompt 过长
                error_msg = (
                    "SQL生成失败：输入内容过长导致请求被拒绝。"
                    "建议简化查询条件或缩小数据范围后重试。"
                )
                logger.warning(
                    f"⚠️ Prompt 过长 ({len(prompt)} 字符) 导致 400 错误 | "
                    f"question={len(question)} schema={len(full_schema_text)}"
                )
            else:
                error_msg = (
                    "SQL生成失败：请求参数无效。可能是表结构信息格式异常，"
                    "建议稍后重试或联系管理员。"
                )
                logger.error(f"❌ SQL 生成遇到 400 错误: {e}")

        return error_msg

    except LLMTimeoutError as e:
        # 超时错误
        error_msg = "SQL生成失败：模型响应超时，请稍后重试或简化查询条件。"
        logger.error(f"❌ SQL 生成超时: {e}")
        return error_msg

    except LLMRateLimitError as e:
        # 限流错误
        retry_after = getattr(e, 'retry_after', None) or "几秒"
        error_msg = f"SQL生成失败：API 调用频率限制，请等待 {retry_after} 后重试。"
        logger.warning(f"⚠️ SQL 生成触发限流: {e}")
        return error_msg

    except Exception as e:
        # 其他未知异常
        logger.error(f"❌ SQL 生成失败（未知错误）: {type(e).__name__}: {e}")
        return f"SQL生成失败：系统内部错误，请稍后重试。（{str(e)[:100]}）"


@tool
async def validate_sql(sql: str) -> str:
    """校验SQL语句的语法正确性和安全性（只允许SELECT语句）。返回校验结果和错误信息。

    Args:
        sql: 待校验的SQL语句
    """
    import sqlparse

    if not sql:
        return "错误：未提供SQL语句"

    sql = sql.strip()

    if not sql.upper().startswith("SELECT"):
        return json.dumps({"valid": False, "error": "SQL 不安全，只允许执行 SELECT 查询"}, ensure_ascii=False)

    try:
        parsed = sqlparse.parse(sql)
        if not parsed:
            return json.dumps({"valid": False, "error": "SQL 语法错误：无法解析"}, ensure_ascii=False)

        for stmt in parsed:
            tokens = list(stmt.flatten())
            if not tokens:
                return json.dumps({"valid": False, "error": "SQL 语法错误：空语句"}, ensure_ascii=False)

        return json.dumps({"valid": True, "error": ""}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"valid": False, "error": f"SQL 语法错误：{str(e)}"}, ensure_ascii=False)


@tool
async def execute_sql(sql: str) -> str:
    """执行SQL查询语句。会先进行EXPLAIN预检，预检通过后执行查询并返回结果数据。

    Args:
        sql: 要执行的SQL查询语句
    """
    from sqlalchemy import text
    from app.core.config.db_config import AsyncSessionLocal

    if not sql:
        return "错误：未提供SQL语句"

    sql = sql.strip()

    if not sql.upper().startswith("SELECT"):
        return "错误：只允许执行 SELECT 查询"

    # 结果集行数上限（防止大数据量导致后续 LLM 调用超限）
    MAX_RESULT_ROWS = 500

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

        total_count = len(data)
        if total_count > MAX_RESULT_ROWS:
            data = data[:MAX_RESULT_ROWS]
            logger.warning(
                f"⚠️ SQL 结果集过大：{total_count} 条 → 截断为前 {MAX_RESULT_ROWS} 条"
            )

        logger.info(f"✅ SQL 执行成功：{len(data)} 条记录")
        return json.dumps(data, ensure_ascii=False, default=str)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ SQL 执行失败：{error_msg}")
        return f"SQL 执行失败：{error_msg}"


@tool
async def format_result(question: str, query_result: str) -> str:
    """将SQL查询结果转换为自然语言回答。当需要把数据结果组织成用户可读的回答时使用。

    Args:
        question: 用户的原始问题
        query_result: SQL查询结果的JSON字符串
    """
    if not question or not query_result:
        return "错误：缺少必要参数"

    try:
        from app.core.llm import get_llm, LLMTimeoutError, LLMClientError, LLMRateLimitError
        from prompts.sql.format import FORMAT_RESULT_PROMPT

        # ── 结果集大小诊断 + 截断保护 ──
        _result_len = len(query_result)
        logger.info(f"📊 format_result 输入：{_result_len} 字符（约 {_result_len // 4} tokens）")

        MAX_RESULT_CHARS = 80000  # ~20K tokens，留足余量给 prompt 模板
        if _result_len > MAX_RESULT_CHARS:
            logger.warning(
                f"⚠️ format_result 结果集过大（{_result_len} 字符），"
                f"截断为 {MAX_RESULT_CHARS} 字符"
            )
            # 尝试解析 JSON 后只保留前 N 条
            try:
                _rows = json.loads(query_result)
                if isinstance(_rows, list):
                    _truncated = _rows[:100]  # 最多 100 条
                    query_result = json.dumps(_truncated, ensure_ascii=False, default=str)
                    query_result += f"\n\n...（共 {len(_rows)} 条记录，已截断显示前 {len(_truncated)} 条）"
            except (json.JSONDecodeError, TypeError):
                # 不是 JSON，硬截断
                query_result = query_result[:MAX_RESULT_CHARS] + "\n\n...（结果已截断）"

        messages = FORMAT_RESULT_PROMPT.invoke({
            "question": question,
            "query_result": query_result,
        })

        # ✨ 使用场景化超时配置（格式化结果应该快一些）
        # 加 skip_stream tag，让 on_chat_model_stream handler 跳过此调用（避免产生多余的“推理过程”）
        llm = get_llm(timeout_key='chat').with_config(tags=["skip_stream"])

        # 从 LangGraph config 获取 emitter，用于流式输出最终答案
        _config = get_config()
        _emitter = _config.get("configurable", {}).get("emitter") if _config else None

        # 流式输出：逐 token 发射 ANSWER_STREAM（最终答案实时展示）
        _stream_id = f"fmt_{uuid.uuid4().hex[:8]}"
        _full_text = ""
        async for chunk in llm.astream(messages):
            token = chunk.content if isinstance(chunk.content, str) else str(chunk.content or "")
            if token:
                _full_text += token
                if _emitter:
                    _emitter.emit(EventType.ANSWER_STREAM, {
                        "stream_id": _stream_id,
                        "text": _full_text
                    })
        return _full_text.strip()

    except LLMTimeoutError as e:
        error_msg = "结果格式化超时，请重试。"
        logger.error(f"❌ 结果格式化超时: {e}")
        return f"{error_msg}\n\n原始查询结果：{query_result[:500]}..."

    except LLMClientError as e:
        error_msg = f"结果格式化失败（客户端错误 {e.status_code}），请重试。"
        logger.error(f"❌ 结果格式化遇到客户端错误: {e}")
        return f"{error_msg}\n\n原始查询结果：{query_result[:500]}..."

    except LLMRateLimitError as e:
        retry_after = getattr(e, 'retry_after', None) or "几秒"
        error_msg = f"结果格式化失败：API 调用频率限制，请等待 {retry_after} 后重试。"
        logger.warning(f"⚠️ 结果格式化触发限流: {e}")
        return f"{error_msg}\n\n原始查询结果：{query_result[:500]}..."

    except Exception as e:
        logger.error(f"❌ 结果格式化失败（未知错误）: {type(e).__name__}: {e}")
        return f"查询结果：{query_result}"


@tool
async def ask_user(question: str) -> str:
    """当用户问题存在歧义或信息不足时，向用户提问以获取澄清。返回提问内容，等待用户回答。

    Args:
        question: 向用户提出的澄清问题
    """
    if not question:
        return "错误：question 参数不能为空"
    return question


@tool
async def reflect_error(sql: str, error: str) -> str:
    """分析SQL执行错误原因并给出修正建议。当SQL执行失败时使用。

    Args:
        sql: 执行失败的SQL语句
        error: 错误信息
    """
    if not sql or not error:
        return "错误：缺少必要参数"

    try:
        from langchain_core.messages import HumanMessage
        from app.core.llm import get_structured_llm, LLMError, LLMTimeoutError, LLMRateLimitError
        from prompts.sql.reflect import build_reflect_prompt

        prompt = build_reflect_prompt(
            sql=sql,
            error=error,
        )

        # ✨ 使用 get_structured_llm 获取带自动重试能力的包装对象
        # 加 skip_stream tag 隐藏内部反思过程（不展示为“推理过程”）
        structured_llm = get_structured_llm(ReflectionResult, timeout_key='chat', tags=["skip_stream"])
        reflection: ReflectionResult = await structured_llm.ainvoke([HumanMessage(content=prompt)])

        return json.dumps(reflection.model_dump(), ensure_ascii=False)

    except (LLMTimeoutError, LLMRateLimitError) as e:
        # 超时或限流错误，返回默认的降级策略（建议重新生成 SQL）
        logger.warning(f"⚠️ 反思分析遇到 {type(e).__name__}: {e}")
        fallback_result = ReflectionResult(
            root_cause=f"反思分析失败（{type(e).__name__}）",
            next_action="regenerate_sql",
            fix_hint="重新生成SQL",
            schema_keywords=[]
        )
        return json.dumps(fallback_result.model_dump(), ensure_ascii=False)

    except LLMError as e:
        # 其他 LLM 错误（客户端错误、服务端错误等）
        logger.warning(f"⚠️ 反思分析失败（LLM错误）: {e}")
        fallback_result = ReflectionResult(
            root_cause=f"反思分析失败（LLM错误 {e.status_code or 'unknown'}）",
            next_action="regenerate_sql",
            fix_hint="重新生成SQL，如持续失败请联系管理员",
            schema_keywords=[]
        )
        return json.dumps(fallback_result.model_dump(), ensure_ascii=False)

    except Exception as e:
        # 其他未知异常
        logger.error(f"❌ 反思分析失败（未知错误）: {type(e).__name__}: {e}")
        fallback_result = ReflectionResult(
            root_cause="未知错误",
            next_action="regenerate_sql",
            fix_hint="重新生成SQL",
            schema_keywords=[]
        )
        return json.dumps(fallback_result.model_dump(), ensure_ascii=False)


# ──────────────────────────── Memory 工具 ────────────────────────────

@tool
async def build_context(session_id: str, current_question: str = "") -> str:
    """构建短期记忆上下文，获取当前会话的最近对话历史。用于理解用户的上下文语境。

    Args:
        session_id: 会话ID
        current_question: 当前用户问题
    """
    if not session_id:
        return ""

    try:
        from app.memory.session import smart_session_memory

        context = await smart_session_memory.build_context(
            session_id=session_id,
            current_question=current_question
        )
        return context
    except Exception as e:
        logger.error(f"❌ 构建上下文失败：{e}")
        return ""


# ──────────────────────────── MCP 时间工具 ────────────────────────────

@tool
async def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间。当用户问现在几点、当前时间时使用。

    Args:
        timezone: 时区，默认 Asia/Shanghai
    """
    try:
        result = await _call_mcp_time_tool("get_current_time", {"timezone": timezone})
        return result
    except Exception as e:
        logger.error(f"❌ MCP 获取当前时间失败：{e}")
        return f"获取当前时间失败：{e}"


@tool
async def get_date_range(period: str) -> str:
    """获取自然语言描述的日期范围。当用户问"上个月"、"本周"、"今年"等时间范围时使用。

    Args:
        period: 时间段描述，如"上个月"、"本周"、"今年"
    """
    try:
        result = await _call_mcp_time_tool("get_date_range", {"period": period})
        return result
    except Exception as e:
        logger.error(f"❌ MCP 获取日期范围失败：{e}")
        return f"获取日期范围失败：{e}"


@tool
async def get_relative_date(expression: str) -> str:
    """获取相对日期。当用户问"3天前"、"下周一"等相对时间时使用。

    Args:
        expression: 相对日期表达式，如"3天前"、"下周一"
    """
    try:
        result = await _call_mcp_time_tool("get_relative_date", {"expression": expression})
        return result
    except Exception as e:
        logger.error(f"❌ MCP 获取相对日期失败：{e}")
        return f"获取相对日期失败：{e}"


@tool
async def is_workday(date: str) -> str:
    """判断某日期是否为工作日。当用户需要判断某天是否上班时使用。

    Args:
        date: 日期字符串，如"2025-01-01"
    """
    try:
        result = await _call_mcp_time_tool("is_workday", {"date": date})
        return result
    except Exception as e:
        logger.error(f"❌ MCP 判断工作日失败：{e}")
        return f"判断工作日失败：{e}"


@tool
async def get_workdays(start_date: str, end_date: str) -> str:
    """计算两个日期之间的工作日天数。当用户问某段时间有多少工作日时使用。

    Args:
        start_date: 起始日期，如"2025-01-01"
        end_date: 结束日期，如"2025-01-31"
    """
    try:
        result = await _call_mcp_time_tool("get_workdays", {"start_date": start_date, "end_date": end_date})
        return result
    except Exception as e:
        logger.error(f"❌ MCP 获取工作日天数失败：{e}")
        return f"获取工作日天数失败：{e}"
