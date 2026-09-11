"""
sql_simple 子图 —— LangGraph 版 SimpleQuerySkill

将原有 SimpleQuerySkill 的流程迁移为 LangGraph StateGraph：
  START → build_context → search_schema → search_examples → sql_loop → END

其中 sql_loop 节点内部实现 generate → validate → execute → reflect 的重试循环（最多 3 次），
避免将循环建模为 LangGraph 边（过于复杂且不必要）。
"""

import json
import logging
import re
from typing import Optional

from langgraph.config import get_config
from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState
from app.agent.tools.langchain_tools import (
    search_schema,
    generate_sql,
    validate_sql,
    execute_sql,
    format_result,
    reflect_error,
    build_context as build_context_tool,
    search_examples,
)
from app.events import EventType, NullEmitter

logger = logging.getLogger(__name__)

# ──────────────────────── 关键词 / 正则（与 SimpleQuerySkill 一致） ────────────────────────

_METADATA_KEYWORDS = [
    "统计", "对比", "排名", "多少", "哪些", "所有", "全部", "列表",
    "平均", "总和", "数量", "部门", "设备", "考勤", "打卡",
    "迟到", "早退", "加班", "上班", "下班", "在职",
]

_EXAMPLES_PATTERNS = [
    r"对比.*和",
    r"既.*又",
    r"最.*前\d+",
    r"趋势",
    r"异常",
    r"变化",
    r"连续",
    r"跨.*协作",
    r"每.*平均",
    r"分析.*各",
]


def _format_query_result(exec_result: str, max_table_rows: int = 20) -> str:
    """将 SQL 查询结果格式化为 Markdown 表格。

    策略：
    - 0 条：提示无数据
    - ≤ max_table_rows：完整表格
    - > max_table_rows：展示前 max_table_rows 行 + “共 N 条”摘要
    """
    try:
        rows = json.loads(exec_result) if isinstance(exec_result, str) else exec_result
    except (json.JSONDecodeError, TypeError):
        return f"查询完成。\n\n```\n{exec_result[:500]}\n```"

    if not isinstance(rows, list) or len(rows) == 0:
        return "✅ 查询完成，未找到匹配的记录。"

    total = len(rows)
    display_rows = rows[:max_table_rows]
    columns = list(display_rows[0].keys()) if display_rows else []

    # 构建 Markdown 表格
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = []
    for row in display_rows:
        cells = [str(row.get(c, "")) for c in columns]
        body_lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join([header, separator] + body_lines)

    if total > max_table_rows:
        return f"✅ 查询完成，共 **{total}** 条记录（展示前 {max_table_rows} 条）：\n\n{table}"
    else:
        return f"✅ 查询完成，共 **{total}** 条记录：\n\n{table}"

def _need_schema_context(question: str) -> bool:
    """判断是否需要搜索 schema（问题含元数据关键词时跳过，否则搜索）"""
    return any(kw in question for kw in _METADATA_KEYWORDS)


def _need_examples(question: str) -> bool:
    """判断是否需要搜索历史示例"""
    return any(re.search(p, question) for p in _EXAMPLES_PATTERNS)


def _get_emitter():
    """从 LangGraph configurable 中获取 emitter，不存在则返回 NullEmitter"""
    config = get_config()
    configurable = config.get("configurable", {})
    emitter = configurable.get("emitter")
    return emitter if emitter else NullEmitter()


# ──────────────────────── 节点实现 ────────────────────────


async def build_context(state: AgentState) -> dict:
    """构建短期记忆上下文（会话历史）"""
    emitter = _get_emitter()
    session_id = state.get("session_id", "")
    current_question = state.get("user_input", "")

    if not session_id:
        return {"metadata_context": ""}

    emitter.emit(EventType.TOOL_CALL, {
        "tool": "build_context",
        "message": "正在构建会话上下文...",
    })

    context_text = await build_context_tool.ainvoke({
        "session_id": session_id,
        "current_question": current_question,
    })

    # build_context_tool 返回空字符串表示无上下文或出错
    if context_text and not context_text.startswith("错误"):
        logger.info(f"🧠 短期记忆上下文：{len(context_text)} 字符")
    else:
        context_text = ""

    emitter.emit(EventType.TOOL_RESULT, {
        "tool": "build_context",
        "success": bool(context_text),
    })

    return {"metadata_context": context_text}


async def search_schema_node(state: AgentState) -> dict:
    """搜索相关表结构，追加到 metadata_context"""
    emitter = _get_emitter()
    question = state.get("user_input", "")

    if not _need_schema_context(question):
        logger.info("🔍 问题不含元数据关键词，跳过 schema 搜索")
        return {}

    emitter.emit(EventType.TOOL_CALL, {
        "tool": "search_schema",
        "message": "正在搜索相关表结构...",
    })

    schema_result = await search_schema.ainvoke({"question": question})

    existing_context = state.get("metadata_context", "")
    if schema_result and not schema_result.startswith(("未找到", "表结构检索失败")):
        new_context = (existing_context + "\n" + schema_result).strip() if existing_context else schema_result
        emitter.emit(EventType.TOOL_RESULT, {
            "tool": "search_schema",
            "success": True,
            "preview": schema_result[:200],
        })
        return {"metadata_context": new_context}

    emitter.emit(EventType.TOOL_RESULT, {
        "tool": "search_schema",
        "success": False,
    })
    return {}


async def search_examples_node(state: AgentState) -> dict:
    """搜索相似查询示例"""
    emitter = _get_emitter()
    question = state.get("user_input", "")

    if not _need_examples(question):
        logger.info("📚 问题不匹配示例模式，跳过示例搜索")
        return {}

    emitter.emit(EventType.TOOL_CALL, {
        "tool": "search_examples",
        "message": "正在搜索相似查询示例...",
    })

    examples_result = await search_examples.ainvoke({
        "question": question,
        "user_id": state.get("user_id", "default"),
    })

    if examples_result and not examples_result.startswith("示例检索失败"):
        emitter.emit(EventType.TOOL_RESULT, {
            "tool": "search_examples",
            "success": True,
        })
        return {"examples_context": examples_result}

    emitter.emit(EventType.TOOL_RESULT, {
        "tool": "search_examples",
        "success": False,
    })
    return {}


async def sql_loop(state: AgentState) -> dict:
    """
    SQL 生成 → 校验 → 执行 → 反思 重试循环

    在单个节点内部完成整个重试逻辑（最多 max_retries 次），
    避免将循环建模为 LangGraph 边带来的状态管理复杂度。
    """
    emitter = _get_emitter()
    max_retries = state.get("max_retries", 3)
    retry_count = state.get("retry_count", 0)
    sql_error = state.get("sql_error", "")
    metadata_context = state.get("metadata_context", "")
    examples_context = state.get("examples_context", "")
    user_input = state.get("user_input", "")
    user_id = state.get("user_id", "default")

    # 长期记忆上下文（如果主图已检索）
    memory_context = ""
    memories = state.get("memories")
    if memories and memories.get("results"):
        memory_texts = []
        for mem in memories["results"][:3]:
            text = mem.get("memory", "")
            if text:
                memory_texts.append(text)
        memory_context = "\n".join(memory_texts)

    # 短期记忆上下文已在 metadata_context 中（build_context 节点写入）

    sql_query = state.get("sql_query", "")

    while retry_count < max_retries:
        retry_count += 1
        error_context = ""

        # ── 反思上一轮错误 ──
        if sql_error:
            emitter.emit(EventType.TOOL_CALL, {
                "tool": "reflect_error",
                "message": f"正在反思第 {retry_count - 1} 次错误...",
            })

            reflection_raw = await reflect_error.ainvoke({
                "user_question": user_input,
                "sql": sql_query,
                "error": sql_error,
                "schema_context": metadata_context,
            })

            try:
                reflection = json.loads(reflection_raw) if isinstance(reflection_raw, str) else reflection_raw
            except (json.JSONDecodeError, TypeError):
                reflection = {}

            next_action = reflection.get("next_action", "regenerate_sql")
            fix_hint = reflection.get("fix_hint", "")
            root_cause = reflection.get("root_cause", "")
            schema_keywords = reflection.get("schema_keywords", []) or []

            emitter.emit(EventType.TOOL_RESULT, {
                "tool": "reflect_error",
                "success": True,
                "next_action": next_action,
                "root_cause": root_cause,
            })

            # 根据反思结论走不同修复路径
            if next_action == "ask_user":
                clarification = fix_hint or "请补充更多查询条件"
                logger.info(f"🤔 反思建议向用户澄清：{clarification}")
                return {
                    "clarification_needed": True,
                    "clarification_question": clarification,
                    "response": clarification,
                    "retry_count": retry_count,
                }

            if next_action == "unrecoverable":
                logger.warning(f"💥 反思判定为不可恢复错误：{root_cause}")
                return {
                    "query_status": "fail",
                    "response": f"查询失败：{root_cause}。{fix_hint}",
                    "retry_count": retry_count,
                }

            if next_action == "need_more_schema" and schema_keywords:
                logger.info(f"🔍 反思建议补搜 schema：{schema_keywords}")
                emitter.emit(EventType.TOOL_CALL, {
                    "tool": "search_schema",
                    "message": f"按反思建议补搜 schema：{schema_keywords}",
                })
                for kw in schema_keywords[:3]:
                    extra = await search_schema.ainvoke({"question": kw})
                    if extra and not extra.startswith(("未找到", "表结构检索失败")):
                        metadata_context = (metadata_context + "\n" + extra).strip()
                emitter.emit(EventType.TOOL_RESULT, {
                    "tool": "search_schema",
                    "success": True,
                    "preview": metadata_context[-300:] if metadata_context else None,
                })

            if next_action == "need_more_examples":
                logger.info("📚 反思建议补搜历史示例")
                extra = await search_examples.ainvoke({
                    "question": user_input,
                    "user_id": user_id,
                })
                if extra and not extra.startswith("示例检索失败"):
                    examples_context = (examples_context + "\n" + extra).strip()

            error_context = (
                f"上次错误：{sql_error}\n"
                f"根因：{root_cause}\n"
                f"修复提示：{fix_hint}"
            )

        # ── 生成 SQL ──
        emitter.emit(EventType.TOOL_CALL, {
            "tool": "generate_sql",
            "message": "正在生成 SQL...",
        })

        sql = await generate_sql.ainvoke({
            "question": user_input,
            "schema_context": metadata_context,
            "examples_context": examples_context,
            "memory_context": memory_context,
            "error_context": error_context,
        })

        if sql.startswith("SQL 生成失败") or sql.startswith("错误"):
            sql_error = sql
            emitter.emit(EventType.TOOL_RESULT, {
                "tool": "generate_sql",
                "success": False,
                "error": sql,
            })
            continue

        sql_query = sql
        emitter.emit(EventType.TOOL_RESULT, {
            "tool": "generate_sql",
            "success": True,
            "sql": sql,
        })

        # ── 校验 SQL ──
        emitter.emit(EventType.TOOL_CALL, {
            "tool": "validate_sql",
            "message": "正在校验 SQL...",
        })

        validation_raw = await validate_sql.ainvoke({"sql": sql})

        try:
            val_result = json.loads(validation_raw) if isinstance(validation_raw, str) else validation_raw
        except (json.JSONDecodeError, TypeError):
            val_result = {"valid": False, "error": "校验结果解析失败"}

        if not val_result.get("valid", False):
            sql_error = val_result.get("error", "校验失败")
            logger.warning(f"⚠️ SQL 校验失败：{sql_error}")
            emitter.emit(EventType.TOOL_RESULT, {
                "tool": "validate_sql",
                "success": False,
                "error": sql_error,
            })
            continue

        emitter.emit(EventType.TOOL_RESULT, {
            "tool": "validate_sql",
            "success": True,
        })

        # ── 执行 SQL ──
        emitter.emit(EventType.TOOL_CALL, {
            "tool": "execute_sql",
            "message": "正在执行 SQL...",
        })

        exec_result = await execute_sql.ainvoke({"sql": sql})

        if exec_result.startswith("SQL 执行失败") or exec_result.startswith("错误"):
            sql_error = exec_result
            logger.warning(f"⚠️ SQL 执行失败：{sql_error}")
            emitter.emit(EventType.TOOL_RESULT, {
                "tool": "execute_sql",
                "success": False,
                "error": exec_result,
            })
            continue

        # ── 执行成功，格式化结果 ──
        emitter.emit(EventType.TOOL_RESULT, {
            "tool": "execute_sql",
            "success": True,
        })

        # 根据结果集大小选择格式化策略：
        # - 小结果集：通过 LLM 生成自然语言回答（流式输出）
        # - 大结果集：直接用代码生成 Markdown 表格（避免 LLM 处理大量数据）
        _MAX_LLM_FORMAT_CHARS = 3000  # 约 750 tokens，超过则用表格
        if len(exec_result) <= _MAX_LLM_FORMAT_CHARS:
            formatted = await format_result.ainvoke({
                "question": user_input,
                "query_result": exec_result,
            })
        else:
            logger.info(f"📊 结果集较大 ({len(exec_result)} 字符)，使用 Markdown 表格直接展示")
            formatted = _format_query_result(exec_result)

        return {
            "response": formatted,
            "sql_query": sql_query,
            "query_status": "success",
            "sql_valid": True,
            "sql_error": "",
            "retry_count": retry_count,
        }

    # ── 超过最大重试次数 ──
    logger.error(f"❌ 超过最大重试次数 ({max_retries})")
    return {
        "query_status": "fail",
        "response": f"查询失败，已重试 {max_retries} 次仍无法成功。最后错误：{sql_error}",
        "retry_count": retry_count,
    }


# ──────────────────────── 构建子图 ────────────────────────


def build_sql_simple_graph() -> StateGraph:
    """
    构建 sql_simple 子图并返回编译后的图。

    流程：START → build_context → search_schema → search_examples → sql_loop → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("build_context", build_context)
    graph.add_node("search_schema", search_schema_node)
    graph.add_node("search_examples", search_examples_node)
    graph.add_node("sql_loop", sql_loop)

    graph.add_edge(START, "build_context")
    graph.add_edge("build_context", "search_schema")
    graph.add_edge("search_schema", "search_examples")
    graph.add_edge("search_examples", "sql_loop")
    graph.add_edge("sql_loop", END)

    return graph.compile()
