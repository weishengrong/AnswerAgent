import logging
import re
import uuid
from typing import Optional

from app.agent.skills.base import BaseSkill
from app.agent.state import AgentState
from app.agent.tools.schema_tool import SearchSchemaTool
from app.agent.tools.sql_tool import (
    SearchExamplesTool, SearchMemoryTool, GenerateSQLTool,
    ValidateSQLTool, ExecuteSQLTool, FormatResultTool, ReflectErrorTool
)
from app.agent.tools.memory_tool import BuildContextTool
from app.mcp.bridge import get_lazy_mcp_tools
from app.agent.llm_stream import stream_llm
from app.events import EventType, NullEmitter

logger = logging.getLogger(__name__)


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


def _need_schema_context(question: str) -> bool:
    return any(kw in question for kw in _METADATA_KEYWORDS)


def _need_examples(question: str) -> bool:
    return any(re.search(p, question) for p in _EXAMPLES_PATTERNS)


class SimpleQuerySkill(BaseSkill):
    name = "simple_query"
    description = "简单的单表或两表查询，一条SQL即可完成。适合明确的、不含歧义的数据查询请求，如'查询所有用户'、'张三的打卡记录'。"
    tools = [
        SearchSchemaTool(), SearchExamplesTool(), SearchMemoryTool(),
        GenerateSQLTool(), ValidateSQLTool(), ExecuteSQLTool(),
        FormatResultTool(), ReflectErrorTool(), BuildContextTool(),
        *get_lazy_mcp_tools()
    ]

    async def execute(self, state: AgentState, session_id: Optional[str] = None, emitter=None) -> AgentState:
        logger.info(f"🔧 SimpleQuerySkill 执行：{state.user_input}")

        search_schema = self.get_tool("search_schema")
        search_examples = self.get_tool("search_examples")
        search_memory = self.get_tool("search_memory")
        generate_sql = self.get_tool("generate_sql")
        validate_sql = self.get_tool("validate_sql")
        execute_sql = self.get_tool("execute_sql")
        format_result = self.get_tool("format_result")
        reflect_error = self.get_tool("reflect_error")
        build_context = self.get_tool("build_context")

        schema_context = ""
        examples_context = ""
        memory_context = ""
        short_term_context = ""

        if session_id:
            if emitter:
                emitter.emit(EventType.TOOL_CALL, {"tool": "build_context", "message": "正在构建会话上下文..."})
            ctx_result = await build_context.execute(
                session_id=session_id, current_question=state.user_input
            )
            if ctx_result.success and ctx_result.data:
                short_term_context = ctx_result.data
                logger.info(f"🧠 短期记忆上下文：{len(short_term_context)} 字符")
            if emitter:
                emitter.emit(EventType.TOOL_RESULT, {"tool": "build_context", "success": ctx_result.success})

        use_metadata = _need_schema_context(state.user_input)
        if use_metadata:
            if emitter:
                emitter.emit(EventType.TOOL_CALL, {"tool": "search_schema", "message": "正在搜索相关表结构..."})
            result = await search_schema.execute(question=state.user_input)
            if result.success and result.data:
                schema_context = result.data
                state.update(metadata_context=schema_context)
            if emitter:
                emitter.emit(EventType.TOOL_RESULT, {
                    "tool": "search_schema",
                    "success": result.success,
                    "preview": schema_context[:200] if schema_context else None
                })

        use_examples = _need_examples(state.user_input)
        if use_examples:
            if emitter:
                emitter.emit(EventType.TOOL_CALL, {"tool": "search_examples", "message": "正在搜索相似查询示例..."})
            result = await search_examples.execute(
                question=state.user_input, user_id=state.user_id
            )
            if result.success and result.data:
                examples_context = result.data
                state.update(examples_context=examples_context)
            if emitter:
                emitter.emit(EventType.TOOL_RESULT, {"tool": "search_examples", "success": result.success})

        if state.need_memory and state.memories and state.memories.get("results"):
            memory_texts = []
            for mem in state.memories["results"][:3]:
                text = mem.get("memory", "")
                if text:
                    memory_texts.append(text)
            memory_context = "\n".join(memory_texts)

        for attempt in range(state.max_retries):
            state.update(retry_count=attempt + 1)

            error_context = ""
            if attempt > 0 and state.sql_error:
                if emitter:
                    emitter.emit(EventType.TOOL_CALL, {"tool": "reflect_error", "message": f"正在反思第{attempt}次错误..."})
                reflect_result = await reflect_error.execute(
                    user_question=state.user_input,
                    sql=state.sql_query or "",
                    error=state.sql_error,
                    schema_context=schema_context
                )

                if reflect_result.success and isinstance(reflect_result.data, dict):
                    reflection = reflect_result.data
                    next_action = reflection.get("next_action", "regenerate_sql")
                    fix_hint = reflection.get("fix_hint", "")
                    root_cause = reflection.get("root_cause", "")
                    schema_keywords = reflection.get("schema_keywords", []) or []

                    state.update(reflection=reflection)

                    if emitter:
                        emitter.emit(EventType.TOOL_RESULT, {
                            "tool": "reflect_error",
                            "success": True,
                            "next_action": next_action,
                            "root_cause": root_cause,
                        })

                    # 根据反思结论走不同的修复路径，而不是把反思文本无差别拼进 prompt
                    if next_action == "ask_user":
                        # 问题本身歧义，重试也是浪费——直接中止重试，向用户澄清
                        clarification = fix_hint or "请补充更多查询条件"
                        logger.info(f"🤔 反思建议向用户澄清：{clarification}")
                        state.update(
                            clarification_needed=True,
                            clarification_question=clarification,
                            response=clarification,
                        )
                        state.add_node_history("simple_query_skill")
                        return state

                    if next_action == "unrecoverable":
                        # 环境问题，重试无意义，直接失败返回
                        logger.warning(f"💥 反思判定为不可恢复错误：{root_cause}")
                        state.update(
                            query_status="fail",
                            response=f"查询失败：{root_cause}。{fix_hint}",
                        )
                        state.add_node_history("simple_query_skill")
                        return state

                    if next_action == "need_more_schema" and schema_keywords:
                        # 主动按反思给的关键词补搜 schema，让下一轮 generate_sql 拿到更全的上下文
                        logger.info(f"🔍 反思建议补搜 schema：{schema_keywords}")
                        if emitter:
                            emitter.emit(EventType.TOOL_CALL, {
                                "tool": "search_schema",
                                "message": f"按反思建议补搜 schema：{schema_keywords}"
                            })
                        for kw in schema_keywords[:3]:
                            extra = await search_schema.execute(question=kw)
                            if extra.success and extra.data:
                                schema_context = (schema_context + "\n" + extra.data).strip()
                        state.update(metadata_context=schema_context)
                        if emitter:
                            emitter.emit(EventType.TOOL_RESULT, {
                                "tool": "search_schema",
                                "success": True,
                                "preview": schema_context[-300:] if schema_context else None,
                            })

                    if next_action == "need_more_examples":
                        logger.info("📚 反思建议补搜历史示例")
                        extra = await search_examples.execute(
                            question=state.user_input, user_id=state.user_id
                        )
                        if extra.success and extra.data:
                            examples_context = (examples_context + "\n" + extra.data).strip()
                            state.update(examples_context=examples_context)

                    error_context = (
                        f"上次错误：{state.sql_error}\n"
                        f"根因：{root_cause}\n"
                        f"修复提示：{fix_hint}"
                    )
                else:
                    error_context = state.sql_error
                    if emitter:
                        emitter.emit(EventType.TOOL_RESULT, {"tool": "reflect_error", "success": False})

            if emitter:
                emitter.emit(EventType.TOOL_CALL, {"tool": "generate_sql", "message": "正在生成 SQL..."})
            gen_result = await generate_sql.execute(
                question=state.user_input,
                schema_context=schema_context,
                examples_context=examples_context,
                memory_context=memory_context,
                short_term_context=short_term_context,
                error_context=error_context
            )

            if not gen_result.success:
                state.update(query_status="fail", sql_error=gen_result.error or "SQL生成失败")
                if emitter:
                    emitter.emit(EventType.TOOL_RESULT, {"tool": "generate_sql", "success": False, "error": gen_result.error})
                continue

            sql = gen_result.data
            state.update(sql_query=sql)
            if emitter:
                emitter.emit(EventType.TOOL_RESULT, {"tool": "generate_sql", "success": True, "sql": sql})

            if emitter:
                emitter.emit(EventType.TOOL_CALL, {"tool": "validate_sql", "message": "正在校验 SQL..."})
            val_result = await validate_sql.execute(sql=sql)
            if not val_result.success:
                state.update(sql_valid=False, sql_error=val_result.error or "校验工具调用失败")
                if emitter:
                    emitter.emit(EventType.TOOL_RESULT, {"tool": "validate_sql", "success": False, "error": val_result.error})
                continue

            val_data = val_result.data
            if not val_data.get("valid", False):
                state.update(sql_valid=False, sql_error=val_data.get("error", "校验失败"))
                logger.warning(f"⚠️ SQL 校验失败：{val_data.get('error')}")
                if emitter:
                    emitter.emit(EventType.TOOL_RESULT, {"tool": "validate_sql", "success": False, "error": val_data.get("error")})
                continue

            state.update(sql_valid=True, sql_error="")
            if emitter:
                emitter.emit(EventType.TOOL_RESULT, {"tool": "validate_sql", "success": True})

            if emitter:
                emitter.emit(EventType.TOOL_CALL, {"tool": "execute_sql", "message": "正在执行 SQL..."})
            exec_result = await execute_sql.execute(sql=sql)
            if exec_result.success:
                state.update(query_result=exec_result.data, query_status="success")
                if emitter:
                    emitter.emit(EventType.TOOL_RESULT, {
                        "tool": "execute_sql",
                        "success": True,
                        "row_count": len(exec_result.data) if isinstance(exec_result.data, list) else None
                    })

                await self._stream_format_result(
                    state, str(exec_result.data), emitter
                )

                state.add_node_history("simple_query_skill")
                return state
            else:
                state.update(query_status="fail", sql_error=exec_result.error or "执行失败")
                logger.warning(f"⚠️ SQL 执行失败：{exec_result.error}")
                if emitter:
                    emitter.emit(EventType.TOOL_RESULT, {"tool": "execute_sql", "success": False, "error": exec_result.error})

        logger.error(f"❌ 超过最大重试次数 ({state.max_retries})")
        state.update(
            query_status="fail",
            response=f"查询失败，已重试 {state.max_retries} 次仍无法成功。最后错误：{state.sql_error}"
        )
        state.add_node_history("simple_query_skill")
        return state

    async def _stream_format_result(self, state: AgentState, query_result: str, emitter=None):
        prompt = f"""用户问题：{state.user_input}

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
        stream_id = f"answer_{uuid.uuid4().hex[:6]}"
        em = emitter or NullEmitter()

        response = await stream_llm(
            prompt, em, stream_id,
            event_type=EventType.ANSWER_STREAM
        )

        state.update(response=response.strip())

    def should_handle(self, state: AgentState) -> float:
        if state.intent != "database_query":
            return 0.0
        if state.needs_react or state.need_memory:
            return 0.3
        return 0.7
