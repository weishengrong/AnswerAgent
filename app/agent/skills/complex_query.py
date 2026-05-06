import logging
import json
import uuid
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.agent.skills.base import BaseSkill
from app.agent.state import AgentState
from app.agent.tools.base import ToolResult
from app.agent.tools.schema_tool import SearchSchemaTool
from app.agent.tools.sql_tool import (
    SearchExamplesTool, SearchMemoryTool, GenerateSQLTool,
    ValidateSQLTool, ExecuteSQLTool, FormatResultTool,
    AskUserTool, ReflectErrorTool
)
from app.agent.tools.memory_tool import BuildContextTool
from app.mcp.bridge import get_lazy_mcp_tools
from app.agent.llm_stream import stream_llm
from app.events import EventType
from config.settings import llm_settings

logger = logging.getLogger(__name__)


class ReactStep(BaseModel):
    """ReAct 单步推理的结构化输出。

    通过 pydantic-ai 的 structured output 机制，强制 LLM 把每一步推理输出成
    {thought, action, parameters} 三字段的合法 JSON。相比纯文本 ReAct（输出后
    用正则/json.loads 二次解析），这里把格式约束放到协议层，从根本上消除
    "JSON 解析失败 → parameters 静默置空" 这类静默错误。
    """

    thought: str = Field(
        description="当前步的推理过程：基于已有信息分析下一步要做什么"
    )
    action: str = Field(
        description="下一步动作：必须是可用工具列表中的工具名，或 'done'/'ask_user'"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="工具参数 JSON 对象，键名必须严格匹配工具的参数定义；done 时为空 {}"
    )


_react_llm_model = OpenAIChatModel(
    model_name=llm_settings.LLM_MODEL_NAME,
    provider=OpenAIProvider(
        base_url=llm_settings.LLM_BASE_URL,
        api_key=llm_settings.LLM_API_KEY,
    ),
)

_react_agent = Agent(
    model=_react_llm_model,
    output_type=ReactStep,
    model_settings={"extra_body": {"options": {"num_ctx": 8192}}},
)


class ComplexQuerySkill(BaseSkill):
    name = "complex_query"
    description = "需要多步推理、多表关联、数据对比的复杂查询。适合含歧义、需拆解、需验证的请求，如'对比研发部和市场部的考勤'、'他上个月考勤怎么样'。"
    tools = [
        SearchSchemaTool(), SearchExamplesTool(), SearchMemoryTool(),
        GenerateSQLTool(), ValidateSQLTool(), ExecuteSQLTool(),
        FormatResultTool(), AskUserTool(), ReflectErrorTool(),
        BuildContextTool(), *get_lazy_mcp_tools()
    ]

    MAX_STEPS = 8

    async def execute(self, state: AgentState, session_id: Optional[str] = None, emitter=None) -> AgentState:
        logger.info(f"🔧 ComplexQuerySkill 执行（ReAct模式）：{state.user_input}")

        context_parts = [f"用户问题：{state.user_input}"]

        if state.memories and state.memories.get("results"):
            memory_texts = []
            for mem in state.memories["results"][:3]:
                text = mem.get("memory", "")
                if text:
                    memory_texts.append(text)
            if memory_texts:
                context_parts.append(f"长期记忆：\n" + "\n".join(memory_texts))

        if session_id:
            build_context = self.get_tool("build_context")
            if emitter:
                emitter.emit(EventType.TOOL_CALL, {"tool": "build_context", "message": "正在构建会话上下文..."})
            ctx_result = await build_context.execute(
                session_id=session_id, current_question=state.user_input
            )
            if ctx_result.success and ctx_result.data:
                context_parts.append(f"最近对话：\n{ctx_result.data}")
            if emitter:
                emitter.emit(EventType.TOOL_RESULT, {"tool": "build_context", "success": ctx_result.success})

        context = "\n\n".join(context_parts)

        accumulated_data = []

        for step in range(self.MAX_STEPS):
            logger.info(f"🔄 ReAct 第 {step + 1} 步")

            if emitter:
                emitter.emit(EventType.THINKING, {
                    "phase": "react_think",
                    "message": f"ReAct 第 {step + 1} 步，正在推理...",
                    "step": step + 1
                })

            thought_result = await self._think(context, state, emitter=emitter, step=step + 1)
            if not thought_result:
                break

            thought_text = thought_result.get("thought", "")
            action = thought_result.get("action", "")
            parameters = thought_result.get("parameters", {})

            logger.info(f"💭 Thought: {thought_text}")
            logger.info(f"🎯 Action: {action}({parameters})")

            if action == "done":
                if accumulated_data and not state.response:
                    await self._stream_format_result(
                        state, accumulated_data, emitter
                    )

                if emitter:
                    emitter.emit(EventType.REACT_STEP, {
                        "step": step + 1,
                        "thought": thought_text,
                        "action": "done",
                        "parameters": {},
                        "observation": "推理完成"
                    })
                break

            if action == "ask_user":
                ask_tool = self.get_tool("ask_user")
                question = parameters.get("question", "请提供更多信息")
                if emitter:
                    emitter.emit(EventType.TOOL_CALL, {"tool": "ask_user", "message": "向用户提问..."})
                ask_result = await ask_tool.execute(question=question)
                if ask_result.success:
                    state.update(
                        clarification_needed=True,
                        clarification_question=ask_result.data,
                        response=ask_result.data
                    )
                if emitter:
                    emitter.emit(EventType.TOOL_RESULT, {"tool": "ask_user", "success": ask_result.success})
                state.add_react_step(thought_text, action, parameters, "等待用户回答")
                state.add_node_history("complex_query_skill")
                return state

            tool = self.get_tool(action)
            if not tool:
                available = ", ".join([t.name for t in self.tools] + ["done", "ask_user"])
                observation = f"错误：工具 '{action}' 不存在。可用工具：{available}。请下一步选择正确的工具名。"
                logger.warning(f"⚠️ 工具不存在：{action}")
                context += f"\n\nThought: {thought_text}\nAction: {action}({parameters})\nObservation: {observation}"
                state.add_react_step(thought_text, action, parameters, observation)
                if emitter:
                    emitter.emit(EventType.REACT_STEP, {
                        "step": step + 1,
                        "thought": thought_text,
                        "action": action,
                        "parameters": parameters,
                        "observation": observation,
                        "error": True
                    })
                continue

            if emitter:
                emitter.emit(EventType.TOOL_CALL, {
                    "tool": action,
                    "message": f"正在执行 {action}...",
                    "step": step + 1
                })

            tool_result: Optional[ToolResult] = None
            try:
                tool_result = await tool.execute(**parameters)

                if tool_result.success:
                    obs_data = tool_result.data
                    observation = self._format_observation(action, obs_data)

                    if action == "execute_sql" and isinstance(obs_data, list):
                        accumulated_data.extend(obs_data)
                        state.update(
                            sql_query=parameters.get("sql", state.sql_query),
                            query_result=accumulated_data,
                            query_status="success"
                        )

                    if action == "generate_sql" and isinstance(obs_data, str):
                        state.update(sql_query=obs_data)

                    if action == "reflect_error" and isinstance(obs_data, dict):
                        state.update(reflection=obs_data)
                        # 反思给出 ask_user / unrecoverable 信号时，提前结束 ReAct
                        # 避免无效的下一轮思考浪费 step 配额
                        next_action = obs_data.get("next_action", "")
                        fix_hint = obs_data.get("fix_hint", "")
                        root_cause = obs_data.get("root_cause", "")

                        if next_action == "ask_user":
                            clarification = fix_hint or "请补充更多查询条件"
                            logger.info(f"🤔 反思建议向用户澄清：{clarification}")
                            state.update(
                                clarification_needed=True,
                                clarification_question=clarification,
                                response=clarification,
                            )
                            state.add_react_step(thought_text, action, parameters, observation)
                            if emitter:
                                emitter.emit(EventType.REACT_STEP, {
                                    "step": step + 1,
                                    "thought": thought_text,
                                    "action": action,
                                    "parameters": parameters,
                                    "observation": observation,
                                    "early_exit": "ask_user",
                                })
                            state.add_node_history("complex_query_skill")
                            return state

                        if next_action == "unrecoverable":
                            logger.warning(f"💥 反思判定为不可恢复错误：{root_cause}")
                            state.update(
                                query_status="fail",
                                response=f"查询失败：{root_cause}。{fix_hint}",
                            )
                            state.add_react_step(thought_text, action, parameters, observation)
                            if emitter:
                                emitter.emit(EventType.REACT_STEP, {
                                    "step": step + 1,
                                    "thought": thought_text,
                                    "action": action,
                                    "parameters": parameters,
                                    "observation": observation,
                                    "early_exit": "unrecoverable",
                                })
                            state.add_node_history("complex_query_skill")
                            return state

                else:
                    observation = f"工具执行失败：{tool_result.error}"
                    if action == "execute_sql":
                        state.update(sql_error=tool_result.error or "执行失败")

            except TypeError as e:
                # 参数名/类型不匹配（execute() 收到不识别的关键字参数等）
                # 把错误信息注入下一轮上下文，让 LLM 自纠
                expected = list((tool.parameters or {}).get("properties", {}).keys())
                observation = (
                    f"参数错误：{str(e)}。"
                    f"工具 '{action}' 期望的参数为 {expected}，"
                    f"你给的参数为 {list(parameters.keys())}。"
                    f"请检查参数名后重试。"
                )
                logger.warning(f"⚠️ 参数错误：{e}")
            except Exception as e:
                observation = f"工具执行异常：{str(e)}"
                logger.error(f"❌ 工具执行异常：{e}")

            logger.info(f"👁️ Observation: {observation[:200]}")
            state.add_react_step(thought_text, action, parameters, observation)
            context += f"\n\nThought: {thought_text}\nAction: {action}({json.dumps(parameters, ensure_ascii=False)})\nObservation: {observation}"

            if emitter:
                emitter.emit(EventType.TOOL_RESULT, {
                    "tool": action,
                    "success": bool(tool_result and tool_result.success),
                    "step": step + 1
                })
                emitter.emit(EventType.REACT_STEP, {
                    "step": step + 1,
                    "thought": thought_text,
                    "action": action,
                    "parameters": parameters,
                    "observation": observation[:500]
                })

        if not state.response:
            state.update(response="抱歉，无法完成您的查询请求。")

        state.add_node_history("complex_query_skill")
        return state

    async def _think(self, context: str, state: AgentState, emitter=None, step: int = 1) -> Optional[Dict[str, Any]]:
        """ReAct 单步思考——通过 pydantic-ai structured output 强约束输出格式。

        - thought 字段流式推送到前端（保留 ReAct 的可观测性）
        - action / parameters 由 pydantic schema 在协议层校验，告别 JSON 解析失败
        """
        tool_names = [t.name for t in self.tools] + ["done", "ask_user"]

        prompt = f"""你是一个数据查询 Agent，正在通过 Think-Action-Observation 循环解决用户的数据查询问题。

## 可用工具列表
{self.format_tools_for_prompt()}

## 特殊动作
- `done`：已经获得足够信息可以回答用户，parameters 填空对象
- `ask_user`：需要向用户澄清，parameters 填 {{"question": "你想问用户的问题"}}

## 当前上下文
{context}

## 输出要求
请基于当前上下文，分析下一步动作，输出三个字段：
- `thought`：你的推理过程（说清楚为什么选这个工具/动作）
- `action`：必须从以下列表中选一个：{tool_names}
- `parameters`：工具参数 JSON 对象，键名必须严格匹配工具定义的参数名

## 重要规则
1. 如果已经执行了 SQL 并获得了结果数据，请直接 `action=done`，不要重复查询。
2. 如果上下文中出现 "工具执行失败"、"参数错误" 或 "工具不存在"，请仔细看错误信息，调整下一步：换工具、换参数名、或调用 `reflect_error` 反思。
3. parameters 的键名必须严格匹配工具描述里列出的参数名（例如 execute_sql 必须用 "sql" 而非 "query"）。
"""

        stream_id = f"react_think_{step}_{uuid.uuid4().hex[:6]}"

        try:
            async with _react_agent.run_stream(prompt) as result:
                last_thought = ""
                async for partial in result.stream_output(debounce_by=0.05):
                    thought = getattr(partial, "thought", "") or ""
                    if thought and thought != last_thought:
                        delta = thought[len(last_thought):]
                        last_thought = thought
                        if emitter:
                            emitter.emit(EventType.THINKING_STREAM, {
                                "stream_id": stream_id,
                                "delta": delta,
                                "text": last_thought,
                            })

                final: ReactStep = await result.get_output()

            return {
                "thought": final.thought,
                "action": (final.action or "").strip(),
                "parameters": final.parameters or {},
            }
        except Exception as e:
            logger.error(f"❌ Think 推理失败：{e}")
            if emitter:
                emitter.emit(EventType.THINKING, {
                    "phase": "react_think_error",
                    "message": f"结构化推理失败：{str(e)[:200]}",
                    "step": step,
                })
            return None

    async def _stream_format_result(self, state: AgentState, data: list, emitter=None):
        # 把 ReAct 实际执行的 SQL 和最后一步 thought 注入 prompt，
        # 防止 format LLM 脱离 ReAct 上下文凭 user_input 字面意思瞎归因
        # （典型 bug：fallback 查询里 user_input 提到 A、B 两个名字，
        #  ReAct 实际查的是 B，但 format LLM 把数据套到 A 头上）
        executed_sql = state.sql_query or ""
        last_thought = ""
        if state.react_trace:
            last_thought = state.react_trace[-1].get("thought", "") or ""

        sql_section = (
            f"\n实际执行的 SQL（这是事实证据，回答必须基于此 SQL 的 WHERE 条件，"
            f"不要被『用户问题』里提到的其他名字/条件迷惑）：\n```sql\n{executed_sql}\n```\n"
            if executed_sql else ""
        )
        thought_section = (
            f"\nReAct 最后一步的推理结论（解释了为什么最终查的是这个对象）：\n{last_thought}\n"
            if last_thought else ""
        )

        prompt = f"""用户问题：{state.user_input}
{sql_section}{thought_section}
查询结果数据（JSON格式）：
{json.dumps(data, ensure_ascii=False, default=str)}

请根据用户问题、实际执行的 SQL 和查询结果数据，用自然语言回答用户。
要求：
1. 严格基于实际执行的 SQL 的 WHERE 条件来描述查询主体；如果 user_input 里提到的对象在 SQL 里没出现，要说明"未找到 X，已为您查 Y"
2. 直接回答用户的问题
3. 如果是统计类问题（如有多少人、总数），先给出具体数字
4. 如果是列表类问题，列出所有关键数据
5. 保持准确，如果数据量超过一百条，就返回总结性的信息
6. 如果数据为空或异常，也请说明情况

请直接给出回答，不要说明你是什么模型或解释过程。
"""
        stream_id = f"answer_{uuid.uuid4().hex[:6]}"

        response = await stream_llm(
            prompt, emitter, stream_id,
            event_type=EventType.ANSWER_STREAM
        )

        state.update(response=response.strip())

    def _format_observation(self, action: str, data: Any) -> str:
        if action == "execute_sql" and isinstance(data, list):
            if len(data) > 10:
                return f"查询返回 {len(data)} 条记录，前10条：{json.dumps(data[:10], ensure_ascii=False, default=str)}"
            return json.dumps(data, ensure_ascii=False, default=str)

        if action == "search_schema" and isinstance(data, str):
            if len(data) > 2000:
                return data[:2000] + "\n...(已截断)"
            return data

        # 反思工具返回结构化 dict，转成对 LLM 清晰的多行文本，
        # 让下一轮 think 时能基于 next_action 字段做精准决策
        if action == "reflect_error" and isinstance(data, dict):
            lines = [
                "反思结论：",
                f"- 错误根因：{data.get('root_cause', '')}",
                f"- 建议下一步：{data.get('next_action', '')}",
                f"- 修复提示：{data.get('fix_hint', '')}",
            ]
            kws = data.get("schema_keywords") or []
            if kws:
                lines.append(f"- 需补搜的 schema 关键词：{kws}")
            lines.append(
                "请基于上述反思的 `建议下一步` 选择合适的工具：need_more_schema → 用 search_schema 补搜；"
                "regenerate_sql → 用 generate_sql 按修复提示重新生成；ask_user → 用 ask_user 澄清。"
            )
            return "\n".join(lines)

        if isinstance(data, str):
            if len(data) > 1000:
                return data[:1000] + "\n...(已截断)"
            return data

        return str(data)[:1000]

    def should_handle(self, state: AgentState) -> float:
        if state.intent != "database_query":
            return 0.0
        if state.needs_react or state.need_memory:
            return 0.9
        return 0.3
