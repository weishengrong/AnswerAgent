from pydantic import BaseModel, Field
from enum import Enum
from pydantic_ai import Agent
import logging

from app.events import EventType
from app.agent.state import AgentState
from app.memory.management import memory_manager
from config.settings import llm_settings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    QUERY = "database_query"
    CHAT = "daily_chat"


class IntentResult(BaseModel):
    intent: IntentType = Field(description="识别出的意图类型")
    reason: str = Field(description="判断理由")
    need_memory: bool = Field(description="是否需要检索长期记忆")
    memory_reason: str = Field(description="是否需要检索记忆的理由")
    needs_react: bool = Field(description="是否需要多步推理（仅 database_query 才可能为 true）")
    react_reason: str = Field(description="是否需要多步推理的理由")


llm_model = OpenAIChatModel(
    model_name=llm_settings.LLM_MODEL_NAME,
    provider=OpenAIProvider(
        base_url=llm_settings.LLM_BASE_URL,
        api_key=llm_settings.LLM_API_KEY
    ),
)

# Ollama 默认 num_ctx=2048，加了 few-shot 后 prompt 约 4-5k tokens 会触发 502。
# 这里通过 OpenAI 兼容层的 extra_body 把 ollama 自有的 options.num_ctx 透传过去。
# 对非 ollama 后端无副作用：OpenAI 官方 API 会忽略未知字段。
_OLLAMA_OPTIONS = {"num_ctx": 8192}

intent_agent = Agent(
    model=llm_model,
    output_type=IntentResult,
    model_settings={"extra_body": {"options": _OLLAMA_OPTIONS}},
    system_prompt="""
    你是一个智能意图识别助手。请分析用户输入，完成三项任务。

### 核心原则（最高优先级）
**"宁严勿宽"原则**：如果用户的问题涉及获取具体数据、列表、记录或统计信息，**必须**归类为 `database_query`。只有纯粹的寒暄、问候或与业务完全无关的闲聊才归类为 `daily_chat`。

---

### 任务1：判断意图类型

#### 数据库查询意图 (`database_query`) 的特征
- **获取列表/明细**：询问"全部"、"所有"、"列表"、"名字"、"记录"等（例如："告诉我所有人的名字"、"列出所有设备"）。
- **包含查询条件**：包含具体的人名、部门、时间、设备等筛选条件。
- **统计与聚合**：询问"多少"、"统计"、"总和"、"平均"等。
- **业务实体**：涉及考勤、打卡、用户、部门、设备、订单等业务领域。
- **关键词**：查、找、列出、显示、统计、多少、哪些、全部、所有、名字、记录、信息。

#### 日常聊天意图 (`daily_chat`) 的特征
- **社交礼仪**：你好、再见、谢谢、早上好。
- **能力询问**：你能做什么？你是谁？
- **纯闲聊**：今天天气不错（非查询天气数据）、讲个笑话、心情表达。
- **关于系统/制度的开放性问题**：如"考勤系统好用吗"、"打卡机怎么用"、"考勤数据准不准"、"工资什么时候发"——这些是对系统/制度的主观评价或流程咨询，不是数据查询。
- **注意**：如果用户问"有哪些人"，这是查询，不是聊天。

---

### 任务2：判断是否需要检索长期记忆 (`need_memory`)

`need_memory` 关注的是**这条问题能否独立成立**——是否必须依赖历史对话才能理解。

#### 需要检索 (true) 的情况
- **代词指代**："他"、"她"、"他们"、"那些"、"那个"、"刚才那个"、"上次说的"——指代对象在历史对话中。
- **上下文省略**：问题不完整，必须结合上文（例如："那他们的考勤呢？"、"再给我看一下"）。
- **第一人称归属**："我的"、"我们部门的"——需要从会话/用户上下文确定具体对象。

#### 不需要检索 (false) 的情况
- **独立查询**：问题包含完整的主语和条件（例如："告诉我全部人的名字"、"张三的打卡记录"——人名是常驻实体，不属于代词指代）。
- **通用闲聊**：问候语、能力询问。

**重要**：`need_memory` 只看"指代/省略"，不看"复杂度"。"对比研发部和市场部"虽然复杂但不依赖历史，应为 false。

---

### 任务3：判断是否需要多步推理 (`needs_react`)

`needs_react` 关注的是**这条查询能否用一条 SQL 直接搞定**——还是需要拆解成多步、多次查询、互相依赖。

#### 需要多步推理 (true) 的情况（仅在 intent=database_query 时考虑）
- **多对象对比**：含"对比"、"vs"、"和...哪个"、"差异"，需要分别查再对比。
- **趋势/异常分析**：含"趋势"、"变化"、"异常"、"连续"、"分布"，需要先找规律再筛选。
- **嵌套筛选**：条件之间互相依赖（"既...又"、"在A中且不在B中"、"打卡时间和上班时间相差超过30分钟"）。
- **多步聚合**：需要先聚合再聚合（"按周统计每个部门的平均工作时长"、"出勤率排名"）。
- **跨实体关联**：需要 3 张及以上表关联且含子查询（"跨部门协作项目中各成员的考勤"）。

#### 不需要多步推理 (false) 的情况
- 单表查询：所有用户、设备列表、张三的打卡记录。
- 简单聚合：总人数、研发部多少人、今天迟到几人。
- 简单两表 JOIN：李四在哪个部门、每个部门的平均工资。
- 所有 `daily_chat` 意图——`needs_react` 必须为 false。

---

### Few-shot 示例（重点关注边界场景）

#### 示例1：业务词 ≠ 查询（这些是闲聊）
- 输入：`考勤系统好用吗`
  → `{intent: daily_chat, need_memory: false, needs_react: false}`
  理由：对系统的主观评价，不需要查任何数据。
- 输入：`工资什么时候发`
  → `{intent: daily_chat, need_memory: false, needs_react: false}`
  理由：流程咨询，不是数据查询。
- 输入：`考勤数据准不准`
  → `{intent: daily_chat, need_memory: false, needs_react: false}`
  理由：对系统准确性的疑问，不是要获取具体数据。

#### 示例2：口语化查询（这些是 query，不要被语气骗）
- 输入：`今天出勤怎么样`
  → `{intent: database_query, need_memory: false, needs_react: false}`
  理由：要"今天的出勤数据"，是查询不是闲聊。
- 输入：`有没有什么异常`
  → `{intent: database_query, need_memory: false, needs_react: false}`
  理由：在业务上下文中"异常"指考勤异常记录，是查询。
- 输入：`最近加班多不多`
  → `{intent: database_query, need_memory: false, needs_react: false}`
  理由：要加班数据的统计判断。

#### 示例3：需要记忆 vs 不需要记忆
- 输入：`张三的打卡记录`
  → `{need_memory: false}`
  理由：人名是独立实体，不是代词指代。
- 输入：`我的考勤怎么样`
  → `{need_memory: true}`
  理由："我的"是第一人称归属，需从会话上下文确定 user_id。
- 输入：`我们部门的打卡数据`
  → `{need_memory: true}`
  理由："我们部门"需要从用户上下文确定具体部门。
- 输入：`那他们的考勤呢`
  → `{need_memory: true, needs_react: false}`
  理由：代词指代要查记忆；但本身只是一条简单查询，不需要 ReAct。

#### 示例4：需要 ReAct vs 不需要 ReAct
- 输入：`对比研发部和市场部的考勤情况`
  → `{needs_react: true}`
  理由：两个对象对比，需要分别查再比较。
- 输入：`分析各部门每月的加班趋势`
  → `{needs_react: true}`
  理由：含"趋势"，需要多步聚合 + 时间维度分析。
- 输入：`查询既在A部门又在B部门兼职的员工`
  → `{needs_react: true}`
  理由：嵌套筛选条件互相依赖。
- 输入：`查询打卡时间与上班时间相差超过30分钟的异常记录`
  → `{needs_react: true}`
  理由：需要先计算差值再筛选，单条 SQL 写起来也复杂、需要先理解 schema。
- 输入：`显示每个部门的平均工资`
  → `{needs_react: false}`
  理由：一条 GROUP BY 即可，不需要拆步。
- 输入：`今天有多少人迟到`
  → `{needs_react: false}`
  理由：单条聚合查询。
- 输入：`查询打卡次数最多的前10名员工`
  → `{needs_react: false}`
  理由：ORDER BY + LIMIT 一条 SQL 搞定。

---

请按以下 JSON 格式返回（所有字段必须填写）：
{
  "intent": "database_query 或 daily_chat",
  "reason": "意图判断理由",
  "need_memory": true 或 false,
  "memory_reason": "是否需要检索记忆的理由（聚焦指代/省略，不要混入复杂度）",
  "needs_react": true 或 false,
  "react_reason": "是否需要多步推理的理由（聚焦能否一条SQL搞定）"
}
"""
)


class IntentNode:
    async def retrieve_long_term_memory(
        self,
        user_input: str,
        user_id: str
    ) -> dict:
        try:
            from app.memory.filter import memory_filter

            optimized_query = await memory_filter.extract_search_query(user_input)

            memories = memory_manager.search_memory(
                query=optimized_query,
                user_id=user_id
            )

            if memories and memories.get("results"):
                logger.info(f"🧠 检索到 {len(memories['results'])} 条长期记忆 (query: '{user_input}' → '{optimized_query}')")
                return memories
            else:
                logger.info(f"🧠 未检索到长期记忆")
                return {}
        except Exception as e:
            logger.error(f"❌ 长期记忆检索失败：{e}")
            return {}

    async def execute(
        self,
        state: AgentState,
        session_id: str = None,
        emitter=None
    ) -> AgentState:
        logger.info(f"开始意图识别：{state.user_input}, session_id={session_id}")

        prompt_base = f"用户输入：{state.user_input}\n"

        try:
            result = await intent_agent.run(prompt_base)
            intent_data = result.output

            need_memory = intent_data.need_memory
            memories = {}

            if need_memory:
                if emitter:
                    emitter.emit(EventType.THINKING, {
                        "phase": "memory_retrieval",
                        "message": "正在检索长期记忆..."
                    })
                memories = await self.retrieve_long_term_memory(
                    user_input=state.user_input,
                    user_id=state.user_id
                )
                logger.info(f"📋 需要检索长期记忆：{intent_data.memory_reason}")
            else:
                logger.info(f"📋 无需检索长期记忆：{intent_data.memory_reason}")

            needs_react = intent_data.needs_react and intent_data.intent == IntentType.QUERY

            state.update(
                intent=intent_data.intent.value,
                intent_reason=intent_data.reason,
                need_memory=intent_data.need_memory,
                memory_reason=intent_data.memory_reason,
                needs_react=needs_react,
                react_reason=intent_data.react_reason,
                memories=memories or {}
            )

            if emitter:
                emitter.emit(EventType.THINKING, {
                    "phase": "intent_result",
                    "message": f"意图：{intent_data.intent.value}",
                    "intent": intent_data.intent.value,
                    "need_memory": intent_data.need_memory,
                    "needs_react": needs_react
                })

            logger.info(f"意图识别结果：{intent_data.intent.value}")
            logger.info(f"判断理由：{intent_data.reason}")
            logger.info(f"需要记忆：{intent_data.need_memory} - {intent_data.memory_reason}")
            logger.info(f"需要多步推理：{needs_react} - {intent_data.react_reason}")

        except Exception as e:
            logger.error(f"意图识别失败：{e}")
            state.update(
                intent="daily_chat",
                intent_reason=f"识别失败，降级处理：{str(e)}",
                need_memory=False,
                memory_reason="识别失败，默认不检索",
                needs_react=False,
                react_reason="识别失败，默认不进入 ReAct",
                memories={}
            )
            if emitter:
                emitter.emit(EventType.THINKING, {
                    "phase": "intent_result",
                    "message": f"意图识别失败，降级为日常对话",
                    "intent": "daily_chat"
                })

        state.add_node_history("intent_node")
        return state
