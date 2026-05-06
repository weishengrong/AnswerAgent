import json
import logging
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field
from config.redis import get_redis_client
from config.settings import memory_settings
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from config.settings import llm_settings

logger = logging.getLogger(__name__)


class SessionMessage(BaseModel):
    role: str = Field(description="user 或 assistant")
    content: str = Field(description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")


class SmartSessionMemory:
    """
    智能短期记忆管理器

    基于 Redis List 实现滑动窗口，并集成 LLM 摘要和长期记忆评估。

    存储策略：
    - Hard Limit: 30 条消息（Redis List 最大长度）
    - Soft Limit: 20 条消息（读取基准上下文）
    - Token 阈值: 5000 tokens（超过触发摘要）
    """

    HARD_LIMIT = memory_settings.SESSION_HARD_LIMIT
    SOFT_LIMIT = memory_settings.SESSION_SOFT_LIMIT
    KEEP_RAW_COUNT = memory_settings.SESSION_KEEP_RAW
    SUMMARIZE_COUNT = memory_settings.SESSION_SUMMARIZE
    TOKEN_THRESHOLD = memory_settings.SESSION_TOKEN_THRESHOLD
    TTL_SECONDS = memory_settings.SESSION_TTL_SECONDS

    def __init__(self):
        self._redis = None
        self._eval_counter = {}
        self._llm = None

    async def _get_redis(self):
        if self._redis is None:
            self._redis = await get_redis_client()
        return self._redis

    def _get_llm(self):
        if self._llm is None:
            llm_model = OpenAIChatModel(
                model_name=llm_settings.LLM_MODEL_NAME,
                provider=OpenAIProvider(
                    base_url=llm_settings.LLM_BASE_URL,
                    api_key=llm_settings.LLM_API_KEY
                ),
            )
            self._llm = Agent(llm_model, output_type=str)
        return self._llm

    def _key(self, session_id: str) -> str:
        return f"smart:session:{session_id}"

    def _user_sessions_key(self, user_id: str) -> str:
        return f"user:{user_id}:sessions"

    async def add_message(
            self,
            session_id: str,
            role: str,
            content: str,
            user_id: str = "default"
    ):
        redis = await self._get_redis()
        key = self._key(session_id)

        msg = SessionMessage(role=role, content=content, timestamp=datetime.now())
        msg_json = msg.model_dump_json()

        await redis.rpush(key, msg_json)
        await redis.expire(key, self.TTL_SECONDS)

        user_sessions_key = self._user_sessions_key(user_id)
        await redis.sadd(user_sessions_key, session_id)
        await redis.expire(user_sessions_key, 86400 * 30)

        session_meta_key = f"session:{session_id}:meta"
        first_msg_key = f"session:{session_id}:first_msg"
        exists = await redis.exists(first_msg_key)
        if not exists:
            preview = content[:50] if content else ""
            await redis.set(first_msg_key, preview)
            await redis.expire(first_msg_key, self.TTL_SECONDS)
            await redis.hset(session_meta_key, mapping={
                "session_id": session_id,
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "first_msg": preview
            })
            await redis.expire(session_meta_key, self.TTL_SECONDS)

        current_len = await redis.llen(key)

        if current_len > self.HARD_LIMIT:
            await redis.lpop(key)

        if current_len == self.HARD_LIMIT:
            if session_id not in self._eval_counter:
                self._eval_counter[session_id] = 0
                await self._trigger_batch_evaluate(
                    await self.get_all_messages(session_id),
                    user_id,
                    session_id
                )
            else:
                self._eval_counter[session_id] += 1
                if self._eval_counter[session_id] >= self.HARD_LIMIT:
                    await self._trigger_batch_evaluate(
                        await self.get_all_messages(session_id),
                        user_id,
                        session_id
                    )
                    self._eval_counter[session_id] = 0

    async def add_conversation(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        user_id: str = "default"
    ):
        await self.add_message(session_id, "user", user_message, user_id)
        await self.add_message(session_id, "assistant", assistant_message, user_id)
        logger.info(f"💾 保存对话到短期记忆：{session_id}")

    async def get_recent_messages(
        self,
        session_id: str,
        limit: int = None
    ) -> List[SessionMessage]:
        redis = await self._get_redis()
        key = self._key(session_id)

        if limit is None:
            limit = self.SOFT_LIMIT

        raw_messages = await redis.lrange(key, -limit, -1)

        messages = []
        for raw in raw_messages:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                msg_data = json.loads(raw)
                messages.append(SessionMessage(**msg_data))
            except Exception as e:
                logger.warning(f"⚠️ 解析消息失败：{e}")
                continue

        return messages

    async def get_all_messages(self, session_id: str) -> List[dict]:
        redis = await self._get_redis()
        key = self._key(session_id)

        raw_messages = await redis.lrange(key, 0, -1)

        messages = []
        for raw in raw_messages:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                msg_data = json.loads(raw)
                messages.append(msg_data)
            except Exception as e:
                logger.warning(f"⚠️ 解析消息失败：{e}")
                continue

        return messages

    async def _summarize_messages(self, messages: List[SessionMessage]) -> str:
        llm = self._get_llm()

        conversation_text = "\n".join([
            f"{'用户' if msg.role == 'user' else '助手'}: {msg.content}"
            for msg in messages
        ])

        prompt = f"""请将以下对话内容压缩为一段简洁的摘要，保留关键信息。

对话内容：
{conversation_text}

要求：
1. 保留关键事实和数据
2. 保留用户偏好和习惯
3. 保留重要决策和结论
4. 保留未完成的任务或问题
5. 忽略寒暄和礼貌性对话
6. 摘要不超过200字

请直接输出摘要内容："""

        try:
            result = await llm.run(prompt)
            summary = result.output.strip()
            logger.info(f"📝 生成摘要：{len(summary)} 字符")
            return summary
        except Exception as e:
            logger.error(f"❌ 生成摘要失败：{e}")
            return ""

    async def build_context(
        self,
        session_id: str,
        current_question: str = ""
    ) -> str:
        messages = await self.get_recent_messages(session_id, limit=self.SOFT_LIMIT)

        if not messages:
            return ""

        total_chars = sum(len(msg.content) for msg in messages)
        estimated_tokens = total_chars // 4

        if estimated_tokens <= self.TOKEN_THRESHOLD:
            context_lines = []
            for msg in messages:
                role = "用户" if msg.role == "user" else "助手"
                context_lines.append(f"{role}: {msg.content}")
            return "\n".join(context_lines)

        recent_messages = messages[-self.KEEP_RAW_COUNT:]
        older_messages = messages[:-self.KEEP_RAW_COUNT]

        if older_messages:
            to_summarize = older_messages[-self.SUMMARIZE_COUNT:]
            summary = await self._summarize_messages(to_summarize)
        else:
            summary = ""

        context_parts = []

        if summary:
            context_parts.append(f"【早前对话摘要】\n{summary}")

        if recent_messages:
            context_parts.append("【近期对话（保持连贯）】")
            for msg in recent_messages:
                role = "用户" if msg.role == "user" else "助手"
                context_parts.append(f"{role}: {msg.content}")

        return "\n".join(context_parts)

    async def _trigger_batch_evaluate(self, messages: list, user_id: str, session_id: str):
        from app.memory.management import memory_manager
        from app.memory.filter import memory_filter

        try:
            result = await memory_filter.evaluate_batch(messages, user_id, session_id)

            if result.get("memories_to_store"):
                for memory_text in result["memories_to_store"]:
                    memory_manager.add_memory(
                        [{"role": "system", "content": memory_text}],
                        user_id=user_id
                    )
                logger.info(
                    f"💾 批量评估存储：{result['count']} 条到长期记忆 | "
                    f"过滤：{len(result.get('filtered_out', []))} 条 | "
                    f"去重：{len(result.get('duplicates', []))} 条"
                )
            else:
                logger.info(f"⏭️ 批量评估：无值得存储的内容")

        except Exception as e:
            logger.error(f"❌ 批量评估失败：{e}")

    async def clear(self, session_id: str, user_id: str = "default"):
        redis = await self._get_redis()
        key = self._key(session_id)

        messages = await self.get_all_messages(session_id)

        if messages:
            await self._trigger_batch_evaluate(messages, user_id, session_id)

        await redis.delete(key)

        if session_id in self._eval_counter:
            del self._eval_counter[session_id]

        logger.info(f"🗑️ 清理会话短期记忆：{session_id}")

    async def list_sessions(self, user_id: str = "default") -> list:
        redis = await self._get_redis()
        user_sessions_key = self._user_sessions_key(user_id)

        session_ids = await redis.smembers(user_sessions_key)
        if not session_ids:
            return []

        sessions = []
        for sid in session_ids:
            if isinstance(sid, bytes):
                sid = sid.decode("utf-8")

            session_meta_key = f"session:{sid}:meta"
            meta = await redis.hgetall(session_meta_key)
            if not meta:
                continue

            decoded_meta = {}
            for k, v in meta.items():
                key = k.decode("utf-8") if isinstance(k, bytes) else k
                val = v.decode("utf-8") if isinstance(v, bytes) else v
                decoded_meta[key] = val

            first_msg = decoded_meta.get("first_msg", "")
            created_at = decoded_meta.get("created_at", "")

            msg_count = await redis.llen(self._key(sid))

            sessions.append({
                "session_id": sid,
                "first_msg": first_msg,
                "created_at": created_at,
                "message_count": msg_count
            })

        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return sessions

    async def get_session_history(self, session_id: str) -> list:
        messages = await self.get_all_messages(session_id)
        result = []
        for msg in messages:
            result.append({
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", "")
            })
        return result


smart_session_memory = SmartSessionMemory()
