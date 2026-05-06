import math
from typing import Dict, Optional, Literal, List
from datetime import datetime
from pydantic import BaseModel
import logging
import numpy as np
import json

from config.settings import embedding
from app.memory.management import memory_manager

logger = logging.getLogger(__name__)


class ContentAnalysis(BaseModel):
    persistence: float = 0.0
    uniqueness: float = 0.0
    actionability: float = 0.0
    entity_density: float = 0.0
    memory_type: Literal["state", "event", "fact", "chat"] = "chat"
    raw_importance: float = 0.0


class ImportanceScore(BaseModel):
    content_score: float = 0.0
    relevance_score: float = 0.0
    recency_score: float = 0.0
    final_score: float = 0.0
    decision: Literal["WRITE", "IGNORE", "KEEP"] = "IGNORE"
    decision_reason: str = ""
    content_details: Optional[ContentAnalysis] = None


class MemoryFilter:
    """
    长期记忆过滤器 - 基于多维度评分机制

    评估维度：
    1. 内容重要性: LLM 结构化评估（持久性、独特性、可操作性）
    2. 相关性: 动态上下文窗口 + 混合相似度
    3. 时效性: 半衰期模型（基于类型动态衰减）

    综合得分 = 0.6×内容分 + 0.25×时效分 + 0.15×相关分

    双阈值机制：
    - 高水位线 0.75: 立即存储
    - 低水位线 0.60: 坚决不存
    - 中间地带: 保持现状
    """

    HIGH_WATERMARK = 0.75
    LOW_WATERMARK = 0.60

    CONTENT_WEIGHT = 0.6
    RECENCY_WEIGHT = 0.25
    RELEVANCE_WEIGHT = 0.15

    RELEVANCE_THRESHOLD = 0.3

    DECAY_LAMBDA_STATE = 10.0
    DECAY_LAMBDA_EVENT = 0.5
    DECAY_LAMBDA_FACT = 0.01
    DECAY_LAMBDA_CHAT = 2.0

    DEDUP_SIMILARITY_THRESHOLD = 0.9

    def __init__(self):
        self._llm = None
        self._embedding_client = None
        self._context_window = 30
        logger.info("✅ MemoryFilter 初始化完成")
        logger.info(f"   - 高水位线: {self.HIGH_WATERMARK}")
        logger.info(f"   - 低水位线: {self.LOW_WATERMARK}")
        logger.info(f"   - 权重: 内容={self.CONTENT_WEIGHT} 时效={self.RECENCY_WEIGHT} 相关={self.RELEVANCE_WEIGHT}")
        logger.info(f"   - 去重阈值: {self.DEDUP_SIMILARITY_THRESHOLD}")

    def _get_llm(self):
        if self._llm is None:
            from pydantic_ai import Agent
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
            from config.settings import llm_settings

            llm_model = OpenAIChatModel(
                model_name=llm_settings.LLM_MODEL_NAME,
                provider=OpenAIProvider(
                    base_url=llm_settings.LLM_BASE_URL,
                    api_key=llm_settings.LLM_API_KEY
                ),
            )
            self._llm = Agent(llm_model)
        return self._llm

    def _get_embedding_client(self):
        if self._embedding_client is None:
            from openai import OpenAI
            self._embedding_client = OpenAI(
                api_key=embedding.EMBEDDING_MODEL_API_KEY,
                base_url=embedding.EMBEDDING_MODEL_URL
            )
        return self._embedding_client

    async def analyze_content(self, memory_text: str) -> ContentAnalysis:
        llm = self._get_llm()

        prompt = f"""你是一个内容重要性评估专家。请分析以下记忆内容，输出一个 JSON 对象。

记忆内容：{memory_text}

评分标准：
1. persistence (持久性, 0-10):
   - "我现在饿了" → 0 (马上过时)
   - "我是素食主义者" → 10 (长期有效)
   - "刚才报错了" → 5 (短期有效)

2. uniqueness (独特性, 0-10):
   - "今天天气不错" → 0 (通用常识)
   - "我喜欢吃辣" → 8 (用户偏好)
   - "我叫韦胜荣" → 10 (用户特有事实)

3. actionability (可操作性, 0-10):
   - "帮我查一下考勤" → 10 (明确指令)
   - "我想要减肥" → 7 (目标陈述)
   - "哈哈" → 0 (无操作)

4. entity_density (实体密度, 0-10):
   - 包含人名、时间、数字 → 高
   - 纯主观感受 → 低

5. memory_type (记忆类型):
   - state: 临时状态 ("我现在很生气")
   - event: 事件记录 ("刚才报错")
   - fact: 永久事实 ("我叫张三")
   - chat: 闲聊 ("你好")

6. raw_importance (原始重要性, 1-10):
   基于以上综合给出 1-10 的重要性评分

请严格按以下 JSON 格式输出，不要输出其他内容：
{{
    "persistence": 0-10的数字,
    "uniqueness": 0-10的数字,
    "actionability": 0-10的数字,
    "entity_density": 0-10的数字,
    "memory_type": "state/event/fact/chat",
    "raw_importance": 1-10的数字
}}
"""

        try:
            result = await llm.run(prompt)
            output = result.output.strip()

            data = json.loads(output)

            analysis = ContentAnalysis(
                persistence=float(data.get("persistence", 5)),
                uniqueness=float(data.get("uniqueness", 5)),
                actionability=float(data.get("actionability", 5)),
                entity_density=float(data.get("entity_density", 5)),
                memory_type=data.get("memory_type", "chat"),
                raw_importance=float(data.get("raw_importance", 5))
            )

            logger.debug(
                f"📊 内容分析 - 持久性:{analysis.persistence} 独特性:{analysis.uniqueness} "
                f"可操作性:{analysis.actionability} 类型:{analysis.memory_type}"
            )
            return analysis

        except Exception as e:
            logger.error(f"❌ 内容分析失败：{e}")
            return ContentAnalysis(
                persistence=5, uniqueness=5, actionability=5,
                entity_density=5, memory_type="chat", raw_importance=5
            )

    def calculate_content_score(self, analysis: ContentAnalysis) -> float:
        if analysis.persistence < 2:
            return 0.0

        score = (
            analysis.persistence * 0.5 +
            analysis.uniqueness * 0.3 +
            analysis.actionability * 0.2
        )

        score = max(0.0, min(10.0, score))

        logger.debug(f"📊 内容分计算: {score} = {analysis.persistence}×0.5 + {analysis.uniqueness}×0.3 + {analysis.actionability}×0.2")
        return score

    async def build_context_vector(self, session_id: str) -> Optional[str]:
        if not session_id:
            return None

        try:
            from app.memory.session import smart_session_memory

            recent_messages = await smart_session_memory.get_recent_messages(
                session_id, limit=self._context_window
            )

            if not recent_messages:
                return None

            conversation_text = "\n".join([
                f"{'用户' if msg.role == 'user' else '助手'}: {msg.content}"
                for msg in recent_messages[-self._context_window:]
            ])

            llm = self._get_llm()
            prompt = f"""请用一句话概括以下对话的主题：
{conversation_text}

要求：
- 不超过20个字
- 只输出主题描述，不要其他内容
- 例如："用户询问Python异步编程"、"用户查询考勤记录"
"""

            result = await llm.run(prompt)
            summary = result.output.strip()

            logger.debug(f"📊 对话摘要: {summary}")
            return summary

        except Exception as e:
            logger.error(f"❌ 构建上下文摘要失败：{e}")
            return None

    def calculate_relevance(
        self,
        current_message: str,
        existing_memories: list,
        context_summary: Optional[str] = None
    ) -> float:
        if not existing_memories:
            return 5.0

        client = self._get_embedding_client()

        try:
            texts_to_embed = [current_message]

            if context_summary:
                texts_to_embed.append(context_summary)

            memories_text = []
            for mem in existing_memories[:5]:
                if isinstance(mem, dict):
                    memories_text.append(mem.get('memory', ''))
                else:
                    memories_text.append(str(mem))

            texts_to_embed.extend(memories_text)

            response = client.embeddings.create(
                model=embedding.EMBEDDING_MODEL_NAME,
                input=texts_to_embed
            )

            vectors = [np.array(v.embedding) for v in response.data]

            current_vector = vectors[0]

            if len(vectors) > 1 and context_summary:
                context_vector = vectors[1]
                combined_vector = (current_vector + context_vector) / 2
            else:
                combined_vector = current_vector

            vector_norm = np.linalg.norm(combined_vector)
            if vector_norm > 0:
                combined_vector = combined_vector / vector_norm

            similarities = []
            for i in range(len(vectors) - len(memories_text), len(vectors)):
                mem_vector = vectors[i]

                mem_norm = np.linalg.norm(mem_vector)
                if mem_norm > 0:
                    mem_vector = mem_vector / mem_norm

                cosine_sim = np.dot(combined_vector, mem_vector)
                similarities.append(cosine_sim)

            if similarities:
                max_similarity = max(similarities)

                if max_similarity < self.RELEVANCE_THRESHOLD:
                    score = max_similarity * 5
                else:
                    score = max_similarity * 10
            else:
                score = 5.0

            logger.debug(f"📊 相关性评分: {score:.2f}")
            return score

        except Exception as e:
            logger.error(f"❌ 相关性评分失败：{e}")
            return 5.0

    def calculate_recency(self, message_timestamp: datetime, memory_type: str) -> float:
        lambda_val = {
            "state": self.DECAY_LAMBDA_STATE,
            "event": self.DECAY_LAMBDA_EVENT,
            "fact": self.DECAY_LAMBDA_FACT,
            "chat": self.DECAY_LAMBDA_CHAT
        }.get(memory_type, self.DECAY_LAMBDA_EVENT)

        time_diff = (datetime.now() - message_timestamp).total_seconds() / 3600.0

        recency_score = math.exp(-lambda_val * time_diff)
        score = recency_score * 10

        logger.debug(f"📊 时效性评分: {score:.2f} (类型:{memory_type}, λ:{lambda_val}, Δt:{time_diff:.1f}小时)")
        return score

    async def should_store_memory(
        self,
        memory_text: str,
        user_id: str,
        session_id: str = "",
        message_timestamp: Optional[datetime] = None
    ) -> ImportanceScore:
        if message_timestamp is None:
            message_timestamp = datetime.now()

        analysis = await self.analyze_content(memory_text)

        if analysis.persistence < 2:
            result = ImportanceScore(
                content_score=0.0,
                relevance_score=0.0,
                recency_score=0.0,
                final_score=0.0,
                decision="IGNORE",
                decision_reason="持久性极低，一票否决",
                content_details=analysis
            )
            logger.info(f"📊 记忆过滤 → 丢弃: 持久性{analysis.persistence}<2，一票否决")
            return result

        content_score = self.calculate_content_score(analysis)

        context_summary = await self.build_context_vector(session_id)

        existing_memories = memory_manager.get_all(user_id=user_id)
        relevance_score = self.calculate_relevance(
            memory_text,
            existing_memories.get('results', []) if existing_memories else [],
            context_summary
        )

        recency_score = self.calculate_recency(message_timestamp, analysis.memory_type)

        final_score = (
            content_score * self.CONTENT_WEIGHT +
            recency_score * self.RECENCY_WEIGHT +
            relevance_score * self.RELEVANCE_WEIGHT
        )

        if final_score >= self.HIGH_WATERMARK:
            decision = "WRITE"
            reason = f"超过高水位线({self.HIGH_WATERMARK})"
        elif final_score <= self.LOW_WATERMARK:
            decision = "IGNORE"
            reason = f"低于低水位线({self.LOW_WATERMARK})"
        else:
            decision = "KEEP"
            reason = f"中间地带({self.LOW_WATERMARK}~{self.HIGH_WATERMARK})，保持现状"

        reasons_detail = []
        if content_score >= 7:
            reasons_detail.append("内容重要")
        if relevance_score >= 7:
            reasons_detail.append("相关性高")
        if recency_score >= 7:
            reasons_detail.append("时效性强")

        decision_reason = f"{reason} | {' + '.join(reasons_detail) if reasons_detail else ''}"

        result = ImportanceScore(
            content_score=content_score,
            relevance_score=relevance_score,
            recency_score=recency_score,
            final_score=final_score,
            decision=decision,
            decision_reason=decision_reason,
            content_details=analysis
        )

        logger.info(
            f"📊 记忆过滤评分 - "
            f"内容:{content_score:.1f} 相关:{relevance_score:.1f} 时效:{recency_score:.1f} "
            f"综合:{final_score:.2f} → {decision} | {reason}"
        )

        return result

    async def evaluate_batch(
        self,
        messages: list,
        user_id: str,
        session_id: str = ""
    ) -> dict:
        llm = self._get_llm()

        conversation_text = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            if isinstance(msg, dict) else
            f"{msg.role}: {msg.content}"
            for msg in messages
        ])

        prompt = f"""你是一个记忆提取专家。请分析以下对话，提取值得长期记忆的关键信息。

对话内容：
{conversation_text}

分析要求：
1. 提取关键事实（如用户名、部门、时间、数据等）
2. 提取用户偏好和习惯
3. 忽略寒暄和礼貌性对话
4. 判断哪些信息值得永久保存
5. 每条事实用一句简洁的话描述，避免冗余

请按以下JSON格式返回：
{{
    "memories_to_store": [
        "关键事实1的简洁描述",
        "关键事实2的简洁描述"
    ],
    "summary": "这批对话的简短总结（50字以内）",
    "count": 值得存储的事实数量
}}

如果没有任何值得存储的信息，memories_to_store 返回空数组 []。
"""

        try:
            result = await llm.run(prompt)
            output = result.output.strip()

            try:
                data = json.loads(output)
                raw_memories = data.get("memories_to_store", [])
                summary = data.get("summary", "")

                logger.info(f"📊 批量提取完成：{len(raw_memories)} 条候选记忆")

                memories_to_store = []
                filtered_out = []
                duplicates = []

                for memory_text in raw_memories:
                    is_dup, dup_reason = self.check_duplicate(memory_text, user_id)
                    if is_dup:
                        duplicates.append({"text": memory_text, "reason": dup_reason})
                        logger.info(f"🔄 去重跳过：{memory_text[:30]}... ({dup_reason})")
                        continue

                    score_result = await self.should_store_memory(
                        memory_text=memory_text,
                        user_id=user_id,
                        session_id=session_id
                    )

                    if score_result.decision == "WRITE":
                        memories_to_store.append(memory_text)
                        logger.info(f"✅ 评分通过：{memory_text[:30]}... (综合:{score_result.final_score:.2f})")
                    elif score_result.decision == "KEEP":
                        memories_to_store.append(memory_text)
                        logger.info(f"⚠️ 中间地带，保留：{memory_text[:30]}... (综合:{score_result.final_score:.2f})")
                    else:
                        filtered_out.append({
                            "text": memory_text,
                            "reason": score_result.decision_reason,
                            "score": score_result.final_score
                        })
                        logger.info(f"❌ 评分过滤：{memory_text[:30]}... (综合:{score_result.final_score:.2f})")

                logger.info(
                    f"📊 批量评估最终结果 - "
                    f"候选:{len(raw_memories)} 存储:{len(memories_to_store)} "
                    f"过滤:{len(filtered_out)} 重复:{len(duplicates)}"
                )

                return {
                    "memories_to_store": memories_to_store,
                    "filtered_out": filtered_out,
                    "duplicates": duplicates,
                    "summary": summary,
                    "count": len(memories_to_store)
                }
            except json.JSONDecodeError:
                logger.warning(f"⚠️ LLM返回格式错误，尝试提取文本")
                return {
                    "memories_to_store": [],
                    "filtered_out": [],
                    "duplicates": [],
                    "summary": output[:100],
                    "count": 0
                }

        except Exception as e:
            logger.error(f"❌ 批量评估失败：{e}")
            return {
                "memories_to_store": [],
                "filtered_out": [],
                "duplicates": [],
                "summary": "",
                "count": 0
            }

    def check_duplicate(self, memory_text: str, user_id: str) -> tuple:
        try:
            existing = memory_manager.search_memory(query=memory_text, user_id=user_id)

            if not existing or not existing.get("results"):
                return False, ""

            for mem in existing["results"][:3]:
                existing_text = mem.get("memory", "")
                if not existing_text:
                    continue

                similarity = self._compute_text_similarity(memory_text, existing_text)
                if similarity >= self.DEDUP_SIMILARITY_THRESHOLD:
                    return True, f"与已有记忆相似度{similarity:.2f}>={self.DEDUP_SIMILARITY_THRESHOLD}"

            return False, ""

        except Exception as e:
            logger.error(f"❌ 去重检查失败：{e}")
            return False, ""

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        client = self._get_embedding_client()

        try:
            response = client.embeddings.create(
                model=embedding.EMBEDDING_MODEL_NAME,
                input=[text1, text2]
            )

            vectors = [np.array(v.embedding) for v in response.data]

            v1 = vectors[0]
            v2 = vectors[1]

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0 and norm2 > 0:
                cosine_sim = np.dot(v1, v2) / (norm1 * norm2)
                return float(cosine_sim)

            return 0.0

        except Exception as e:
            logger.error(f"❌ 相似度计算失败：{e}")
            return 0.0

    async def extract_search_query(self, user_message: str) -> str:
        llm = self._get_llm()

        prompt = f"""用户说了以下内容，请提取出需要从长期记忆中查找的信息。

用户消息：{user_message}

分析要求：
1. 如果用户使用了代词（他、那个、上次等），提取代词指代的内容
2. 如果用户引用了之前的对话，提取引用的具体内容
3. 如果消息中没有需要从记忆中查找的信息，返回空字符串

请只输出提取的检索关键词或短语，不要输出其他内容。
如果不需要检索记忆，直接输出：无"""

        try:
            result = await llm.run(prompt)
            output = result.output.strip()

            if output == "无" or not output:
                logger.debug(f"📊 记忆检索query：无需优化，使用原始消息")
                return user_message

            logger.info(f"📊 记忆检索query优化：'{user_message}' → '{output}'")
            return output

        except Exception as e:
            logger.error(f"❌ 检索query优化失败：{e}")
            return user_message

    def cleanup_expired_memories(self, user_id: str) -> dict:
        try:
            all_memories = memory_manager.get_all(user_id=user_id)

            if not all_memories or 'results' not in all_memories:
                return {"deleted": 0, "kept": 0, "details": []}

            memories = all_memories['results']
            deleted = 0
            kept = 0
            details = []

            for mem in memories:
                memory_text = mem.get('memory', '')
                memory_id = mem.get('id', '')
                updated_at = mem.get('updated_at', None)

                if not memory_text or not updated_at:
                    kept += 1
                    continue

                try:
                    if isinstance(updated_at, (int, float)):
                        mem_time = datetime.fromtimestamp(updated_at)
                    elif isinstance(updated_at, str):
                        mem_time = datetime.fromisoformat(updated_at.replace('Z', '+00:00')).replace(tzinfo=None)
                    else:
                        kept += 1
                        continue
                except Exception:
                    kept += 1
                    continue

                hours_elapsed = (datetime.now() - mem_time).total_seconds() / 3600.0

                should_delete = False
                reason = ""

                state_score = math.exp(-self.DECAY_LAMBDA_STATE * hours_elapsed) * 10
                chat_score = math.exp(-self.DECAY_LAMBDA_CHAT * hours_elapsed) * 10
                event_score = math.exp(-self.DECAY_LAMBDA_EVENT * hours_elapsed) * 10

                if state_score < 0.1 and hours_elapsed > 1:
                    should_delete = True
                    reason = f"state类型过期(时效分:{state_score:.3f}, 已过{hours_elapsed:.1f}小时)"
                elif chat_score < 0.1 and hours_elapsed > 6:
                    should_delete = True
                    reason = f"chat类型过期(时效分:{chat_score:.3f}, 已过{hours_elapsed:.1f}小时)"
                elif event_score < 0.1 and hours_elapsed > 48:
                    should_delete = True
                    reason = f"event类型过期(时效分:{event_score:.3f}, 已过{hours_elapsed:.1f}小时)"

                if should_delete:
                    try:
                        memory_manager.memory.delete(memory_id)
                        deleted += 1
                        details.append({"text": memory_text[:50], "reason": reason, "action": "deleted"})
                        logger.info(f"🗑️ 清理过期记忆：{memory_text[:30]}... ({reason})")
                    except Exception as e:
                        logger.error(f"❌ 删除记忆失败：{e}")
                        kept += 1
                else:
                    kept += 1

            logger.info(f"🧹 记忆清理完成：删除{deleted}条，保留{kept}条")
            return {"deleted": deleted, "kept": kept, "details": details}

        except Exception as e:
            logger.error(f"❌ 记忆清理失败：{e}")
            return {"deleted": 0, "kept": 0, "details": []}


memory_filter = MemoryFilter()
