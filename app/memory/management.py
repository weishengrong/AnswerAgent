from mem0 import Memory
from mem0.configs.base import MemoryConfig
from config.settings import embedding, llm_settings, chroma_settings
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

COLLECTION_NOT_LOADED_KEYWORDS = ["not loaded", "collection not loaded", "not fully loaded"]


def _is_recoverable_error(error: Exception) -> bool:
    error_msg = str(error).lower()
    return any(kw in error_msg for kw in COLLECTION_NOT_LOADED_KEYWORDS)


class MemoryService:
    def __init__(self):
        self._memory = None
        self._connected = False
        logger.info("✅ MemoryService 延迟初始化（未连接 Chroma）")

    @property
    def memory(self):
        if self._memory is None:
            self._connect()
        return self._memory

    def _connect(self):
        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": llm_settings.LLM_MODEL_NAME,
                    "ollama_base_url": "http://localhost:11434",
                    "api_key": llm_settings.LLM_API_KEY
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": embedding.EMBEDDING_MODEL_NAME,
                    "openai_base_url": embedding.EMBEDDING_MODEL_URL,
                    "api_key": embedding.EMBEDDING_MODEL_API_KEY,
                }
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "path": chroma_settings.CHROMA_PERSIST_DIR,
                    "collection_name": "answer_agent_memory",
                }
            }
        }

        try:
            config_obj = MemoryConfig(**config)
            self._memory = Memory(config=config_obj)
            self._connected = True
            logger.info("✅ MemoryService 延迟连接 Chroma 成功")
        except Exception as e:
            self._connected = False
            logger.error(f"❌ MemoryService 连接 Chroma 失败：{e}")
            from app.middleware import service_health
            service_health.mark_milvus_down()
            raise

    def _handle_error(self, operation: str, error: Exception, fallback=None):
        if _is_recoverable_error(error):
            logger.warning(f"⚠️ {operation} 遇到可恢复错误：{error}")
            from app.middleware import service_health
            service_health.mark_milvus_up()
            return fallback
        else:
            logger.error(f"❌ {operation} 失败：{error}")
            from app.middleware import service_health
            service_health.mark_milvus_down()
            return fallback

    def add_memory(
            self,
            messages,
            user_id: str,
            run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            result = self.memory.add(
                messages,
                user_id=user_id,
                run_id=run_id
            )

            if run_id:
                logger.info(f"💾 添加会话记忆：{user_id} - {run_id} ({len(messages)}条)")
            else:
                logger.info(f"💾 添加全局记忆：{user_id} ({len(messages)}条)")

            return result
        except Exception as e:
            return self._handle_error("添加记忆", e, fallback={})

    def search_memory(
            self,
            query: str,
            user_id: str,
            run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            result = self.memory.search(
                query,
                user_id=user_id,
                run_id=run_id
            )

            if run_id:
                logger.debug(f"🔍 搜索会话记忆：{user_id} - {run_id}")
            else:
                logger.debug(f"🔍 搜索全局记忆：{user_id}")

            return result
        except Exception as e:
            return self._handle_error("搜索记忆", e, fallback={"results": []})

    def get_session_memories(
            self,
            user_id: str,
            run_id: str,
            limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        all_memories = self.memory.get_all(
            user_id=user_id,
            run_id=run_id
        )

        if not all_memories or 'results' not in all_memories:
            return []

        memories = all_memories['results']

        sorted_memories = sorted(
            memories,
            key=lambda x: x.get('updated_at', 0),
            reverse=True
        )

        if limit:
            sorted_memories = sorted_memories[:limit]

        logger.debug(f"📖 获取会话记忆：{user_id} - {run_id} ({len(sorted_memories)}条)")

        return sorted_memories

    def get_recent_context(
            self,
            user_id: str,
            run_id: str,
            limit: int = 5
    ) -> str:
        memories = self.get_session_memories(user_id, run_id, limit=limit * 2)

        if not memories:
            return ""

        context_lines = []
        for mem in memories:
            memory_text = mem.get('memory', '')
            if memory_text:
                context_lines.append(memory_text)

        context = "\n".join(context_lines)
        logger.debug(f"🧠 构建会话上下文：{user_id} - {run_id} ({len(context)}字符)")

        return context

    def get_all(
            self,
            user_id: str,
            run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            result = self.memory.get_all(
                user_id=user_id,
                run_id=run_id
            )
            return result
        except Exception as e:
            return self._handle_error("获取全部记忆", e, fallback={"results": []})

    def delete_session(self, run_id: str, user_id: Optional[str] = None) -> None:
        self.memory.delete(
            user_id=user_id,
            run_id=run_id
        )

        logger.info(f"🗑️ 删除会话：{run_id}")


memory_manager = MemoryService()
