"""
中间件模块 - 日志追踪 + 异常降级

5.2 异常处理与降级：
- Milvus 连接失败 → 降级为无 RAG 模式
- Redis 连接失败 → 降级为无短期记忆模式
- LLM 超时 → 重试 + 降级

5.3 日志追踪：
- 每次请求分配 trace_id
- 每个节点记录输入/输出/耗时
"""
import logging
import time
import uuid
import functools
from contextvars import ContextVar
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def new_trace_id() -> str:
    """生成新的 trace_id"""
    return str(uuid.uuid4())[:12]


def get_trace_id() -> str:
    """获取当前 trace_id"""
    return trace_id_var.get()


def set_trace_id(tid: str):
    """设置当前 trace_id"""
    trace_id_var.set(tid)


class TraceLogger:
    """带 trace_id 的日志记录器"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def _format_msg(self, msg: str) -> str:
        tid = get_trace_id()
        if tid:
            return f"[{tid}] {msg}"
        return msg

    def info(self, msg: str, **kwargs):
        self.logger.info(self._format_msg(msg), **kwargs)

    def warning(self, msg: str, **kwargs):
        self.logger.warning(self._format_msg(msg), **kwargs)

    def error(self, msg: str, **kwargs):
        self.logger.error(self._format_msg(msg), **kwargs)

    def debug(self, msg: str, **kwargs):
        self.logger.debug(self._format_msg(msg), **kwargs)


class NodeTimer:
    """节点耗时计时器"""

    def __init__(self, node_name: str):
        self.node_name = node_name
        self.start_time = None

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time if self.start_time else 0
        tid = get_trace_id()
        if exc_type:
            logger.warning(f"[{tid}] ⏱️ {self.node_name} 失败 ({elapsed:.2f}s): {exc_val}")
        else:
            logger.info(f"[{tid}] ⏱️ {self.node_name} 完成 ({elapsed:.2f}s)")
        return False


def with_fallback(fallback_value=None, log_error=True):
    """
    降级装饰器

    当被装饰的函数/方法抛出异常时，返回 fallback_value 而不是抛出异常。

    用法：
        @with_fallback(fallback_value="")
        async def search_memory(query, user_id):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    tid = get_trace_id()
                    logger.warning(f"[{tid}] ⚠️ {func.__name__} 降级: {e}")
                return fallback_value

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    tid = get_trace_id()
                    logger.warning(f"[{tid}] ⚠️ {func.__name__} 降级: {e}")
                return fallback_value

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class ServiceHealth:
    """服务健康状态检查"""

    def __init__(self):
        self._milvus_healthy = True
        self._redis_healthy = True
        self._llm_healthy = True

    @property
    def milvus_healthy(self) -> bool:
        return self._milvus_healthy

    @property
    def redis_healthy(self) -> bool:
        return self._redis_healthy

    @property
    def llm_healthy(self) -> bool:
        return self._llm_healthy

    def mark_milvus_down(self):
        self._milvus_healthy = False
        logger.warning("🔴 Milvus 标记为不可用，降级为无 RAG 模式")

    def mark_milvus_up(self):
        if not self._milvus_healthy:
            self._milvus_healthy = True
            logger.info("🟢 Milvus 恢复可用")

    def try_recover_milvus(self) -> bool:
        if self._milvus_healthy:
            return True
        try:
            from config.settings import chroma_settings
            import os
            chroma_dir = chroma_settings.CHROMA_PERSIST_DIR
            if os.path.exists(chroma_dir):
                self.mark_milvus_up()
                return True
            else:
                self.mark_milvus_up()
                return True
        except Exception as e:
            logger.debug(f"Chroma 健康探测失败：{e}")
            return False

    def mark_redis_down(self):
        self._redis_healthy = False
        logger.warning("🔴 Redis 标记为不可用，降级为无短期记忆模式")

    def mark_redis_up(self):
        if not self._redis_healthy:
            self._redis_healthy = True
            logger.info("🟢 Redis 恢复可用")

    def mark_llm_down(self):
        self._llm_healthy = False
        logger.warning("🔴 LLM 标记为不可用")

    def mark_llm_up(self):
        if not self._llm_healthy:
            self._llm_healthy = True
            logger.info("🟢 LLM 恢复可用")

    def status(self) -> dict:
        return {
            "milvus": "healthy" if self._milvus_healthy else "down",
            "redis": "healthy" if self._redis_healthy else "down",
            "llm": "healthy" if self._llm_healthy else "down",
        }


service_health = ServiceHealth()
