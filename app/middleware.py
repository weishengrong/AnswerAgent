"""
中间件模块 - 日志追踪 + 异常降级 + Prometheus 监控指标

5.2 异常处理与降级：
- Redis 连接失败 → 降级为无短期记忆模式
- LLM 超时 → 重试 + 降级

5.3 日志追踪：
- 每次请求分配 trace_id
- 每个节点记录输入/输出/耗时

5.4 Prometheus 指标：
- LLM 调用指标（token/延迟/错误率）
- RAG 检索指标（命中数/融合数/耗时）
- 降级事件计数器
- 服务健康状态 Gauge
- 工作流节点耗时 Histogram
"""
import logging
import time
import uuid
import functools
from contextvars import ContextVar
from typing import Optional, Callable, Any

# Windows 兼容性：prometheus-client 在 Windows 上因 resource.getpagesize() 报错
import sys
if sys.platform == 'win32':
    import resource as _resource
    if not hasattr(_resource, 'getpagesize'):
        _resource.getpagesize = lambda: 4096

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


# ============================================================
# Prometheus 自定义业务指标定义
# ============================================================

# --- LLM 调用指标 ---
LLM_CALLS_TOTAL = Counter(
    'llm_calls_total',
    'LLM 调用总数',
    ['skill', 'action', 'status']
)

LLM_TOKENS_TOTAL = Counter(
    'llm_tokens_total',
    'LLM Token 用量',
    ['skill', 'action', 'type']  # type: input / output
)

LLM_DURATION_MS = Histogram(
    'llm_duration_ms',
    'LLM 调用延迟 (ms)',
    ['skill', 'action'],
    buckets=(50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000, 60000)
)

# --- RAG 检索指标 ---
RAG_BM25_HITS = Histogram(
    'rag_bm25_hits',
    'BM25 命中文档数',
    buckets=(0, 1, 3, 5, 10, 20, 50)
)
RAG_VECTOR_HITS = Histogram(
    'rag_vector_hits',
    '向量检索命中文档数',
    buckets=(0, 1, 3, 5, 10, 20, 50)
)
RAG_RRF_RESULTS = Histogram(
    'rag_rrf_results',
    'RRF 融合后结果数',
    buckets=(0, 3, 5, 10, 20, 50)
)
RAG_RETRIEVAL_LATENCY_MS = Histogram(
    'rag_retrieval_duration_ms',
    'RAG 各阶段耗时 (ms)',
    ['stage'],  # query_rewrite / vector_search / bm25_search / rrf_merge / table_rerank / context_assemble
    buckets=(10, 50, 100, 200, 500, 1000, 2000, 5000)
)

# --- 降级事件指标 ---
FALLBACK_TOTAL = Counter(
    'fallback_total',
    '降级事件计数',
    ['component', 'reason']
)  # component: chroma / redis / bm25 / memory / llm / mcp / sql / unknown

# --- 服务健康指标 ---
SERVICE_HEALTH = Gauge(
    'service_health',
    '服务健康状态 (1=healthy, 0=down)',
    ['service']
)  # service: chroma / redis / llm

# --- 工作流节点耗时指标 ---
WORKFLOW_NODE_DURATION_MS = Histogram(
    'workflow_node_duration_ms',
    '工作流节点耗时 (ms)',
    ['node_name'],
    buckets=(50, 100, 200, 500, 1000, 2000, 5000, 10000, 30000)
)


# --- 辅助函数：从函数/异常推断标签值 ---

_COMPONENT_KEYWORDS = {
    'chroma': ('chroma', 'vector', 'embedding'),
    'redis': ('redis', 'session', 'memory', 'cache'),
    'bm25': ('bm25', 'index', 'search'),
    'memory': ('memory', 'mem0', 'long_term'),
    'llm': ('llm', 'agent', 'model', 'openai', 'generate', 'stream'),
    'mcp': ('mcp', 'calendar', 'chart', 'tool'),
    'sql': ('sql', 'query', 'database', 'db', 'execute'),
}

_REASON_KEYWORDS = {
    'connection_error': ('connection', 'connect', 'refused', 'unreachable'),
    'timeout': ('timeout', 'timed out'),
    'parse_error': ('parse', 'json', 'decode', 'invalid'),
    'rate_limited': ('rate', 'limit', '429', 'throttl'),
    'unavailable': ('unavailable', '503', 'service', 'down'),
}


def _extract_component(func: Callable) -> str:
    """从被装饰的函数名和模块名推断 component 标签值"""
    func_name = getattr(func, '__name__', '').lower()
    module = getattr(func, '__module__', '').lower()
    combined = f"{func_name} {module}"
    for comp, keywords in _COMPONENT_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return comp
    return 'unknown'


def _classify_reason(exception: Exception) -> str:
    """从异常类型和消息推断 reason 标签值"""
    exc_str = str(exception).lower() + ' ' + type(exception).__name__.lower()
    for reason, keywords in _REASON_KEYWORDS.items():
        if any(kw in exc_str for kw in keywords):
            return reason
    return 'exception'


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
        elapsed_ms = elapsed * 1000
        tid = get_trace_id()
        if exc_type:
            logger.warning(f"[{tid}] ⏱️ {self.node_name} 失败 ({elapsed:.2f}s): {exc_val}")
        else:
            logger.info(f"[{tid}] ⏱️ {self.node_name} 完成 ({elapsed:.2f}s)")
        # 写入 Prometheus Histogram
        WORKFLOW_NODE_DURATION_MS.labels(node_name=self.node_name).observe(elapsed_ms)
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
                # 记录降级到 Prometheus
                FALLBACK_TOTAL.labels(
                    component=_extract_component(func),
                    reason=_classify_reason(e)
                ).inc()
                return fallback_value

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    tid = get_trace_id()
                    logger.warning(f"[{tid}] ⚠️ {func.__name__} 降级: {e}")
                # 记录降级到 Prometheus
                FALLBACK_TOTAL.labels(
                    component=_extract_component(func),
                    reason=_classify_reason(e)
                ).inc()
                return fallback_value

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class ServiceHealth:
    """服务健康状态检查"""

    def __init__(self):
        self._chroma_healthy = True
        self._redis_healthy = True
        self._llm_healthy = True

    @property
    def chroma_healthy(self) -> bool:
        return self._chroma_healthy

    @property
    def redis_healthy(self) -> bool:
        return self._redis_healthy

    @property
    def llm_healthy(self) -> bool:
        return self._llm_healthy

    def mark_chroma_down(self):
        self._chroma_healthy = False
        logger.warning("🔴 Chroma 标记为不可用，降级为无 RAG 模式")
        SERVICE_HEALTH.labels(service='chroma').set(0)

    def mark_chroma_up(self):
        if not self._chroma_healthy:
            self._chroma_healthy = True
            logger.info("🟢 Chroma 恢复可用")
            SERVICE_HEALTH.labels(service='chroma').set(1)

    def try_recover_chroma(self) -> bool:
        if self._chroma_healthy:
            return True
        try:
            from app.core.config.settings import chroma_settings
            import os
            chroma_dir = chroma_settings.CHROMA_PERSIST_DIR
            if os.path.exists(chroma_dir):
                self.mark_chroma_up()
                return True
            else:
                self.mark_chroma_up()
                return True
        except Exception as e:
            logger.debug(f"Chroma 健康探测失败：{e}")
            return False

    def mark_redis_down(self):
        self._redis_healthy = False
        logger.warning("🔴 Redis 标记为不可用，降级为无短期记忆模式")
        SERVICE_HEALTH.labels(service='redis').set(0)

    def mark_redis_up(self):
        if not self._redis_healthy:
            self._redis_healthy = True
            logger.info("🟢 Redis 恢复可用")
            SERVICE_HEALTH.labels(service='redis').set(1)

    def mark_llm_down(self):
        self._llm_healthy = False
        logger.warning("🔴 LLM 标记为不可用")
        SERVICE_HEALTH.labels(service='llm').set(0)

    def mark_llm_up(self):
        if not self._llm_healthy:
            self._llm_healthy = True
            logger.info("🟢 LLM 恢复可用")
            SERVICE_HEALTH.labels(service='llm').set(1)

    def status(self) -> dict:
        return {
            "chroma": "healthy" if self._chroma_healthy else "down",
            "redis": "healthy" if self._redis_healthy else "down",
            "llm": "healthy" if self._llm_healthy else "down",
        }


service_health = ServiceHealth()
