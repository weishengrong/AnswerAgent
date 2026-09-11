import time
import warnings
import httpx
from langchain_openai import ChatOpenAI

from app.core.config.settings import llm_settings


# ==================== LLM API 错误处理框架 - 异常类层次结构 ====================
# 提供统一的异常类型，便于上层调用者进行精确的错误分类和处理
# 所有 LLM 相关的异常都继承自 LLMError 基类

class LLMError(Exception):
    """
    LLM API 调用基础异常
    
    特性：
    - 包含错误类型标识（error_type），用于程序化判断
    - 记录 HTTP 状态码（status_code）
    - 标记是否可重试（retryable）
    - 支持请求 ID 追踪（request_id）
    
    用法示例：
        try:
            result = await call_llm()
        except LLMError as e:
            if e.retryable:
                # 可重试，执行重试逻辑
                pass
            else:
                # 不可重试，直接报错给用户
                pass
    """
    def __init__(self, message: str, error_type: str = "unknown", status_code: int = None, retryable: bool = False):
        super().__init__(message)
        self.error_type = error_type          # 错误类型：timeout, rate_limit, client_error, server_error, connection_error
        self.status_code = status_code         # HTTP 状态码（如果有）
        self.retryable = retryable             # 是否可重试
        self.request_id = None                 # 请求 ID（可在后续设置，用于日志追踪）


class LLMTimeoutError(LLMError):
    """
    请求超时异常
    
    触发场景：
    - LLM 响应时间超过配置的超时阈值
    - 网络读取/写入超时
    - 连接池等待超时
    
    默认可重试：是
    """
    def __init__(self, message: str = "LLM API request timeout"):
        super().__init__(message, error_type="timeout", retryable=True)


class LLMRateLimitError(LLMError):
    """
    速率限制异常 (HTTP 429)
    
    触发场景：
    - API 调用频率超出限制
    - 并发请求数超过配额
    
    特殊属性：
    - retry_after: 服务端建议的重试等待时间（秒）
    
    默认可重试：是
    """
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        super().__init__(message, error_type="rate_limit", status_code=429, retryable=True)
        self.retry_after = retry_after       # 服务端返回的 Retry-After 头部值


class LLMClientError(LLMError):
    """
    客户端错误 (HTTP 4xx)
    
    触发场景：
    - 请求参数错误 (400)
    - 认证失败 (401)
    - 权限不足 (403)
    - 资源不存在 (404)
    - 请求超时 (408) - 特殊情况，可重试
    
    默认可重试：否（除了 408 和 429）
    """
    def __init__(self, message: str, status_code: int):
        retryable = status_code in (408, 429)   # 408 Request Timeout 和 429 Rate Limit 可重试
        super().__init__(message, error_type="client_error", status_code=status_code, retryable=retryable)


class LLMServerError(LLMError):
    """
    服务端错误 (HTTP 5xx)
    
    触发场景：
    - LLM 服务内部错误 (500)
    - 网关错误 (502)
    - 服务不可用 (503)
    - 网关超时 (504)
    
    默认可重试：是（服务端问题通常是瞬态的）
    """
    def __init__(self, message: str, status_code: int):
        super().__init__(message, error_type="server_error", status_code=status_code, retryable=True)


class LLMConnectionError(LLMError):
    """
    连接错误（网络层面的问题）
    
    触发场景：
    - DNS 解析失败
    - TCP 连接被拒绝
    - 网络中断
    - SSL/TLS 握手失败
    
    默认可重试：是（网络问题通常是瞬态的）
    """
    def __init__(self, message: str = "Failed to connect to LLM API"):
        super().__init__(message, error_type="connection_error", retryable=True)

# 抑制 LangChain with_structured_output 的 Pydantic 序列化警告
# 原因：LangChain 内部在 parsed 字段中存储 Pydantic 对象，但 schema 声明为 None，
# 导致 Pydantic v2 序列化时产生 UserWarning，属于 LangChain 已知问题，不影响功能
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
)

_llm_instance: ChatOpenAI | None = None

# 全局共享的 httpx 异步客户端（带连接池 + keep-alive）
# 所有 LLM 调用复用同一个 TCP 连接，避免每次都 DNS+TCP+TLS
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """获取全局共享 httpx 客户端（连接池复用）"""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
            verify=True,
        )
    return _http_client


def get_llm(timeout_key: str = 'default') -> ChatOpenAI:
    """
    获取 LLM 单例实例（懒加载，使用共享 httpx 连接池）
    
    Args:
        timeout_key: 超时场景标识符，可选值：
            - 'intent_recognition': 60s (意图识别)
            - 'sql_generation': 120s (SQL生成)
            - 'chat': 30s (日常聊天)
            - 'default': 90s (默认)
    
    Returns:
        配置好超时的 ChatOpenAI 实例
    """
    global _llm_instance
    
    if _llm_instance is None:
        # 使用场景化超时配置
        timeout_seconds = DEFAULT_TIMEOUT.get(timeout_key, DEFAULT_TIMEOUT['default'])
        
        _llm_instance = ChatOpenAI(
            model=llm_settings.LLM_MODEL_NAME,
            api_key=llm_settings.LLM_API_KEY,
            base_url=llm_settings.LLM_BASE_URL,
            streaming=True,
            http_async_client=_get_http_client(),
            timeout=timeout_seconds,  # ✨ 新增：使用场景化超时
            request_timeout=timeout_seconds,  # ✨ 新增：确保超时传递到 httpx
        )
        
        logger.info(f"✅ LLM 实例已创建 (model={llm_settings.LLM_MODEL_NAME}, timeout={timeout_seconds}s)")
    
    return _llm_instance


def get_structured_llm(output_model, timeout_key: str = 'default', max_retries: int = None, tags: list = None):
    """
    获取带结构化输出的 LLM 实例（带自动重试能力）
    
    Args:
        output_model: Pydantic 输出模型类
        timeout_key: 超时场景标识符（同 get_llm）
        max_retries: 最大重试次数（None 表示使用默认值 DEFAULT_MAX_RETRIES）
        tags: 附加到 LLM 事件的标签列表（如 ["skip_stream"] 用于隐藏流式输出）
    
    Returns:
        带有重试装饰器的结构化输出 LLM 包装对象
    """
    t0 = time.perf_counter()
    
    # 获取基础 LLM 实例（使用指定的超时配置）
    llm = get_llm(timeout_key=timeout_key)
    
    # 创建结构化输出包装器
    structured_llm = llm.with_structured_output(output_model)
    
    # 如果指定了 tags，附加到内部链上（事件会携带这些 tags）
    if tags:
        structured_llm = structured_llm.with_config(tags=tags)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if elapsed_ms > 10:
        logger.warning(f"⏱️ with_structured_output 耗时 {elapsed_ms:.1f}ms（{output_model.__name__}）")
    
    # ✨ 新增：创建带有重试能力的包装函数
    retries = max_retries or DEFAULT_MAX_RETRIES
    
    class RetryableStructuredLLM:
        """
        带自动重试的结构化输出 LLM 包装器
        
        使用方式与原始 structured_llm 完全兼容，
        但在调用 ainvoke() 时会自动处理瞬态错误。
        """
        
        def __init__(self, inner_llm, model_name, retry_count):
            self._inner = inner_llm
            self._model_name = model_name
            self._retry_count = retry_count
        
        async def ainvoke(self, messages, config=None):
            """
            异步调用 LLM，带自动重试
            
            内部使用 @retry_on_transient_error 装饰器的逻辑
            """
            import uuid
            request_id = str(uuid.uuid4())[:8]  # 用于日志追踪
            
            last_exception = None
            
            for attempt in range(self._retry_count + 1):
                try:
                    if attempt > 0:
                        delay = min(DEFAULT_RETRY_BASE_DELAY * (2 ** (attempt - 1)), DEFAULT_RETRY_MAX_DELAY)
                        logger.warning(
                            f"🔄 结构化LLM调用第 {attempt}/{self._retry_count} 次重试 "
                            f"(等待 {delay:.1f}s) | request_id={request_id} | "
                            f"原因: {last_exception}"
                        )
                        await asyncio.sleep(delay)
                    
                    t_start = time.perf_counter()
                    result = await self._inner.ainvoke(messages, config=config)
                    latency_ms = (time.perf_counter() - t_start) * 1000
                    
                    if attempt > 0:
                        logger.info(
                            f"✅ 结构化LLM调用成功 | request_id={request_id} | "
                            f"attempt={attempt+1} | latency={latency_ms:.0f}ms"
                        )
                    else:
                        logger.debug(
                            f"📞 结构化LLM调用完成 | request_id={request_id} | "
                            f"model={self._model_name} | latency={latency_ms:.0f}ms"
                        )

                    return result
                    
                except Exception as e:
                    last_exception = f"{type(e).__name__}: {str(e)[:150]}"
                    
                    # 判断是否可重试
                    status_code = getattr(getattr(e, 'response', None), 'status_code', None)
                    
                    if status_code and status_code in RETRYABLE_STATUS_CODES:
                        if attempt < self._retry_count:
                            continue
                        # 达到最大重试次数
                        if status_code == 429:
                            raise LLMRateLimitError(
                                f"结构化输出调用触发限流 ({self._model_name})",
                                retry_after=None
                            )
                        raise LLMServerError(
                            f"结构化输出调用服务端错误 {status_code} ({self._model_name})",
                            status_code
                        )
                    
                    elif isinstance(e, tuple(RETRYABLE_EXCEPTIONS)):
                        if attempt < self._retry_count:
                            continue
                        if isinstance(e, (ReadTimeout, WriteTimeout, PoolTimeout)):
                            raise LLMTimeoutError(f"结构化输出调用超时 ({self._model_name})")
                        raise LLMConnectionError(f"结构化输出连接失败 ({self._model_name}): {e}")
                    
                    elif isinstance(e, LLMError) and e.retryable:
                        if attempt < self._retry_count:
                            continue
                        raise
                    
                    else:
                        # 不可重试的错误，直接抛出
                        if status_code and status_code not in RETRYABLE_STATUS_CODES:
                            raise LLMClientError(
                                f"结构化输出客户端错误 {status_code}: {e}",
                                status_code
                            )
                        logger.error(
                            f"❌ 结构化LLM调用遇到未预期异常 | "
                            f"request_id={request_id} | error={type(e).__name__}: {e}"
                        )
                        raise
            
            # 理论上不应到达这里
            raise LLMError(f"未知错误 (after {self._retry_count} retries): {last_exception}")
        
        def invoke(self, messages, config=None):
            """同步调用（暂不支持重试）"""
            return self._inner.invoke(messages, config=config)
    
    # 返回包装后的对象
    return RetryableStructuredLLM(
        inner_llm=structured_llm,
        model_name=output_model.__name__,
        retry_count=retries
    )


async def close_llm():
    """关闭 LLM 相关资源（用于 lifespan 清理）"""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


# ==================== LLM API 错误处理框架 - 重试机制与配置 ====================
# 提供自动重试、超时控制、日志记录等基础设施

import asyncio
import functools
import logging
from typing import Type, Tuple, List, Optional, Callable, Any
from httpx import HTTPStatusError, ConnectError, ReadTimeout, WriteTimeout, PoolTimeout

logger = logging.getLogger(__name__)


# ------------------- 配置常量 -------------------

# 默认重试配置
DEFAULT_MAX_RETRIES = 3                    # 最大重试次数
DEFAULT_RETRY_BASE_DELAY = 1.0             # 基础延迟时间（秒）
DEFAULT_RETRY_MAX_DELAY = 10.0             # 最大延迟时间（秒）
DEFAULT_RETRY_EXPONENTIAL_BASE = 2.0       # 指数退避基数

# 不同场景的超时配置（单位：秒）
# 根据业务场景调整：意图识别较快，SQL生成可能较慢，聊天需要快速响应
DEFAULT_TIMEOUT = {
    'intent_recognition': 60,   # 意图识别（通常较快，但需要处理复杂查询）
    'sql_generation': 120,       # SQL生成（可能需要多轮推理，较慢）
    'chat': 30,                  # 日常聊天（用户期望快速响应）
    'default': 90                # 默认值（适用于大多数场景）
}

# 可重试的 HTTP 状态码集合
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# 可重试的异常类型元组（网络层面和超时相关的异常）
RETRYABLE_EXCEPTIONS = (
    ConnectError,      # 连接失败（DNS、TCP、SSL等）
    ReadTimeout,       # 读取超时
    WriteTimeout,      # 写入超时
    PoolTimeout,       # 连接池耗尽
    ConnectionError,   # 通用连接错误
    OSError            # 系统级网络错误
)


# ------------------- 重试装饰器 -------------------

def retry_on_transient_error(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    exponential_base: float = DEFAULT_RETRY_EXPONENTIAL_BASE,
    retryable_exceptions: tuple = RETRYABLE_EXCEPTIONS,
):
    """
    重试装饰器 - 针对瞬态错误的自动重试机制
    
    设计理念：
    - 遵循"幂等性假设"：被装饰的函数应该是幂等的或至少是安全的
    - 指数退避策略：避免在服务端压力大时雪崩式重试
    - 精细的错误分类：只对瞬态错误重试，对逻辑错误立即失败
    
    特性：
    ✅ 指数退避策略（1s → 2s → 4s，受 max_delay 限制）
    ✅ 可配置的最大重试次数
    ✅ 仅对可重试错误进行重试（HTTP 429/5xx、网络错误、超时）
    ✅ 详细的重试日志（包含重试次数、等待时间、错误原因）
    ✅ 自动将原始异常转换为语义化的 LLMError 子类
    
    参数说明：
        max_retries: 最大重试次数（不含首次调用），默认 3 次
        base_delay: 基础延迟时间（秒），默认 1.0s
        max_delay: 最大延迟时间（秒），防止退避时间过长，默认 10.0s
        exponential_base: 指数退避基数，默认 2.0（即每次翻倍）
        retryable_exceptions: 触发重试的异常类型元组
    
    用法示例：
        # 方式1：使用默认配置
        @retry_on_transient_error()
        async def call_llm_simple():
            return await llm.ainvoke(messages)
        
        # 方式2：自定义配置
        @retry_on_transient_error(max_retries=5, base_delay=0.5)
        async def call_llm_custom():
            return await llm.ainvoke(messages)
        
        # 方式3：在业务代码中使用
        try:
            result = await call_llm_with_retry()
        except LLMRateLimitError as e:
            # 处理速率限制
            logger.warning(f"触发限流，建议等待 {e.retry_after} 秒")
        except LLMTimeoutError:
            # 处理超时
            logger.error("LLM 响应超时")
    
    返回：
        装饰后的异步函数，具有自动重试能力
    """
    def decorator(func):
        @functools.wraps(func)  # 保留原函数的元信息（名称、文档字符串等）
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            # 循环尝试：首次调用 + max_retries 次重试
            for attempt in range(max_retries + 1):  # +1 因为第一次不算重试
                try:
                    # 如果不是首次调用，执行退避等待
                    if attempt > 0:
                        # 计算指数退避延迟时间
                        delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
                        
                        # 记录重试日志（使用 emoji 提高可读性）
                        logger.warning(
                            f"🔄 LLM 调用第 {attempt}/{max_retries} 次重试 "
                            f"(等待 {delay:.1f}s) | 原因: {last_exception}"
                        )
                        
                        # 异步等待（不阻塞事件循环）
                        await asyncio.sleep(delay)
                    
                    # 执行实际的函数调用
                    result = await func(*args, **kwargs)
                    
                    # 如果经历了重试后成功，记录成功日志
                    if attempt > 0:
                        logger.info(f"✅ LLM 调用在第 {attempt + 1} 次尝试成功")
                    
                    return result
                    
                except HTTPStatusError as e:
                    """
                    处理 HTTP 状态码错误（4xx/5xx）
                    
                    分类逻辑：
                    - 429 (Rate Limit): 可重试，转换为 LLMRateLimitError
                    - 5xx (Server Error): 可重试，转换为 LLMServerError
                    - 其他 4xx: 不可重试，转换为 LLMClientError
                    """
                    status_code = e.response.status_code if hasattr(e, 'response') else None
                    
                    if status_code and status_code in RETRYABLE_STATUS_CODES:
                        # 可重试的 HTTP 错误
                        last_exception = f"HTTP {status_code}: {str(e)[:100]}"
                        
                        if attempt < max_retries:
                            # 还有重试机会，继续循环
                            continue
                        
                        # 达到最大重试次数，抛出对应类型的异常
                        if status_code == 429:
                            raise LLMRateLimitError(f"Rate limit exceeded after {max_retries} retries")
                        raise LLMServerError(f"Server error {status_code} after {max_retries} retries", status_code)
                    else:
                        # 不可重试的 HTTP 错误（如 400/401/403/404），立即抛出
                        if status_code:
                            raise LLMClientError(str(e), status_code)
                        raise
                        
                except retryable_exceptions as e:
                    """
                    处理可重试的网络/超时异常
                    
                    包括：
                    - ConnectError: 连接失败
                    - ReadTimeout/WriteTimeout: 读写超时
                    - PoolTimeout: 连接池耗尽
                    - ConnectionError/OSError: 通用网络错误
                    """
                    last_exception = f"{type(e).__name__}: {str(e)[:100]}"
                    
                    if attempt < max_retries:
                        # 还有重试机会，继续循环
                        continue
                    
                    # 达到最大重试次数，根据具体类型抛出对应的 LLMError
                    if isinstance(e, (ReadTimeout, WriteTimeout, PoolTimeout)):
                        raise LLMTimeoutError(f"Timeout after {max_retries} retries: {str(e)}")
                    raise LLMConnectionError(f"Connection error after {max_retries} retries: {str(e)}")
                    
                except LLMError as e:
                    """
                    处理已经是 LLMError 类型的异常
                    
                    场景：内部函数已经转换过一次异常，或者手动抛出的 LLMError
                    逻辑：根据异常的 retryable 属性决定是否继续重试
                    """
                    if e.retryable and attempt < max_retries:
                        last_exception = str(e)[:100]
                        continue
                    raise
                    
                except Exception as e:
                    """
                    处理其他未预期的异常
                    
                    策略：不重试，直接记录并抛出
                    原因：未知异常可能是程序逻辑错误，重试无法解决
                    """
                    logger.error(f"❌ LLM 调用遇到未预期异常: {type(e).__name__}: {e}")
                    raise
            
            # 理论上不应该到达这里（所有路径都应该 return 或 raise）
            # 如果到达这里，说明存在未处理的边界情况
            raise last_exception or "Unknown error after retries"
            
        return wrapper
    return decorator
