"""
API 错误处理与重试机制测试套件

覆盖范围：
- 异常类层次结构
- 重试装饰器逻辑
- RetryableStructuredLLM 行为
- EventEmitter 错误事件
- 完整集成流程模拟

运行方式：pytest tests/test_error_handling.py -v
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from httpx import HTTPStatusError, ConnectError, ReadTimeout


# ==================== 1. 单元测试：重试机制验证 ====================

class TestRetryMechanism:
    """验证 @retry_on_transient_error 装饰器的重试逻辑"""

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self):
        """验证遇到 500 错误时自动重试 3 次"""
        from app.core.llm import retry_on_transient_error

        call_count = 0

        async def mock_llm_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # 模拟 HTTP 500 错误
                response = MagicMock()
                response.status_code = 500
                raise HTTPStatusError("Server Error", request=MagicMock(), response=response)
            return {"result": "success"}

        # 应用装饰器（使用快速模式：base_delay=0.01s）
        decorated_func = retry_on_transient_error(max_retries=3, base_delay=0.01)(mock_llm_call)

        result = await decorated_func()

        assert result == {"result": "success"}
        assert call_count == 3  # 应该调用了 3 次（2次失败 + 1次成功）

    @pytest.mark.asyncio
    async def test_no_retry_on_client_error(self):
        """验证遇到 400 错误时不重试，立即抛出异常"""
        from app.core.llm import retry_on_transient_error, LLMClientError

        async def mock_llm_call():
            response = MagicMock()
            response.status_code = 400
            raise HTTPStatusError("Bad Request", request=MagicMock(), response=response)

        decorated_func = retry_on_transient_error(max_retries=3)(mock_llm_call)

        with pytest.raises(LLMClientError) as exc_info:
            await decorated_func()

        assert exc_info.value.status_code == 400
        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """验证超时错误会触发重试"""
        from app.core.llm import retry_on_transient_error

        call_count = 0

        async def mock_llm_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ReadTimeout("Timeout")
            return "success"

        decorated_func = retry_on_transient_error(max_retries=3, base_delay=0.01)(mock_llm_call)

        result = await decorated_func()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_429(self):
        """验证遇到 429 限流错误时会重试并最终抛出 LLMRateLimitError"""
        from app.core.llm import retry_on_transient_error, LLMRateLimitError

        call_count = 0

        async def mock_llm_call():
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            response.status_code = 429
            raise HTTPStatusError("Rate Limit", request=MagicMock(), response=response)

        decorated_func = retry_on_transient_error(max_retries=2, base_delay=0.01)(mock_llm_call)

        with pytest.raises(LLMRateLimitError) as exc_info:
            await decorated_func()

        assert exc_info.value.error_type == "rate_limit"
        assert call_count == 3  # 首次调用 + 2次重试

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """验证连接错误会触发重试"""
        from app.core.llm import retry_on_transient_error, LLMConnectionError

        call_count = 0

        async def mock_llm_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectError("Connection failed")
            return "connected"

        decorated_func = retry_on_transient_error(max_retries=3, base_delay=0.01)(mock_llm_call)

        result = await decorated_func()

        assert result == "connected"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """验证指数退避延迟时间的正确性"""
        from app.core.llm import retry_on_transient_error
        import time

        call_times = []

        async def mock_llm_call():
            call_times.append(time.perf_counter())
            if len(call_times) < 4:
                raise ReadTimeout("Timeout")
            return "success"

        # 使用 base_delay=0.05, exponential_base=2
        # 预期延迟: attempt 1: 0.05s, attempt 2: 0.1s, attempt 3: 0.2s
        decorated_func = retry_on_transient_error(
            max_retries=3,
            base_delay=0.05,
            exponential_base=2.0,
            max_delay=10.0
        )(mock_llm_call)

        start_time = time.perf_counter()
        result = await decorated_func()
        total_time = time.perf_counter() - start_time

        assert result == "success"
        assert len(call_times) == 4
        # 总时间应该接近 0.05 + 0.1 + 0.2 = 0.35s（允许一定误差）
        assert total_time >= 0.30  # 至少等待了退避时间
        assert total_time < 1.0   # 不应该超过 1 秒

    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        """验证最大延迟时间限制生效"""
        from app.core.llm import retry_on_transient_error
        import time

        call_count = 0

        async def mock_llm_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ReadTimeout("Timeout")
            return "success"

        # 使用很大的 base_delay 和 exponential_base，但限制 max_delay=0.1s
        decorated_func = retry_on_transient_error(
            max_retries=3,
            base_delay=1.0,
            exponential_base=10.0,
            max_delay=0.1  # 强制限制最大延迟
        )(mock_llm_call)

        start_time = time.perf_counter()
        result = await decorated_func()
        total_time = time.perf_counter() - start_time

        assert result == "success"
        # 即使 base_delay=1.0, exponential_base=10.0，总延迟也不应超过 0.3s (3次 * 0.1s)
        assert total_time < 0.5


# ==================== 2. 单元测试：异常类层次结构验证 ====================

class TestExceptionHierarchy:
    """验证所有自定义异常类的属性和行为"""

    def test_base_exception_attributes(self):
        """验证基础异常 LLMError 的属性设置正确"""
        from app.core.llm import LLMError

        error = LLMError(
            message="test message",
            error_type="test",
            status_code=500,
            retryable=True
        )

        assert str(error) == "test message"
        assert error.error_type == "test"
        assert error.status_code == 500
        assert error.retryable is True
        assert error.request_id is None  # 默认值

    def test_timeout_exception_defaults(self):
        """验证超时异常的默认属性"""
        from app.core.llm import LLMTimeoutError

        timeout = LLMTimeoutError()

        assert timeout.error_type == "timeout"
        assert timeout.retryable is True
        assert "timeout" in str(timeout).lower()

    def test_rate_limit_exception(self):
        """验证限流异常的属性"""
        from app.core.llm import LLMRateLimitError

        rate_limit = LLMRateLimitError(retry_after=5)

        assert rate_limit.error_type == "rate_limit"
        assert rate_limit.retry_after == 5
        assert rate_limit.status_code == 429
        assert rate_limit.retryable is True

        # 测试不带 retry_after 参数
        rate_limit_default = LLMRateLimitError()
        assert rate_limit_default.retry_after is None

    def test_client_error_retryable_logic(self):
        """验证客户端异常的重试逻辑（408 可重试，其他不可重试）"""
        from app.core.llm import LLMClientError

        # 408 Request Timeout 可重试
        client_408 = LLMClientError("timeout", 408)
        assert client_408.retryable is True
        assert client_408.status_code == 408

        # 429 Rate Limit 可重试
        client_429 = LLMClientError("rate limit", 429)
        assert client_429.retryable is True

        # 400 Bad Request 不可重试
        client_400 = LLMClientError("bad request", 400)
        assert client_400.retryable is False
        assert client_400.status_code == 400

        # 401 Unauthorized 不可重试
        client_401 = LLMClientError("unauthorized", 401)
        assert client_401.retryable is False

        # 403 Forbidden 不可重试
        client_403 = LLMClientError("forbidden", 403)
        assert client_403.retryable is False

        # 404 Not Found 不可重试
        client_404 = LLMClientError("not found", 404)
        assert client_404.retryable is False

    def test_server_error_always_retryable(self):
        """验证服务端异常默认可重试"""
        from app.core.llm import LLMServerError

        server_500 = LLMServerError("internal error", 500)
        assert server_500.error_type == "server_error"
        assert server_500.retryable is True
        assert server_500.status_code == 500

        server_502 = LLMServerError("bad gateway", 502)
        assert server_502.retryable is True
        assert server_502.status_code == 502

        server_503 = LLMServerError("service unavailable", 503)
        assert server_503.retryable is True

        server_504 = LLMServerError("gateway timeout", 504)
        assert server_504.retryable is True

    def test_connection_error_defaults(self):
        """验证连接错误的默认属性"""
        from app.core.llm import LLMConnectionError

        conn_error = LLMConnectionError()
        assert conn_error.error_type == "connection_error"
        assert conn_error.retryable is True

        custom_message = LLMConnectionError("DNS resolution failed")
        assert "DNS" in str(custom_message)

    def test_exception_inheritance(self):
        """验证异常类的继承关系"""
        from app.core.llm import (
            LLMError, LLMTimeoutError, LLMRateLimitError,
            LLMClientError, LLMServerError, LLMConnectionError
        )

        # 所有异常都继承自 LLMError
        assert issubclass(LLMTimeoutError, LLMError)
        assert issubclass(LLMRateLimitError, LLMError)
        assert issubclass(LLMClientError, LLMError)
        assert issubclass(LLMServerError, LLMError)
        assert issubclass(LLMConnectionError, LLMError)

        # 都继承自 Exception
        assert issubclass(LLMError, Exception)


# ==================== 3. 集成测试：EventEmitter 错误事件传播 ====================

class TestEventEmitterErrorHandling:
    """验证 EventEmitter 能正确发射和处理 ERROR 事件"""

    def test_emit_error_method(self):
        """验证 emit_error() 能正确发射高优先级错误事件"""
        from app.events import EventEmitter, EventType

        emitter = EventEmitter()

        # 发射错误事件
        emitter.emit_error(
            error_type="llm_timeout",
            message="AI 模型响应超时",
            suggestion="请稍后重试",
            details={"request_id": "test123"}
        )

        # 从队列取出事件
        event = emitter._queue.get_nowait()

        assert event["type"] == EventType.ERROR
        assert event["data"]["error_type"] == "llm_timeout"
        assert event["data"]["message"] == "AI 模型响应超时"
        assert event["data"]["suggestion"] == "请稍后重试"
        # 注意：details 会被展开到 data 字段中（见 events.py 实现）
        assert event["data"]["request_id"] == "test123"
        assert event["priority"] == "high"

    def test_emit_error_default_suggestion(self):
        """验证不提供 suggestion 时使用默认值"""
        from app.events import EventEmitter, EventType

        emitter = EventEmitter()
        emitter.emit_error(
            error_type="unknown_error",
            message="发生了未知错误"
            # 不提供 suggestion
        )

        event = emitter._queue.get_nowait()

        assert event["data"]["suggestion"] == "请稍后重试"  # 默认建议

    def test_emit_error_when_queue_full(self):
        """验证即使队列已满，ERROR 事件也能强制插入"""
        from app.events import EventEmitter, EventType

        # 创建一个容量很小的队列
        emitter = EventEmitter()
        emitter._queue = asyncio.Queue(maxsize=2)

        # 填满队列
        emitter.emit(EventType.THINKING, {"message": "event 1"})
        emitter.emit(EventType.THINKING, {"message": "event 2"})
        assert emitter._queue.full()

        # 尝试发射错误事件（应该成功插入，可能清理旧事件）
        emitter.emit_error(
            error_type="critical_error",
            message="严重错误",
            suggestion="立即检查"
        )

        # 取出所有事件
        events = []
        while not emitter._queue.empty():
            events.append(emitter._queue.get_nowait())

        # 至少应该包含错误事件
        error_events = [e for e in events if e.get("type") == EventType.ERROR]
        assert len(error_events) >= 1
        assert error_events[0]["data"]["error_type"] == "critical_error"
        assert error_events[0]["priority"] == "high"

    def test_emit_error_preserves_details(self):
        """验证 ERROR 事件的 details 字段被正确保留"""
        from app.events import EventEmitter, EventType

        emitter = EventEmitter()
        test_details = {
            "request_id": "req-12345",
            "model": "gpt-4",
            "latency_ms": 5000,
            "tokens_used": 150
        }

        emitter.emit_error(
            error_type="performance_issue",
            message="响应时间过长",
            details=test_details
        )

        event = emitter._queue.get_nowait()

        # 验证所有 details 字段都被展开到 data 中（见 events.py 实现）
        assert event["data"]["request_id"] == "req-12345"
        assert event["data"]["model"] == "gpt-4"
        assert event["data"]["latency_ms"] == 5000
        assert event["data"]["tokens_used"] == 150

    def test_emit_done_closes_emitter(self):
        """验证 emit_done 会关闭发射器"""
        from app.events import EventEmitter, EventType

        emitter = EventEmitter()

        # 发送 DONE 事件
        emitter.emit_done(response_type="success", message="完成")

        # 验证发射器已关闭
        assert emitter._closed is True

        # 验证 DONE 事件在队列中
        event = emitter._queue.get_nowait()
        assert event["type"] == EventType.DONE
        assert event["data"]["response_type"] == "success"

        # 关闭后再发送事件应该被忽略
        emitter.emit(EventType.THINKING, {"message": "should be ignored"})
        assert emitter._queue.empty()


# ==================== 4. 集成测试：完整流程模拟 ====================

class TestFullFlowSimulation:
    """模拟完整业务流程中的错误处理"""

    @pytest.mark.asyncio
    async def test_full_flow_with_sql_generation_error(self):
        """
        模拟完整流程：
        用户提问 → 意图识别成功 → SQL生成返回400 →
        前端收到ERROR事件 → 流正常关闭
        """
        from app.events import EventEmitter, EventType
        from app.core.llm import LLMClientError

        emitter = EventEmitter()

        # 模拟 run_and_save() 函数的行为
        async def simulate_run_and_save():
            try:
                # 模拟意图识别成功
                emitter.emit(EventType.THINKING, {
                    "phase": "chain_start",
                    "name": "intent_router",
                    "message": "意图识别"
                })

                # 模拟进入 sql_simple 节点
                emitter.emit(EventType.THINKING, {
                    "phase": "chain_start",
                    "name": "sql_simple",
                    "message": "SQL 查询处理"
                })

                # 模拟调用 generate_sql 工具
                emitter.emit(EventType.TOOL_CALL, {
                    "tool": "generate_sql",
                    "message": "调用工具: generate_sql"
                })

                # 模拟 400 错误发生
                raise LLMClientError("Prompt too long", status_code=400)

            except Exception as e:
                # 捕获并发送错误事件
                if isinstance(e, LLMClientError) and getattr(e, 'status_code') == 400:
                    emitter.emit_error(
                        error_type="client_error_400",
                        message="请求参数无效，可能是查询条件过于复杂",
                        suggestion="请简化问题后重试"
                    )

                # 发送 DONE 事件
                emitter.emit_done(
                    response_type="error",
                    message=str(e)[:200],
                    error_type=type(e).__name__
                )

        # 执行模拟
        await simulate_run_and_save()

        # 收集所有发射的事件
        events = []
        while not emitter.is_empty():
            event = emitter._queue.get_nowait()
            events.append(event)

        # 验证事件序列
        assert len(events) >= 4  # 至少有：intent_start, sql_start, ERROR, DONE

        # 找到 ERROR 和 DONE 事件
        error_event = next((e for e in events if e.get("type") == EventType.ERROR), None)
        done_event = next((e for e in events if e.get("type") == EventType.DONE), None)

        # 验证 ERROR 事件
        assert error_event is not None
        assert error_event["data"]["error_type"] == "client_error_400"
        assert "请求参数无效" in error_event["data"]["message"]
        assert "简化问题" in error_event["data"]["suggestion"]

        # 验证 DONE 事件
        assert done_event is not None
        assert done_event["data"]["response_type"] == "error"
        assert done_event["data"]["error_type"] == "LLMClientError"

    @pytest.mark.asyncio
    async def test_full_flow_with_server_error_and_retry_success(self):
        """
        模拟服务端错误后重试成功的流程：
        用户提问 → SQL生成遇到500 → 重试 → 成功 → 返回结果
        """
        from app.events import EventEmitter, EventType
        from app.core.llm import retry_on_transient_error

        emitter = EventEmitter()
        call_count = 0

        async def mock_generate_sql():
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # 第一次调用失败
                emitter.emit(EventType.TOOL_CALL, {
                    "tool": "generate_sql",
                    "message": f"第 {call_count} 次尝试调用 generate_sql"
                })
                response = MagicMock()
                response.status_code = 500
                raise HTTPStatusError("Internal Server Error", request=MagicMock(), response=response)

            # 第二次调用成功
            emitter.emit(EventType.TOOL_RESULT, {
                "tool": "generate_sql",
                "message": f"第 {call_count} 次调用成功",
                "result": "SELECT * FROM users"
            })
            return {"sql": "SELECT * FROM users"}

        # 使用带重试的装饰器
        decorated_generate = retry_on_transient_error(max_retries=2, base_delay=0.01)(mock_generate_sql)

        # 模拟流程
        async def simulate_flow():
            try:
                emitter.emit(EventType.THINKING, {
                    "phase": "chain_start",
                    "name": "sql_simple",
                    "message": "开始 SQL 生成"
                })

                result = await decorated_generate()

                emitter.emit(EventType.ANSWER, {
                    "content": f"查询结果: {result['sql']}",
                    "confidence": 0.95
                })

                emitter.emit_done(response_type="success")

            except Exception as e:
                emitter.emit_error(
                    error_type="unexpected_error",
                    message=f"未预期的错误: {str(e)}"
                )
                emitter.emit_done(response_type="error", error_type=type(e).__name__)

        await simulate_flow()

        # 收集并验证事件
        events = []
        while not emitter.is_empty():
            events.append(emitter._queue.get_nowait())

        # 应该有：THINKING(start), TOOL_CALL(1st), TOOL_RESULT(2nd success), ANSWER, DONE
        assert len(events) >= 4

        # 不应该有 ERROR 事件（因为重试成功了）
        error_events = [e for e in events if e.get("type") == EventType.ERROR]
        assert len(error_events) == 0

        # 应该有成功的 DONE 事件
        done_event = next((e for e in events if e.get("type") == EventType.DONE), None)
        assert done_event is not None
        assert done_event["data"]["response_type"] == "success"

        # 验证确实调用了 2 次
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_full_flow_with_timeout_error(self):
        """
        模拟超时错误的完整流程：
        多次超时后最终失败 → 发送超时错误事件
        """
        from app.events import EventEmitter, EventType
        from app.core.llm import retry_on_transient_error, LLMTimeoutError

        emitter = EventEmitter()

        async def always_timeout():
            emitter.emit(EventType.TOOL_CALL, {
                "tool": "llm_query",
                "message": "调用 LLM"
            })
            raise ReadTimeout("Request timeout after 120s")

        # 设置较少的重试次数以加速测试
        decorated_call = retry_on_transient_error(max_retries=2, base_delay=0.01)(always_timeout)

        async def simulate_flow():
            try:
                emitter.emit(EventType.THINKING, {
                    "phase": "chain_start",
                    "name": "sql_agent",
                    "message": "开始处理"
                })

                result = await decorated_call()

                emitter.emit(EventType.ANSWER, {"content": result})
                emitter.emit_done(response_type="success")

            except LLMTimeoutError as e:
                emitter.emit_error(
                    error_type="llm_timeout",
                    message="AI 模型响应超时，可能是查询过于复杂或服务负载过高",
                    suggestion="请稍后重试，或简化问题后再次尝试",
                    details={"original_error": str(e)}
                )
                emitter.emit_done(
                    response_type="error",
                    message=str(e)[:200],
                    error_type="LLMTimeoutError"
                )

        await simulate_flow()

        # 收集事件
        events = []
        while not emitter.is_empty():
            events.append(emitter._queue.get_nowait())

        # 验证有 ERROR 事件
        error_event = next((e for e in events if e.get("type") == EventType.ERROR), None)
        assert error_event is not None
        assert error_event["data"]["error_type"] == "llm_timeout"
        assert "超时" in error_event["data"]["message"]
        assert error_event["priority"] == "high"

        # 验证有 DONE 事件
        done_event = next((e for e in events if e.get("type") == EventType.DONE), None)
        assert done_event is not None
        assert done_event["data"]["response_type"] == "error"


# ==================== 5. 边界情况和特殊场景测试 ====================

class TestEdgeCasesAndSpecialScenarios:
    """测试边界条件和特殊情况"""

    @pytest.mark.asyncio
    async def test_immediate_success_no_retry(self):
        """验证首次调用成功时不触发任何重试逻辑"""
        from app.core.llm import retry_on_transient_error

        call_count = 0

        async def successful_call():
            nonlocal call_count
            call_count += 1
            return "immediate success"

        decorated = retry_on_transient_error(max_retries=3)(successful_call)
        result = await decorated()

        assert result == "immediate success"
        assert call_count == 1  # 只调用了一次

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """验证所有重试次数用尽后抛出正确的异常"""
        from app.core.llm import retry_on_transient_error, LLMServerError

        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            response.status_code = 502
            raise HTTPStatusError("Bad Gateway", request=MagicMock(), response=response)

        decorated = retry_on_transient_error(max_retries=2, base_delay=0.01)(always_fail)

        with pytest.raises(LLMServerError) as exc_info:
            await decorated()

        assert exc_info.value.status_code == 502
        assert call_count == 3  # 首次 + 2次重试

    @pytest.mark.asyncio
    async def test_unexpected_exception_not_retried(self):
        """验证非预期异常不会触发重试"""
        from app.core.llm import retry_on_transient_error

        call_count = 0

        async def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Unexpected logic error")

        decorated = retry_on_transient_error(max_retries=3)(raise_value_error)

        with pytest.raises(ValueError, match="Unexpected logic error"):
            await decorated()

        assert call_count == 1  # 不应该重试

    @pytest.mark.asyncio
    async def test_retryable_llm_error_respects_retry_flag(self):
        """验证可重试的 LLMError 会触发重试，不可重试的不会"""
        from app.core.llm import retry_on_transient_error, LLMClientError

        # 测试可重试的情况（408）
        call_count_retryable = 0

        async def raise_retryable_error():
            nonlocal call_count_retryable
            call_count_retryable += 1
            if call_count_retryable < 2:
                raise LLMClientError("Timeout", 408)
            return "recovered"

        decorated_retryable = retry_on_transient_error(max_retries=2, base_delay=0.01)(raise_retryable_error)
        result = await decorated_retryable()
        assert result == "recovered"
        assert call_count_retryable == 2

        # 测试不可重试的情况（400）
        call_count_nonretryable = 0

        async def raise_nonretryable_error():
            nonlocal call_count_nonretryable
            call_count_nonretryable += 1
            raise LLMClientError("Bad request", 400)

        decorated_nonretryable = retry_on_transient_error(max_retries=3)(raise_nonretryable_error)

        with pytest.raises(LLMClientError):
            await decorated_nonretryable()

        assert call_count_nonretryable == 1  # 不应该重试

    def test_null_emitter_methods(self):
        """验证 NullEmitter 的所有方法都不会报错"""
        from app.events import NullEmitter

        emitter = NullEmitter()

        # 这些调用都不应该抛出异常
        emitter.emit("thinking", {"message": "test"})
        emitter.emit_done(response_type="success")
        emitter.emit_error(error_type="test", message="test error")
        emitter.emit_error(error_type="test", message="test", suggestion="custom suggestion", details={"key": "value"})

        assert emitter.is_empty() is True
        assert emitter.queue_size() == 0

    @pytest.mark.asyncio
    async def test_event_ordering_in_error_scenario(self):
        """验证错误场景下的事件顺序符合预期"""
        from app.events import EventEmitter, EventType
        from app.core.llm import LLMRateLimitError

        emitter = EventEmitter()

        async def simulate_ordered_flow():
            try:
                emitter.emit(EventType.THINKING, {"step": 1, "message": "开始"})
                emitter.emit(EventType.THINKING, {"step": 2, "message": "处理中"})

                raise LLMRateLimitError(retry_after=10)

            except LLMRateLimitError as e:
                emitter.emit_error(
                    error_type="rate_limited",
                    message="API 调用频率超限",
                    suggestion=f"请等待 {e.retry_after} 秒后重试"
                )
                emitter.emit_done(response_type="error", error_type="LLMRateLimitError")

        await simulate_ordered_flow()

        events = []
        while not emitter.is_empty():
            events.append(emitter._queue.get_nowait())

        # 验证顺序：THINKING(1), THINKING(2), ERROR, DONE
        assert len(events) == 4
        assert events[0]["type"] == EventType.THINKING
        assert events[0]["data"]["step"] == 1
        assert events[1]["type"] == EventType.THINKING
        assert events[1]["data"]["step"] == 2
        assert events[2]["type"] == EventType.ERROR
        assert events[3]["type"] == EventType.DONE


# ==================== 运行入口 ====================

if __name__ == "__main__":
    # 直接运行此文件执行所有测试
    pytest.main([__file__, "-v", "--tb=short"])
