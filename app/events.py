import asyncio
import json
import time
from typing import Any, Dict, Optional, AsyncGenerator


class EventType:
    THINKING = "thinking"
    THINKING_STREAM = "thinking_stream"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REACT_STEP = "react_step"
    ANSWER = "answer"
    ANSWER_STREAM = "answer_stream"
    CLARIFICATION = "clarification"
    ERROR = "error"
    DONE = "done"


class EventEmitter:
    """SSE 事件发射器 —— 生产者-消费者模式中的生产者

    工作流执行过程中，各节点通过 emit() 将中间事件推入 asyncio.Queue；
    /ask/stream 接口通过 stream() 异步迭代消费队列，将事件以 SSE 格式实时推送给前端。
    典型事件流：THINKING → TOOL_CALL → TOOL_RESULT → ANSWER_STREAM → DONE
    """

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._closed = False
        # 思考流共享累积（单次请求所有 LLM token 累积到同一个卡片）
        self._thinking_stream_id: Optional[str] = None
        self._thinking_text: str = ""

    def init_thinking_stream(self, stream_id: str):
        """初始化思考流 ID（在请求开始时调用一次）"""
        self._thinking_stream_id = stream_id
        self._thinking_text = ""

    def emit_thinking_token(self, token: str):
        """累积单个 token 到思考流，发射更新后的完整文本。
        所有 LLM 调用共享同一个 stream_id，确保前端只显示一个卡片。"""
        if self._closed or not self._thinking_stream_id or not token:
            return
        self._thinking_text += token
        self.emit(EventType.THINKING_STREAM, {
            "stream_id": self._thinking_stream_id,
            "text": self._thinking_text
        })

    def emit(self, event_type: str, data: Any = None):
        """将事件推入队列，供 stream() 消费。closed 后不再接受新事件。"""
        if self._closed:
            return
        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        self._queue.put_nowait(event)

    def emit_done(self, **kwargs):
        """发送 DONE 事件并关闭发射器，stream() 收到后终止迭代。"""
        self.emit(EventType.DONE, kwargs)
        self._closed = True

    def emit_error(self, error_type: str, message: str, suggestion: str = None, details: dict = None):
        """
        发射 ERROR 事件（高优先级，即使队列满也会强制插入）

        Args:
            error_type: 错误类型标识符（如 'llm_timeout', 'client_error', 'server_error'）
            message: 用户可见的错误描述（中文）
            suggestion: 给用户的操作建议（可选）
            details: 技术细节字典（可选，用于调试，不展示给用户）

        Usage:
            emitter.emit_error(
                error_type="llm_timeout",
                message="AI 模型响应超时，请稍后重试",
                suggestion="可以尝试简化问题或稍后再试",
                details={"request_id": "abc123", "timeout": 60}
            )
        """
        import logging
        logger = logging.getLogger(__name__)

        error_event = {
            "type": EventType.ERROR,
            "data": {
                "error_type": error_type,
                "message": message,
                "suggestion": suggestion or "请稍后重试",
                "timestamp": time.time(),
                **(details or {})
            },
            "priority": "high",  # 标记为高优先级
        }

        # 如果队列已满，尝试清空非错误事件再插入
        try:
            self._queue.put_nowait(error_event)
        except asyncio.QueueFull:
            logger.warning(
                f"⚠️ 事件队列已满，正在清理旧事件以插入 ERROR 事件 "
                f"(error_type={error_type})"
            )
            # 清空队列（丢弃旧的事件，保留最新的错误信息）
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            # 重新插入错误事件
            self._queue.put_nowait(error_event)

        logger.error(
            f"🚨 ERROR 事件已发射 | type={error_type} | msg={message[:50]}..."
        )

    async def stream(self) -> AsyncGenerator[str, None]:
        """
        异步生成器 —— 从队列消费事件，格式化为 SSE 文本流。

        改进：
        - 更好的异常处理
        - 确保资源释放
        - 心跳机制保持活跃
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.debug("📡 SSE stream 开始消费事件队列")

        try:
            while True:
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=300.0)
                except asyncio.TimeoutError:
                    # 发送心跳防止连接被代理/负载均衡器断开
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    continue

                # 序列化事件（确保所有对象都可 JSON 化）
                try:
                    event_json = json.dumps(event, ensure_ascii=False, default=str)
                except (TypeError, ValueError) as e:
                    logger.error(f"❌ 事件序列化失败: {e}, event={event}")
                    event_json = json.dumps({
                        "type": EventType.ERROR,
                        "data": {
                            "error_type": "serialization_error",
                            "message": "内部事件格式错误",
                            "suggestion": "请联系管理员"
                        },
                        "timestamp": time.time()
                    }, ensure_ascii=False)

                yield f"data: {event_json}\n\n"

                # 检查是否应该结束
                if event.get("type") == EventType.DONE:
                    logger.info("✅ 收到 DONE 事件，SSE stream 正常关闭")
                    break

        except GeneratorExit:
            # 前端断开连接
            logger.info("🔌 前端断开连接，SSE stream 被迫关闭")
        except asyncio.CancelledError:
            # 任务被取消
            logger.warning("⚠️ SSE stream 被取消")
        except Exception as e:
            # 未预期的异常
            logger.error(f"❌ SSE stream 异常中断: {type(e).__name__}: {e}")
            # 尝试发送最后的错误事件
            try:
                error_data = {
                    'type': EventType.ERROR,
                    'data': {
                        'error_type': 'stream_error',
                        'message': '连接异常中断',
                        'suggestion': '请刷新页面重试'
                    },
                    'timestamp': time.time()
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            except:
                pass  # 如果连这个都失败了，就放弃
        finally:
            logger.debug("🔚 SSE stream 已退出清理")

    def is_empty(self) -> bool:
        """检查队列是否为空（用于测试和监控）"""
        return self._queue.empty()

    def queue_size(self) -> int:
        """获取当前队列大小（用于监控和调试）"""
        return self._queue.qsize()


class NullEmitter:
    """空事件发射器 —— 空对象模式"""

    def init_thinking_stream(self, stream_id: str):
        pass

    def emit_thinking_token(self, token: str):
        pass

    def emit(self, event_type: str, data: Any = None):
        pass

    def emit_done(self, **kwargs):
        pass

    def emit_error(self, error_type: str, message: str, suggestion: str = None, details: dict = None):
        pass  # 空实现，不抛出异常

    async def stream(self):
        # 返回空生成器
        return
        yield  # 让它成为异步生成器（语法要求）

    def is_empty(self) -> bool:
        return True

    def queue_size(self) -> int:
        return 0
