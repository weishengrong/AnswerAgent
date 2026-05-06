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

    async def stream(self) -> AsyncGenerator[str, None]:
        """异步生成器 —— 从队列消费事件，格式化为 SSE 文本流。

        每个事件格式：data: {"type": "xxx", "data": {...}, "timestamp": ...}\\n\\n
        300秒无事件时发送 ping 心跳，防止连接被中间代理超时断开。
        收到 DONE 事件后结束迭代。
        """
        while True:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=300.0)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                continue

            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

            if event.get("type") == EventType.DONE:
                break


class NullEmitter:
    """空事件发射器 —— 空对象模式（Null Object Pattern）

    /ask 接口不需要实时推送中间事件，但工作流代码中大量调用 emitter.emit()。
    传入 NullEmitter 后所有 emit() 调用变为空操作，避免在工作流中到处写
    if emitter: emitter.emit(...) 的防御性判断。
    """

    def emit(self, event_type: str, data: Any = None):
        pass

    def emit_done(self, **kwargs):
        pass
