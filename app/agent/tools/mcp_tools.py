import json
import logging
import subprocess
import sys
from typing import Any, Dict, Optional

from app.agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class MCPTimeTool(BaseTool):
    name = "mcp_time"
    description = "时间工具集：获取当前时间、解析自然语言日期范围、判断工作日、计算工作日天数。当用户问题涉及相对时间（如'上个月'、'本周'、'3天前'）或需要判断工作日时使用。"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "要执行的操作：get_current_time / get_date_range / get_relative_date / is_workday / get_workdays"
            },
            "params": {
                "type": "object",
                "description": "操作参数，不同 action 需要不同参数"
            }
        },
        "required": ["action"]
    }

    _ACTION_PARAMS = {
        "get_current_time": {"timezone": "Asia/Shanghai"},
        "get_date_range": {"period": ""},
        "get_relative_date": {"expression": ""},
        "is_workday": {"date": ""},
        "get_workdays": {"start_date": "", "end_date": ""},
    }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "")
        params = kwargs.get("params", {})

        if action not in self._ACTION_PARAMS:
            return ToolResult(
                success=False,
                error=f"不支持的操作：{action}，可选：{list(self._ACTION_PARAMS.keys())}"
            )

        try:
            result = await self._call_mcp_tool(action, params)
            return ToolResult(success=True, data=result)
        except Exception as e:
            logger.error(f"❌ MCP 时间工具调用失败：{e}")
            return ToolResult(success=False, error=str(e))

    async def _call_mcp_tool(self, tool_name: str, params: dict) -> str:
        import asyncio

        script_path = "mcp_servers/time_server.py"

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            }
        }

        proc = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        proc.stdin.write((json.dumps(request) + "\n").encode())
        proc.stdin.write(b"\n")
        await proc.stdin.drain()

        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=30.0)
            response = json.loads(line.decode())

            if "result" in response:
                content = response["result"].get("content", [])
                if content and isinstance(content, list):
                    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    return "\n".join(texts)
                return json.dumps(response["result"], ensure_ascii=False)
            elif "error" in response:
                return json.dumps(response["error"], ensure_ascii=False)
            else:
                return json.dumps(response, ensure_ascii=False)
        except asyncio.TimeoutError:
            proc.kill()
            return json.dumps({"error": "MCP 调用超时"}, ensure_ascii=False)
        finally:
            try:
                proc.kill()
            except Exception:
                pass


class MCPCalendarTool(BaseTool):
    name = "mcp_calendar"
    description = "日历工具集：查询节假日、查看假期详情、查询下一个假期、管理自定义日历事件、获取月度概览。当用户问放假安排、节假日、日历事件时使用。"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "要执行的操作：get_holidays / get_holiday_detail / get_next_holiday / add_calendar_event / list_calendar_events / delete_calendar_event / get_monthly_overview"
            },
            "params": {
                "type": "object",
                "description": "操作参数"
            }
        },
        "required": ["action"]
    }

    _ACTION_PARAMS = {
        "get_holidays": {"year": 2025, "month": None},
        "get_holiday_detail": {"date": ""},
        "get_next_holiday": {"from_date": None},
        "add_calendar_event": {"title": "", "date": "", "event_type": "reminder", "description": ""},
        "list_calendar_events": {"start_date": None, "end_date": None, "event_type": None},
        "delete_calendar_event": {"event_id": ""},
        "get_monthly_overview": {"year": 2025, "month": 1},
    }

    def __init__(self, host: str = "127.0.0.1", port: int = 9001):
        self.host = host
        self.port = port

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "")
        params = kwargs.get("params", {})

        if action not in self._ACTION_PARAMS:
            return ToolResult(
                success=False,
                error=f"不支持的操作：{action}，可选：{list(self._ACTION_PARAMS.keys())}"
            )

        try:
            result = await self._call_sse_tool(action, params)
            return ToolResult(success=True, data=result)
        except Exception as e:
            logger.error(f"❌ MCP 日历工具调用失败：{e}")
            return ToolResult(success=False, error=str(e))

    async def _call_sse_tool(self, tool_name: str, params: dict) -> str:
        import httpx

        base_url = f"http://{self.host}:{self.port}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            init_request = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "answer-agent", "version": "1.0"}
                }
            }

            sse_resp = await client.get(f"{base_url}/sse", headers={"Accept": "text/event-stream"})
            message_endpoint = None

            for line in sse_resp.text.split("\n"):
                if line.startswith("data:") and "messages" in line:
                    message_endpoint = line.split("data:")[1].strip().strip('"')
                    break

            if not message_endpoint:
                return json.dumps({"error": "无法获取 SSE 消息端点"}, ensure_ascii=False)

            if not message_endpoint.startswith("http"):
                message_endpoint = f"{base_url}{message_endpoint}"

            await client.post(message_endpoint, json=init_request)

            tool_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": params
                }
            }

            resp = await client.post(message_endpoint, json=tool_request)
            response = resp.json()

            if "result" in response:
                content = response["result"].get("content", [])
                if content and isinstance(content, list):
                    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    return "\n".join(texts)
                return json.dumps(response["result"], ensure_ascii=False)
            elif "error" in response:
                return json.dumps(response["error"], ensure_ascii=False)
            else:
                return json.dumps(response, ensure_ascii=False)
