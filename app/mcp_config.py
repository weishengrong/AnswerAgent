import logging
from typing import Optional

from pydantic_ai.mcp import MCPServerStdio, MCPServerSSE

logger = logging.getLogger(__name__)


def create_time_server() -> MCPServerStdio:
    """创建 Stdio 模式的时间工具 MCP Server

    作为本地子进程启动，通过 stdin/stdout 通信。
    提供：get_current_time, get_date_range, get_relative_date, is_workday, get_workdays
    """
    server = MCPServerStdio(
        "python",
        args=["mcp_servers/time_server.py"],
    )
    logger.info("✅ 时间工具 MCP Server (Stdio) 已配置")
    return server


def create_calendar_server(host: str = "127.0.0.1", port: int = 9001) -> MCPServerSSE:
    """创建 SSE 模式的日历工具 MCP Server

    连接远程日历服务，通过 HTTP/SSE 通信。
    提供：get_holidays, get_holiday_detail, get_next_holiday,
          add_calendar_event, list_calendar_events, delete_calendar_event, get_monthly_overview

    需要先启动日历服务：python mcp_servers/calendar_server.py
    """
    server = MCPServerSSE(
        url=f"http://{host}:{port}/sse",
    )
    logger.info(f"✅ 日历工具 MCP Server (SSE) 已配置，连接 {host}:{port}")
    return server


def get_all_mcp_servers(
    calendar_host: str = "127.0.0.1",
    calendar_port: int = 9001,
    enable_calendar: bool = True,
) -> list:
    """获取所有 MCP Server 实例，用于注入到 pydantic_ai Agent

    用法：
        from app.mcp_config import get_all_mcp_servers
        servers = get_all_mcp_servers()
        agent = Agent(model=llm_model, toolsets=servers, ...)

    Args:
        calendar_host: 日历服务地址
        calendar_port: 日历服务端口
        enable_calendar: 是否启用日历服务（需要先启动 calendar_server.py）
    """
    servers = [create_time_server()]

    if enable_calendar:
        servers.append(create_calendar_server(calendar_host, calendar_port))

    return servers
