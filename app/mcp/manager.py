import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client

from app.mcp.registry import MCPServerConfig, get_registry

logger = logging.getLogger(__name__)


class MCPConnection:
    """单个 MCP Server 的连接状态"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session: Optional[ClientSession] = None
        self.connected = False
        self.tools: List[Dict[str, Any]] = []
        self.error: Optional[str] = None

    @property
    def name(self) -> str:
        return self.config.name


class MCPManager:
    """MCP Client 生命周期管理器

    负责管理所有 MCP Server 的连接、断开、健康检查。
    使用 mcp Python SDK 的 ClientSession 进行标准 MCP 通信。
    """

    def __init__(self):
        self._connections: Dict[str, MCPConnection] = {}
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return

        registry = get_registry()
        for config in registry.get_enabled():
            conn = MCPConnection(config)
            self._connections[config.name] = conn

        self._initialized = True
        logger.info(f"🔌 MCP Manager 初始化完成，管理 {len(self._connections)} 个连接")

    async def connect(self, server_name: str) -> Optional[MCPConnection]:
        conn = self._connections.get(server_name)
        if not conn:
            logger.warning(f"⚠️ 未注册的 MCP Server: {server_name}")
            return None

        if conn.connected:
            return conn

        try:
            config = conn.config

            if config.transport == "stdio":
                await self._connect_stdio(conn)
            elif config.transport == "sse":
                await self._connect_sse(conn)
            else:
                conn.error = f"不支持的传输模式: {config.transport}"
                logger.error(f"❌ {conn.error}")

        except Exception as e:
            conn.connected = False
            conn.error = str(e)
            logger.error(f"❌ MCP Server {server_name} 连接失败: {e}")

        return conn

    async def _connect_stdio(self, conn: MCPConnection):
        config = conn.config
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env if config.env else None,
        )

        read_stream, write_stream = await self._enter_stdio(server_params)
        session = ClientSession(read_stream, write_stream)
        await session.initialize()

        conn.session = session
        conn.connected = True
        conn.error = None

        tools_result = await session.list_tools()
        conn.tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema if t.inputSchema else {},
            }
            for t in tools_result.tools
        ]

        logger.info(f"✅ MCP Server {conn.name} (stdio) 已连接，发现 {len(conn.tools)} 个工具")

    async def _connect_sse(self, conn: MCPConnection):
        config = conn.config

        read_stream, write_stream = await self._enter_sse(
            url=config.url,
            headers=config.headers,
        )
        session = ClientSession(read_stream, write_stream)
        await session.initialize()

        conn.session = session
        conn.connected = True
        conn.error = None

        tools_result = await session.list_tools()
        conn.tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema if t.inputSchema else {},
            }
            for t in tools_result.tools
        ]

        logger.info(f"✅ MCP Server {conn.name} (sse) 已连接，发现 {len(conn.tools)} 个工具")

    async def _enter_stdio(self, server_params: StdioServerParameters):
        cm = stdio_client(server_params)
        return await cm.__aenter__()

    async def _enter_sse(self, url: str, headers: dict = None):
        cm = sse_client(url=url, headers=headers)
        return await cm.__aenter__()

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict = None) -> Any:
        conn = self._connections.get(server_name)
        if not conn or not conn.connected or not conn.session:
            raise RuntimeError(f"MCP Server {server_name} 未连接")

        result = await conn.session.call_tool(tool_name, arguments)

        if result.isError:
            error_text = ""
            for content in result.content:
                if hasattr(content, "text"):
                    error_text += content.text
            raise RuntimeError(f"MCP 工具调用失败: {error_text}")

        output = ""
        for content in result.content:
            if hasattr(content, "text"):
                output += content.text

        return output

    async def disconnect(self, server_name: str):
        conn = self._connections.get(server_name)
        if conn:
            conn.connected = False
            conn.session = None
            logger.info(f"🔌 MCP Server {server_name} 已断开")

    async def disconnect_all(self):
        for name in list(self._connections.keys()):
            await self.disconnect(name)

    def get_connection(self, server_name: str) -> Optional[MCPConnection]:
        return self._connections.get(server_name)

    def get_all_tools(self) -> List[Dict[str, Any]]:
        tools = []
        for conn in self._connections.values():
            if conn.connected:
                for tool in conn.tools:
                    tools.append({
                        "server": conn.name,
                        "transport": conn.config.transport,
                        **tool
                    })
        return tools

    def health_check(self) -> Dict[str, Any]:
        result = {}
        for name, conn in self._connections.items():
            result[name] = {
                "connected": conn.connected,
                "transport": conn.config.transport,
                "enabled": conn.config.enabled,
                "tools_count": len(conn.tools) if conn.connected else 0,
                "error": conn.error
            }
        return result


_manager: Optional[MCPManager] = None


def get_manager() -> MCPManager:
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager
