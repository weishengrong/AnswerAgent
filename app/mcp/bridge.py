import json
import logging
from typing import Any, Dict, List, Optional

from app.agent.tools.base import BaseTool, ToolResult
from app.mcp.manager import get_manager

logger = logging.getLogger(__name__)


class MCPToolBridge(BaseTool):
    """MCP 工具桥接器

    将 MCP Server 上的单个工具包装为 BaseTool，
    使其可以无缝接入现有的 ReAct 循环和 Skill 体系。

    设计模式：Adapter Pattern
    - 每个远端 MCP 工具 → 一个 MCPToolBridge 实例
    - execute() 时通过 MCPManager 调用远端工具
    - 自动处理连接、错误、降级
    """

    def __init__(self, server_name: str, tool_name: str, tool_description: str, input_schema: dict):
        self.server_name = server_name
        self._tool_name = tool_name
        self._input_schema = input_schema

        self.name = f"mcp_{server_name}_{tool_name}"
        self.description = f"[MCP:{server_name}] {tool_description}"
        self.parameters = input_schema if input_schema else {
            "type": "object",
            "properties": {},
        }

    async def execute(self, **kwargs) -> ToolResult:
        manager = get_manager()

        try:
            conn = manager.get_connection(self.server_name)
            if not conn or not conn.connected:
                await manager.connect(self.server_name)
                conn = manager.get_connection(self.server_name)

            if not conn or not conn.connected:
                return ToolResult(
                    success=False,
                    error=f"MCP Server {self.server_name} 连接失败: {conn.error if conn else '未注册'}"
                )

            result = await manager.call_tool(self.server_name, self._tool_name, kwargs)
            return ToolResult(success=True, data=result)

        except Exception as e:
            logger.error(f"❌ MCP 工具 {self.name} 调用失败: {e}")
            return ToolResult(success=False, error=str(e))

    def to_prompt_format(self) -> str:
        params_desc = []
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])
        for pname, pinfo in props.items():
            req_mark = "（必填）" if pname in required else "（可选）"
            desc = pinfo.get("description", pinfo.get("type", ""))
            params_desc.append(f"  - {pname}: {desc}{req_mark}")
        params_text = "\n".join(params_desc) if params_desc else "  无参数"
        return f"工具名：{self.name}\n描述：{self.description}\n参数：\n{params_text}"


async def get_mcp_tools() -> List[MCPToolBridge]:
    """自动发现所有已启用 MCP Server 的工具，并包装为 BaseTool 列表

    流程：
    1. 从 Registry 获取所有已启用的 MCP Server
    2. 通过 Manager 连接每个 Server
    3. 调用 list_tools 发现可用工具
    4. 将每个工具包装为 MCPToolBridge

    Returns:
        List[MCPToolBridge]: 可直接注册到 Skill 的工具列表
    """
    manager = get_manager()
    await manager.initialize()

    tools = []

    for server_name in [c.config.name for c in manager._connections.values()]:
        try:
            conn = await manager.connect(server_name)
            if not conn or not conn.connected:
                logger.warning(f"⚠️ 跳过不可用的 MCP Server: {server_name}")
                continue

            for tool_info in conn.tools:
                bridge = MCPToolBridge(
                    server_name=server_name,
                    tool_name=tool_info["name"],
                    tool_description=tool_info.get("description", ""),
                    input_schema=tool_info.get("inputSchema", {}),
                )
                tools.append(bridge)
                logger.info(f"🔗 桥接 MCP 工具: {bridge.name}")

        except Exception as e:
            logger.warning(f"⚠️ MCP Server {server_name} 工具发现失败: {e}，已跳过")

    logger.info(f"📋 MCP 工具桥接完成：共 {len(tools)} 个工具")
    return tools


def get_mcp_tools_sync() -> List[MCPToolBridge]:
    """同步版本的 MCP 工具发现（用于 Skill 类定义时）

    注意：此方法不连接 Server，仅根据配置预创建 Bridge 实例。
    工具的 inputSchema 会在首次 execute 时动态获取。

    Returns:
        List[MCPToolBridge]: 预创建的工具桥接列表
    """
    from app.mcp.registry import get_registry

    registry = get_registry()
    tools = []

    for config in registry.get_enabled():
        bridge = MCPToolBridge(
            server_name=config.name,
            tool_name="*",
            tool_description=config.description,
            input_schema={
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": f"要调用的工具名称（{config.name} 提供的工具）"
                    },
                    "arguments": {
                        "type": "object",
                        "description": "工具参数"
                    }
                },
                "required": ["tool_name"]
            }
        )
        bridge.name = f"mcp_{config.name}"
        bridge.description = f"[MCP:{config.name}] {config.description}。通过 tool_name 指定要调用的具体工具。"
        tools.append(bridge)

    return tools


class MCPLazyToolBridge(BaseTool):
    """懒加载 MCP 工具桥接器

    每个 MCP Server 对应一个 MCPLazyToolBridge 实例。
    首次 execute 时才连接 Server 并调用工具。
    适合在 Skill 类定义时使用，无需提前连接。
    """

    def __init__(self, server_name: str, description: str):
        self.server_name = server_name
        self.name = f"mcp_{server_name}"
        self.description = f"[MCP:{server_name}] {description}。通过 tool_name 指定具体工具，arguments 传递参数。"
        self.parameters = {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": f"要调用的 {server_name} 工具名称"
                },
                "arguments": {
                    "type": "object",
                    "description": "工具参数（JSON 对象）"
                }
            },
            "required": ["tool_name"]
        }

    async def execute(self, **kwargs) -> ToolResult:
        tool_name = kwargs.get("tool_name", "")
        arguments = kwargs.get("arguments", {})

        if not tool_name:
            return ToolResult(success=False, error="必须指定 tool_name")

        manager = get_manager()

        try:
            conn = manager.get_connection(self.server_name)
            if not conn or not conn.connected:
                await manager.initialize()
                conn = await manager.connect(self.server_name)

            if not conn or not conn.connected:
                return ToolResult(
                    success=False,
                    error=f"MCP Server {self.server_name} 不可用: {conn.error if conn else '未注册'}"
                )

            available = [t["name"] for t in conn.tools]
            if tool_name not in available:
                return ToolResult(
                    success=False,
                    error=f"工具 {tool_name} 不存在于 {self.server_name}，可用工具: {available}"
                )

            result = await manager.call_tool(self.server_name, tool_name, arguments)
            return ToolResult(success=True, data=result)

        except Exception as e:
            logger.error(f"❌ MCP 工具调用失败 [{self.server_name}/{tool_name}]: {e}")
            return ToolResult(success=False, error=str(e))


def get_lazy_mcp_tools() -> List[MCPLazyToolBridge]:
    """获取懒加载 MCP 工具列表（每个 Server 一个 Bridge）

    适合在 Skill 类定义时使用，无需提前连接 MCP Server。
    首次调用 execute 时才会连接。
    """
    from app.mcp.registry import get_registry

    registry = get_registry()
    tools = []

    for config in registry.get_enabled():
        bridge = MCPLazyToolBridge(
            server_name=config.name,
            description=config.description,
        )
        tools.append(bridge)

    return tools
