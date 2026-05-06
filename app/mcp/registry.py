import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    name: str = Field(description="MCP Server 唯一标识")
    transport: str = Field(description="传输模式：stdio | sse | streamable_http")
    enabled: bool = Field(default=True, description="是否启用")
    description: str = Field(default="", description="服务描述")

    command: Optional[str] = Field(default=None, description="Stdio 模式的可执行命令")
    args: List[str] = Field(default_factory=list, description="Stdio 模式的命令参数")
    env: Dict[str, str] = Field(default_factory=dict, description="环境变量")

    url: Optional[str] = Field(default=None, description="SSE/HTTP 模式的服务地址")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP 请求头")

    startup_hint: str = Field(default="", description="启动提示信息")


class MCPRegistry:
    """MCP Server 注册表

    从 YAML 配置文件加载所有 MCP Server 定义，提供查询和过滤功能。
    支持环境变量替换（${VAR_NAME} 语法）。
    """

    def __init__(self):
        self._servers: Dict[str, MCPServerConfig] = {}
        self._loaded = False

    def load(self, config_path: str = None):
        if self._loaded:
            return

        if config_path is None:
            config_path = str(Path(__file__).parent.parent.parent / "config" / "mcp_servers.yaml")

        if not os.path.exists(config_path):
            logger.warning(f"⚠️ MCP 配置文件不存在：{config_path}")
            self._loaded = True
            return

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not raw or "servers" not in raw:
            logger.warning("⚠️ MCP 配置文件为空或格式错误")
            self._loaded = True
            return

        for item in raw["servers"]:
            item["env"] = {k: self._resolve_env(v) for k, v in item.get("env", {}).items()}
            item["url"] = self._resolve_env(item["url"]) if item.get("url") else None

            config = MCPServerConfig(**item)
            self._servers[config.name] = config
            status = "✅ 启用" if config.enabled else "⏸️ 禁用"
            logger.info(f"{status} MCP Server: {config.name} ({config.transport}) - {config.description}")

        self._loaded = True
        logger.info(f"📋 MCP 注册表加载完成：{len(self._servers)} 个 Server，{len(self.get_enabled())} 个已启用")

    def _resolve_env(self, value: str) -> str:
        if not isinstance(value, str):
            return value
        if value.startswith("${") and value.endswith("}"):
            env_key = value[2:-1]
            resolved = os.environ.get(env_key, "")
            if not resolved:
                logger.warning(f"⚠️ 环境变量 {env_key} 未设置")
            return resolved
        return value

    def get(self, name: str) -> Optional[MCPServerConfig]:
        return self._servers.get(name)

    def get_enabled(self) -> List[MCPServerConfig]:
        return [s for s in self._servers.values() if s.enabled]

    def get_all(self) -> List[MCPServerConfig]:
        return list(self._servers.values())

    def get_by_transport(self, transport: str) -> List[MCPServerConfig]:
        return [s for s in self._servers.values() if s.transport == transport and s.enabled]

    def enable(self, name: str):
        if name in self._servers:
            self._servers[name].enabled = True
            logger.info(f"✅ 已启用 MCP Server: {name}")

    def disable(self, name: str):
        if name in self._servers:
            self._servers[name].enabled = False
            logger.info(f"⏸️ 已禁用 MCP Server: {name}")

    def register(self, config: MCPServerConfig):
        self._servers[config.name] = config
        logger.info(f"➕ 动态注册 MCP Server: {config.name}")

    def summary(self) -> Dict[str, Any]:
        enabled = self.get_enabled()
        return {
            "total": len(self._servers),
            "enabled": len(enabled),
            "disabled": len(self._servers) - len(enabled),
            "servers": {
                name: {
                    "transport": cfg.transport,
                    "enabled": cfg.enabled,
                    "description": cfg.description
                }
                for name, cfg in self._servers.items()
            }
        }


_registry: Optional[MCPRegistry] = None


def get_registry() -> MCPRegistry:
    global _registry
    if _registry is None:
        _registry = MCPRegistry()
        _registry.load()
    return _registry
