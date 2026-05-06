from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    parameters: Dict = {}

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        ...

    def to_prompt_format(self) -> str:
        params_desc = []
        for pname, pinfo in self.parameters.get("properties", {}).items():
            required = pname in self.parameters.get("required", [])
            req_mark = "（必填）" if required else "（可选）"
            params_desc.append(f"  - {pname}: {pinfo.get('description', '')}{req_mark}")
        params_text = "\n".join(params_desc) if params_desc else "  无参数"
        return f"工具名：{self.name}\n描述：{self.description}\n参数：\n{params_text}"


class ToolResult:
    def __init__(self, success: bool, data: Any = None, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error

    def to_dict(self) -> Dict:
        result = {"success": self.success}
        if self.data is not None:
            result["data"] = self.data
        if self.error is not None:
            result["error"] = self.error
        return result
