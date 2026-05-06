import logging
import json
from typing import Any, Dict, Optional

from app.agent.tools.base import BaseTool, ToolResult
from app.rag.retrieval import retrieve_schema, retrieve_table_info_simple
from config.redis import get_redis_client

logger = logging.getLogger(__name__)


class SearchSchemaTool(BaseTool):
    name = "search_schema"
    description = "检索与用户问题相关的数据库表结构信息，包括字段详情、表关系等。当需要了解数据库有哪些表、某个表的字段结构时使用。"
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "用于检索表结构的问题文本"
            }
        },
        "required": ["question"]
    }

    async def _get_relation_map(self) -> Dict[str, list]:
        redis_client = await get_redis_client()
        relation_map = {}
        async for key in redis_client.scan_iter("rag:relation:*"):
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            table_name = key.replace("rag:relation:", "")
            data = await redis_client.get(key)
            if data:
                relation_map[table_name] = json.loads(data)
        return relation_map

    async def execute(self, **kwargs) -> ToolResult:
        question = kwargs.get("question", "")
        if not question:
            return ToolResult(success=False, error="question 参数不能为空")

        try:
            relation_map = await self._get_relation_map()

            if relation_map:
                metadata = await retrieve_schema(question, relation_map)
            else:
                metadata = await retrieve_table_info_simple(question)

            if not metadata or "未找到" in metadata:
                return ToolResult(success=True, data="未找到相关表结构信息")

            logger.info(f"📚 表结构检索成功：{len(metadata)} 字符")
            return ToolResult(success=True, data=metadata)

        except Exception as e:
            logger.error(f"❌ 表结构检索失败：{e}")
            return ToolResult(success=False, error=str(e))
