import json
from typing import Dict, List, Optional
from utils.chunk_builder import ParentChunk


class ParentStore:
    def __init__(self):
        self.redis_client = None
        self._relation_map: Dict[str, List[Dict]] = {}

    async def init(self):
        from config.redis import get_redis_client
        self.redis_client = await get_redis_client()

    def set_relation_map(self, relation_map: Dict[str, List[Dict]]):
        self._relation_map = relation_map

    async def store_parent(self, chunk: ParentChunk):
        if not self.redis_client:
            await self.init()

        key = f"rag:parent:{chunk.table_name}"
        await self.redis_client.set(key, chunk.content)

    async def store_parents(self, chunks: List[ParentChunk]):
        if not self.redis_client:
            await self.init()

        pipe = self.redis_client.pipeline()
        for chunk in chunks:
            key = f"rag:parent:{chunk.table_name}"
            pipe.set(key, chunk.content)
        await pipe.execute()

    async def get_parent(self, table_name: str) -> Optional[str]:
        if not self.redis_client:
            await self.init()

        key = f"rag:parent:{table_name}"
        return await self.redis_client.get(key)

    async def get_parents(self, table_names: List[str]) -> Dict[str, str]:
        if not self.redis_client:
            await self.init()

        result = {}
        for table_name in table_names:
            content = await self.get_parent(table_name)
            if content:
                result[table_name] = content
        return result

    async def store_relation_map(self):
        if not self.redis_client:
            await self.init()

        for table_name, relations in self._relation_map.items():
            key = f"rag:relation:{table_name}"
            await self.redis_client.set(key, json.dumps(relations))

    async def get_relation(self, table_name: str) -> List[Dict]:
        if not self.redis_client:
            await self.init()

        key = f"rag:relation:{table_name}"
        data = await self.redis_client.get(key)
        if data:
            return json.loads(data)
        return []

    async def get_relation_map(self) -> Dict[str, List[Dict]]:
        if not self.redis_client:
            await self.init()

        result = {}
        for table_name in self._relation_map.keys():
            relations = await self.get_relation(table_name)
            if relations:
                result[table_name] = relations
        return result

    async def clear_all(self):
        if not self.redis_client:
            await self.init()

        keys = []
        async for key in self.redis_client.scan_iter("rag:parent:*"):
            keys.append(key)
        async for key in self.redis_client.scan_iter("rag:relation:*"):
            keys.append(key)

        if keys:
            await self.redis_client.delete(*keys)


parent_store = ParentStore()
