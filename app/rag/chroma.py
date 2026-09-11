import chromadb
import logging
from typing import Optional, List, Dict, Any

from app.core.config.settings import chroma_settings, rag_settings

DIMENSION = 1024

_client: Optional[chromadb.PersistentClient] = None
_rag_client: Optional[chromadb.PersistentClient] = None  # RAG Schema 专用客户端（独立目录）
logger = logging.getLogger(__name__)


def get_chroma_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=chroma_settings.CHROMA_PERSIST_DIR
        )
    return _client


def close_chroma():
    global _client
    _client = None


def init_chroma(collection_name: str) -> chromadb.Collection:
    """初始化集合：删除旧集合后重建（用于全量重建索引场景）"""
    client = get_chroma_client()

    try:
        collections = client.list_collections()
        if collection_name in [c.name for c in collections]:
            print(f"集合 {collection_name} 已存在，将删除重建以确保数据干净...")
            client.delete_collection(collection_name)
    except Exception as e:
        print(f"检查集合时出错：{e}")

    client.create_collection(
        name=collection_name,
        metadata={"dimension": DIMENSION},
        get_or_create=False
    )

    print("Chroma 集合初始化完成。")
    return client.get_collection(collection_name)


def get_rag_chroma_client() -> chromadb.PersistentClient:
    """获取 RAG Schema 专用 Chroma 客户端（独立目录，与 mem0 隔离）"""
    global _rag_client
    if _rag_client is None:
        _rag_client = chromadb.PersistentClient(path=rag_settings.RAG_CHROMA_PERSIST_DIR)
    return _rag_client


def get_or_init_collection(collection_name: str) -> chromadb.Collection:
    """获取或创建集合（用于 lifespan 内自动重建场景）

    核心约束：lifespan 中 mem0 可能已用不同配置初始化了同目录的 Chroma 单例客户端，
    通过单例做任何操作（get/create/delete_collection）都会触发
    "already exists with different settings" 错误。

    因此：创建独立的 PersistentClient 实例，绕过单例，避免设置冲突。
    """
    import os

    # 使用 RAG 专用客户端（独立目录，与 mem0 隔离）
    try:
        rebuild_client = get_rag_chroma_client()
    except Exception as e:
        logger.error(f"❌ 创建独立 Chroma 客户端失败：{e}")
        raise

    # 尝试获取已有集合
    try:
        collection = rebuild_client.get_collection(collection_name)
        # 集合存在，清空旧数据后复用
        try:
            total = collection.count()
            if total > 0:
                all_ids = []
                offset = 0
                batch_size = 1000
                while True:
                    result = collection.get(limit=batch_size, offset=offset, include=[])
                    if not result["ids"]:
                        break
                    all_ids.extend(result["ids"])
                    offset += batch_size
                if all_ids:
                    collection.delete(ids=all_ids)
                logger.info(f"🧹 集合 {collection_name} 已清空 {total} 条旧数据")
        except Exception as e:
            logger.warning(f"⚠️ 清空集合数据失败（非致命）：{e}")
        return collection
    except Exception as e:
        # 集合不存在，创建新的
        logger.info(f"📦 集合 {collection_name} 不存在，正在创建...")
        return rebuild_client.create_collection(
            name=collection_name,
            metadata={"dimension": DIMENSION},
        )


def chroma_search(
    collection: chromadb.Collection,
    query_embeddings: List[List[float]],
    n_results: int = 10,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        where=where,
        include=["metadatas", "documents", "distances"]
    )

    formatted_results = []
    if results and results.get("ids") and len(results["ids"]) > 0:
        for i in range(len(results["ids"][0])):
            item = {
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else 0,
            }
            if results.get("metadatas") and results["metadatas"][0]:
                item.update(results["metadatas"][0][i])
            if results.get("documents") and results["documents"][0]:
                item["content"] = results["documents"][0][i]
            formatted_results.append(item)

    return formatted_results


class ChromaSessionLocal:
    def __call__(self) -> chromadb.PersistentClient:
        return get_chroma_client()

    def close(self) -> None:
        close_chroma()
