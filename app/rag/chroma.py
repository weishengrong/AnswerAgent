import chromadb
from typing import Optional, List, Dict, Any

from config.settings import chroma_settings, rag_settings

DIMENSION = 1024

_client: Optional[chromadb.PersistentClient] = None


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
