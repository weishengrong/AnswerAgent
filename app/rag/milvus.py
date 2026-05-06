from pymilvus import MilvusClient
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema
from pymilvus import DataType
from typing import Optional

from config.settings import milvus_settings

DIMENSION = 1024


class MilvusClientFactory:

    _instance: Optional["MilvusClientFactory"] = None
    _client: Optional[MilvusClient] = None

    def __new__(cls) -> "MilvusClientFactory":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(
                # f"http://{milvus_settings.MILVUS_HOST}:{milvus_settings.MILVUS_PORT}"
                path=milvus_settings.MILVUS_LITE_PATH
            )
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None


MilvusSessionLocal = MilvusClientFactory()


def create_schema() -> CollectionSchema:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="field_name", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    ]
    return CollectionSchema(fields=fields, description="Database schema collection (parent-child)")


def init_milvus(collection_name: str) -> Collection:
    # 在服务器上用milvuslite，不用connect，本地则需要
    # connections.connect(host=milvus_settings.MILVUS_HOST, port=milvus_settings.MILVUS_PORT)

    # if utility.has_collection(collection_name):
    #     print(f"集合 {collection_name} 已存在，将删除重建以确保数据干净...")
    #     utility.drop_collection(collection_name)
    #
    # schema = create_schema()
    # collection = Collection(collection_name, schema)
    #
    # index_params = {
    #     "metric_type": "COSINE",
    #     "index_type": "HNSW",
    #     "params": {"M": 8, "efConstruction": 200}
    # }
    # collection.create_index(field_name="vector", index_params=index_params)
    # print("Milvus 集合初始化完成。")
    # return collection
    """初始化 Milvus Lite 集合"""
    client = MilvusSessionLocal()

    # 检查集合并删除（如果存在）
    try:
        collections = client.list_collections()
        if collection_name in collections:
            print(f"集合 {collection_name} 已存在，将删除重建以确保数据干净...")
            client.drop_collection(collection_name)
    except Exception as e:
        print(f"检查集合时出错：{e}")

    # 创建集合并创建索引
    client.create_collection(
        collection_name=collection_name,
        dimension=DIMENSION,
        metric_type="COSINE",
        auto_id=True,
        index_params={
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 200}
        }
    )

    print("Milvus 集合初始化完成。")
    return client
