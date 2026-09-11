from pymilvus import MilvusClient
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema
from pymilvus import DataType
from typing import Optional

from app.core.config.settings import milvus_settings

DIMENSION = 1024


class MilvusClientFactory:
    _instance: Optional["MilvusClientFactory"] = None
    _client: Optional[MilvusClient] = None

    def __new__(cls) -> "MilvusClientFactory":
        if cls._instance is None:  # ← 改这里：self → cls
            cls._instance = super().__new__(cls)
        return cls._instance

    def __call__(self) -> MilvusClient:
        if self._client is None:
            import os
            abs_path = os.path.abspath(milvus_settings.MILVUS_LITE_PATH)
            print(f" Milvus Lite 路径：{abs_path}")

            # 确保父目录存在
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

            # 直接传路径字符串
            print(f"🔧 初始化 MilvusClient: {abs_path}")
            self._client = MilvusClient(abs_path)
            print("✅ MilvusClient 初始化成功")

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
    print(f"🔍 步骤 1: 准备初始化集合 {collection_name}")
    print(f"   Milvus Lite 路径：{milvus_settings.MILVUS_LITE_PATH}")

    try:
        print("🔍 步骤 2: 创建 MilvusClient...")
        client = MilvusSessionLocal()
        print("✅ MilvusClient 创建成功")
    except Exception as e:
        print(f"❌ MilvusClient 创建失败：{e}")
        raise

    try:
        print("🔍 步骤 3: 列出当前集合...")
        collections = client.list_collections()
        print(f"✅ 当前集合列表：{collections}")

        if collection_name in collections:
            print(f"🔍 步骤 4: 删除已存在的集合...")
            client.drop_collection(collection_name)
            print("✅ 集合已删除")
    except Exception as e:
        print(f"⚠️ 检查/删除集合时出错：{e}")

    try:
        print("🔍 步骤 5: 创建新集合...")
        print(f"   参数：dimension={DIMENSION}, metric_type=COSINE")

        client.create_collection(
            collection_name=collection_name,
            dimension=DIMENSION,
            metric_type="COSINE"
        )
        print("✅ 集合创建成功！")
    except Exception as e:
        print(f"❌ 创建集合失败：{e}")
        import traceback
        traceback.print_exc()
        raise

    print("✅ Milvus 集合初始化完成。")
    return client
