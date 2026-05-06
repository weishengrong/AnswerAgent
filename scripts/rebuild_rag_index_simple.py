import asyncio
import sys
sys.path.insert(0, ".")

from utils.docx_parser import parse_docx
from utils.chunk_builder import build_chunks
from app.rag.milvus import init_milvus, MilvusSessionLocal
from app.rag.embedding import get_embeddings
from app.rag.milvus import DIMENSION
from config.settings import rag_settings

COLLECTION_NAME = rag_settings.RAG_COLLECTION_NAME


def prepare_child_data(chunks, vectors):
    data = [[], [], [], [], []]

    for i, chunk in enumerate(chunks):
        data[0].append(vectors[i])
        data[1].append(chunk.chunk_type)
        data[2].append(chunk.parent_id)
        data[3].append(chunk.field_name)
        data[4].append(chunk.content)

    return data


async def rebuild_index(docx_path: str):
    print(f"开始重建索引: {docx_path}")

    print("\n[1/4] 解析文档...")
    tables = parse_docx(docx_path)
    print(f"解析完成，共 {len(tables)} 张表")

    total_fields = sum(len(t.fields) for t in tables)
    total_relations = sum(len(t.relations) for t in tables)
    print(f"  总字段数: {total_fields}, 总关系数: {total_relations}")

    print("\n[2/4] 生成父子块...")
    parent_chunks, child_chunks, relation_map = build_chunks(tables)
    print(f"生成完成: {len(parent_chunks)} 父块, {len(child_chunks)} 子块")

    print("\n[3/4] 初始化 Milvus...")
    collection = init_milvus(COLLECTION_NAME)
    client = MilvusSessionLocal()

    print("\n[4/4] 向量化和写入 Milvus...")
    if not child_chunks:
        print("❌ 没有子块，跳过 Milvus 写入")
        return

    child_contents = [c.content for c in child_chunks]
    print(f"  正在向量化 {len(child_contents)} 个子块...")

    vectors = get_embeddings(child_contents)

    if vectors is None:
        print("❌ 向量化失败")
        return

    print(f"  向量化完成: {len(vectors)} 个向量")

    data = prepare_child_data(child_chunks, vectors)

    try:
        mr = collection.insert(data)
        collection.flush()
        print(f"✅ 成功插入 {mr.insert_count} 条数据到 Milvus")
    except Exception as e:
        print(f"❌ 插入失败: {e}")
        return

    print("\n" + "=" * 50)
    print("🎉 Milvus 索引重建完成!")
    print("=" * 50)
    print(f"\n统计:")
    print(f"  表数量: {len(parent_chunks)}")
    print(f"  子块数量: {len(child_chunks)}")


if __name__ == "__main__":
    docx_path = "resource/htower.docx"
    asyncio.run(rebuild_index(docx_path))
