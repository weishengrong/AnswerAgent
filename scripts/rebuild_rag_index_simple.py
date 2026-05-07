import asyncio
import sys
sys.path.insert(0, ".")

from utils.docx_parser import parse_docx
from utils.chunk_builder import build_chunks
from app.rag.chroma import init_chroma, ChromaSessionLocal
from app.rag.embedding import get_embeddings
from config.settings import rag_settings

COLLECTION_NAME = rag_settings.RAG_COLLECTION_NAME


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

    print("\n[3/4] 初始化 Chroma...")
    collection = init_chroma(COLLECTION_NAME)

    print("\n[4/4] 向量化和写入 Chroma...")
    if not child_chunks:
        print("❌ 没有子块，跳过 Chroma 写入")
        return

    child_contents = [c.content for c in child_chunks]
    print(f"  正在向量化 {len(child_contents)} 个子块...")

    vectors = get_embeddings(child_contents)

    if vectors is None:
        print("❌ 向量化失败")
        return

    print(f"  向量化完成: {len(vectors)} 个向量")

    ids = [f"child_{c.parent_id}_{c.field_name}_{i}" for i, c in enumerate(child_chunks)]
    documents = [c.content for c in child_chunks]
    metadatas = [
        {
            "chunk_type": c.chunk_type,
            "parent_id": c.parent_id,
            "field_name": c.field_name,
        }
        for c in child_chunks
    ]

    try:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas
        )
        print(f"✅ 成功插入 {len(child_chunks)} 条数据到 Chroma")
    except Exception as e:
        print(f"❌ 插入失败: {e}")
        return

    print("\n" + "=" * 50)
    print("🎉 Chroma 索引重建完成!")
    print("=" * 50)
    print(f"\n统计:")
    print(f"  表数量: {len(parent_chunks)}")
    print(f"  子块数量: {len(child_chunks)}")


if __name__ == "__main__":
    docx_path = "resource/htower.docx"
    asyncio.run(rebuild_index(docx_path))
