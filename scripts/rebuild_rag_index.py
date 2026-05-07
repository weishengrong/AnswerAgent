import asyncio
import sys
import argparse

sys.path.insert(0, ".")

from utils.chunk_builder import build_chunks
from utils.parent_store import parent_store
from app.rag.chroma import init_chroma, ChromaSessionLocal
from app.rag.embedding import get_embeddings
from config.settings import rag_settings

COLLECTION_NAME = rag_settings.RAG_COLLECTION_NAME


async def rebuild_from_docx(docx_path: str):
    from utils.docx_parser import parse_docx

    print(f"开始重建索引（docx）: {docx_path}")

    print("\n[1/6] 解析文档...")
    tables = parse_docx(docx_path)
    print(f"解析完成，共 {len(tables)} 张表")

    for t in tables:
        print(f"  - {t.table_name}: {len(t.fields)} 字段, {len(t.relations)} 关系")

    return tables


async def rebuild_from_sql(sql_path: str, enrich: bool = False):
    from utils.sql_parser import SQLParser, parse_sql_with_enrichment

    print(f"开始重建索引（sql）: {sql_path}")

    print("\n[1/6] 解析 SQL DDL...")
    with open(sql_path, "r", encoding="utf-8") as f:
        sql_content = f.read()

    if enrich:
        tables = await parse_sql_with_enrichment(sql_content)
    else:
        from utils.sql_parser import parse_sql
        tables = parse_sql(sql_content)

    print(f"解析完成，共 {len(tables)} 张表")

    for t in tables:
        print(f"  - {t.table_name}: {len(t.fields)} 字段, {len(t.relations)} 关系")

    return tables


async def rebuild_index(tables):
    print("\n[2/7] 生成父子块...")
    parent_chunks, child_chunks, relation_map = build_chunks(tables)
    print(f"生成完成: {len(parent_chunks)} 父块, {len(child_chunks)} 子块")

    print("\n[3/7] 初始化 Chroma...")
    collection = init_chroma(COLLECTION_NAME)
    client = ChromaSessionLocal()

    print("\n[4/7] 存储父块到 Redis...")
    parent_store.set_relation_map(relation_map)
    await parent_store.store_parents(parent_chunks)
    await parent_store.store_relation_map()
    print(f"已存储 {len(parent_chunks)} 个父块")

    print("\n[5/7] 向量化子块...")
    child_contents = [c.content for c in child_chunks]
    vectors = get_embeddings(child_contents)

    if vectors is None:
        print("❌ 向量化失败")
        return

    print(f"向量化完成: {len(vectors)} 个向量")
    print("\n[6/7] 写入 Chroma...")

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
        print(f"❌ 插入失败：{e}")
        return
    print("\n[7/7] 构建 BM25 索引...")
    from app.rag.bm25_index import bm25_index

    corpus = [
        {
            "parent_id": c.parent_id,
            "field_name": c.field_name,
            "content": c.content,
        }
        for c in child_chunks
    ]
    bm25_index.build(corpus)
    await bm25_index.save_to_redis()
    print(f"✅ BM25 索引已构建并保存到 Redis（{len(corpus)} 条文档）")

    print("\n" + "=" * 50)
    print("🎉 索引重建完成!")
    print("=" * 50)

    print("\n父子块统计:")
    print(f"  父块数量: {len(parent_chunks)}")
    print(f"  子块数量: {len(child_chunks)}")
    print(f"  关系数量: {sum(len(r) for r in relation_map.values())}")
    print(f"  BM25 索引: {len(corpus)} 条文档")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重建 RAG 索引")
    parser.add_argument("--source", choices=["docx", "sql"], default="docx", help="数据源类型")
    parser.add_argument("--file", default=None, help="数据源文件路径")
    parser.add_argument("--enrich", action="store_true", help="SQL模式：是否用LLM补全描述")
    args = parser.parse_args()

    if args.source == "docx":
        file_path = args.file or "resource/htower.docx"
        tables = asyncio.run(rebuild_from_docx(file_path))
    else:
        file_path = args.file or "resource/schema.sql"
        tables = asyncio.run(rebuild_from_sql(file_path, enrich=args.enrich))

    asyncio.run(rebuild_index(tables))
