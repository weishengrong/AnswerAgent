import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.livesqlbench.livesqlbench_parser import load_database_schema, get_all_database_names, load_livesqlbench_data
from utils.rag.chunk_builder import build_chunks
from utils.rag.parent_store import parent_store
from app.rag.chroma import init_chroma
from app.rag.embedding import get_embeddings
from app.core.config.settings import rag_settings

COLLECTION_NAME = rag_settings.RAG_COLLECTION_NAME


async def rebuild_livesqlbench_index(db_path: str, jsonl_path: str):
    print(f"开始重建 LiveSQLBench RAG 索引")
    print(f"  数据库路径: {db_path}")
    print(f"  数据文件: {jsonl_path}")

    data_list = load_livesqlbench_data(jsonl_path)
    db_names = get_all_database_names(data_list)
    print(f"  共 {len(db_names)} 个数据库: {', '.join(db_names)}")
    print(f"  共 {len(data_list)} 个任务")

    all_parent_chunks = []
    all_child_chunks = []
    all_relation_maps = {}

    for db_name in db_names:
        print(f"\n[{db_names.index(db_name) + 1}/{len(db_names)}] 处理数据库: {db_name}")
        tables, column_meanings, knowledge = load_database_schema(db_path, db_name)

        if not tables:
            print(f"  ⚠️  {db_name} 没有解析到表结构，跳过")
            continue

        print(f"  解析到 {len(tables)} 张表")

        parent_chunks, child_chunks, relation_map = build_chunks(tables, knowledge)

        for chunk in child_chunks:
            chunk.parent_id = f"{db_name}.{chunk.parent_id}"
            chunk.db_name = db_name

        for chunk in parent_chunks:
            chunk.table_name = f"{db_name}.{chunk.table_name}"
            chunk.content = f"数据库：{db_name}\n" + chunk.content

        all_parent_chunks.extend(parent_chunks)
        all_child_chunks.extend(child_chunks)
        all_relation_maps[db_name] = relation_map

        print(f"  生成 {len(parent_chunks)} 父块, {len(child_chunks)} 子块")

    if not all_child_chunks:
        print("❌ 没有生成任何子块，退出")
        return

    print(f"\n总计: {len(all_parent_chunks)} 父块, {len(all_child_chunks)} 子块")

    print("\n[1/5] 初始化 Chroma...")
    collection = init_chroma(COLLECTION_NAME)

    print("\n[2/5] 存储父块到 Redis...")
    combined_relation_map = {}
    for db_name, rel_map in all_relation_maps.items():
        for table_name, relations in rel_map.items():
            combined_relation_map[f"{db_name}.{table_name}"] = relations
    parent_store.set_relation_map(combined_relation_map)
    await parent_store.store_parents(all_parent_chunks)
    await parent_store.store_relation_map()
    print(f"  已存储 {len(all_parent_chunks)} 个父块")

    print("\n[3/5] 向量化子块...")
    child_contents = [c.content for c in all_child_chunks]
    vectors = get_embeddings(child_contents)
    if vectors is None:
        print("❌ 向量化失败")
        return
    print(f"  向量化完成: {len(vectors)} 个向量")

    print("\n[4/5] 写入 Chroma...")
    ids = []
    documents = []
    metadatas = []

    for i, c in enumerate(all_child_chunks):
        ids.append(f"livesql_{i}")
        documents.append(c.content)
        metadatas.append({
            "chunk_type": c.chunk_type,
            "parent_id": c.parent_id,
            "field_name": c.field_name,
            "field_type": c.field_type,
            "field_desc": c.field_desc,
            "db_name": c.db_name,
        })

    try:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas
        )
        print(f"  ✅ 成功插入 {len(all_child_chunks)} 条数据到 Chroma")
    except Exception as e:
        print(f"  ❌ 插入失败: {e}")
        return

    print("\n[5/5] 构建 BM25 索引...")
    from app.rag.bm25_index import bm25_index

    corpus = [
        {
            "parent_id": c.parent_id,
            "field_name": c.field_name,
            "content": f"{c.db_name} {c.parent_id.split('.', 1)[-1]} {c.field_name} {c.field_desc}",
        }
        for c in all_child_chunks
    ]
    bm25_index.build(corpus)
    await bm25_index.save_to_redis()
    print(f"  ✅ BM25 索引已构建并保存到 Redis（{len(corpus)} 条文档）")

    print("\n" + "=" * 50)
    print("🎉 LiveSQLBench RAG 索引重建完成!")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="重建 LiveSQLBench RAG 索引")
    parser.add_argument("--db_path", default=None, help="SQLite 数据库文件夹路径")
    parser.add_argument("--jsonl", default=None, help="LiveSQLBench JSONL 数据文件路径")
    args = parser.parse_args()

    db_path = args.db_path or input("请输入 SQLite 数据库文件夹路径: ")
    jsonl_path = args.jsonl or input("请输入 JSONL 数据文件路径: ")

    asyncio.run(rebuild_livesqlbench_index(db_path, jsonl_path))