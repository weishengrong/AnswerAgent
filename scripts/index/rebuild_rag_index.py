import asyncio
import json
import sys
import argparse
import logging
from typing import List, Dict

sys.path.insert(0, ".")

from utils.rag.chunk_builder import build_chunks
from utils.rag.parent_store import parent_store
from app.rag.chroma import get_or_init_collection, get_rag_chroma_client
from app.rag.embedding import get_embeddings
from app.core.config.settings import rag_settings

COLLECTION_NAME = rag_settings.RAG_COLLECTION_NAME
logger = logging.getLogger(__name__)


async def translate_field_names_to_cn(fields_to_translate: List[Dict]) -> Dict[str, str]:
    """用 LLM 批量将英文字段名翻译为中文描述

    Args:
        fields_to_translate: [{"parent_id": "table_name", "field_name": "afternoon_out"}, ...]

    Returns:
        {f"{parent_id}.{field_name}": "中文描述", ...}
    """
    from app.core.llm import get_llm

    if not fields_to_translate:
        return {}

    # 按表分组，让 LLM 有上下文
    table_groups: Dict[str, List[str]] = {}
    for item in fields_to_translate:
        key = item["parent_id"]
        if key not in table_groups:
            table_groups[key] = []
        table_groups[key].append(item["field_name"])

    # 构建提示词
    group_lines = []
    for table_name, field_names in table_groups.items():
        field_list = ", ".join(field_names)
        group_lines.append(f"- 表 {table_name} 的字段：{field_list}")

    prompt = f"""你是一个数据库专家。请将以下英文数据库字段名翻译为简体中文的业务含义描述。
要求：
1. 翻译要准确反映字段的业务含义（不是逐词翻译，而是结合常见数据库命名惯例）
2. 返回纯 JSON 格式，key 为"表名.字段名"，value 为中文描述
3. 不要有多余的说明，只返回 JSON

待翻译的字段：
{chr(10).join(group_lines)}

示例输出格式：
{{"work_hour_record.afternoon_out": "下午下班时间", "area.city_id": "城市ID"}}"""

    llm = get_llm(timeout_key='chat')
    response = await llm.ainvoke(prompt)

    result_text = response.content.strip()
    # 提取 JSON（处理可能的 markdown 包裹）
    if "```json" in result_text:
        json_str = result_text.split("```json")[1].split("```")[0].strip()
    elif "```" in result_text:
        json_str = result_text.split("```")[1].split("```")[0].strip()
    else:
        json_str = result_text

    try:
        translated = json.loads(json_str)
        logger.info(f"✅ LLM 翻译完成：{len(translated)} 个字段")
        return translated
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ LLM 翻译结果解析失败，返回空映射：{e}")
        return {}


async def rebuild_from_docx(docx_path: str):
    from utils.docx.parser import parse_docx

    msg = f"开始重建索引（docx）: {docx_path}"
    print(msg)
    logger.info(msg)

    msg = "[1/8] 解析文档..."
    print(f"\n{msg}")
    logger.info(msg)
    tables = parse_docx(docx_path)
    msg = f"解析完成，共 {len(tables)} 张表"
    print(msg)
    logger.info(msg)

    for t in tables:
        detail = f"  - {t.table_name}: {len(t.fields)} 字段, {len(t.relations)} 关系"
        print(detail)
        logger.debug(detail)

    return tables


async def rebuild_from_sql(sql_path: str, enrich: bool = False):
    from utils.db.sql_parser import SQLParser, parse_sql_with_enrichment

    msg = f"开始重建索引（sql）: {sql_path}"
    print(msg)
    logger.info(msg)

    msg = "[1/8] 解析 SQL DDL..."
    print(f"\n{msg}")
    logger.info(msg)
    with open(sql_path, "r", encoding="utf-8") as f:
        sql_content = f.read()

    if enrich:
        tables = await parse_sql_with_enrichment(sql_content)
    else:
        from utils.db.sql_parser import parse_sql
        tables = parse_sql(sql_content)

    msg = f"解析完成，共 {len(tables)} 张表"
    print(msg)
    logger.info(msg)

    for t in tables:
        detail = f"  - {t.table_name}: {len(t.fields)} 字段, {len(t.relations)} 关系"
        print(detail)
        logger.debug(detail)

    return tables


async def rebuild_from_db(enrich: bool = False):
    from utils.db.connector import load_db_schema

    msg = "开始重建索引（db：直连数据库）..."
    print(msg)
    logger.info(msg)

    msg = "[1/8] 从数据库加载 Schema..."
    print(f"\n{msg}")
    logger.info(msg)
    tables = await load_db_schema(enrich=enrich)
    msg = f"加载完成，共 {len(tables)} 张表"
    print(msg)
    logger.info(msg)

    for t in tables:
        detail = f"  - {t.table_name}: {len(t.fields)} 字段, {len(t.relations)} 关系"
        print(detail)
        logger.debug(detail)

    return tables


async def rebuild_index(tables):
    # [2/8] 父子块构建
    msg = "[2/8] 构建父子块..."
    print(f"\n{msg}")
    logger.info(msg)
    parent_chunks, child_chunks, relation_map = build_chunks(tables)
    msg = f"✅ 父子块生成完成：{len(parent_chunks)} 个父块, {len(child_chunks)} 个子块"
    print(msg)
    logger.info(msg)

    # [2.5/8] 收集无描述字段并 LLM 翻译（用于 BM25 中文分词）
    fields_to_translate = []
    for c in child_chunks:
        if not c.field_desc or not c.field_desc.strip():
            fields_to_translate.append({
                "parent_id": c.parent_id,
                "field_name": c.field_name,
            })

    cn_desc_map: Dict[str, str] = {}
    if fields_to_translate:
        msg = f"[2.5/8] LLM 翻译 {len(fields_to_translate)} 个无描述字段..."
        print(msg)
        logger.info(msg)
        cn_desc_map = await translate_field_names_to_cn(fields_to_translate)
        msg = f"✅ 翻译完成，获得 {len(cn_desc_map)} 个中文描述"
        print(msg)
        logger.info(msg)
    else:
        msg = "[2.5/8] 所有字段均有描述，跳过 LLM 翻译"
        print(msg)
        logger.info(msg)

    # [3/8] 初始化 ChromaDB
    msg = "[3/8] 初始化 ChromaDB 集合..."
    print(msg)
    logger.info(msg)
    collection = get_or_init_collection(COLLECTION_NAME)

    # [4/8] 父块存 Redis
    msg = "[4/8] 存储父块到 Redis..."
    print(msg)
    logger.info(msg)
    parent_store.set_relation_map(relation_map)
    await parent_store.store_parents(parent_chunks)
    await parent_store.store_relation_map()
    msg = f"✅ Redis 父块存储完成：{len(parent_chunks)} 个"
    print(msg)
    logger.info(msg)

    # [5/8] Parent 块向量化 + 入 ChromaDB
    msg = "[5/8] 向量化父块并写入 ChromaDB..."
    print(msg)
    logger.info(msg)
    parent_contents = [c.content for c in parent_chunks]
    parent_vectors = get_embeddings(parent_contents)

    if parent_vectors:
        parent_ids = [f"parent_{c.table_name}" for c in parent_chunks]
        parent_metadatas = [
            {
                "chunk_type": "parent",
                "table_name": c.table_name,
            }
            for c in parent_chunks
        ]
        collection.add(
            ids=parent_ids,
            documents=parent_contents,
            embeddings=parent_vectors,
            metadatas=parent_metadatas
        )
        msg = f"✅ ChromaDB 父块入库完成：{len(parent_chunks)} 个"
        print(msg)
        logger.info(msg)
    else:
        msg = "⚠️ 父块向量化失败，跳过 ChromaDB 入库"
        print(msg)
        logger.warning(msg)

    # [6/8] Child 块向量化
    msg = "[6/8] 向量化子块..."
    print(msg)
    logger.info(msg)
    child_contents = [c.content for c in child_chunks]
    vectors = get_embeddings(child_contents)

    if vectors is None:
        msg = "❌ 子块向量化失败"
        print(msg)
        logger.error(msg)
        return

    msg = f"✅ 子块向量化完成：{len(vectors)} 个向量"
    print(msg)
    logger.info(msg)

    # [7/8] Child 块入 ChromaDB
    msg = "[7/8] 写入子块到 ChromaDB..."
    print(msg)
    logger.info(msg)

    ids = [f"child_{c.parent_id}_{c.field_name}_{i}" for i, c in enumerate(child_chunks)]
    documents = [c.content for c in child_chunks]
    metadatas = [
        {
            "chunk_type": c.chunk_type,
            "parent_id": c.parent_id,
            "field_name": c.field_name,
            "field_type": c.field_type,
            "field_desc": c.field_desc,
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
        msg = f"✅ ChromaDB 子块入库完成：{len(child_chunks)} 条"
        print(msg)
        logger.info(msg)
    except Exception as e:
        msg = f"❌ 子块插入 ChromaDB 失败：{e}"
        print(msg)
        logger.error(msg)
        return

    # [8/8] BM25 索引
    msg = "[8/8] 构建 BM25 索引并持久化到 Redis..."
    print(msg)
    logger.info(msg)
    from app.rag.bm25_index import bm25_index

    corpus = []
    for c in child_chunks:
        cn_desc = ""
        if c.field_desc and c.field_desc.strip():
            # 有字段描述，直接用作中文 token 来源
            cn_desc = c.field_desc.strip()
        else:
            # 无描述，使用 LLM 翻译结果
            key = f"{c.parent_id}.{c.field_name}"
            cn_desc = cn_desc_map.get(key, "")

        corpus.append({
            "parent_id": c.parent_id,
            "field_name": c.field_name,
            # 使用完整中文内容（含表名+字段名+描述），而非仅英文标识符
            # 否则中文 query 经 jieba 分词后与英文 corpus 零匹配 → BM25 永远返回 0
            "content": c.content,
            "cn_desc": cn_desc,
        })
    bm25_index.build(corpus)
    await bm25_index.save_to_redis()
    msg = f"✅ BM25 索引完成：{len(corpus)} 条文档"
    print(msg)
    logger.info(msg)

    # ChromaDB 总文档数统计
    try:
        chroma_client = get_rag_chroma_client()
        chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
        total_docs = chroma_collection.count()
        msg = f"📊 ChromaDB 集合总文档数：{total_docs}（Parent {len(parent_chunks)} + Child {len(child_chunks)}）"
        print(msg)
        logger.info(msg)
    except Exception as e:
        logger.warning(f"⚠️ ChromaDB 文档数统计失败（非致命）：{e}")

    # 保存 schema 指纹
    try:
        from utils.db.connector import DBConnector
        connector = DBConnector()
        await connector.save_fingerprint()
    except Exception as e:
        msg = f"⚠️ Schema 指纹保存失败（非致命）：{e}"
        print(msg)
        logger.warning(msg)

    # 最终汇总
    separator = "=" * 50
    print(f"\n{separator}")
    print("🎉 RAG 索引重建全部完成!")
    print(separator)
    summary = (
        f"\n📋 索引汇总:\n"
        f"  📦 父块 (Parent):     {len(parent_chunks)} 个 → Redis + ChromaDB\n"
        f"  🔹 子块 (Child):      {len(child_chunks)} 条 → ChromaDB\n"
        f"  🔍 BM25 索引:         {len(corpus)} 条 → Redis\n"
        f"  🔗 表关系:             {sum(len(r) for r in relation_map.values())} 条 → Redis"
    )
    print(summary)
    logger.info(summary.replace("\n", " | "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重建 RAG 索引")
    parser.add_argument("--source", choices=["docx", "sql", "db"], default="docx", help="数据源类型")
    parser.add_argument("--file", default=None, help="数据源文件路径（docx/sql模式需要）")
    parser.add_argument("--enrich", action="store_true", help="是否用LLM补全描述（sql/db模式生效）")
    args = parser.parse_args()

    if args.source == "docx":
        file_path = args.file or "resource/htower.docx"
        tables = asyncio.run(rebuild_from_docx(file_path))
    elif args.source == "sql":
        file_path = args.file or "resource/schema.sql"
        tables = asyncio.run(rebuild_from_sql(file_path, enrich=args.enrich))
    elif args.source == "db":
        tables = asyncio.run(rebuild_from_db(enrich=args.enrich))

    asyncio.run(rebuild_index(tables))
