import asyncio
import json
import logging
import os
import sys
import re
from collections import defaultdict
from typing import Dict, Set, Tuple

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import sqlglot
from sqlglot import exp

from app.rag.chroma import get_chroma_client
from app.rag.bm25_index import bm25_index
from app.rag.retrieval import rrf_merge, aggregate_hits_by_table, complete_foreign_keys, _get_embedding_vector_async, multi_path_retrieve_async
from app.rag.query_rewriter import query_rewriter
from utils.rag.context_assembler import assemble_context
from utils.rag.parent_store import parent_store
from app.core.config.settings import rag_settings
from scripts.livesqlbench.livesqlbench_parser import load_database_schema


def build_schema_columns(db_path: str, db_name: str) -> Dict[str, Set[str]]:
    tables, _, _ = load_database_schema(db_path, db_name)
    return {
        table.table_name.lower(): {field.name.lower() for field in table.fields}
        for table in tables
    }


def extract_sql_references(sql: str, schema_columns: Dict[str, Set[str]]) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Extract real schema table/column references from SQL.

    The result is bound to the current database schema. CTE names, CTE output
    columns, computed aliases, and columns created by ALTER TABLE are excluded
    because they do not have child chunks in the RAG index.
    """
    needed_tables: Set[str] = set()
    table_columns: Dict[str, Set[str]] = defaultdict(set)

    try:
        expressions = sqlglot.parse(sql, dialect="sqlite")
    except Exception as e:
        logger.warning("SQL 解析失败: %s, sql=%s", e, sql)
        return needed_tables, table_columns

    for tree in expressions:
        if tree is None:
            continue

        cte_names = {cte.alias.lower() for cte in tree.find_all(exp.CTE) if cte.alias}
        alias_to_table: Dict[str, str] = {}

        for table in tree.find_all(exp.Table):
            table_name = table.name.lower()
            if table_name in cte_names or table_name not in schema_columns:
                continue
            needed_tables.add(table_name)
            alias_to_table[table.alias_or_name.lower()] = table_name
            alias_to_table[table_name] = table_name

        for column in tree.find_all(exp.Column):
            col_name = column.name.lower()
            qualifier = column.table.lower() if column.table else ""

            if qualifier:
                table_name = alias_to_table.get(qualifier, qualifier)
                if table_name in schema_columns and col_name in schema_columns[table_name]:
                    needed_tables.add(table_name)
                    table_columns[table_name].add(col_name)
                continue

            matching_tables = [
                table_name
                for table_name in needed_tables
                if col_name in schema_columns.get(table_name, set())
            ]
            for table_name in matching_tables:
                table_columns[table_name].add(col_name)

    return needed_tables, table_columns


def extract_tables_from_schema(schema_text: str) -> Set[str]:
    tables = set()
    for m in re.finditer(r'【表：[^】]+\.(\w+)】', schema_text):
        tables.add(m.group(1).lower())
    return tables


def extract_columns_from_schema(schema_text: str) -> Set[str]:
    columns = set()
    for m in re.finditer(r'-\s+(\w+)：', schema_text):
        columns.add(m.group(1).lower())
    return columns


def build_relevant_chunk_keys(table_columns: Dict[str, Set[str]], db_name: str) -> Set[str]:
    """Build expected child chunk keys: {db}.{table}.{column}"""
    keys = set()
    for table, columns in table_columns.items():
        for col in columns:
            keys.add(f"{db_name}.{table}.{col}")
    return keys


async def evaluate_rag_for_task(
    task: Dict,
    db_name: str,
    schema_columns: Dict[str, Set[str]],
    rerank_mode: str = None,
) -> Dict:
    question = task.get("query", "")
    sol_sql_list = task.get("sol_sql", [])
    if isinstance(sol_sql_list, str):
        sol_sql_list = [sol_sql_list]

    instance_id = task.get("instance_id", "unknown")

    needed_tables = set()
    table_columns: Dict[str, Set[str]] = defaultdict(set)
    for sql in sol_sql_list:
        if sql:
            sql_tables, sql_table_columns = extract_sql_references(sql, schema_columns)
            needed_tables.update(sql_tables)
            for table, columns in sql_table_columns.items():
                table_columns[table].update(columns)

    all_sql_columns = {col for columns in table_columns.values() for col in columns}
    relevant_chunk_keys = build_relevant_chunk_keys(table_columns, db_name)

    collection_name = rag_settings.RAG_COLLECTION_NAME
    client = get_chroma_client()
    collection = client.get_collection(collection_name)

    # 1. Query 分解
    sub_queries = query_rewriter.decompose_query(question)

    # 2. 混合检索：向量用分解子查询多路检索，BM25 用原始问题
    sub_vector_k = max(rag_settings.RAG_TOP_K_VECTOR // len(sub_queries), 10)

    seen_vector = set()
    vector_chunks = []
    for sq in sub_queries:
        v = await _get_embedding_vector_async(sq)
        collection = client.get_collection(collection_name)
        results = collection.query(
            query_embeddings=[v],
            n_results=sub_vector_k,
            where={"$and": [{"chunk_type": {"$eq": "child"}}, {"db_name": {"$eq": db_name}}]},
            include=["metadatas", "documents", "distances"]
        )
        if results and results.get("ids") and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                item = {"id": results["ids"][0][i]}
                if results.get("distances"):
                    item["distance"] = results["distances"][0][i]
                if results.get("metadatas") and results["metadatas"][0]:
                    item.update(results["metadatas"][0][i])
                if results.get("documents") and results["documents"][0]:
                    item["content"] = results["documents"][0][i]
                key = f"{item.get('parent_id','')}.{item.get('field_name','')}"
                if key not in seen_vector:
                    seen_vector.add(key)
                    vector_chunks.append(item)

    bm25_chunks = bm25_index.search(question, top_k=rag_settings.RAG_TOP_K_BM25, db_name=db_name)

    # --- 子块级评测 ---
    vector_chunk_keys = set()
    for c in vector_chunks:
        key = f"{c.get('parent_id','')}.{c.get('field_name','')}"
        vector_chunk_keys.add(key)

    bm25_chunk_keys = set()
    for c in bm25_chunks:
        key = f"{c.get('parent_id','')}.{c.get('field_name','')}"
        bm25_chunk_keys.add(key)

    # 粗排：RRF 融合
    merged_chunks = rrf_merge(vector_chunks, bm25_chunks) if vector_chunks and bm25_chunks else (vector_chunks or bm25_chunks)

    # 表级优先重排
    from app.rag.retrieval import table_level_rerank
    relation_map = await parent_store.get_relation_map()
    table_ranked_chunks = table_level_rerank(
        question,
        merged_chunks,
        relation_map,
        mode=rerank_mode,
    )

    # 评测指标计算（分别计算粗排和表级重排的结果）
    merged_chunk_keys = set()
    for c in merged_chunks:
        key = f"{c.get('parent_id','')}.{c.get('field_name','')}"
        merged_chunk_keys.add(key)

    table_ranked_chunk_keys = set()
    for c in table_ranked_chunks:
        key = f"{c.get('parent_id','')}.{c.get('field_name','')}"
        table_ranked_chunk_keys.add(key)

    if relevant_chunk_keys:
        vector_chunk_recall = len(relevant_chunk_keys & vector_chunk_keys) / len(relevant_chunk_keys)
        bm25_chunk_recall = len(relevant_chunk_keys & bm25_chunk_keys) / len(relevant_chunk_keys)
        merged_chunk_recall = len(relevant_chunk_keys & merged_chunk_keys) / len(relevant_chunk_keys)
    else:
        vector_chunk_recall = bm25_chunk_recall = merged_chunk_recall = 0.0

    if vector_chunk_keys:
        vector_chunk_precision = len(relevant_chunk_keys & vector_chunk_keys) / len(vector_chunk_keys)
    else:
        vector_chunk_precision = 0.0

    if bm25_chunk_keys:
        bm25_chunk_precision = len(relevant_chunk_keys & bm25_chunk_keys) / len(bm25_chunk_keys)
    else:
        bm25_chunk_precision = 0.0

    if merged_chunk_keys:
        merged_chunk_precision = len(relevant_chunk_keys & merged_chunk_keys) / len(merged_chunk_keys)
    else:
        merged_chunk_precision = 0.0

    # 表级重排后的召回率和精确率
    if relevant_chunk_keys:
        table_ranked_chunk_recall = len(relevant_chunk_keys & table_ranked_chunk_keys) / len(relevant_chunk_keys)
    else:
        table_ranked_chunk_recall = 0.0

    if table_ranked_chunk_keys:
        table_ranked_chunk_precision = len(relevant_chunk_keys & table_ranked_chunk_keys) / len(table_ranked_chunk_keys)
    else:
        table_ranked_chunk_precision = 0.0

    # --- 表级和列级评测（使用表级重排结果） ---
    hit_fields = aggregate_hits_by_table(table_ranked_chunks)
    relation_map = await parent_store.get_relation_map()
    hit_fields = complete_foreign_keys(hit_fields, relation_map)
    hit_tables = list(hit_fields.keys())
    parent_contents = await parent_store.get_parents(hit_tables)
    context = assemble_context(hit_fields, parent_contents, relation_map)

    retrieved_tables = extract_tables_from_schema(context)
    retrieved_columns = extract_columns_from_schema(context)

    if needed_tables:
        table_recall = len(needed_tables & retrieved_tables) / len(needed_tables)
    else:
        table_recall = 0.0

    if retrieved_tables:
        table_precision = len(needed_tables & retrieved_tables) / len(retrieved_tables)
    else:
        table_precision = 0.0

    if all_sql_columns:
        column_recall = len(all_sql_columns & retrieved_columns) / len(all_sql_columns)
    else:
        column_recall = 0.0

    if retrieved_columns:
        column_precision = len(all_sql_columns & retrieved_columns) / len(retrieved_columns)
    else:
        column_precision = 0.0

    return {
        "instance_id": instance_id,
        "db_name": db_name,
        "question": question[:200],
        "sol_sql": sol_sql_list[0] if sol_sql_list else "",
        "needed_tables": sorted(needed_tables),
        "needed_columns": sorted(all_sql_columns),
        "needed_table_columns": {table: sorted(columns) for table, columns in sorted(table_columns.items())},
        "relevant_chunk_count": len(relevant_chunk_keys),
        "vector_chunk_recall": round(vector_chunk_recall, 3),
        "vector_chunk_precision": round(vector_chunk_precision, 3),
        "bm25_chunk_recall": round(bm25_chunk_recall, 3),
        "bm25_chunk_precision": round(bm25_chunk_precision, 3),
        "merged_chunk_recall": round(merged_chunk_recall, 3),
        "merged_chunk_precision": round(merged_chunk_precision, 3),
        "vector_chunk_count": len(vector_chunk_keys),
        "bm25_chunk_count": len(bm25_chunk_keys),
        "merged_chunk_count": len(merged_chunk_keys),
        "table_ranked_chunk_recall": round(table_ranked_chunk_recall, 3),
        "table_ranked_chunk_precision": round(table_ranked_chunk_precision, 3),
        "table_ranked_chunk_count": len(table_ranked_chunk_keys),
        "retrieved_tables": sorted(retrieved_tables),
        "retrieved_table_count": len(retrieved_tables),
        "retrieved_column_count": len(retrieved_columns),
        "table_recall": round(table_recall, 3),
        "table_precision": round(table_precision, 3),
        "column_recall": round(column_recall, 3),
        "column_precision": round(column_precision, 3),
        "context_length": len(context),
        "_vector_hit_keys": sorted(vector_chunk_keys & relevant_chunk_keys),
        "_bm25_hit_keys": sorted(bm25_chunk_keys & relevant_chunk_keys),
        "_merged_hit_keys": sorted(merged_chunk_keys & relevant_chunk_keys),
        "_table_ranked_hit_keys": sorted(table_ranked_chunk_keys & relevant_chunk_keys),
    }


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--rerank_mode",
        choices=["fixed", "dynamic", "dynamic_fields"],
        default=None,
        help="表级重排策略；默认使用配置中的 RAG_TABLE_RERANK_MODE",
    )
    args = parser.parse_args()

    print("=" * 70)
    active_rerank_mode = args.rerank_mode or rag_settings.RAG_TABLE_RERANK_MODE
    print("LiveSQLBench RAG 评测（子块级 + 表级 + 列级）")
    print(f"重排模式: {active_rerank_mode}")
    print("=" * 70)

    await parent_store.init()
    await bm25_index.load_from_redis()

    with open(args.jsonl, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        tasks = tasks[:args.limit]

    print(f"评测任务数: {len(tasks)}")
    print()

    schema_cache: Dict[str, Dict[str, Set[str]]] = {}
    for task in tasks:
        db_name = task.get("selected_database", "")
        if db_name and db_name not in schema_cache:
            schema_cache[db_name] = build_schema_columns(args.db_path, db_name)

    results = []
    for i, task in enumerate(tasks):
        db_name = task.get("selected_database", "")
        instance_id = task.get("instance_id", f"task_{i}")
        print(f"  [{i+1}/{len(tasks)}] {instance_id} ({db_name})...", end=" ", flush=True)

        result = await evaluate_rag_for_task(
            task,
            db_name,
            schema_cache.get(db_name, {}),
            rerank_mode=active_rerank_mode,
        )
        results.append(result)

        vr = result["vector_chunk_recall"]
        vp = result["vector_chunk_precision"]
        mr = result["merged_chunk_recall"]
        mp = result["merged_chunk_precision"]
        rr = result["table_ranked_chunk_recall"]
        rp = result["table_ranked_chunk_precision"]
        tr = result["table_recall"]
        cr = result["column_recall"]
        print(f"子块V-R={vr:.1%} V-P={vp:.1%} | M-R={mr:.1%} M-P={mp:.1%} | R-R={rr:.1%} R-P={rp:.1%} | 表R={tr:.1%} 列R={cr:.1%}")

        # 详细输出
        print(f"\n  [问题] {result['question']}")
        print(f"  [答案SQL] {result.get('sol_sql', 'N/A')[:300]}")
        print(f"  [需要字段] {result['needed_table_columns']}")
        print(f"  [向量召回字段] {result.get('_vector_hit_keys', [])}")
        print(f"  [BM25召回字段] {result.get('_bm25_hit_keys', [])}")
        print(f"  [融合召回字段] {result.get('_merged_hit_keys', [])}")
        print(f"  [表级重排召回字段] {result.get('_table_ranked_hit_keys', [])}")
        print()

    print()
    print("=" * 70)
    print("汇总统计")
    print("=" * 70)

    n = len(results)

    avg_vr = sum(r["vector_chunk_recall"] for r in results) / n
    avg_vp = sum(r["vector_chunk_precision"] for r in results) / n
    avg_br = sum(r["bm25_chunk_recall"] for r in results) / n
    avg_bp = sum(r["bm25_chunk_precision"] for r in results) / n
    avg_mr = sum(r["merged_chunk_recall"] for r in results) / n
    avg_mp = sum(r["merged_chunk_precision"] for r in results) / n
    avg_rr = sum(r["table_ranked_chunk_recall"] for r in results) / n
    avg_rp = sum(r["table_ranked_chunk_precision"] for r in results) / n
    avg_tr = sum(r["table_recall"] for r in results) / n
    avg_tp = sum(r["table_precision"] for r in results) / n
    avg_cr = sum(r["column_recall"] for r in results) / n
    avg_cp = sum(r["column_precision"] for r in results) / n
    avg_tables = sum(r["retrieved_table_count"] for r in results) / n
    avg_columns = sum(r["retrieved_column_count"] for r in results) / n
    avg_context_length = sum(r["context_length"] for r in results) / n

    print(f"  ┌──────────────────────┬──────────┬──────────┐")
    print(f"  │ 层级                  │ 召回率    │ 精确率    │")
    print(f"  ├──────────────────────┼──────────┼──────────┤")
    print(f"  │ 子块级 (向量检索)     │ {avg_vr:.1%}    │ {avg_vp:.1%}    │")
    print(f"  │ 子块级 (BM25)         │ {avg_br:.1%}    │ {avg_bp:.1%}    │")
    print(f"  │ 子块级 (RRF 融合后)   │ {avg_mr:.1%}    │ {avg_mp:.1%}    │")
    print(f"  │ 子块级 (表级重排后)   │ {avg_rr:.1%}    │ {avg_rp:.1%}    │")
    print(f"  │ 表级 (最终上下文)     │ {avg_tr:.1%}    │ {avg_tp:.1%}    │")
    print(f"  │ 列级 (最终上下文)     │ {avg_cr:.1%}    │ {avg_cp:.1%}    │")
    print(f"  └──────────────────────┴──────────┴──────────┘")
    print()

    full_vr = sum(1 for r in results if r["vector_chunk_recall"] >= 1.0)
    full_mr = sum(1 for r in results if r["merged_chunk_recall"] >= 1.0)
    full_rr = sum(1 for r in results if r["table_ranked_chunk_recall"] >= 1.0)
    full_tr = sum(1 for r in results if r["table_recall"] >= 1.0)
    print(f"  子块完全召回 (向量 Recall=100%): {full_vr}/{n} ({full_vr/n:.1%})")
    print(f"  子块完全召回 (融合 Recall=100%): {full_mr}/{n} ({full_mr/n:.1%})")
    print(f"  子块完全召回 (表级重排 Recall=100%): {full_rr}/{n} ({full_rr/n:.1%})")
    print(f"  表级完全召回 (Recall=100%):      {full_tr}/{n} ({full_tr/n:.1%})")
    print(f"  平均上下文表数:                  {avg_tables:.1f}")
    print(f"  平均上下文字段数:                {avg_columns:.1f}")
    print(f"  平均上下文长度:                  {avg_context_length:.0f} chars")
    print()

    by_db = defaultdict(list)
    for r in results:
        by_db[r["db_name"]].append(r)

    print(f"按数据库统计:")
    print(f"  {'数据库':20s} {'子块V-R':>8s} {'子块V-P':>8s} {'子块M-R':>8s} {'子块M-P':>8s} {'子块R-R':>8s} {'子块R-P':>8s} {'表R':>6s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for db_name in sorted(by_db.keys()):
        db_r = by_db[db_name]
        dvr = sum(r["vector_chunk_recall"] for r in db_r) / len(db_r)
        dvp = sum(r["vector_chunk_precision"] for r in db_r) / len(db_r)
        dmr = sum(r["merged_chunk_recall"] for r in db_r) / len(db_r)
        dmp = sum(r["merged_chunk_precision"] for r in db_r) / len(db_r)
        drr = sum(r["table_ranked_chunk_recall"] for r in db_r) / len(db_r)
        drp = sum(r["table_ranked_chunk_precision"] for r in db_r) / len(db_r)
        dtr = sum(r["table_recall"] for r in db_r) / len(db_r)
        print(f"  {db_name:20s} {dvr:.1%}    {dvp:.1%}    {dmr:.1%}    {dmp:.1%}    {drr:.1%}    {drp:.1%}    {dtr:.1%}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total": n,
                    "rerank_mode": active_rerank_mode,
                    "avg_vector_chunk_recall": avg_vr,
                    "avg_vector_chunk_precision": avg_vp,
                    "avg_bm25_chunk_recall": avg_br,
                    "avg_bm25_chunk_precision": avg_bp,
                    "avg_merged_chunk_recall": avg_mr,
                    "avg_merged_chunk_precision": avg_mp,
                    "avg_table_ranked_chunk_recall": avg_rr,
                    "avg_table_ranked_chunk_precision": avg_rp,
                    "avg_table_recall": avg_tr,
                    "avg_table_precision": avg_tp,
                    "avg_column_recall": avg_cr,
                    "avg_column_precision": avg_cp,
                    "avg_retrieved_table_count": avg_tables,
                    "avg_retrieved_column_count": avg_columns,
                    "avg_context_length": avg_context_length,
                },
                "results": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
