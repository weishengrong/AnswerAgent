import asyncio
import json
import os
import sys
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.livesqlbench.livesqlbench_parser import load_livesqlbench_data, group_tasks_by_database
from scripts.livesqlbench.livesqlbench_eval_utils import evaluate_single_task, split_field
from app.rag.chroma import get_chroma_client
from app.rag.bm25_index import bm25_index
from app.rag.retrieval import rrf_merge, aggregate_hits_by_table, complete_foreign_keys, _get_embedding_vector_async
from utils.rag.context_assembler import assemble_context
from utils.rag.parent_store import parent_store
from app.core.config.settings import rag_settings, llm_settings
from openai import OpenAI


def _get_llm_client():
    return OpenAI(
        api_key=llm_settings.LLM_API_KEY,
        base_url=llm_settings.LLM_BASE_URL
    )


async def retrieve_schema_for_db(question: str, db_name: str, limit: int = 30) -> str:
    collection_name = rag_settings.RAG_COLLECTION_NAME
    client = get_chroma_client()
    collection = client.get_collection(collection_name)

    vector = await _get_embedding_vector_async(question)

    results = collection.query(
        query_embeddings=[vector],
        n_results=limit,
        where={"$and": [{"chunk_type": {"$eq": "child"}}, {"db_name": {"$eq": db_name}}]},
        include=["metadatas", "documents", "distances"]
    )

    vector_chunks = []
    if results and results.get("ids") and len(results["ids"]) > 0:
        for i in range(len(results["ids"][0])):
            item = {"id": results["ids"][0][i]}
            if results.get("distances"):
                item["distance"] = results["distances"][0][i]
            if results.get("metadatas") and results["metadatas"][0]:
                item.update(results["metadatas"][0][i])
            if results.get("documents") and results["documents"][0]:
                item["content"] = results["documents"][0][i]
            vector_chunks.append(item)

    bm25_chunks = bm25_index.search(question, top_k=limit)

    if vector_chunks and bm25_chunks:
        merged_chunks = rrf_merge(vector_chunks, bm25_chunks)
    elif vector_chunks:
        merged_chunks = vector_chunks
    elif bm25_chunks:
        merged_chunks = bm25_chunks
    else:
        return ""

    hit_fields = aggregate_hits_by_table(merged_chunks)

    relation_map = await parent_store.get_relation_map()
    hit_fields = complete_foreign_keys(hit_fields, relation_map)

    hit_tables = list(hit_fields.keys())
    parent_contents = await parent_store.get_parents(hit_tables)

    if not parent_contents:
        return ""

    context = assemble_context(hit_fields, parent_contents, relation_map)
    return context


def generate_sql_with_llm(question: str, schema_context: str, model: str = None) -> Optional[str]:
    client = _get_llm_client()
    model_name = model or llm_settings.LLM_MODEL_NAME

    system_prompt = """你是一个精通 SQLite 的数据库专家。根据提供的表结构信息生成 SQL 查询语句。

规则：
1. 只使用提供的表结构中的表名和列名
2. 输出 SQLite 兼容的 SQL 语法
3. 只输出 SQL 代码，不要包含任何解释
4. 必须生成 SQL，不能拒绝回答"""

    user_prompt = f"【表结构】\n{schema_context}\n\n【问题】\n{question}\n\n请生成 SQLite SQL 查询语句："

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        sql = response.choices[0].message.content.strip()
        print(f"  📨 LLM 原始响应: {sql[:300]}")

        import re
        match = re.search(r"```sql\s*(.*?)\s*```", sql, re.DOTALL)
        if match:
            sql = match.group(1).strip()
        if sql.startswith("--"):
            print(f"  ⚠️  LLM 返回了注释: {sql[:200]}")
            return None
        if not sql:
            print(f"  ⚠️  LLM 返回了空内容")
            return None
        return sql
    except Exception as e:
        print(f"  ❌ LLM 调用失败: {e}")
        return None


async def evaluate_task(task: Dict, db_path: str, db_name: str, use_rag: bool = True) -> Dict:
    instance_id = task.get("instance_id", "unknown")
    question = task.get("query", "")
    sol_sql = task.get("sol_sql", "")
    test_cases = task.get("test_cases", None)
    preprocess_sql = split_field(task, "preprocess_sql")
    clean_up_sql = split_field(task, "clean_up_sql")
    conditions = task.get("conditions", {})
    category = task.get("category", "Query")

    print(f"\n  📝 [{instance_id}] {question[:80]}...")

    if use_rag:
        schema_context = await retrieve_schema_for_db(question, db_name)
        if not schema_context:
            print(f"  ⚠️  未检索到 schema，跳过")
            return {
                "instance_id": instance_id,
                "status": "skipped",
                "error": "no_schema",
                "category": category,
            }
    else:
        schema_path = os.path.join(db_path, db_name, f"{db_name}_schema.txt")
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_context = f.read()
        else:
            schema_context = ""

    pred_sql = generate_sql_with_llm(question, schema_context)
    if not pred_sql:
        print(f"  ⚠️  SQL 生成失败")
        return {
            "instance_id": instance_id,
            "status": "failed",
            "error": "sql_generation_failed",
            "category": category,
        }

    print(f"  🔍 生成 SQL: {pred_sql[:100]}...")

    db_file = os.path.join(db_path, db_name, f"{db_name}_template.sqlite")
    if not os.path.exists(db_file):
        db_file = os.path.join(db_path, db_name, f"{db_name}.sqlite")
    if not os.path.exists(db_file):
        print(f"  ⚠️  数据库文件不存在: {db_file}")
        return {
            "instance_id": instance_id,
            "status": "failed",
            "error": "db_not_found",
            "category": category,
            "pred_sql": pred_sql,
        }

    eval_result = evaluate_single_task(
        pred_sql=pred_sql,
        sol_sql=sol_sql,
        db_path=db_file,
        db_name=db_name,
        test_cases=test_cases,
        preprocess_sql=preprocess_sql,
        clean_up_sql=clean_up_sql,
        conditions=conditions,
    )

    print(f"  {'✅' if eval_result['success'] else '❌'} 测试通过: {eval_result['passed']}/{eval_result['total']}")

    return {
        "instance_id": instance_id,
        "status": "success" if eval_result["success"] else "failed",
        "category": category,
        "passed": eval_result["passed"],
        "total": eval_result["total"],
        "failed_tests": eval_result["failed"],
        "pred_sql": pred_sql,
        "sol_sql": sol_sql,
    }


async def main():
    parser = argparse.ArgumentParser(description="在 LiveSQLBench 上评测 AnswerAgent")
    parser.add_argument("--db_path", required=True, help="SQLite 数据库文件夹路径")
    parser.add_argument("--jsonl", required=True, help="LiveSQLBench JSONL 数据文件路径")
    parser.add_argument("--limit", type=int, default=None, help="限制评测任务数量")
    parser.add_argument("--skip", type=int, default=0, help="跳过前 N 个任务")
    parser.add_argument("--db", default=None, help="只评测指定数据库")
    parser.add_argument("--no-rag", action="store_true", help="不使用 RAG（直接用全量 schema）")
    parser.add_argument("--output", default=None, help="输出结果文件路径")
    args = parser.parse_args()

    print("=" * 60)
    print("LiveSQLBench 评测 - AnswerAgent")
    print("=" * 60)
    print(f"数据库路径: {args.db_path}")
    print(f"数据文件: {args.jsonl}")
    print(f"RAG 模式: {'关闭' if args.no_rag else '开启'}")

    if not args.no_rag:
        print("\n初始化 RAG 组件...")
        await parent_store.init()
        loaded = await bm25_index.load_from_redis()
        if loaded:
            print("  ✅ BM25 索引已从 Redis 加载")
        else:
            print("  ⚠️  BM25 索引未找到，仅使用向量检索")

    data_list = load_livesqlbench_data(args.jsonl)
    print(f"加载 {len(data_list)} 个任务")

    if args.db:
        data_list = [d for d in data_list if d.get("selected_database") == args.db]
        print(f"过滤后: {len(data_list)} 个任务 (数据库: {args.db})")

    if args.skip > 0:
        data_list = data_list[args.skip:]
    if args.limit:
        data_list = data_list[:args.limit]

    print(f"实际评测: {len(data_list)} 个任务\n")

    grouped = group_tasks_by_database(data_list)
    all_results = []
    total_start = time.time()

    for db_name, tasks in grouped.items():
        print(f"\n{'=' * 40}")
        print(f"数据库: {db_name} ({len(tasks)} 个任务)")
        print(f"{'=' * 40}")

        for task in tasks:
            result = await evaluate_task(task, args.db_path, db_name, use_rag=not args.no_rag)
            all_results.append(result)

    total_elapsed = time.time() - total_start

    print(f"\n\n{'=' * 60}")
    print("评测结果汇总")
    print("=" * 60)

    total = len(all_results)
    passed = sum(1 for r in all_results if r["status"] == "success")
    failed = sum(1 for r in all_results if r["status"] == "failed")
    skipped = sum(1 for r in all_results if r["status"] == "skipped")

    query_tasks = [r for r in all_results if r.get("category") == "Query"]
    management_tasks = [r for r in all_results if r.get("category") != "Query"]
    query_passed = sum(1 for r in query_tasks if r["status"] == "success")
    management_passed = sum(1 for r in management_tasks if r["status"] == "success")

    print(f"总任务数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"跳过: {skipped}")
    print(f"总耗时: {total_elapsed:.1f}s")
    print(f"平均耗时: {total_elapsed / max(total, 1):.1f}s/任务")
    print()
    print(f"SELECT 任务: {len(query_tasks)} 通过 {query_passed} ({query_passed / max(len(query_tasks), 1) * 100:.1f}%)")
    print(f"Management 任务: {len(management_tasks)} 通过 {management_passed} ({management_passed / max(len(management_tasks), 1) * 100:.1f}%)")
    print(f"总体准确率 (EX): {passed / max(total, 1) * 100:.1f}%")

    if args.output:
        output_data = {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "accuracy": round(passed / max(total, 1) * 100, 1),
                "query_accuracy": round(query_passed / max(len(query_tasks), 1) * 100, 1),
                "management_accuracy": round(management_passed / max(len(management_tasks), 1) * 100, 1),
                "total_time_seconds": round(total_elapsed, 1),
                "timestamp": datetime.now().isoformat(),
            },
            "results": all_results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())