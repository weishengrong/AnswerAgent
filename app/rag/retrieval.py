from openai import OpenAI
from typing import List, Dict, Tuple, Set
import asyncio
import logging
import time
from utils.rag.context_assembler import FieldSelection, assemble_context
from utils.rag.parent_store import parent_store
from app.rag.chroma import get_rag_chroma_client, chroma_search
from app.rag.bm25_index import bm25_index
from app.rag.query_rewriter import query_rewriter
from app.rag.reranker import reranker
from app.core.config.settings import embedding, rag_settings

logger = logging.getLogger(__name__)

_chroma_client = None

RRF_K = 60

JOIN_HINTS = [
    "join", "group", "by each", "for each", "across", "breakdown", "report",
    "summary", "ranked", "classify", "classification", "compare"
]
KEY_FIELD_HINTS = [
    "id", "name", "title", "code", "number", "registry", "ref", "key"
]


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = get_rag_chroma_client()
    return _chroma_client


def _get_collection():
    client = _get_chroma_client()
    return client.get_collection(rag_settings.RAG_COLLECTION_NAME)


def _get_embedding_vector(text: str) -> List[float]:
    client = OpenAI(
        api_key=embedding.EMBEDDING_MODEL_API_KEY,
        base_url=embedding.EMBEDDING_MODEL_URL
    )

    response = client.embeddings.create(
        model=embedding.EMBEDDING_MODEL_NAME,
        input=text
    )

    return response.data[0].embedding


async def _get_embedding_vector_async(text: str) -> List[float]:
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_embedding_vector, text)


def retrieve_parent_chunks(question: str, limit: int = 10) -> List[Dict]:
    """Parent 块向量检索：返回命中的表名和分数"""
    vector = _get_embedding_vector(question)
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[vector],
        n_results=limit,
        where={"chunk_type": {"$eq": "parent"}},
        include=["metadatas", "documents", "distances"]
    )

    formatted = []
    if results and results.get("ids") and len(results["ids"]) > 0:
        for i in range(len(results["ids"][0])):
            item = {
                "table_name": "",
                "distance": results["distances"][0][i] if results.get("distances") else 0,
                "content": "",
            }
            if results.get("metadatas") and results["metadatas"][0]:
                item["table_name"] = results["metadatas"][0][i].get("table_name", "")
            if results.get("documents") and results["documents"][0]:
                item["content"] = results["documents"][0][i]
            formatted.append(item)
    return formatted


async def retrieve_parent_chunks_async(question: str, limit: int = 10) -> List[Dict]:
    """Parent 块向量检索（异步版本）"""
    vector = await _get_embedding_vector_async(question)
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[vector],
        n_results=limit,
        where={"chunk_type": {"$eq": "parent"}},
        include=["metadatas", "documents", "distances"]
    )

    formatted = []
    if results and results.get("ids") and len(results["ids"]) > 0:
        for i in range(len(results["ids"][0])):
            item = {
                "table_name": "",
                "distance": results["distances"][0][i] if results.get("distances") else 0,
                "content": "",
            }
            if results.get("metadatas") and results["metadatas"][0]:
                item["table_name"] = results["metadatas"][0][i].get("table_name", "")
            if results.get("documents") and results["documents"][0]:
                item["content"] = results["documents"][0][i]
            formatted.append(item)
    return formatted


def _identify_foreign_keys(relation_map: Dict[str, List[Dict]]) -> Dict[str, str]:
    fk_map = {}

    for table_name, relations in relation_map.items():
        for rel in relations:
            via_field = rel.get("via_field", "")
            target_table = _qualify_relation_target(table_name, rel.get("target_table", ""))
            if via_field and target_table:
                fk_map[f"{table_name}.{via_field}"] = target_table

    return fk_map


def _qualify_relation_target(source_table: str, target_table: str) -> str:
    if not source_table or not target_table or "." in target_table:
        return target_table

    if "." not in source_table:
        return target_table

    db_name, _ = source_table.split(".", 1)
    return f"{db_name}.{target_table}"


def retrieve_child_chunks(question: str, limit: int = 10) -> List[Dict]:
    vector = _get_embedding_vector(question)

    collection = _get_collection()
    results = collection.query(
        query_embeddings=[vector],
        n_results=limit,
        where={"chunk_type": {"$eq": "child"}},
        include=["metadatas", "documents", "distances"]
    )

    formatted = []
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
            formatted.append(item)

    return formatted


async def retrieve_child_chunks_async(question: str, limit: int = 10) -> List[Dict]:
    vector = await _get_embedding_vector_async(question)

    collection = _get_collection()
    results = collection.query(
        query_embeddings=[vector],
        n_results=limit,
        where={"chunk_type": {"$eq": "child"}},
        include=["metadatas", "documents", "distances"]
    )

    formatted = []
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
            formatted.append(item)

    return formatted


def bm25_search(question: str, top_k: int = 10) -> List[Dict]:
    results = bm25_index.search(question, top_k=top_k)
    if results:
        logger.info(f"📊 BM25 检索命中 {len(results)} 条：{[r['field_name'] for r in results[:3]]}")
    # 记录 Prometheus 指标
    try:
        from app.middleware import RAG_BM25_HITS
        RAG_BM25_HITS.observe(len(results))
    except Exception:
        pass
    return results


def rrf_merge(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    k: int = RRF_K,
) -> List[Dict]:
    """Reciprocal Rank Fusion 融合向量检索和 BM25 检索结果

    公式：RRF_score(d) = Σ 1/(k + rank_i(d))

    两路结果都只看排名，不看绝对分数，天然解决量纲不同的问题。
    在两路都命中的文档会获得更高分数。
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}

    for rank, item in enumerate(vector_results):
        parent_id = item.get("parent_id", "")
        field_name = item.get("field_name", "")
        key = f"{parent_id}.{field_name}"

        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = {
                "parent_id": parent_id,
                "field_name": field_name,
                "content": item.get("content", ""),
                "distance": item.get("distance", 0),
            }

    for rank, item in enumerate(bm25_results):
        parent_id = item.get("parent_id", "")
        field_name = item.get("field_name", "")
        key = f"{parent_id}.{field_name}"

        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = {
                "parent_id": parent_id,
                "field_name": field_name,
                "content": item.get("content", ""),
                "distance": 0,
            }

    sorted_keys = sorted(scores, key=scores.get, reverse=True)

    merged = []
    for key in sorted_keys:
        chunk = chunk_map[key].copy()
        chunk["rrf_score"] = scores[key]
        merged.append(chunk)

    vector_only = len(vector_results) - len(set(
        f"{r.get('parent_id', '')}.{r.get('field_name', '')}" for r in bm25_results
    ) & set(
        f"{r.get('parent_id', '')}.{r.get('field_name', '')}" for r in vector_results
    ))
    overlap = len(vector_results) + len(bm25_results) - len(sorted_keys)
    logger.info(
        f"🔀 RRF 融合完成：向量 {len(vector_results)} 条 + BM25 {len(bm25_results)} 条 "
        f"→ 融合 {len(merged)} 条（重叠 {overlap} 条）"
    )

    # 记录 Prometheus 指标
    try:
        from app.middleware import RAG_RRF_RESULTS
        RAG_RRF_RESULTS.observe(len(merged))
    except Exception:
        pass

    return merged


def fuse_anchor_mode(parent_tables: List[str], child_chunks: List[Dict]) -> List[Dict]:
    """模式 A（Parent 锚定）：只保留 Child 中 parent_id 在候选表集合内的字段

    Args:
        parent_tables: Parent 检索命中的表名列表
        child_chunks: Child 检索命中的字段 chunk 列表

    Returns:
        过滤后的 child_chunks 列表
    """
    if not parent_tables:
        logger.warning("⚠️ Parent 检索未命中任何表，anchor 模式返回空")
        return []

    parent_set = set(parent_tables)
    filtered = [c for c in child_chunks if c.get("parent_id", "") in parent_set]

    dropped = len(child_chunks) - len(filtered)
    if dropped > 0:
        logger.info(f"🔒 Anchor 模式过滤：保留 {len(filtered)} 条，丢弃 {dropped} 条越界字段")

    return filtered


def fuse_union_mode(parent_tables: List[str], child_chunks: List[Dict]) -> List[Dict]:
    """模式 C（并集去重）：Parent + Child 并集，按表合并去重

    Args:
        parent_tables: Parent 检索命中的表名列表
        child_chunks: Child 检索命中的字段 chunk 列表

    Returns:
        合并后的 chunk 列表（包含所有命中的表和字段）
    """
    # 将 Parent 未命中但 Child 命中的表也纳入
    # 本质上就是不过滤，但记录额外信息
    extra_tables = set()
    for c in child_chunks:
        tid = c.get("parent_id", "")
        if tid and tid not in parent_tables:
            extra_tables.add(tid)

    if extra_tables:
        logger.info(f"🔓 Union 模式扩展：Parent 未命中但 Child 命中的表 {extra_tables}")

    return list(child_chunks)


def aggregate_hits_by_table(chunks: List[Dict]) -> Dict[str, List[FieldSelection]]:
    hit_fields: Dict[str, List[FieldSelection]] = {}

    for chunk in chunks:
        table_name = chunk.get("parent_id", "unknown")
        field_name = chunk.get("field_name", "")
        content = chunk.get("content", "")

        rrf_score = chunk.get("rrf_score", 0)
        distance = chunk.get("distance", 0)

        if rrf_score > 0:
            relevance_score = min(rrf_score * 30, 1.0)
        else:
            relevance_score = 1.0 / (1.0 + distance) if distance else 1.0

        field_selection = FieldSelection(
            table_name=table_name,
            field_name=field_name,
            relevance_score=relevance_score,
            is_foreign_key=False
        )

        if table_name not in hit_fields:
            hit_fields[table_name] = []

        if not any(f.field_name == field_name for f in hit_fields[table_name]):
            hit_fields[table_name].append(field_selection)

    for table_name in hit_fields:
        hit_fields[table_name].sort(key=lambda f: f.relevance_score, reverse=True)

    return hit_fields


def complete_foreign_keys(
    hit_fields: Dict[str, List[FieldSelection]],
    relation_map: Dict[str, List[Dict]]
) -> Dict[str, List[FieldSelection]]:
    KEY_DISPLAY_FIELDS = ["name", "title", "code", "number"]

    fk_map = _identify_foreign_keys(relation_map)

    for table_name, fields in hit_fields.items():
        for field in fields:
            fk_key = f"{table_name}.{field.field_name}"
            if fk_key in fk_map:
                field.is_foreign_key = True

    completed_fields = dict(hit_fields)

    for table_name, fields in hit_fields.items():
        for field in fields:
            if field.is_foreign_key:
                fk_key = f"{table_name}.{field.field_name}"
                if fk_key not in fk_map:
                    continue

                target_table = fk_map[fk_key]

                if target_table not in completed_fields:
                    completed_fields[target_table] = []

                existing_field_names = [f.field_name for f in completed_fields[target_table]]

                primary_key_field = "id"
                if primary_key_field not in existing_field_names:
                    completed_fields[target_table].append(
                        FieldSelection(
                            table_name=target_table,
                            field_name=primary_key_field,
                            relevance_score=0.5,
                            is_foreign_key=True
                        )
                    )

                for display_field in KEY_DISPLAY_FIELDS:
                    if display_field not in existing_field_names:
                        completed_fields[target_table].append(
                            FieldSelection(
                                table_name=target_table,
                                field_name=display_field,
                                relevance_score=0.6,
                                is_foreign_key=False
                            )
                        )
                        break

    return completed_fields


def _chunk_score(chunk: Dict) -> float:
    score = chunk.get("rrf_score", 0)
    if score:
        return score

    distance = chunk.get("distance", 0)
    return 1.0 / (1.0 + distance) if distance else 0.5


def _is_multi_table_question(question: str) -> bool:
    q = question.lower()
    return any(hint in q for hint in JOIN_HINTS)


def _relation_fields(relation_map: Dict[str, List[Dict]]) -> Dict[str, Set[str]]:
    fields: Dict[str, Set[str]] = {}

    for table_name, relations in relation_map.items():
        for rel in relations:
            via_field = rel.get("via_field", "")
            if via_field:
                fields.setdefault(table_name, set()).add(via_field)

    return fields


def _expand_with_relation_tables(selected_tables: List[str], relation_map: Dict[str, List[Dict]]) -> List[str]:
    expanded = list(selected_tables)
    seen = set(selected_tables)
    max_tables = max(rag_settings.RAG_TABLE_DYNAMIC_MIN, rag_settings.RAG_TABLE_DYNAMIC_MAX)

    for table_name in selected_tables:
        for rel in relation_map.get(table_name, []):
            target_table = _qualify_relation_target(table_name, rel.get("target_table", ""))
            if target_table and target_table not in seen and len(expanded) < max_tables:
                expanded.append(target_table)
                seen.add(target_table)

    return expanded


def _is_key_like_field(field_name: str) -> bool:
    field = field_name.lower()
    return any(hint in field for hint in KEY_FIELD_HINTS)


def _select_dynamic_tables(question: str, sorted_tables: List[Tuple[str, float]]) -> List[str]:
    if not sorted_tables:
        return []

    min_tables = max(rag_settings.RAG_TABLE_DYNAMIC_MIN, 2 if _is_multi_table_question(question) else 1)
    max_tables = max(min_tables, rag_settings.RAG_TABLE_DYNAMIC_MAX)

    selected = []
    max_score = sorted_tables[0][1]
    prev_score = max_score

    for index, (table_name, score) in enumerate(sorted_tables):
        keep = False
        if index < min_tables:
            keep = True
        elif score >= max_score * rag_settings.RAG_TABLE_SCORE_RATIO:
            keep = True
        elif prev_score and score >= prev_score * rag_settings.RAG_TABLE_PREV_SCORE_RATIO:
            keep = True

        if keep and len(selected) < max_tables:
            selected.append(table_name)
            prev_score = score
            continue

        break

    return selected


def _filter_fields_within_tables(
    table_chunks: Dict[str, List[Dict]],
    selected_tables: List[str],
    relation_map: Dict[str, List[Dict]],
) -> List[Dict]:
    result_chunks = []
    relation_field_map = _relation_fields(relation_map)

    for table_name in selected_tables:
        chunks_for_table = sorted(
            table_chunks.get(table_name, []),
            key=_chunk_score,
            reverse=True
        )
        if not chunks_for_table:
            continue

        top_score = _chunk_score(chunks_for_table[0])
        kept = []
        seen_fields = set()

        for index, chunk in enumerate(chunks_for_table):
            field_name = chunk.get("field_name", "")
            score = _chunk_score(chunk)
            is_relation_field = field_name in relation_field_map.get(table_name, set())
            is_key_field = _is_key_like_field(field_name)
            protected_field = is_relation_field or is_key_field
            keep = (
                index < rag_settings.RAG_FIELD_MIN_PER_TABLE
                or score >= top_score * rag_settings.RAG_FIELD_SCORE_RATIO
                or protected_field
            )
            if len(kept) >= rag_settings.RAG_FIELD_SOFT_MAX_PER_TABLE and not protected_field:
                keep = False

            if not keep or field_name in seen_fields:
                continue

            kept.append(chunk)
            seen_fields.add(field_name)

            if len(kept) >= rag_settings.RAG_FIELD_HARD_MAX_PER_TABLE:
                break

        result_chunks.extend(kept)

    return result_chunks


def table_level_rerank(
    question: str,
    chunks: List[Dict],
    relation_map: Dict[str, List[Dict]],
    mode: str = None,
) -> List[Dict]:
    """表级重排：支持固定 top-N、动态表数和表内字段过滤。"""
    if not chunks:
        return []

    rerank_mode = mode or rag_settings.RAG_TABLE_RERANK_MODE

    # 1. 按表聚合，计算表级分数
    table_scores = {}
    table_chunks = {}

    for chunk in chunks:
        table_name = chunk.get("parent_id", "")
        if not table_name:
            continue

        score = _chunk_score(chunk)

        if table_name not in table_scores:
            table_scores[table_name] = 0
            table_chunks[table_name] = []

        table_scores[table_name] += score
        table_chunks[table_name].append(chunk)

    # 2. 按表分数排序，取 top 8
    sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
    if rerank_mode == "fixed":
        top_tables = [name for name, _ in sorted_tables[:8]]
    else:
        top_tables = _select_dynamic_tables(question, sorted_tables)
        top_tables = _expand_with_relation_tables(top_tables, relation_map)

    top_table_set = set(top_tables)

    logger.info(
        f"📊 表级筛选({rerank_mode})：共 {len(table_scores)} 个表 → 保留 {len(top_tables)} 个表"
    )

    # 3. 保留 top 表内的所有字段
    if rerank_mode == "dynamic_fields":
        result_chunks = _filter_fields_within_tables(table_chunks, top_tables, relation_map)
    else:
        result_chunks = []
        for table_name in top_tables:
            result_chunks.extend(table_chunks[table_name])

    # 4. 补齐关联表（外键目标表）
    fk_map = _identify_foreign_keys(relation_map)
    selected_chunk_keys = {
        f"{chunk.get('parent_id', '')}.{chunk.get('field_name', '')}"
        for chunk in result_chunks
    }
    for chunk in chunks:
        table_name = chunk.get("parent_id", "")
        field_name = chunk.get("field_name", "")
        fk_key = f"{table_name}.{field_name}"

        if fk_key in fk_map:
            target_table = fk_map[fk_key]
            chunk_key = f"{table_name}.{field_name}"
            if (
                table_name in top_table_set
                and target_table in top_table_set
                and chunk_key not in selected_chunk_keys
            ):
                result_chunks.append(chunk)
                selected_chunk_keys.add(chunk_key)

    return result_chunks


def multi_path_retrieve(
    question: str,
    sub_queries: List[str],
    bm25_top_k: int,
    vector_limit: int,
) -> Tuple[List[Dict], List[Dict]]:
    """多路检索：对每个子查询分别做向量检索和 BM25 检索，合并去重

    复杂问题的向量被多个方面稀释，分解为多个子查询后各自检索，
    每个子查询的向量更聚焦，与字段 chunk 的粒度更匹配。

    Returns:
        (all_vector_chunks, all_bm25_chunks) 已去重合并
    """
    seen_vector = set()
    all_vector_chunks = []
    for sq in sub_queries:
        chunks = retrieve_child_chunks(sq, limit=vector_limit)
        for c in chunks:
            key = f"{c.get('parent_id','')}.{c.get('field_name','')}"
            if key not in seen_vector:
                seen_vector.add(key)
                all_vector_chunks.append(c)

    seen_bm25 = set()
    all_bm25_chunks = []
    for sq in sub_queries:
        chunks = bm25_search(sq, top_k=bm25_top_k)
        for c in chunks:
            key = f"{c.get('parent_id','')}.{c.get('field_name','')}"
            if key not in seen_bm25:
                seen_bm25.add(key)
                all_bm25_chunks.append(c)

    logger.info(
        f"🔀 多路检索（{len(sub_queries)} 个子查询）："
        f"向量 {len(all_vector_chunks)} 条（去重后）, "
        f"BM25 {len(all_bm25_chunks)} 条（去重后）"
    )
    return all_vector_chunks, all_bm25_chunks


async def multi_path_retrieve_async(
    question: str,
    sub_queries: List[str],
    bm25_top_k: int,
    vector_limit: int,
) -> Tuple[List[Dict], List[Dict]]:
    """多路检索的异步版本（并发执行子查询的向量检索）"""
    # 并发执行所有子查询的向量检索（原来逐个 await 串行，现在 gather 并发）
    if sub_queries:
        tasks = [retrieve_child_chunks_async(sq, limit=vector_limit) for sq in sub_queries]
        all_results = await asyncio.gather(*tasks)
    else:
        all_results = []

    seen_vector = set()
    all_vector_chunks = []
    for chunks in all_results:
        for c in chunks:
            key = f"{c.get('parent_id','')}.{c.get('field_name','')}"
            if key not in seen_vector:
                seen_vector.add(key)
                all_vector_chunks.append(c)

    # BM25 检索是纯本地计算，不需要并发
    seen_bm25 = set()
    all_bm25_chunks = []
    for sq in sub_queries:
        chunks = bm25_search(sq, top_k=bm25_top_k)
        for c in chunks:
            key = f"{c.get('parent_id','')}.{c.get('field_name','')}"
            if key not in seen_bm25:
                seen_bm25.add(key)
                all_bm25_chunks.append(c)

    logger.info(
        f"🔀 多路检索（{len(sub_queries)} 个子查询）："
        f"向量 {len(all_vector_chunks)} 条（去重后）, "
        f"BM25 {len(all_bm25_chunks)} 条（去重后）"
    )
    return all_vector_chunks, all_bm25_chunks


async def retrieve_schema(question: str, relation_map: Dict[str, List[Dict]]) -> str:
    # 1. Query 分解：将复杂问题拆成多个聚焦单一主题的子查询
    t0 = time.time()
    sub_queries = query_rewriter.decompose_query(question)
    logger.info(f"🔪 Query 分解为 {len(sub_queries)} 个子查询")
    # 兜底：子查询为空时使用原始问题（防止检索全空）
    if not sub_queries:
        sub_queries = [question]
        logger.info(f"🔪 子查询为空，回退为原始问题")
    try:
        from app.middleware import RAG_RETRIEVAL_LATENCY_MS
        RAG_RETRIEVAL_LATENCY_MS.labels(stage="query_rewrite").observe((time.time() - t0) * 1000)
    except Exception:
        pass

    # ===== 新增：Parent 路检索（与 Child 路并行） =====
    import asyncio as _asyncio
    parent_table_names = []

    async def _run_parent_search():
        nonlocal parent_table_names
        try:
            parent_results = await retrieve_parent_chunks_async(
                question, limit=rag_settings.RAG_PARENT_TOP_K
            )
            parent_table_names = [r["table_name"] for r in parent_results if r.get("table_name")]
            if parent_table_names:
                logger.info(f"📊 Parent 检索命中 {len(parent_table_names)} 张表: {parent_table_names[:5]}")
            else:
                logger.warning("⚠️ Parent 检索未命中（可能是旧索引无 parent 数据），将降级为纯 Child 检索")
        except Exception as e:
            logger.warning(f"⚠️ Parent 检索异常，降级为纯 Child 检索: {e}")

    # ===== Child 路检索（保持原有逻辑） =====
    child_task_result = {}

    async def _run_child_search():
        nonlocal child_task_result
        sub_vector_k = max(rag_settings.RAG_TOP_K_VECTOR // max(len(sub_queries), 1), 10)

        t1 = time.time()
        vector_chunks, _ = await multi_path_retrieve_async(
            question, sub_queries, 0, sub_vector_k
        )
        try:
            from app.middleware import RAG_VECTOR_HITS, RAG_RETRIEVAL_LATENCY_MS
            RAG_VECTOR_HITS.observe(len(vector_chunks))
            RAG_RETRIEVAL_LATENCY_MS.labels(stage="vector_search").observe((time.time() - t1) * 1000)
        except Exception:
            pass

        t2 = time.time()
        bm25_chunks = bm25_search(question, top_k=rag_settings.RAG_TOP_K_BM25)
        try:
            from app.middleware import RAG_RETRIEVAL_LATENCY_MS
            RAG_RETRIEVAL_LATENCY_MS.labels(stage="bm25_search").observe((time.time() - t2) * 1000)
        except Exception:
            pass

        if vector_chunks and bm25_chunks:
            t3 = time.time()
            merged_chunks = rrf_merge(vector_chunks, bm25_chunks)
            try:
                from app.middleware import RAG_RETRIEVAL_LATENCY_MS
                RAG_RETRIEVAL_LATENCY_MS.labels(stage="rrf_merge").observe((time.time() - t3) * 1000)
            except Exception:
                pass
        elif vector_chunks:
            merged_chunks = vector_chunks
        elif bm25_chunks:
            merged_chunks = bm25_chunks
        else:
            child_task_result["chunks"] = []
            return

        logger.info(f"🔍 粗排完成：向量 {len(vector_chunks)} 条 + BM25 {len(bm25_chunks)} 条 → 合并 {len(merged_chunks)} 条")

        # 表级重排
        t4 = time.time()
        table_ranked_chunks = table_level_rerank(question, merged_chunks, relation_map)
        try:
            from app.middleware import RAG_RETRIEVAL_LATENCY_MS
            RAG_RETRIEVAL_LATENCY_MS.labels(stage="table_rerank").observe((time.time() - t4) * 1000)
        except Exception:
            pass

        child_task_result["chunks"] = table_ranked_chunks if table_ranked_chunks else []

    # ===== 并行执行两路检索 =====
    await _asyncio.gather(_run_parent_search(), _run_child_search())

    child_chunks = child_task_result.get("chunks", [])

    if not child_chunks:
        return "未找到相关表信息"

    logger.info(f"🎯 Child 路表级重排完成：{len(child_chunks)} 条")

    # ===== 融合策略 =====
    fusion_mode = rag_settings.RAG_FUSION_MODE  # 新增配置项

    if parent_table_names and child_chunks:
        if fusion_mode == "anchor":
            final_chunks = fuse_anchor_mode(parent_table_names, child_chunks)
            logger.info(f"🔒 使用 Anchor 融合: {len(final_chunks)} 条（来自 {len(parent_table_names)} 张候选表）")
        elif fusion_mode == "union":
            final_chunks = fuse_union_mode(parent_table_names, child_chunks)
            logger.info(f"🔓 使用 Union 融合: {len(final_chunks)} 条")
        else:
            logger.warning(f"⚠️ 未知融合模式 '{fusion_mode}'，默认使用 anchor")
            final_chunks = fuse_anchor_mode(parent_table_names, child_chunks)
    else:
        # 降级：无 parent 数据时直接使用 child 结果
        if not parent_table_names:
            logger.warning("⚠️ 无 Parent 数据可用，跳过融合，使用纯 Child 检索结果")
        final_chunks = child_chunks

    if not final_chunks:
        return "未找到相关表信息"

    # ===== 组装上下文（保持现有逻辑不变） =====
    t5 = time.time()
    hit_fields = aggregate_hits_by_table(final_chunks)
    hit_fields = complete_foreign_keys(hit_fields, relation_map)
    hit_tables = list(hit_fields.keys())
    parent_contents = await parent_store.get_parents(hit_tables)

    if not parent_contents:
        return "未找到相关表信息"

    context = assemble_context(hit_fields, parent_contents, relation_map)

    try:
        from app.middleware import RAG_RETRIEVAL_LATENCY_MS
        RAG_RETRIEVAL_LATENCY_MS.labels(stage="context_assemble").observe((time.time() - t5) * 1000)
    except Exception:
        pass

    return context


async def retrieve_table_info_simple(question: str) -> str:
    vector = await _get_embedding_vector_async(question)

    collection = _get_collection()
    results = collection.query(
        query_embeddings=[vector],
        n_results=rag_settings.RAG_TOP_K_TABLE_VECTOR,
        include=["metadatas", "documents", "distances"]
    )

    vector_tables = []
    seen = set()

    if results and results.get("ids") and len(results["ids"]) > 0:
        for i in range(len(results["ids"][0])):
            item = {}
            if results.get("metadatas") and results["metadatas"][0]:
                item = results["metadatas"][0][i]
            if results.get("documents") and results["documents"][0]:
                item["content"] = results["documents"][0][i]

            table_name = item.get("table_name", "未知表")
            if table_name not in seen:
                vector_tables.append(table_name)
                seen.add(table_name)

    bm25_results = bm25_search(question, top_k=rag_settings.RAG_TOP_K_TABLE_BM25)
    for r in bm25_results:
        table_name = r.get("parent_id", "")
        if table_name and table_name not in seen:
            vector_tables.append(table_name)
            seen.add(table_name)

    formatted_content = []
    for table_name in vector_tables[:3]:
        parent_content = await parent_store.get_parent(table_name)
        if parent_content:
            formatted_content.append(f"表名：{table_name}\n{parent_content}")

    return "\n\n".join(formatted_content) if formatted_content else "未找到相关表信息"
