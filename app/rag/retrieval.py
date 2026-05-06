from openai import OpenAI
from typing import List, Dict, Tuple
import time
import logging
from utils.context_assembler import FieldSelection, assemble_context
from utils.parent_store import parent_store
from app.rag.milvus import MilvusSessionLocal
from app.rag.bm25_index import bm25_index
from config.settings import embedding, rag_settings

logger = logging.getLogger(__name__)

_milvus_client = None

RRF_K = 60


def _get_milvus_client():
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusSessionLocal()
    return _milvus_client


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


def _identify_foreign_keys(relation_map: Dict[str, List[Dict]]) -> Dict[str, str]:
    fk_map = {}

    for table_name, relations in relation_map.items():
        for rel in relations:
            via_field = rel.get("via_field", "")
            target_table = rel.get("target_table", "")
            if via_field and target_table:
                fk_map[f"{table_name}.{via_field}"] = target_table

    return fk_map


def _ensure_collection_loaded(collection_name: str, retries: int = 3) -> bool:
    client = _get_milvus_client()
    for i in range(retries):
        try:
            client.load_collection(collection_name=collection_name)
            logger.info(f"✅ 集合 {collection_name} 已加载到内存")
            return True
        except Exception as e:
            if i < retries - 1:
                wait = 2 ** i
                logger.warning(f"⚠️ 加载集合 {collection_name} 第{i+1}次失败，{wait}s 后重试：{e}")
                time.sleep(wait)
            else:
                logger.error(f"❌ 加载集合 {collection_name} 失败（已重试{retries}次）：{e}")
                from app.middleware import service_health
                service_health.mark_milvus_down()
                return False


def retrieve_child_chunks(question: str, limit: int = 10) -> List[Dict]:
    # Milvus Lite 不需要 connections.connect，直接使用客户端

    # from pymilvus import connections
    # from config.settings import milvus_settings
    #
    # try:
    #     connections.connect(
    #         host=milvus_settings.MILVUS_HOST,
    #         port=milvus_settings.MILVUS_PORT
    #     )
    # except Exception:
    pass

    vector = _get_embedding_vector(question)

    if not _ensure_collection_loaded(rag_settings.RAG_COLLECTION_NAME):
        return []

    client = _get_milvus_client()
    result = client.search(
        collection_name=rag_settings.RAG_COLLECTION_NAME,
        data=[vector],
        limit=limit,
        search_params={"metric_type": "COSINE"},
        filter="chunk_type == 'child'",
        output_fields=["parent_id", "field_name", "content", "chunk_type"]
    )

    return result[0] if result else []


async def retrieve_child_chunks_async(question: str, limit: int = 10) -> List[Dict]:
    # Milvus Lite 不需要 connections.connect，直接使用客户端

    # from pymilvus import connections
    # from config.settings import milvus_settings
    #
    # try:
    #     connections.connect(
    #         host=milvus_settings.MILVUS_HOST,
    #         port=milvus_settings.MILVUS_PORT
    #     )
    # except Exception:
    pass

    vector = await _get_embedding_vector_async(question)

    if not _ensure_collection_loaded(rag_settings.RAG_COLLECTION_NAME):
        return []

    client = _get_milvus_client()
    result = client.search(
        collection_name=rag_settings.RAG_COLLECTION_NAME,
        data=[vector],
        limit=limit,
        search_params={"metric_type": "COSINE"},
        filter="chunk_type == 'child'",
        output_fields=["parent_id", "field_name", "content", "chunk_type"]
    )

    return result[0] if result else []


def bm25_search(question: str, top_k: int = 10) -> List[Dict]:
    results = bm25_index.search(question, top_k=top_k)
    if results:
        logger.info(f"📊 BM25 检索命中 {len(results)} 条：{[r['field_name'] for r in results[:3]]}")
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

    return merged


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


async def retrieve_schema(question: str, relation_map: Dict[str, List[Dict]]) -> str:
    vector_chunks = await retrieve_child_chunks_async(question, limit=10)

    bm25_chunks = bm25_search(question, top_k=10)

    if vector_chunks and bm25_chunks:
        merged_chunks = rrf_merge(vector_chunks, bm25_chunks)
    elif vector_chunks:
        merged_chunks = vector_chunks
    elif bm25_chunks:
        merged_chunks = bm25_chunks
    else:
        return "未找到相关表信息"

    hit_fields = aggregate_hits_by_table(merged_chunks)

    hit_fields = complete_foreign_keys(hit_fields, relation_map)

    hit_tables = list(hit_fields.keys())
    parent_contents = await parent_store.get_parents(hit_tables)

    if not parent_contents:
        return "未找到相关表信息"

    context = assemble_context(hit_fields, parent_contents, relation_map)

    return context


async def retrieve_table_info_simple(question: str) -> str:
    vector = await _get_embedding_vector_async(question)

    if not _ensure_collection_loaded(rag_settings.RAG_COLLECTION_NAME):
        return "加载集合失败"

    client = _get_milvus_client()
    result = client.search(
        collection_name=rag_settings.RAG_COLLECTION_NAME,
        data=[vector],
        limit=3,
        search_params={"metric_type": "COSINE"},
        output_fields=["table_name", "content"]
    )

    vector_tables = []
    seen = set()
    for item in result[0]:
        table_name = item.get("table_name", "未知表")
        if table_name not in seen:
            vector_tables.append(table_name)
            seen.add(table_name)

    bm25_results = bm25_search(question, top_k=5)
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
        else:
            for item in result[0]:
                if item.get("table_name") == table_name:
                    formatted_content.append(f"表名：{table_name}\n{item.get('content', '')}")
                    break

    return "\n\n".join(formatted_content) if formatted_content else "未找到相关表信息"
