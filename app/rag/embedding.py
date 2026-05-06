from openai import OpenAI
from typing import List, Optional

from config.settings import embedding
from app.rag.milvus import init_milvus, DIMENSION
from utils.docx_util import slice_document


def get_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    if not texts:
        return []

    print(f"🧠 正在调用硅基流动模型：{embedding.EMBEDDING_MODEL_NAME} ...")

    client = OpenAI(
        api_key=embedding.EMBEDDING_MODEL_API_KEY,
        base_url=embedding.EMBEDDING_MODEL_URL
    )

    all_vectors = []
    batch_size = 10

    detected_dim = None

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        current_batch_num = (i // batch_size) + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size

        print(f"   📦 处理批次 {current_batch_num}/{total_batches}...")

        try:
            response = client.embeddings.create(
                model=embedding.EMBEDDING_MODEL_NAME,
                input=batch_texts
            )

            batch_vectors = [item.embedding for item in response.data]

            if len(batch_vectors) != len(batch_texts):
                print(f"❌ 批次返回数量不匹配！")
                return None

            if detected_dim is None and batch_vectors:
                detected_dim = len(batch_vectors[0])
                print(f"✅ 检测到模型维度：{detected_dim}")

            all_vectors.extend(batch_vectors)

        except Exception as e:
            print(f"❌ API 调用失败：{e}")
            return None

    print(f"✅ 向量化完成！共 {len(all_vectors)} 条。")
    return all_vectors


async def process_document(collection_name: str, FILE_PATH: str) -> None:
    collection = init_milvus(collection_name)

    texts, metas = slice_document(FILE_PATH)
    if not texts:
        print("❌ 切片结果为空，退出。")
        return

    vectors = get_embeddings(texts)
    if vectors is None:
        print("❌ 向量化失败，退出。")
        return

    if len(vectors) != len(texts) or len(vectors) != len(metas):
        print(f"❌ 致命错误：数据长度不匹配！Vectors:{len(vectors)}, Texts:{len(texts)}, Metas:{len(metas)}")
        return

    clean_vectors = []
    for i, v in enumerate(vectors):
        if isinstance(v, list):
            clean_v = [float(x) for x in v]
            if len(clean_v) != DIMENSION:
                print(f"❌ 第 {i} 条向量维度错误：{len(clean_v)} != {DIMENSION}")
                return
            clean_vectors.append(clean_v)
        else:
            try:
                clean_v = [float(x) for x in v]
                if len(clean_v) != DIMENSION:
                    print(f"❌ 第 {i} 条向量维度错误：{len(clean_v)} != {DIMENSION}")
                    return
                clean_vectors.append(clean_v)
            except Exception as e:
                print(f"❌ 第 {i} 条向量无法转换为 float 列表: {e}")
                return

    vectors = clean_vectors

    total_floats = sum(len(v) for v in vectors)
    expected_floats = len(vectors) * DIMENSION
    print(
        f"   [Debug] 向量校验: 条数={len(vectors)}, 维度={DIMENSION}, 总浮点数={total_floats} (期望: {expected_floats})")

    if total_floats != expected_floats:
        print(f"❌ 致命错误：向量总浮点数不匹配！实际: {total_floats}, 期望: {expected_floats}")
        return

    print("💾 正在写入 Milvus...")

    table_names = [m["table_name"] for m in metas]

    data_to_insert = [
        vectors,
        table_names,
        texts
    ]

    try:
        mr = collection.insert(data_to_insert)
        collection.flush()
        print(f"✅ 成功！已插入 {mr.insert_count} 条数据到 Milvus 集合 '{collection_name}'。")
    except Exception as e:
        print(f"❌ 插入 Milvus 失败: {e}")
