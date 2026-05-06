import json
import logging
import pickle
import re
from typing import List, Dict, Optional

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

STOPWORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "他", "她", "它", "们", "那", "些", "什么", "怎么", "如何", "哪",
    "哪些", "多少", "几", "可以", "能", "请", "帮", "查", "查询", "统计", "显示",
    "列出", "找出", "搜索", "获取", "知道", "告诉", "问", "想", "需要", "看看",
    "一下", "吗", "呢", "吧", "啊", "哦", "嗯", "呀",
}

IDENTIFIER_PATTERN = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')


def tokenize(text: str) -> List[str]:
    tokens = []

    identifiers = IDENTIFIER_PATTERN.findall(text)
    for ident in identifiers:
        if len(ident) > 1 and ident.lower() not in STOPWORDS:
            tokens.append(ident.lower())

    text_without_identifiers = IDENTIFIER_PATTERN.sub(" ", text)
    for word in jieba.cut(text_without_identifiers):
        word = word.strip()
        if not word:
            continue
        if word in STOPWORDS:
            continue
        if len(word) == 1 and not word.isalpha():
            continue
        tokens.append(word.lower())

    return tokens


class BM25Index:
    """BM25 索引管理器

    离线构建索引，在线检索，索引持久化到 Redis。
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: List[Dict] = []
        self._tokenized_corpus: List[List[str]] = []

    def build(self, corpus: List[Dict]):
        """构建 BM25 索引

        Args:
            corpus: 子块列表，每个子块包含 content, parent_id, field_name 等字段
        """
        self._corpus = corpus
        self._tokenized_corpus = [tokenize(item["content"]) for item in corpus]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"✅ BM25 索引构建完成：{len(corpus)} 条文档")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 检索

        Args:
            query: 查询文本
            top_k: 返回前 K 个结果

        Returns:
            检索结果列表，每个结果包含 parent_id, field_name, content, bm25_score
        """
        if not self._bm25 or not self._corpus:
            logger.warning("⚠️ BM25 索引未构建，返回空结果")
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] <= 0:
                continue
            result = {
                "parent_id": self._corpus[idx].get("parent_id", ""),
                "field_name": self._corpus[idx].get("field_name", ""),
                "content": self._corpus[idx].get("content", ""),
                "bm25_score": float(scores[idx]),
            }
            results.append(result)

        return results

    def serialize(self) -> bytes:
        """序列化索引到 bytes，用于持久化"""
        data = {
            "corpus": self._corpus,
            "tokenized_corpus": self._tokenized_corpus,
        }
        return pickle.dumps(data)

    def deserialize(self, data: bytes):
        """从 bytes 反序列化索引"""
        loaded = pickle.loads(data)
        self._corpus = loaded["corpus"]
        self._tokenized_corpus = loaded["tokenized_corpus"]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"✅ BM25 索引从持久化加载：{len(self._corpus)} 条文档")

    async def save_to_redis(self, redis_client=None):
        """持久化索引到 Redis（必须用 raw 客户端，避免 pickle bytes 被 utf-8 解码）"""
        if redis_client is None:
            from config.redis import get_raw_redis_client
            redis_client = await get_raw_redis_client()

        data = self.serialize()
        await redis_client.set("rag:bm25_index", data)
        logger.info(f"💾 BM25 索引已保存到 Redis（{len(data)} bytes）")

    async def load_from_redis(self, redis_client=None) -> bool:
        """从 Redis 加载索引（必须用 raw 客户端，避免 pickle bytes 被 utf-8 解码）

        Returns:
            True 加载成功，False 加载失败或不存在
        """
        if redis_client is None:
            from config.redis import get_raw_redis_client
            redis_client = await get_raw_redis_client()

        data = await redis_client.get("rag:bm25_index")
        if not data:
            logger.warning("⚠️ Redis 中无 BM25 索引")
            return False

        try:
            self.deserialize(data)
            return True
        except Exception as e:
            logger.error(f"❌ BM25 索引从 Redis 加载失败: {e}")
            return False


bm25_index = BM25Index()
