import logging
from typing import List, Dict
from openai import OpenAI

from app.core.config.settings import llm_settings, rag_settings

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-Encoder 精排器：对粗排结果进行相关性重排序"""

    def __init__(self):
        self.client = OpenAI(
            api_key=llm_settings.LLM_API_KEY,
            base_url=llm_settings.LLM_BASE_URL
        )
        self.model = llm_settings.LLM_MODEL_NAME

    def rerank(self, question: str, chunks: List[Dict]) -> List[Dict]:
        """对粗排结果进行精排

        Args:
            question: 用户问题
            chunks: 粗排结果列表，每个包含 parent_id, field_name, content 等

        Returns:
            按相关性分数排序后的结果列表
        """
        if not chunks:
            return []

        scored_chunks = []
        for chunk in chunks:
            content = chunk.get("content", "")
            field_name = chunk.get("field_name", "")
            table_name = chunk.get("parent_id", "").split(".")[-1]

            # 构建精排文本：表名 + 字段名 + 描述
            rerank_text = f"{table_name}.{field_name}: {content[:200]}"

            score = self._get_relevance_score(question, rerank_text)

            scored_chunk = chunk.copy()
            scored_chunk["rerank_score"] = score
            scored_chunks.append(scored_chunk)

        # 按分数降序排序
        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 过滤低分结果
        threshold = rag_settings.RAG_RERANK_THRESHOLD
        filtered = [c for c in scored_chunks if c["rerank_score"] >= threshold]

        # 取 top_k
        top_k = rag_settings.RAG_RERANK_TOP_K
        result = filtered[:top_k]

        logger.info(
            f"🎯 精排完成：粗排 {len(chunks)} 条 → 精排 {len(result)} 条 "
            f"(阈值 {threshold}, top_k {top_k})"
        )

        return result

    def _get_relevance_score(self, question: str, text: str) -> float:
        """计算问题和文本的相关性分数

        使用 LLM 做零样本相关性判断，返回 0-1 的分数
        """
        prompt = f"""判断以下问题和数据库字段描述的相关性。

问题：{question}

字段描述：{text}

请只返回一个 0-1 之间的数字，表示相关性程度：
- 0.0-0.3：不相关
- 0.3-0.6：部分相关
- 0.6-1.0：高度相关

只返回数字，不要其他内容："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个相关性判断助手，只返回数字。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )

            content = response.choices[0].message.content.strip()
            # 提取数字
            import re
            match = re.search(r'0?\.\d+', content)
            if match:
                score = float(match.group())
                return min(max(score, 0.0), 1.0)

            return 0.5
        except Exception as e:
            logger.warning(f"⚠️ 精排打分失败: {e}")
            return 0.5


reranker = Reranker()
