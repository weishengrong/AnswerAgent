import logging
import re
from typing import List, Optional
from openai import OpenAI

from app.core.config.settings import llm_settings

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Query 改写器：将用户自然语言问题扩展为更适合检索的查询文本"""

    def __init__(self):
        self.client = OpenAI(
            api_key=llm_settings.LLM_API_KEY,
            base_url=llm_settings.LLM_BASE_URL
        )
        self.model = llm_settings.LLM_MODEL_NAME

    def rewrite(self, question: str) -> str:
        """将用户问题改写为检索优化版本"""
        prompt = f"""你是一个数据库查询优化助手。请将用户的问题改写为更适合关键词检索的版本。

要求：
1. 保留原始问题的核心语义
2. 提取问题中提到的所有实体名称（表名、字段名、指标名等）
3. 将缩写展开为全称（如 SNQI -> Signal-to-Noise Quality Indicator）
4. 添加可能的同义词或相关术语
5. 输出格式：原始问题 + 提取的实体和同义词，用空格分隔

用户问题：{question}

改写结果："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个数据库查询优化助手，擅长提取问题中的关键实体和术语。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            rewritten = response.choices[0].message.content.strip()
            logger.info(f"🔄 Query 改写: '{question[:50]}...' -> '{rewritten[:80]}...'")
            return rewritten
        except Exception as e:
            logger.warning(f"⚠️ Query 改写失败，使用原始问题: {e}")
            return question

    def rewrite_for_vector(self, question: str) -> str:
        """专为向量检索改写的版本，更侧重语义扩展"""
        prompt = f"""请将用户的问题改写为更适合语义向量检索的版本。

要求：
1. 保留原始问题的核心语义
2. 将缩写展开为全称
3. 用更正式、更数据库化的术语重述问题
4. 添加问题中隐含的数据库概念（如外键、关联、聚合等）

用户问题：{question}

改写结果："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个数据库查询优化助手，擅长将自然语言问题转换为数据库术语。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            rewritten = response.choices[0].message.content.strip()
            logger.info(f"🔄 Vector Query 改写: '{question[:50]}...' -> '{rewritten[:80]}...'")
            return rewritten
        except Exception as e:
            logger.warning(f"⚠️ Vector Query 改写失败，使用原始问题: {e}")
            return question

    def decompose_query(self, question: str) -> List[str]:
        """将复杂长问题分解为多个聚焦单一主题的子查询

        原始的长问题包含多个方面（如筛选条件、聚合计算、排序等），
        整个问题做向量检索时语义被稀释。分解后每个子查询只聚焦一个方面，
        与字段 chunk 的向量粒度更匹配。

        Args:
            question: 用户的自然语言问题

        Returns:
            子查询列表
        """
        prompt = f"""你是一个数据库查询分解助手。用户的自然语言问题可能包含多个不同的查询意图，请将其分解为多个独立的关键词子查询。

要求：
1. 每个子查询不超过 20 个词，只包含关键词，不要完整句子
2. 子查询之间不能重叠（每个子查询聚焦不同方面）
3. 子查询数量控制在 2-5 个
4. 每个子查询应包含该方面的关键实体词和属性词
5. 如果问题很简单（如只涉及一个方面），可以不分解，直接输出原始问题
6. 输出格式：每行一个子查询，不要序号，不要解释，不要空行

用户问题：{question}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个数据库查询分解助手，擅长将复杂问题拆分为多个单一主题的子查询。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            raw = response.choices[0].message.content.strip()
            # 鲁棒解析：去除序号前缀（"1. "、"1、" "- " 等）和空行
            lines = raw.split("\n")
            sub_queries = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # 去除常见序号前缀
                line = re.sub(r'^[\d]+[\.\、\:\s]+\s*', '', line).strip()
                # 去除 markdown 列表符号
                line = re.sub(r'^[-*+]\s+', '', line).strip()
                if line and len(line) > 1:
                    sub_queries.append(line)

            # 兜底：如果解析结果为空，使用原始问题
            if not sub_queries:
                logger.warning(f"⚠️ Query 分解结果为空，使用原始问题作为子查询")
                sub_queries = [question]

            logger.info(f"🔪 Query 分解: '{question[:50]}...' -> {len(sub_queries)} 个子查询")
            for i, sq in enumerate(sub_queries):
                logger.info(f"   [{i+1}] {sq[:80]}")
            return sub_queries
        except Exception as e:
            logger.warning(f"⚠️ Query 分解失败，使用原始问题: {e}")
            return [question]


query_rewriter = QueryRewriter()
