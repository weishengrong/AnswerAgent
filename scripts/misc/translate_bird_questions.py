"""
将 BIRD-SQL financial 库的 32 条英文问题批量翻译为中文。

用法:
    python scripts/translate_bird_questions.py

依赖:
    - 项目已有的 openai 库
    - 项目 .env 中的 LLM 配置 (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME)
"""

import json
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_questions() -> list[dict]:
    """从 bird_dataset 加载 financial 库的全部问题"""
    from tests.fixtures.datasets.bird_dataset import load_bird_data

    cases = load_bird_data()
    fin_cases = [c for c in cases if c.db_id == "financial"]

    # 去重 (按 question_id)
    seen = set()
    unique = []
    for c in fin_cases:
        if c.question_id not in seen:
            seen.add(c.question_id)
            unique.append(c)

    logger.info(f"financial 库共 {len(fin_cases)} 条, 去重后 {len(unique)} 条")
    return unique


def translate_batch(
    questions: list[dict], batch_size: int = 10
) -> list[dict]:
    """批量翻译为中文"""
    from openai import OpenAI
    from app.core.config.settings import llm_settings

    client = OpenAI(
        api_key=llm_settings.LLM_API_KEY,
        base_url=llm_settings.LLM_BASE_URL,
    )
    model = llm_settings.LLM_MODEL_NAME

    results = []
    total = len(questions)

    for start in range(0, total, batch_size):
        batch = questions[start : start + batch_size]
        batch_idx = start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        logger.info(
            f"翻译批次 {batch_idx}/{total_batches} "
            f"(第 {start+1}-{min(start+batch_size, total)} 条)"
        )

        # 构造翻译 prompt
        items = []
        for i, q in enumerate(batch):
            items.append(
                f'  {start + i + 1}. ID={q.question_id} | 难度={q.difficulty} | "{q.question}"'
            )

        prompt = f"""你是一个 Text-to-SQL 数据集翻译专家。请将以下 {len(batch)} 条英文查询问题逐条翻译为中文。

要求：
1. 保持原意不变，保留表名、字段名、数值、日期等关键信息
2. 中文表达自然通顺，符合数据库查询场景
3. 每条一行，格式为: ID={{{{id}}}} | 难度={{{{difficulty}}}} | "{{{{翻译后的中文问题}}}}"

待翻译的问题：
{chr(10).join(items)}

请按格式逐条输出翻译结果。"""

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                timeout=120,
            )
            content = resp.choices[0].message.content.strip()

            # 解析返回结果
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # 解析格式: ID=xxx | 难度=xxx | "翻译后文本"
                try:
                    parts = line.split("|")
                    id_part = parts[0].strip()  # ID=xxx
                    # diff_part = parts[1].strip()  # 难度=xxx
                    text_part = "|".join(parts[2:]).strip()  # "中文"
                    text_part = text_part.strip("\"'")

                    qid = int(id_part.replace("ID=", "").strip())
                    # 找到对应的原始问题
                    for q in batch:
                        if q.question_id == qid:
                            q.question_zh = text_part
                            break
                except (ValueError, IndexError):
                    continue

            # 处理没有 ID 前缀的额外行（可能模型只输出翻译文本）
            for i, q in enumerate(batch):
                if not hasattr(q, "question_zh"):
                    # 尝试从 content 中按顺序匹配
                    lines = [
                        l.strip().strip("\"'")
                        for l in content.split("\n")
                        if l.strip()
                    ]
                    if i < len(lines):
                        q.question_zh = lines[i]

        except Exception as e:
            logger.error(f"批次 {batch_idx} 翻译失败: {e}")
            # 失败时用英文原文填充
            for q in batch:
                if not hasattr(q, "question_zh"):
                    q.question_zh = q.question

        # 输出本批次翻译结果
        for q in batch:
            logger.info(
                f"  Q{q.question_id:4d} [{q.difficulty}] "
                f"{q.question[:50]}... → {q.question_zh[:50]}..."
            )

        results.extend(batch)

    return results


def save_dataset(questions: list[dict], output_path: str):
    """保存为 JSONL 格式"""
    records = []
    for q in questions:
        records.append(
            {
                "question_id": q.question_id,
                "difficulty": q.difficulty,
                "question_en": q.question,
                "question_zh": getattr(q, "question_zh", q.question),
                "gold_sql": q.gold_sql,
                "db_id": q.db_id,
                "evidence": q.evidence,
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"已保存 {len(records)} 条到 {output_path}")

    # 同时输出中文问题列表方便查看
    txt_path = output_path.replace(".jsonl", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(
                f"Q{r['question_id']:4d} [{r['difficulty']}]\n"
                f"  EN: {r['question_en']}\n"
                f"  ZH: {r['question_zh']}\n\n"
            )
    logger.info(f"可读版本已保存到 {txt_path}")


def main():
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "bird_financial_zh.jsonl"
    )

    # 1. 加载问题
    questions = load_questions()
    logger.info(f"待翻译: {len(questions)} 条")

    # 2. 批量翻译
    questions = translate_batch(questions, batch_size=10)

    # 3. 保存
    save_dataset(questions, output_path)

    # 4. 打印摘要
    zh_count = sum(
        1 for q in questions if getattr(q, "question_zh", None) and q.question_zh != q.question
    )
    logger.info(
        f"\n翻译完成: {zh_count}/{len(questions)} 条已翻译"
    )


if __name__ == "__main__":
    main()