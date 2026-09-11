"""
BIRD-SQL Mini-Dev 数据集加载模块

BIRD-SQL (Big Benchmark for Integrated Database) 是 Text-to-SQL 领域事实标准
Mini-Dev 包含 11 个数据库 × 500 个 question-SQL 对，含 MySQL 原生版本

数据来源:
  - GitHub: https://github.com/bird-bench/mini_dev
  - 数据 ZIP: https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip

目录结构预期:
  data/minidev/MINIDEV/
  ├── mini_dev_mysql.json          # 500 条 MySQL 问答对
  ├── mini_dev_mysql_gold.sql      # Gold SQL (MySQL 方言)
  ├── dev_tables.json              # 11 个数据库的 Schema 定义 (同 Spider 格式)
  ├── dev_databases/               # 11 个 SQLite 数据库
  │   ├── california_schools/
  │   ├── debit_card_specializing/
  │   └── ...
  └── 上层: MINIDEV_mysql/BIRD_dev.sql  # 完整的 MySQL DDL+DML dump (~1GB)
"""

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ── 默认数据路径 ──────────────────────────────────────────────
BIRD_DEFAULT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "minidev", "MINIDEV"
)
BIRD_MYSQL_DUMP = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "minidev", "MINIDEV_mysql", "BIRD_dev.sql"
)


# ── 数据类型 ──────────────────────────────────────────────────


@dataclass
class BirdColumnDef:
    """数据库列定义 (BIRD 格式)"""
    index: int
    name: str               # 英文名
    name_original: str
    ctype: str
    is_primary: bool = False
    table_index: int = -1


@dataclass
class BirdTableDef:
    """数据库表定义"""
    index: int
    name: str
    name_original: str
    columns: List[BirdColumnDef] = field(default_factory=list)


@dataclass
class BirdDatabaseSchema:
    """完整数据库 Schema"""
    db_id: str
    tables: Dict[str, BirdTableDef] = field(default_factory=dict)
    foreign_keys: List[tuple] = field(default_factory=list)
    primary_keys: List[int] = field(default_factory=list)


@dataclass
class BirdTestCase:
    """BIRD-SQL 测试用例"""
    question_id: int
    question: str           # 英文问题
    db_id: str
    gold_sql: str           # MySQL 方言的 Gold SQL
    evidence: str = ""      # 额外提示信息
    difficulty: str = "unspecified"


# ── 数据加载 ──────────────────────────────────────────────────


def load_bird_tables(tables_path: str) -> Dict[str, BirdDatabaseSchema]:
    """加载 dev_tables.json, 返回 db_id -> BirdDatabaseSchema 的映射"""
    with open(tables_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    schemas: Dict[str, BirdDatabaseSchema] = {}
    for db_entry in raw:
        db_id = db_entry["db_id"]
        schema = BirdDatabaseSchema(db_id=db_id)

        # 构建列定义
        col_indices: Dict[int, BirdColumnDef] = {}
        col_names = db_entry.get("column_names", [])
        col_types = db_entry.get("column_types", [])
        col_names_orig = db_entry.get("column_names_original", col_names)

        for idx, (table_idx, col_name) in enumerate(col_names):
            col = BirdColumnDef(
                index=idx,
                name=col_name,
                name_original=col_names_orig[idx][1] if idx < len(col_names_orig) else col_name,
                ctype=col_types[idx] if idx < len(col_types) else "text",
                table_index=table_idx,
            )
            col_indices[idx] = col

        # 主键 (BIRD 中可能有复合主键, 如 [19, 20] 表示两列联合主键)
        pk_indices = db_entry.get("primary_keys", [])
        for pk_idx in pk_indices:
            if isinstance(pk_idx, list):  # 复合主键
                for sub_idx in pk_idx:
                    if sub_idx in col_indices:
                        col_indices[sub_idx].is_primary = True
            elif pk_idx in col_indices:
                col_indices[pk_idx].is_primary = True

        # 构建表
        table_names = db_entry.get("table_names", [])
        table_names_orig = db_entry.get("table_names_original", table_names)
        for t_idx, t_name in enumerate(table_names):
            table = BirdTableDef(
                index=t_idx,
                name=t_name,
                name_original=table_names_orig[t_idx] if t_idx < len(table_names_orig) else t_name,
                columns=[],
            )
            for col in col_indices.values():
                if col.table_index == t_idx:
                    table.columns.append(col)
            schema.tables[t_name.lower()] = table

        # 外键
        fk_raw = db_entry.get("foreign_keys", [])
        for src, tgt in fk_raw:
            schema.foreign_keys.append((src, tgt))

        schema.primary_keys = pk_indices
        schemas[db_id] = schema

    return schemas


def load_bird_questions(json_path: str) -> List[BirdTestCase]:
    """加载 BIRD-SQL 的 JSON 问题文件

    JSON 格式:
    [
        {
            "question_id": 1471,
            "db_id": "debit_card_specializing",
            "question": "What is the ratio of customers ...",
            "evidence": "ratio ...",
            "SQL": "SELECT CAST(SUM(CASE ...) AS DOUBLE) ...",
            "difficulty": "simple"
        },
        ...
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cases = []
    for entry in raw:
        sql = entry.get("SQL", "")
        sql = " ".join(sql.split()) if sql else ""

        case = BirdTestCase(
            question_id=entry.get("question_id", 0),
            question=entry.get("question", ""),
            db_id=entry.get("db_id", ""),
            gold_sql=sql,
            evidence=entry.get("evidence", ""),
            difficulty=entry.get("difficulty", "unspecified"),
        )
        cases.append(case)

    return cases


def load_bird_data(data_dir: str = BIRD_DEFAULT_DIR, split: str = "dev") -> List[BirdTestCase]:
    """加载 BIRD-SQL Mini-Dev 数据集

    Args:
        data_dir: MINIDEV 目录路径
        split: "dev" (开发集, 500 条, 目前仅有)

    Returns:
        List[BirdTestCase]
    """
    questions_path = os.path.join(data_dir, f"mini_dev_mysql.json")
    tables_path = os.path.join(data_dir, "dev_tables.json")

    if not os.path.exists(questions_path):
        logger.error("mini_dev_mysql.json 未找到: %s", questions_path)
        return []

    cases = load_bird_questions(questions_path)
    if os.path.exists(tables_path):
        tables = load_bird_tables(tables_path)
        logger.info("BIRD-SQL %s 集加载完成: %d 条用例, %d 个数据库", split, len(cases), len(tables))
    else:
        logger.warning("dev_tables.json 未找到, 仅加载问题")
        logger.info("BIRD-SQL %s 集加载完成: %d 条用例", split, len(cases))

    return cases


# ── MySQL 建库工具 ───────────────────────────────────────────


def import_bird_mysql_dump(
    mysql_host: str = "localhost",
    mysql_port: int = 3306,
    mysql_user: str = "root",
    mysql_password: str = "",
    dump_path: str = BIRD_MYSQL_DUMP,
    database_prefix: str = "bird_",
):
    """导入 BIRD_dev.sql 到 MySQL

    注意: BIRD_dev.sql (~1GB) 是一个完整的 mysqldump 输出,
    包含所有 11 个数据库的 CREATE DATABASE + CREATE TABLE + INSERT。

    Args:
        mysql_host: MySQL 主机
        mysql_port: MySQL 端口
        mysql_user: MySQL 用户
        mysql_password: MySQL 密码
        dump_path: BIRD_dev.sql 路径
        database_prefix: 数据库名前缀 (避免冲突)
    """
    if not os.path.exists(dump_path):
        logger.error("BIRD_dev.sql 未找到: %s", dump_path)
        logger.info("请确认文件已下载, 或使用 setup_databases() 从 SQLite 创建")
        return False

    logger.info("正在导入 BIRD_dev.sql 到 MySQL (%s:%d)...", mysql_host, mysql_port)
    logger.info("文件大小: %.1f GB, 预计耗时 5-15 分钟", os.path.getsize(dump_path) / 1024**3)

    # 如果不需要前缀, 直接导入
    cmd = [
        "mysql",
        f"-h{mysql_host}",
        f"-P{mysql_port}",
        f"-u{mysql_user}",
    ]
    if mysql_password:
        cmd.append(f"-p{mysql_password}")

    try:
        with open(dump_path, "r", encoding="utf-8") as f:
            result = subprocess.run(
                cmd,
                stdin=f,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 分钟超时
            )
        if result.returncode == 0:
            logger.info("BIRD_dev.sql 导入成功!")
            return True
        else:
            logger.error("导入失败: %s", result.stderr[:500])
            return False
    except subprocess.TimeoutExpired:
        logger.error("导入超时 (30分钟)")
        return False
    except FileNotFoundError:
        logger.error("未找到 mysql 客户端, 请确保 MySQL CLI 在 PATH 中")
        return False


def _get_mysql_conn_args(
    host: str = "localhost",
    port: int = 3306,
    user: str = "root",
    password: str = "",
) -> List[str]:
    """构建 mysql CLI 连接参数"""
    args = ["mysql", f"-h{host}", f"-P{port}", f"-u{user}"]
    if password:
        args.append(f"-p{password}")
    return args


# ── 中文翻译缓存 ──────────────────────────────────────────────


TRANSLATION_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "bird_translations.json"
)


def load_translation_cache() -> Dict[int, str]:
    """加载已有的翻译缓存"""
    if os.path.exists(TRANSLATION_CACHE_PATH):
        with open(TRANSLATION_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_translation_cache(cache: Dict[int, str]):
    """保存翻译缓存"""
    with open(TRANSLATION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logger.info("翻译缓存已保存: %d 条", len(cache))


def translate_questions(
    cases: List[BirdTestCase],
    llm_api_func=None,
    force: bool = False,
) -> List[BirdTestCase]:
    """批量翻译 BIRD-SQL 英文问题为中文

    使用缓存机制避免重复翻译。翻译后直接修改传入对象的 question 字段。

    Args:
        cases: BIRD-SQL 测试用例列表
        llm_api_func: LLM API 调用函数, 接收 (question, evidence) 返回中文翻译
                      若为 None, 使用加载的缓存或占位符
        force: 是否强制重新翻译 (忽略缓存)

    Returns:
        翻译后的测试用例列表
    """
    cache = {} if force else load_translation_cache()
    untranslated = [(i, c) for i, c in enumerate(cases)
                    if str(c.question_id) not in cache]

    if not untranslated:
        logger.info("所有问题已缓存, 直接加载翻译")
        for i, c in enumerate(cases):
            cached = cache.get(str(c.question_id))
            if cached:
                c.question = cached
        return cases

    if llm_api_func is None:
        logger.warning("未提供翻译函数, 使用英文原文占位")
        logger.info("共 %d 条未翻译, 请稍后使用 translate_questions(cases, llm_api_func=...) 补翻", len(untranslated))
        return cases

    logger.info("正在翻译 %d 条英文问题为中文...", len(untranslated))
    for idx, (orig_idx, case) in enumerate(untranslated):
        try:
            translated = llm_api_func(case.question, case.evidence)
            cache[str(case.question_id)] = translated
            cases[orig_idx].question = translated
            if (idx + 1) % 50 == 0:
                logger.info("翻译进度: %d/%d", idx + 1, len(untranslated))
        except Exception as e:
            logger.warning("翻译失败 (question_id=%d): %s", case.question_id, e)

    save_translation_cache(cache)
    logger.info("翻译完成: %d 条", len(untranslated))
    return cases


# ── 统计信息 ──────────────────────────────────────────────────


def print_bird_stats(cases: List[BirdTestCase]):
    """打印 BIRD-SQL 数据集统计信息"""
    from collections import Counter
    logger.info("=" * 50)
    logger.info("BIRD-SQL Mini-Dev 数据集统计")
    logger.info("=" * 50)
    logger.info("总用例数: %d", len(cases))

    difficulty_counter = Counter(c.difficulty for c in cases)
    logger.info("难度分布:")
    for diff, cnt in difficulty_counter.most_common():
        logger.info("  %s: %d (%.1f%%)", diff, cnt, cnt / len(cases) * 100)

    db_counter = Counter(c.db_id for c in cases)
    logger.info("数据库分布:")
    for db_id, cnt in db_counter.most_common():
        logger.info("  %s: %d", db_id, cnt)
    logger.info("=" * 50)


# ── 入口 ──────────────────────────────────────────────────────


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cases = load_bird_data()
    if cases:
        print_bird_stats(cases)

        # 打印第 1 条样例
        sample = cases[0]
        logger.info("\n样例 #%d:", sample.question_id)
        logger.info("  问题: %s", sample.question)
        logger.info("  SQL:  %s", sample.gold_sql)
        logger.info("  数据库: %s", sample.db_id)
        logger.info("  难度: %s", sample.difficulty)
        logger.info("  Evidence: %s", sample.evidence[:100] if sample.evidence else "(无)")