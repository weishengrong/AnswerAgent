"""
CSpider 中文 Text-to-SQL 数据集加载模块

CSpider 是 Spider 的中文翻译版 (西湖大学, EMNLP 2019)
包含 10,181 个中文问题 + 5,693 条 SQL，覆盖 200 个数据库

数据来源:
  - Google Drive: https://drive.google.com/drive/folders/1TxCUq1ydPuBdDdHF3MkHT-8zixluQuLa
  - BaiduNetDisk: https://pan.baidu.com/s/1B84p7Wl8jx9F3nd0xpt4hw (提取码: 9gzb)

目录结构预期:
  data/cspider/
  ├── train.json          # 训练集 (~7000 条)
  ├── train_gold.sql      # Gold SQL (可选)
  ├── dev.json            # 开发集 (~1034 条)
  ├── dev_gold.sql        # Gold SQL (可选)
  ├── tables.json         # 全部 200 个数据库的 Schema 定义
  └── database/           # SQLite 数据库文件 (每个库一个 .db)
      ├── academia/
      │   └── academia.sqlite
      ├── activity_1/
      │   └── activity_1.sqlite
      └── ...
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ── 默认数据路径 ──────────────────────────────────────────────
CSPIDER_DEFAULT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "cspider"
)


# ── 数据类型 ──────────────────────────────────────────────────


@dataclass
class ColumnDef:
    """数据库列定义"""
    index: int
    name: str               # 原始英文名 (如 user_id)
    name_original: str      # 原始名 (通常与 name 相同)
    ctype: str              # 数据类型 (如 number, text)
    is_primary: bool = False
    table_index: int = -1


@dataclass
class TableDef:
    """数据库表定义"""
    index: int
    name: str               # 英文表名 (如 user)
    name_original: str      # 原始表名
    columns: List[ColumnDef] = field(default_factory=list)


@dataclass
class DatabaseSchema:
    """完整数据库 Schema"""
    db_id: str
    tables: Dict[str, TableDef] = field(default_factory=dict)  # table_name -> TableDef
    foreign_keys: List[tuple] = field(default_factory=list)    # (col_idx, ref_col_idx)
    primary_keys: List[int] = field(default_factory=list)      # column indices


@dataclass
class CspiderTestCase:
    """单条测试用例"""
    question: str           # 中文问题
    gold_sql: str           # 标准答案 SQL
    db_id: str              # 数据库 ID
    query_id: int = 0       # 查询 ID (可选)
    question_tok: Optional[List[str]] = None  # 分词结果 (可选)
    difficulty: str = "unspecified"


# ── 数据加载 ──────────────────────────────────────────────────


def load_tables(tables_path: str) -> Dict[str, DatabaseSchema]:
    """加载 tables.json, 返回 db_id -> DatabaseSchema 的映射"""
    with open(tables_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    schemas: Dict[str, DatabaseSchema] = {}
    for db_entry in raw:
        db_id = db_entry["db_id"]
        schema = DatabaseSchema(db_id=db_id)

        # 构建列定义
        col_indices: Dict[int, ColumnDef] = {}
        col_names = db_entry.get("column_names", [])
        col_types = db_entry.get("column_types", [])
        col_names_orig = db_entry.get("column_names_original", col_names)

        for idx, (table_idx, col_name) in enumerate(col_names):
            col = ColumnDef(
                index=idx,
                name=col_name,
                name_original=col_names_orig[idx][1] if idx < len(col_names_orig) else col_name,
                ctype=col_types[idx] if idx < len(col_types) else "text",
                table_index=table_idx,
            )
            col_indices[idx] = col

        # 主键
        pk_indices = db_entry.get("primary_keys", [])
        for pk_idx in pk_indices:
            if pk_idx in col_indices:
                col_indices[pk_idx].is_primary = True

        # 构建表
        table_names = db_entry.get("table_names", [])
        table_names_orig = db_entry.get("table_names_original", table_names)
        for t_idx, t_name in enumerate(table_names):
            table = TableDef(
                index=t_idx,
                name=t_name,
                name_original=table_names_orig[t_idx] if t_idx < len(table_names_orig) else t_name,
                columns=[],
            )
            for col in col_indices.values():
                if col.table_index == t_idx:
                    table.columns.append(col)
            schema.tables[t_name.lower()] = table

        # 外键: (source_col, target_col)
        fk_raw = db_entry.get("foreign_keys", [])
        for src, tgt in fk_raw:
            schema.foreign_keys.append((src, tgt))

        schema.primary_keys = pk_indices
        schemas[db_id] = schema

    return schemas


def load_questions(json_path: str, tables: Dict[str, DatabaseSchema] = None) -> List[CspiderTestCase]:
    """加载 CSpider 的 JSON 问题文件 (train.json / dev.json)

    JSON 格式:
    [
        {
            "db_id": "academia",
            "query_id": 1,
            "question": "有多少个部门负责人?",
            "question_tok": ["有", "多少", "个", "部门", "负责人", "?"],
            "SQL": "SELECT count(*) FROM department ..."
        },
        ...
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cases = []
    for entry in raw:
        # CSpider: "query" 字段才是 SQL 文本, "sql" 是 parsed AST (dict)
        sql = entry.get("query") or entry.get("SQL") or ""
        # 去除 SQL 中的多余空白
        sql = " ".join(sql.split()) if isinstance(sql, str) and sql else ""

        case = CspiderTestCase(
            query_id=entry.get("query_id", 0),
            question=entry.get("question", ""),
            gold_sql=sql,
            db_id=entry.get("db_id", ""),
            question_tok=entry.get("question_tok"),
            difficulty=_infer_difficulty(sql, tables.get(entry.get("db_id", "")) if tables else None),
        )
        cases.append(case)

    return cases


def _infer_difficulty(sql: str, schema: Optional[DatabaseSchema] = None) -> str:
    """粗略推断 SQL 难度 (参考 Spider 的分级逻辑)"""
    if not sql:
        return "unspecified"

    upper = sql.upper()
    has_nested = "SELECT" in upper[upper.index("SELECT") + 6:] if "SELECT" in upper else False
    has_set_op = any(op in upper for op in ["UNION", "INTERSECT", "EXCEPT"])
    has_order = "ORDER BY" in upper
    has_group = "GROUP BY" in upper
    has_having = "HAVING" in upper
    has_limit = "LIMIT" in upper
    has_join = any(kw in upper for kw in ["JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN"])
    has_like = "LIKE" in upper
    has_agg = any(kw in upper for kw in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("])

    if has_nested or has_set_op:
        return "hard"
    if has_having or (has_join and has_agg) or (has_join and has_order and has_limit):
        return "medium"
    if has_join or has_group or has_order or has_agg:
        return "medium"
    return "easy"


def load_cspider_data(data_dir: str = CSPIDER_DEFAULT_DIR, split: str = "dev") -> List[CspiderTestCase]:
    """便捷加载: 返回指定分区的测试用例列表

    Args:
        data_dir: CSpider 数据目录
        split: "dev" (开发集, ~1034 条) 或 "train" (训练集, ~7000 条)

    Returns:
        List[CspiderTestCase]
    """
    tables_path = os.path.join(data_dir, "tables.json")
    questions_path = os.path.join(data_dir, f"{split}.json")

    if not os.path.exists(tables_path):
        logger.error("tables.json 未找到: %s", tables_path)
        return []
    if not os.path.exists(questions_path):
        logger.error("%s.json 未找到: %s", split, questions_path)
        return []

    tables = load_tables(tables_path)
    cases = load_questions(questions_path, tables)
    logger.info("CSpider %s 集加载完成: %d 条用例, %d 个数据库", split, len(cases), len(tables))
    return cases


def get_schema(db_id: str, tables: Dict[str, DatabaseSchema]) -> Optional[DatabaseSchema]:
    """按 db_id 获取数据库 Schema"""
    return tables.get(db_id)


def format_schema_for_prompt(schema: DatabaseSchema) -> str:
    """将 Schema 格式化为 LLM Prompt 可读的文本

    输出示例:
        CREATE TABLE user (user_id INT, name TEXT, age INT, PRIMARY KEY (user_id))
        CREATE TABLE department (department_id INT, name TEXT, PRIMARY KEY (department_id))
    """
    lines = []
    for table in schema.tables.values():
        col_defs = []
        for col in table.columns:
            col_str = f"{col.name_original} {col.ctype.upper()}"
            if col.is_primary:
                col_str += " PRIMARY KEY"
            col_defs.append(col_str)
        fk_refs = []
        for src, tgt in schema.foreign_keys:
            # 简单匹配 - 实际 project 中可完善
            pass
        lines.append(f"CREATE TABLE {table.name_original} ({', '.join(col_defs)});")
    return "\n".join(lines)


# ── SQLite → MySQL DDL 转换 ──────────────────────────────────


def convert_schema_to_mysql_ddl(schema: DatabaseSchema, db_name: str = None) -> str:
    """将 CSpider Schema 转为 MySQL CREATE DATABASE + CREATE TABLE 语句"""
    db_name = db_name or schema.db_id
    lines = [f"CREATE DATABASE IF NOT EXISTS `{db_name}` DEFAULT CHARACTER SET utf8mb4;",
             f"USE `{db_name}`;"]

    for table in schema.tables.values():
        col_defs = []
        for col in table.columns:
            mysql_type = _sqlite_to_mysql_type(col.ctype)
            col_str = f"  `{col.name_original}` {mysql_type}"
            if col.is_primary:
                col_str += " NOT NULL"
            col_defs.append(col_str)

        # 主键约束
        pk_cols = [f"`{schema.tables[t_name].columns[c_idx].name_original}`"
                    for t_name, t in schema.tables.items()
                    for c_idx, c in enumerate(t.columns) if c.is_primary]
        # 更准确的主键提取方式
        pk_cols = []
        for t in schema.tables.values():
            for c in t.columns:
                if c.is_primary:
                    pk_cols.append(f"`{c.name_original}`")

        pk_clause = f",\n  PRIMARY KEY ({', '.join(pk_cols)})" if pk_cols else ""

        # 外键约束
        fk_clauses = []
        for src_idx, tgt_idx in schema.foreign_keys:
            # 查找源列和目标列所属的表 (简化实现)
            src_col = None
            tgt_col = None
            for t in schema.tables.values():
                for c in t.columns:
                    if c.index == src_idx:
                        src_col = c
                    if c.index == tgt_idx:
                        tgt_col = c
            if src_col and tgt_col:
                src_table_name = None
                tgt_table_name = None
                for t_name, t in schema.tables.items():
                    if src_col in t.columns:
                        src_table_name = t.name_original
                    if tgt_col in t.columns:
                        tgt_table_name = t.name_original
                if src_table_name and tgt_table_name:
                    fk_clauses.append(
                        f"  FOREIGN KEY (`{src_col.name_original}`) "
                        f"REFERENCES `{tgt_table_name}`(`{tgt_col.name_original}`)"
                    )

        all_col_defs = ",\n".join(col_defs + ([pk_clause.lstrip(", \n")] if pk_clause else []))
        ddl = f"CREATE TABLE IF NOT EXISTS `{table.name_original}` (\n{all_col_defs}\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"

        # 追加外键 (写在表外)
        lines.append(ddl)

    return "\n\n".join(lines)


def _sqlite_to_mysql_type(sqlite_type: str) -> str:
    """SQLite 类型 → MySQL 类型映射"""
    t = sqlite_type.upper().strip()
    type_map = {
        "INTEGER": "INT",
        "INT": "INT",
        "BIGINT": "BIGINT",
        "TINYINT": "TINYINT",
        "SMALLINT": "SMALLINT",
        "REAL": "DOUBLE",
        "FLOAT": "FLOAT",
        "DOUBLE": "DOUBLE",
        "TEXT": "TEXT",
        "VARCHAR": "VARCHAR(255)",
        "CHAR": "CHAR(1)",
        "BLOB": "BLOB",
        "DATE": "DATE",
        "DATETIME": "DATETIME",
        "TIMESTAMP": "TIMESTAMP",
        "BOOLEAN": "TINYINT(1)",
        "DECIMAL": "DECIMAL(10,2)",
        "NUMERIC": "DECIMAL(10,2)",
    }
    # 处理带精度的情况如 VARCHAR(100)
    if "(" in t:
        base = t.split("(")[0]
        if base in type_map and base in ("VARCHAR", "CHAR", "DECIMAL"):
            return t  # 保留原始精度
    return type_map.get(t, "TEXT")


# ── 统计信息 ──────────────────────────────────────────────────


def print_dataset_stats(cases: List[CspiderTestCase], tables: Dict[str, DatabaseSchema]):
    """打印数据集统计信息"""
    from collections import Counter
    logger.info("=" * 50)
    logger.info("CSpider 数据集统计")
    logger.info("=" * 50)
    logger.info("总用例数: %d", len(cases))
    logger.info("数据库数: %d", len(tables))

    difficulty_counter = Counter(c.difficulty for c in cases)
    logger.info("难度分布:")
    for diff, cnt in difficulty_counter.most_common():
        logger.info("  %s: %d (%.1f%%)", diff, cnt, cnt / len(cases) * 100)

    db_counter = Counter(c.db_id for c in cases)
    logger.info("Top 10 数据库 (按用例数):")
    for db_id, cnt in db_counter.most_common(10):
        logger.info("  %s: %d", db_id, cnt)
    logger.info("=" * 50)


# ── Gold SQL 方言转换 (Spider/SQLite → MySQL) ──────────────


def convert_sql_to_mysql(sql: str, dialect: str = "sqlite") -> str:
    """使用 sqlglot 将 Spider/SQLite 方言的 SQL 转为 MySQL 方言

    处理的差异包括:
      - || 字符串拼接 → CONCAT()
      - CAST(x AS INTEGER) → CAST(x AS SIGNED)
      - INSTR() 行为差异
      - LIMIT/OFFSET 语法差异 (MySQL 5.x vs 8.x)
      - 双引号字符串 → 单引号

    Args:
        sql: 原始 SQL 语句
        dialect: 源方言 ("sqlite" | "spider")

    Returns:
        MySQL 方言的 SQL 语句, 转换失败时返回原始 SQL
    """
    try:
        import sqlglot
        result = sqlglot.transpile(sql, read=dialect, write="mysql")
        if result and result[0]:
            return result[0].sql(dialect="mysql")
        return sql
    except ImportError:
        logger.warning("sqlglot 未安装, 跳过方言转换")
        return sql
    except Exception as e:
        logger.debug("SQL 方言转换失败: %s | SQL: %s", e, sql[:80])
        return sql


def batch_convert_sqls(
    cases: list,
    source_field: str = "gold_sql",
    target_field: str = "gold_sql_mysql",
    dialect: str = "sqlite",
) -> list:
    """批量转换测试用例中的 SQL 方言

    Args:
        cases: 测试用例列表 (支持 getattr/setattr 的 dataclass)
        source_field: 源 SQL 字段名
        target_field: 目标 SQL 字段名
        dialect: 源方言

    Returns:
        转换后的测试用例列表 (原地修改)
    """
    success = 0
    for case in cases:
        sql = getattr(case, source_field, "")
        if not sql:
            continue
        mysql_sql = convert_sql_to_mysql(sql, dialect)
        setattr(case, target_field, mysql_sql)
        if mysql_sql != sql:
            success += 1
    logger.info("SQL 方言转换完成: %d/%d 条已转换", success, len(cases))
    return cases


# ── 入口 ──────────────────────────────────────────────────────


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cases = load_cspider_data(split="dev")
    if cases:
        tables_dict = load_tables(os.path.join(CSPIDER_DEFAULT_DIR, "tables.json"))
        print_dataset_stats(cases, tables_dict)

        # 打印第一条样例
        sample = cases[0]
        logger.info("\n样例 #1:")
        logger.info("  问题: %s", sample.question)
        logger.info("  SQL:  %s", sample.gold_sql)
        logger.info("  数据库: %s", sample.db_id)
        logger.info("  难度: %s", sample.difficulty)

        if sample.db_id in tables_dict:
            ddl = convert_schema_to_mysql_ddl(tables_dict[sample.db_id])
            logger.info("\nMySQL DDL 示例:\n%s", ddl)