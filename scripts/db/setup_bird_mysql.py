"""
BIRD-SQL Mini-Dev MySQL 建库脚本

从 dev_databases/ 下的 SQLite 数据库文件,
自动生成 MySQL DDL + INSERT 语句并导入。

用法:
  python -m scripts.db.setup_bird_mysql                          # 生成 SQL 文件, 手动导入
  python -m scripts.db.setup_bird_mysql --direct                 # 直接导入 MySQL (需 mysql CLI)
"""

import argparse
import logging
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger(__name__)

# ── 路径配置 ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "minidev", "MINIDEV")
DEV_DATABASES = os.path.join(DATA_DIR, "dev_databases")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "bird_mysql")

# ── SQLite 类型 → MySQL 类型映射 ──────────────────────────


SQLITE_TO_MYSQL = {
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
    "DECIMAL": "DECIMAL(10, 2)",
    "NUMERIC": "DECIMAL(10, 2)",
}


def sqlite_to_mysql_type(t: str) -> str:
    """SQLite 类型 → MySQL 类型"""
    base = t.upper().strip().split("(")[0]
    if "(" in t and base in ("VARCHAR", "CHAR", "DECIMAL"):
        return t
    return SQLITE_TO_MYSQL.get(base, "TEXT")


def sanitize_value(val: Any) -> str:
    """将 SQLite 值转为 MySQL 字面量"""
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return str(val)
    if isinstance(val, str):
        escaped = val.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(val, bytes):
        escaped = val.hex()
        return f"x'{escaped}'"
    if isinstance(val, datetime):
        return f"'{val.strftime('%Y-%m-%d %H:%M:%S')}'"
    return f"'{str(val)}'"


def extract_sqlite_schema(sqlite_path: str) -> List[Dict]:
    """从 SQLite 文件中提取所有表的 CREATE TABLE 语句"""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = []
    for name, sql in cursor.fetchall():
        if name.startswith("sqlite_"):
            continue
        tables.append({"name": name, "sql": sql})

    conn.close()
    return tables


def extract_sqlite_data(sqlite_path: str, table_name: str) -> Tuple[List[str], List[List[Any]]]:
    """从 SQLite 表中提取所有数据"""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    cursor.execute(f'SELECT * FROM "{table_name}"')
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    conn.close()
    return columns, rows


def generate_mysql_ddl(table: Dict, sql_schema: str) -> str:
    """将 SQLite CREATE TABLE 转为 MySQL DDL"""
    if not sql_schema:
        return f"-- (no schema for {table['name']})"

    # 简单替换类型
    mysql_sql = sql_schema
    for sqlite_type, mysql_type in sorted(SQLITE_TO_MYSQL.items(), key=lambda x: -len(x[0])):
        mysql_sql = mysql_sql.replace(sqlite_type, mysql_type)

    # SQLite 特有语法 → MySQL
    mysql_sql = mysql_sql.replace("AUTOINCREMENT", "AUTO_INCREMENT")
    mysql_sql = mysql_sql.replace("autoincrement", "AUTO_INCREMENT")
    mysql_sql = mysql_sql.replace("AUTO INCREMENT", "AUTO_INCREMENT")
    # MySQL: AUTO_INCREMENT 必须在 PRIMARY KEY 之前
    mysql_sql = mysql_sql.replace("primary key AUTO_INCREMENT", "AUTO_INCREMENT PRIMARY KEY")
    mysql_sql = mysql_sql.replace("AUTO_INCREMENT primary key", "AUTO_INCREMENT PRIMARY KEY")

    # 确保 DDL 以分号结尾
    mysql_sql = mysql_sql.rstrip() + ";"

    return mysql_sql


def generate_insert_sql(table_name: str, columns: List[str], rows: List[List[Any]]) -> str:
    """生成 MySQL INSERT 语句 (分批生成, 每批 50 行)"""
    if not rows:
        return ""

    col_names = ", ".join(f"`{c}`" for c in columns)
    chunks = []
    batch_size = 50

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        values = []
        for row in batch:
            vals = ", ".join(sanitize_value(v) for v in row)
            values.append(f"({vals})")
        chunks.append(
            f"INSERT INTO `{table_name}` ({col_names}) VALUES\n" +
            ",\n".join(values) + ";"
        )

    return "\n".join(chunks)


def export_database(db_name: str, sqlite_path: str, output_dir: str) -> str:
    """将一个 SQLite 数据库导出为 MySQL SQL 文件

    Returns:
        SQL 文件路径
    """
    tables = extract_sqlite_schema(sqlite_path)
    os.makedirs(output_dir, exist_ok=True)

    lines = [
        f"-- BIRD-SQL Mini-Dev: {db_name}",
        f"-- Source: {sqlite_path}",
        f"-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"CREATE DATABASE IF NOT EXISTS `{db_name}` DEFAULT CHARACTER SET utf8mb4;",
        f"USE `{db_name}`;",
        "",
        "SET FOREIGN_KEY_CHECKS = 0;",
        "",
    ]

    for table in tables:
        ddl = generate_mysql_ddl(table, table["sql"])
        lines.append(ddl)
        lines.append("")

        columns, rows = extract_sqlite_data(sqlite_path, table["name"])
        insert_sql = generate_insert_sql(table["name"], columns, rows)
        if insert_sql:
            lines.append(insert_sql)
            lines.append("")

    lines.append("SET FOREIGN_KEY_CHECKS = 1;")
    lines.append("")

    output_path = os.path.join(output_dir, f"{db_name}.sql")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    table_count = len(tables)
    row_count = sum(extract_sqlite_data(sqlite_path, t["name"])[1].__len__() for t in tables)
    logger.info("  ✔ %s: %d 张表, %d 行 → %s", db_name, table_count, row_count, output_path)
    return output_path


def export_all(output_dir: str) -> List[str]:
    """导出所有 11 个数据库"""
    os.makedirs(output_dir, exist_ok=True)
    db_dirs = sorted(os.listdir(DEV_DATABASES))
    exported_files = []

    for db_name in db_dirs:
        db_path = os.path.join(DEV_DATABASES, db_name)
        if not os.path.isdir(db_path):
            continue

        # 找 .sqlite 文件
        sqlite_files = [f for f in os.listdir(db_path) if f.endswith((".sqlite", ".db"))]
        if not sqlite_files:
            logger.warning("  ⚠ %s: 未找到 SQLite 文件", db_name)
            continue

        sqlite_path = os.path.join(db_path, sqlite_files[0])
        try:
            path = export_database(db_name, sqlite_path, output_dir)
            exported_files.append(path)
        except Exception as e:
            logger.error("  ✗ %s: 导出失败 - %s", db_name, e)

    return exported_files


def import_to_mysql(sql_dir: str, host: str, port: int, user: str, password: str):
    """将生成的 SQL 文件导入 MySQL"""
    mysql_args = ["mysql", f"-h{host}", f"-P{port}", f"-u{user}"]
    if password:
        mysql_args.append(f"-p{password}")

    sql_files = sorted(os.listdir(sql_dir))
    for sql_file in sql_files:
        if not sql_file.endswith(".sql"):
            continue
        db_name = sql_file.replace(".sql", "")
        file_path = os.path.join(sql_dir, sql_file)

        logger.info("导入 %s...", db_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                result = subprocess.run(
                    mysql_args,
                    stdin=f,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            if result.returncode == 0:
                logger.info("  ✔ %s 导入成功", db_name)
            else:
                logger.error("  ✗ %s 导入失败:\n%s", db_name, result.stderr[:500])
        except subprocess.TimeoutExpired:
            logger.error("  ✗ %s 导入超时", db_name)
        except FileNotFoundError:
            logger.error("  ✗ 未找到 mysql CLI, 请手动导入")
            return


def main():
    parser = argparse.ArgumentParser(description="BIRD-SQL Mini-Dev MySQL 建库工具")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="导出后直接导入 MySQL (需 mysql CLI 在 PATH 中)",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DIR,
        help=f"SQL 文件输出目录 (默认: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="MySQL 主机 (默认: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3306,
        help="MySQL 端口 (默认: 3306)",
    )
    parser.add_argument(
        "--user",
        default="root",
        help="MySQL 用户 (默认: root)",
    )
    parser.add_argument(
        "--password",
        default="",
        help="MySQL 密码 (默认: 空)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    logger.info("=" * 50)
    logger.info("BIRD-SQL Mini-Dev MySQL 建库")
    logger.info("=" * 50)

    logger.info("正在从 SQLite 导出 %d 个数据库...", len(os.listdir(DEV_DATABASES)))
    files = export_all(args.output)

    logger.info("\n导出完成: %d 个 SQL 文件 → %s", len(files), args.output)

    for f in files:
        size = os.path.getsize(f)
        logger.info("  %s (%.1f MB)", os.path.basename(f), size / 1024 / 1024)

    if args.direct:
        logger.info("\n正在导入 MySQL (%s:%d)...", args.host, args.port)
        import_to_mysql(args.output, args.host, args.port, args.user, args.password)

    logger.info("\n%s", "=" * 50)
    logger.info("完成!")
    logger.info("手动导入: 用 Navicat 逐个运行 %s/*.sql 文件", args.output)
    logger.info("自动导入: python -m scripts.db.setup_bird_mysql --direct")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()