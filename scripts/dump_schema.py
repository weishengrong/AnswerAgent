"""
导出数据库 DDL 脚本

从 MySQL 数据库导出所有表的 CREATE TABLE 语句，
保存为 .sql 文件供 SQLParser 使用。

用法：
  python scripts/dump_schema.py
  python scripts/dump_schema.py --output resource/schema.sql
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def dump_schema(output_path: str = "resource/schema.sql"):
    """从数据库导出所有表的 DDL"""
    from sqlalchemy import text
    from config.db_config import AsyncSessionLocal

    print(f"开始导出数据库 DDL...")

    async with AsyncSessionLocal() as db:
        result = await db.execute(text("SHOW TABLES"))
        tables = [row[0] for row in result.fetchall()]

    print(f"发现 {len(tables)} 张表")

    ddl_statements = []

    for table_name in tables:
        async with AsyncSessionLocal() as db:
            result = await db.execute(text(f"SHOW CREATE TABLE `{table_name}`"))
            row = result.fetchone()
            if row:
                create_sql = row[1]
                ddl_statements.append(f"{create_sql};")
                print(f"  ✅ {table_name}")

    ddl_content = "\n\n".join(ddl_statements)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ddl_content)

    print(f"\n✅ DDL 已保存到: {output_path}")
    print(f"   共 {len(ddl_statements)} 张表, {len(ddl_content)} 字符")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="导出数据库 DDL")
    parser.add_argument("--output", default="resource/schema.sql", help="输出文件路径")
    args = parser.parse_args()

    asyncio.run(dump_schema(args.output))
