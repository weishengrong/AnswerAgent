"""
数据库连接器 - 从 MySQL information_schema 读取表结构

直连 MySQL 数据库自动读取 schema，替代静态文件解析。
输出与 DocxParser/SQLParser 相同的 TableInfo 对象。

策略：
1. 通过 information_schema 查询表结构（表名、字段、外键）
2. 优先使用 COLUMN_COMMENT，缺少时用命名约定推断
3. 外键优先从 KEY_COLUMN_USAGE 读取，再通过 _id 命名约定补充
4. 可选调用 LLM 补全缺少描述的字段
5. 支持 schema 指纹检测：通过表名+字段名的 hash 判断是否需要重建索引
"""
import hashlib
import json
import logging
from typing import List, Dict, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from utils.docx.parser import TableInfo, FieldInfo, Relation
from utils.db.sql_parser import SQLParser

logger = logging.getLogger(__name__)

# Redis key for storing the last schema fingerprint
SCHEMA_FINGERPRINT_KEY = "rag:schema_fingerprint"


class DBConnector:
    """从 MySQL information_schema 读取表结构"""

    def __init__(self, database_url: str = None):
        """
        初始化数据库连接器

        Args:
            database_url: 数据库连接字符串，为空时从 config.db_config 导入
        """
        if database_url:
            self.database_url = database_url
        else:
            from app.core.config.db_config import database_url, AsyncSessionLocal
            self.database_url = database_url
            self.AsyncSessionLocal = AsyncSessionLocal

        self._sql_parser = SQLParser()
        self._llm = None

    async def _get_session(self) -> AsyncSession:
        """获取数据库会话"""
        if hasattr(self, 'AsyncSessionLocal'):
            return self.AsyncSessionLocal()

        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        engine = create_async_engine(self.database_url, echo=False)
        session_factory = async_sessionmaker(bind=engine)
        return session_factory()

    async def load_schema(self, enrich: bool = False) -> List[TableInfo]:
        """
        主入口：读取完整 schema 返回 TableInfo 列表

        步骤：
        1. 连接数据库
        2. 从 information_schema.TABLES 获取所有表名和 TABLE_COMMENT
        3. 对每张表，从 information_schema.COLUMNS 获取字段详情
        4. 从 information_schema.KEY_COLUMN_USAGE 获取外键关系
        5. 根据 table_name 推断 table_type
        6. 如果 enrich=True，调用 LLM 补全缺少描述的字段
        7. 基于 _id 后缀命名约定推断额外外键关系

        Args:
            enrich: 是否使用 LLM 补全缺少的描述

        Returns:
            List[TableInfo]
        """
        logger.info("开始从数据库加载 schema...")

        # 步骤1-2：获取所有表
        tables_info = await self._fetch_tables()
        logger.info(f"发现 {len(tables_info)} 张表")

        # 步骤3：获取每张表的字段和外键
        result_tables = []
        for table_meta in tables_info:
            table_name = table_meta['TABLE_NAME']
            table_comment = table_meta.get('TABLE_COMMENT', '') or ''

            # 获取字段
            columns = await self._fetch_columns(table_name)

            # 构建字段列表
            fields = []
            for col in columns:
                field_desc = col.get('COLUMN_COMMENT', '') or ''

                # 如果没有注释，用命名约定推断
                if not field_desc:
                    field_desc = self._sql_parser._infer_field_comment(
                        col['COLUMN_NAME'],
                        col['COLUMN_TYPE']
                    )

                field_info = FieldInfo(
                    name=col['COLUMN_NAME'],
                    type=col['COLUMN_TYPE'],
                    desc=field_desc,
                    nullable=col.get('IS_NULLABLE', 'YES') == 'YES'
                )
                fields.append(field_info)

            # 步骤4：获取外键关系
            relations = []
            foreign_keys = await self._fetch_foreign_keys(table_name)

            for fk in foreign_keys:
                relation = Relation(
                    target_table=fk['REFERENCED_TABLE_NAME'],
                    relation_type="多对一",
                    via_field=fk['COLUMN_NAME'],
                    via=f"{fk['COLUMN_NAME']} → {fk.get('REFERENCED_COLUMN_NAME', 'id')}"
                )
                relations.append(relation)

            # 步骤5：推断表类型
            table_type = self._infer_table_type(table_name)

            table_info = TableInfo(
                table_name=table_name,
                table_type=table_type,
                table_desc=table_comment,
                fields=fields,
                relations=relations
            )

            result_tables.append(table_info)

        # 步骤7：基于 _id 后缀推断额外外键关系
        result_tables = self._sql_parser._infer_relations(result_tables)

        # 步骤6：可选的 LLM 补全
        if enrich:
            result_tables = await self._sql_parser.enrich_with_llm(result_tables)

        logger.info(f"Schema 加载完成：{len(result_tables)} 张表")
        return result_tables

    async def _fetch_tables(self) -> List[Dict]:
        """
        查询 information_schema.TABLES，排除系统表

        Returns:
            表信息列表，包含 TABLE_NAME 和 TABLE_COMMENT
        """
        session = await self._get_session()
        try:
            query = text("""
                SELECT TABLE_NAME, TABLE_COMMENT
                FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_TYPE = 'BASE TABLE'
                  AND TABLE_NAME NOT IN ('sys', 'mysql', 'information_schema', 'performance_schema')
                ORDER BY TABLE_NAME
            """)

            result = await session.execute(query)
            rows = result.fetchall()

            tables = []
            for row in rows:
                tables.append({
                    'TABLE_NAME': row[0],
                    'TABLE_COMMENT': row[1] or ''
                })

            return tables
        finally:
            await session.close()

    async def _fetch_columns(self, table_name: str) -> List[Dict]:
        """
        查询单张表的 information_schema.COLUMNS

        Args:
            table_name: 表名

        Returns:
            字段信息列表
        """
        session = await self._get_session()
        try:
            query = text("""
                SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_COMMENT,
                       CHARACTER_MAXIMUM_LENGTH, ORDINAL_POSITION
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table_name
                ORDER BY ORDINAL_POSITION
            """)

            result = await session.execute(query, {'table_name': table_name})
            rows = result.fetchall()

            columns = []
            for row in rows:
                columns.append({
                    'COLUMN_NAME': row[0],
                    'COLUMN_TYPE': row[1],
                    'IS_NULLABLE': row[2],
                    'COLUMN_COMMENT': row[3] or '',
                    'CHARACTER_MAXIMUM_LENGTH': row[4],
                    'ORDINAL_POSITION': row[5]
                })

            return columns
        finally:
            await session.close()

    async def _fetch_foreign_keys(self, table_name: str) -> List[Dict]:
        """
        查询单张表的 information_schema.KEY_COLUMN_USAGE 中的外键

        Args:
            table_name: 表名

        Returns:
            外键信息列表
        """
        session = await self._get_session()
        try:
            query = text("""
                SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME, CONSTRAINT_NAME
                FROM information_schema.KEY_COLUMN_USAGE
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = :table_name
                  AND REFERENCED_TABLE_NAME IS NOT NULL
            """)

            result = await session.execute(query, {'table_name': table_name})
            rows = result.fetchall()

            fks = []
            for row in rows:
                fks.append({
                    'COLUMN_NAME': row[0],
                    'REFERENCED_TABLE_NAME': row[1],
                    'REFERENCED_COLUMN_NAME': row[2],
                    'CONSTRAINT_NAME': row[3]
                })

            return fks
        finally:
            await session.close()

    def _infer_table_type(self, table_name: str) -> str:
        """
        根据表名推断表类型

        Args:
            table_name: 表名

        Returns:
            表类型字符串
        """
        name_lower = table_name.lower()

        if any(kw in name_lower for kw in ["log", "history", "record", "audit"]):
            return "日志表"
        elif any(kw in name_lower for kw in ["config", "setting", "dict", "type"]):
            return "配置表"
        elif any(kw in name_lower for kw in ["rel", "map", "ref", "link"]):
            return "关联表"
        else:
            return "业务表"

    async def get_schema_fingerprint(self) -> str:
        """
        计算当前数据库 schema 的指纹（MD5 hash）

        基于：所有表名 + 每张表的字段名列表，排序后生成 hash。
        字段类型变更、COMMENT 变更不会触发重建（不影响检索结构），
        只有新增/删除表或字段才会触发。

        Returns:
            32位 MD5 hex 字符串
        """
        tables_info = await self._fetch_tables()

        # 收集所有表名和字段名，排序后生成确定性字符串
        schema_parts = []
        for t in sorted(tables_info, key=lambda x: x['TABLE_NAME']):
            table_name = t['TABLE_NAME']
            columns = await self._fetch_columns(table_name)
            field_names = sorted([c['COLUMN_NAME'] for c in columns])
            schema_parts.append(f"{table_name}:{','.join(field_names)}")

        schema_str = "|".join(schema_parts)
        fingerprint = hashlib.md5(schema_str.encode('utf-8')).hexdigest()

        logger.debug(f"Schema 指纹计算完成：{fingerprint} （基于 {len(tables_info)} 张表）")
        return fingerprint

    async def save_fingerprint(self, redis_client=None) -> None:
        """
        将当前 schema 指纹保存到 Redis

        Args:
            redis_client: 可选的 Redis 客户端实例
        """
        fingerprint = await self.get_schema_fingerprint()

        if redis_client is None:
            import redis.asyncio as aioredis
            from app.core.config.settings import redis_settings
            pwd = f":{redis_settings.REDIS_PASSWORD}@" if redis_settings.REDIS_PASSWORD else ""
            redis_url = f"redis://{pwd}{redis_settings.REDIS_HOST}:{redis_settings.REDIS_PORT}/{redis_settings.REDIS_DB}"
            redis_client = aioredis.from_url(redis_url)

        try:
            await redis_client.set(SCHEMA_FINGERPRINT_KEY, fingerprint)
            logger.info(f"✅ Schema 指纹已保存：{fingerprint}")
        except Exception as e:
            logger.warning(f"⚠️ 保存 Schema 指纹失败：{e}")

    @staticmethod
    async def load_stored_fingerprint(redis_client=None) -> Optional[str]:
        """
        从 Redis 加载上次存储的 schema 指纹

        Args:
            redis_client: 可选的 Redis 客户端实例

        Returns:
            存储的指纹字符串，不存在则返回 None
        """
        if redis_client is None:
            import redis.asyncio as aioredis
            from app.core.config.settings import redis_settings
            pwd = f":{redis_settings.REDIS_PASSWORD}@" if redis_settings.REDIS_PASSWORD else ""
            redis_url = f"redis://{pwd}{redis_settings.REDIS_HOST}:{redis_settings.REDIS_PORT}/{redis_settings.REDIS_DB}"
            redis_client = aioredis.from_url(redis_url)

        try:
            stored = await redis_client.get(SCHEMA_FINGERPRINT_KEY)
            if stored:
                return stored.decode('utf-8') if isinstance(stored, bytes) else stored
            return None
        except Exception as e:
            logger.warning(f"⚠️ 读取 Schema 指纹失败：{e}")
            return None

    async def schema_changed(self) -> bool:
        """
        判断数据库 schema 是否发生了变化

        对比当前数据库的 schema 指纹与 Redis 中存储的上次指纹。

        Returns:
            True 表示 schema 有变化需要重建索引，False 表示无变化
        """
        current_fp = await self.get_schema_fingerprint()
        stored_fp = await self.load_stored_fingerprint()

        if stored_fp is None:
            logger.info("🆕 未找到历史 Schema 指纹，视为首次构建")
            return True

        if current_fp != stored_fp:
            logger.info(f"🔄 Schema 已变更（旧：{stored_fp} → 新：{current_fp}），需要重建索引")
            return True

        logger.debug(f"✅ Schema 未变（{current_fp}），无需重建索引")
        return False


async def load_db_schema(enrich: bool = False) -> List[TableInfo]:
    """
    便捷函数：加载数据库 schema

    Args:
        enrich: 是否使用 LLM 补全缺少的描述

    Returns:
        List[TableInfo]
    """
    connector = DBConnector()
    return await connector.load_schema(enrich=enrich)


if __name__ == "__main__":
    import asyncio

    async def test():
        connector = DBConnector()
        tables = await connector.load_schema(enrich=False)

        print(f"\n{'='*60}")
        print(f"共加载 {len(tables)} 张表")
        print(f"{'='*60}")

        for table in tables[:5]:  # 只显示前5张表
            print(f"\n表名: {table.table_name} ({table.table_type})")
            print(f"描述: {table.table_desc}")
            print(f"字段数: {len(table.fields)}, 关系数: {len(table.relations)}")

            for field in table.fields[:8]:  # 每张表只显示前8个字段
                nullable_str = "" if field.nullable else " [NOT NULL]"
                print(f"  - {field.name} ({field.type}){nullable_str}: {field.desc}")
            if len(table.fields) > 8:
                print(f"  ... 还有 {len(table.fields) - 8} 个字段")

            for rel in table.relations:
                print(f"  → {rel.target_table} ({rel.relation_type}) via {rel.via}")

    asyncio.run(test())
