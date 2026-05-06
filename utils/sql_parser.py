"""
SQL DDL 解析器 - 从 SQL DDL 文件解析表结构

替代 DocxParser，从 SQL 文件自动构建 RAG 索引。
输出与 DocxParser 相同的 TableInfo 对象，下游 ChunkBuilder 无需修改。

策略：
1. 用 sqlglot 精准解析 DDL 结构（表名、字段、类型、外键、COMMENT）
2. 有 COMMENT 的字段直接使用
3. 缺少 COMMENT 的字段/表，批量调用 LLM 推断业务含义
"""
import json
import logging
from typing import List, Optional

from utils.docx_parser import TableInfo, FieldInfo, Relation

logger = logging.getLogger(__name__)


class SQLParser:
    def __init__(self):
        self._llm = None

    def parse(self, sql_content: str) -> List[TableInfo]:
        """
        从 SQL DDL 内容解析表结构

        Args:
            sql_content: SQL DDL 语句（可以包含多条 CREATE TABLE）

        Returns:
            List[TableInfo] - 与 DocxParser 输出格式一致
        """
        try:
            import sqlglot
        except ImportError:
            raise ImportError("请安装 sqlglot: pip install sqlglot")

        tables = []

        statements = sql_content.split(";")
        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue

            try:
                parsed = sqlglot.parse_one(stmt, dialect="mysql")
            except Exception:
                continue

            if not parsed:
                continue

            table_info = self._parse_create_table(parsed)
            if table_info:
                tables.append(table_info)

        tables = self._infer_relations(tables)

        logger.info(f"SQL DDL 解析完成：{len(tables)} 张表")
        return tables

    def _parse_create_table(self, parsed) -> Optional[TableInfo]:
        """解析单条 CREATE TABLE 语句"""
        import sqlglot.expressions as exp

        create = parsed.find(exp.Create)
        if not create:
            return None

        table_expr = create.find(exp.Table)
        if not table_expr:
            return None

        table_name = table_expr.name

        table_comment = ""
        table_props = create.find(exp.Properties)
        if table_props:
            for prop in table_props.expressions:
                if hasattr(prop, 'name') and prop.name and 'comment' in prop.name.lower():
                    if hasattr(prop, 'value') and prop.value:
                        table_comment = prop.value.strip("'\"")

        fields = []
        relations = []

        for column in create.find_all(exp.ColumnDef):
            field_info = self._parse_column(column, table_name)
            if field_info:
                fields.append(field_info)

        for constraint in create.find_all(exp.ForeignKey):
            rel = self._parse_foreign_key(constraint, table_name)
            if rel:
                relations.append(rel)

        table_type = "业务表"
        if any(kw in table_name.lower() for kw in ["log", "history", "record", "audit"]):
            table_type = "日志表"
        elif any(kw in table_name.lower() for kw in ["config", "setting", "dict", "type"]):
            table_type = "配置表"
        elif any(kw in table_name.lower() for kw in ["rel", "map", "ref", "link"]):
            table_type = "关联表"

        table_info = TableInfo(
            table_name=table_name,
            table_type=table_type,
            table_desc=table_comment or "",
            fields=fields,
            relations=relations
        )

        return table_info

    def _parse_column(self, column, table_name: str) -> Optional[FieldInfo]:
        """解析单个字段定义"""
        import sqlglot.expressions as exp

        col_name = column.name
        if not col_name:
            return None

        col_type = "unknown"
        kind = column.args.get("kind")
        if kind:
            col_type = kind.sql(dialect="mysql")

        comment = ""
        constraints = column.args.get("constraints", [])
        nullable = True

        for constraint in constraints:
            constraint_sql = constraint.sql(dialect="mysql").upper()

            if "NOT NULL" in constraint_sql:
                nullable = False

            if "COMMENT" in constraint_sql:
                comment_match = constraint_sql.split("COMMENT")
                if len(comment_match) > 1:
                    comment = comment_match[1].strip().strip("'\"")

            if "AUTO_INCREMENT" in constraint_sql:
                if not comment:
                    comment = "主键自增"

        if not comment:
            comment = self._infer_field_comment(col_name, col_type)

        return FieldInfo(
            name=col_name,
            type=col_type,
            desc=comment,
            nullable=nullable
        )

    def _infer_field_comment(self, col_name: str, col_type: str) -> str:
        """根据字段名和类型推断字段描述"""
        name_lower = col_name.lower()

        common_fields = {
            "id": "主键ID",
            "name": "名称",
            "title": "标题",
            "code": "编码",
            "number": "编号",
            "desc": "描述",
            "description": "描述",
            "status": "状态",
            "type": "类型",
            "sort": "排序",
            "order": "排序",
            "create_time": "创建时间",
            "created_at": "创建时间",
            "update_time": "更新时间",
            "updated_at": "更新时间",
            "deleted_at": "删除时间",
            "is_deleted": "是否删除",
            "email": "邮箱",
            "phone": "手机号",
            "password": "密码",
            "address": "地址",
            "ip": "IP地址",
            "url": "URL地址",
            "image_url": "图片URL",
            "video_url": "视频URL",
            "sex": "性别",
            "age": "年龄",
            "birth": "出生日期",
            "user_id": "用户ID",
            "dept_id": "部门ID",
            "department_id": "部门ID",
            "device_id": "设备ID",
            "check_time": "打卡时间",
        }

        if name_lower in common_fields:
            return common_fields[name_lower]

        for suffix, desc in [("_id", "ID"), ("_name", "名称"), ("_type", "类型"),
                             ("_time", "时间"), ("_date", "日期"), ("_count", "数量"),
                             ("_url", "URL"), ("_code", "编码")]:
            if name_lower.endswith(suffix):
                prefix = name_lower[:-len(suffix)]
                return f"{prefix}{desc}"

        return col_name

    def _parse_foreign_key(self, constraint, table_name: str) -> Optional[Relation]:
        """解析外键约束"""
        import sqlglot.expressions as exp

        try:
            fk_columns = []
            for col in constraint.find_all(exp.Column):
                fk_columns.append(col.name)

            ref_table_expr = constraint.find(exp.Table)
            if not ref_table_expr:
                return None

            ref_table = ref_table_expr.name

            ref_columns = []
            for col in constraint.find_all(exp.Column):
                if col.table == ref_table:
                    ref_columns.append(col.name)

            via_field = fk_columns[0] if fk_columns else ""
            via = f"{via_field} → {ref_columns[0] if ref_columns else 'id'}"

            return Relation(
                target_table=ref_table,
                relation_type="多对一",
                via_field=via_field,
                via=via
            )
        except Exception as e:
            logger.debug(f"解析外键失败: {e}")
            return None

    def _infer_relations(self, tables: List[TableInfo]) -> List[TableInfo]:
        """基于命名约定自动推断外键关系

        规则：字段名以 _id 结尾，去掉 _id 后能匹配到已存在的表名，
        则推断该字段是外键，关联到对应表的 id 字段。

        例如：department_id → department 表, user_id → user 表
        """
        table_names = {t.table_name.lower(): t.table_name for t in tables}
        existing_relations = set()
        for table in tables:
            for rel in table.relations:
                existing_relations.add(f"{table.table_name}.{rel.via_field}")

        inferred_count = 0
        for table in tables:
            for field in table.fields:
                fk_key = f"{table.table_name}.{field.name}"
                if fk_key in existing_relations:
                    continue

                if not field.name.lower().endswith("_id"):
                    continue

                potential_table = field.name[:-3].lower()

                if potential_table in table_names:
                    target_table_name = table_names[potential_table]
                    relation = Relation(
                        target_table=target_table_name,
                        relation_type="多对一",
                        via_field=field.name,
                        via=f"{field.name} → id"
                    )
                    table.relations.append(relation)
                    existing_relations.add(fk_key)
                    inferred_count += 1
                    logger.debug(f"推断外键：{table.table_name}.{field.name} → {target_table_name}.id")

        if inferred_count > 0:
            logger.info(f"✅ 自动推断 {inferred_count} 个外键关系")
        else:
            logger.info("未发现可推断的外键关系")

        return tables

    async def enrich_with_llm(self, tables: List[TableInfo]) -> List[TableInfo]:
        """
        LLM 批量补全表/字段描述

        对缺少描述的表和字段，批量调用 LLM 推断业务含义。
        """
        tables_need_enrich = []
        for table in tables:
            needs_enrich = False
            if not table.table_desc:
                needs_enrich = True
            for field in table.fields:
                if field.desc == field.name or not field.desc:
                    needs_enrich = True
                    break
            if needs_enrich:
                tables_need_enrich.append(table)

        if not tables_need_enrich:
            logger.info("所有表和字段都已有描述，无需 LLM 补全")
            return tables

        llm = self._get_llm()

        for table in tables_need_enrich:
            fields_desc = []
            for field in table.fields:
                if field.desc and field.desc != field.name:
                    fields_desc.append(f"  {field.name} ({field.type}): {field.desc}")
                else:
                    fields_desc.append(f"  {field.name} ({field.type}): ???")

            prompt = f"""请根据以下数据库表的字段列表，推断表描述和缺少的字段描述。

表名：{table.table_name}
表类型：{table.table_type}
当前表描述：{table.table_desc or "无"}
字段列表：
{chr(10).join(fields_desc)}

请按以下JSON格式返回：
{{
    "table_desc": "表的业务描述",
    "fields": {{
        "字段名": "字段业务描述"
    }}
}}

注意：
1. 只需要填写缺少描述的字段（标记为???的）
2. 描述要简洁，10字以内
3. 不要输出其他内容"""

            try:
                result = await llm.run(prompt)
                output = result.output.strip()

                data = json.loads(output)

                if not table.table_desc and data.get("table_desc"):
                    table.table_desc = data["table_desc"]

                if data.get("fields"):
                    for field in table.fields:
                        if (field.desc == field.name or not field.desc) and field.name in data["fields"]:
                            field.desc = data["fields"][field.name]

                logger.info(f"✅ LLM 补全表 {table.table_name} 描述成功")

            except Exception as e:
                logger.warning(f"⚠️ LLM 补全表 {table.table_name} 描述失败：{e}")

        return tables

    def _get_llm(self):
        if self._llm is None:
            from pydantic_ai import Agent
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
            from config.settings import llm_settings

            llm_model = OpenAIChatModel(
                model_name=llm_settings.LLM_MODEL_NAME,
                provider=OpenAIProvider(
                    base_url=llm_settings.LLM_BASE_URL,
                    api_key=llm_settings.LLM_API_KEY
                ),
            )
            self._llm = Agent(llm_model)
        return self._llm


def parse_sql(sql_content: str) -> List[TableInfo]:
    """便捷函数：解析 SQL DDL"""
    parser = SQLParser()
    return parser.parse(sql_content)


async def parse_sql_with_enrichment(sql_content: str) -> List[TableInfo]:
    """便捷函数：解析 SQL DDL 并用 LLM 补全描述"""
    parser = SQLParser()
    tables = parser.parse(sql_content)
    tables = await parser.enrich_with_llm(tables)
    return tables


if __name__ == "__main__":
    import asyncio

    sample_sql = """
    CREATE TABLE user (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(64) NOT NULL COMMENT '姓名',
        phone VARCHAR(11) COMMENT '手机号',
        department_id BIGINT COMMENT '部门ID',
        status TINYINT DEFAULT 0 COMMENT '状态 0=正常 1=禁用',
        create_time DATETIME COMMENT '创建时间',
        FOREIGN KEY (department_id) REFERENCES department(id)
    ) COMMENT='用户信息表';

    CREATE TABLE department (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(255) NOT NULL COMMENT '部门名称',
        parent_id BIGINT COMMENT '上级部门ID'
    ) COMMENT='部门信息表';
    """

    tables = parse_sql(sample_sql)
    for table in tables:
        print(f"\n{'='*40}")
        print(f"表名: {table.table_name} ({table.table_type})")
        print(f"描述: {table.table_desc}")
        print(f"字段数: {len(table.fields)}, 关系数: {len(table.relations)}")
        for field in table.fields:
            print(f"  - {field.name} ({field.type}): {field.desc}")
        for rel in table.relations:
            print(f"  → {rel.target_table} ({rel.relation_type}) via {rel.via}")
