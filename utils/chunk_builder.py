from typing import List, Dict, Tuple
from dataclasses import dataclass
from .docx_parser import TableInfo, FieldInfo, Relation


@dataclass
class ParentChunk:
    chunk_type: str = "parent"
    table_name: str = ""
    content: str = ""


@dataclass
class ChildChunk:
    chunk_type: str = "child"
    parent_id: str = ""
    field_name: str = ""
    content: str = ""


class ChunkBuilder:
    def __init__(self):
        self.tables: Dict[str, TableInfo] = {}

    def build(self, tables: List[TableInfo]) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        parent_chunks = []
        child_chunks = []

        for table in tables:
            self.tables[table.table_name] = table

            parent_chunk = self._build_parent_chunk(table)
            parent_chunks.append(parent_chunk)

            child_chunks_for_table = self._build_child_chunks(table)
            child_chunks.extend(child_chunks_for_table)

        return parent_chunks, child_chunks

    def _build_parent_chunk(self, table: TableInfo) -> ParentChunk:
        content_parts = []
        content_parts.append(f"表名：{table.table_name}（{table.table_type}）")
        content_parts.append(f"表描述：{table.table_desc}")

        if table.business_rules:
            content_parts.append("业务规则：")
            for rule in table.business_rules:
                content_parts.append(f"- {rule}")

        if table.fields:
            content_parts.append("字段详情：")
            for field in table.fields:
                nullable_str = "" if field.nullable else "，不可为空"
                content_parts.append(f"- {field.name}：{field.type}（{field.desc}{nullable_str}）")

        if table.relations:
            content_parts.append("表关系：")
            for rel in table.relations:
                content_parts.append(f"- 与 {rel.target_table} 表：{rel.relation_type}（{rel.via}）")

        content = "\n".join(content_parts)

        return ParentChunk(
            chunk_type="parent",
            table_name=table.table_name,
            content=content
        )

    def _build_child_chunks(self, table: TableInfo) -> List[ChildChunk]:
        child_chunks = []

        related_tables = [rel.target_table for rel in table.relations]
        related_tables_str = "、".join(related_tables) if related_tables else "无"

        short_desc = self._shorten_desc(table.table_desc, max_len=30)

        for field in table.fields:
            field_content = self._build_field_content(
                table.table_name,
                short_desc,
                related_tables_str,
                field
            )

            child_chunk = ChildChunk(
                chunk_type="child",
                parent_id=table.table_name,
                field_name=field.name,
                content=field_content
            )
            child_chunks.append(child_chunk)

        return child_chunks

    def _build_field_content(
        self,
        table_name: str,
        table_desc: str,
        related_tables: str,
        field: FieldInfo
    ) -> str:
        nullable_str = "" if field.nullable else "，不可为空"

        content = (
            f"[表:{table_name}] "
            f"[业务:{table_desc}] "
            f"[关联:{related_tables}] "
            f"字段：{field.name} {field.type} {field.desc}{nullable_str}"
        )

        return content

    def _shorten_desc(self, desc: str, max_len: int = 30) -> str:
        if len(desc) <= max_len:
            return desc

        shortened = desc[:max_len - 3] + "..."
        return shortened

    def get_relation_map(self) -> Dict[str, List[Dict]]:
        relation_map = {}

        for table_name, table in self.tables.items():
            relations = []
            for rel in table.relations:
                relations.append({
                    "target_table": rel.target_table,
                    "relation_type": rel.relation_type,
                    "via_field": rel.via_field,
                    "via": rel.via
                })
            relation_map[table_name] = relations

        return relation_map


def build_chunks(tables: List[TableInfo]) -> Tuple[List[ParentChunk], List[ChildChunk], Dict[str, List[Dict]]]:
    builder = ChunkBuilder()
    parent_chunks, child_chunks = builder.build(tables)
    relation_map = builder.get_relation_map()
    return parent_chunks, child_chunks, relation_map


if __name__ == "__main__":
    from utils.docx_parser import parse_docx

    tables = parse_docx("resource/htower.docx")
    parent_chunks, child_chunks, relation_map = build_chunks(tables)

    print(f"生成父块: {len(parent_chunks)} 个")
    print(f"生成子块: {len(child_chunks)} 个")

    print("\n--- 父块示例 ---")
    if parent_chunks:
        p = parent_chunks[0]
        print(f"表名: {p.table_name}")
        print(f"内容:\n{p.content[:300]}...")

    print("\n--- 子块示例 ---")
    if child_chunks:
        c = child_chunks[0]
        print(f"父ID: {c.parent_id}")
        print(f"字段名: {c.field_name}")
        print(f"内容: {c.content}")

    print("\n--- 关系图 ---")
    for table, rels in relation_map.items():
        print(f"{table}: {rels}")
