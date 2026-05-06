from typing import List, Dict, Set, Optional
from dataclasses import dataclass


@dataclass
class FieldSelection:
    table_name: str
    field_name: str
    relevance_score: float
    is_foreign_key: bool = False


@dataclass
class TableSchema:
    table_name: str
    content: str
    selected_fields: List[str]
    relations: List[Dict]


class ContextAssembler:
    def __init__(self, relation_map: Dict[str, List[Dict]]):
        self.relation_map = relation_map

    def assemble(
        self,
        hit_fields: Dict[str, List[FieldSelection]],
        parent_contents: Dict[str, str]
    ) -> List[TableSchema]:
        table_schemas = []

        for table_name, fields in hit_fields.items():
            if table_name not in parent_contents:
                continue

            selected_field_names = [f.field_name for f in fields]

            table_relations = self.relation_map.get(table_name, [])
            relevant_relations = []
            for rel in table_relations:
                for field in fields:
                    if rel.get("via_field") == field.field_name:
                        relevant_relations.append(rel)
                        break

            table_schema = TableSchema(
                table_name=table_name,
                content=parent_contents[table_name],
                selected_fields=selected_field_names,
                relations=relevant_relations
            )
            table_schemas.append(table_schema)

        return table_schemas

    def _get_foreign_key_relations(self, table_name: str, field_name: str) -> List[Dict]:
        relations = self.relation_map.get(table_name, [])
        fk_relations = []

        for rel in relations:
            if rel.get("via_field") == field_name:
                fk_relations.append(rel)

        return fk_relations

    def extract_relevant_fields_from_parent(
        self,
        parent_content: str,
        selected_field_names: List[str]
    ) -> str:
        if not selected_field_names:
            return parent_content

        lines = parent_content.split("\n")
        relevant_lines = []
        current_section = None

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("字段详情"):
                current_section = "fields"
                relevant_lines.append(line)
                continue

            if stripped.startswith("表关系") or stripped.startswith("业务规则"):
                current_section = "other"
                relevant_lines.append(line)
                continue

            if current_section == "fields":
                if stripped.startswith("-"):
                    should_include = any(
                        f"- {fn}：" in stripped for fn in selected_field_names
                    )
                    if should_include:
                        relevant_lines.append(line)
                else:
                    relevant_lines.append(line)
            else:
                relevant_lines.append(line)

        return "\n".join(relevant_lines)

    def format_for_prompt(self, table_schemas: List[TableSchema]) -> str:
        output_parts = []

        for schema in table_schemas:
            output_parts.append(f"【表：{schema.table_name}】")

            if schema.selected_fields:
                relevant_content = self.extract_relevant_fields_from_parent(
                    schema.content,
                    schema.selected_fields
                )
                output_parts.append(relevant_content)
            else:
                output_parts.append(schema.content)

            if schema.relations:
                output_parts.append("表关系：")
                for rel in schema.relations:
                    output_parts.append(f"- 与 {rel['target_table']} 表：{rel['relation_type']}（{rel['via']}）")

            output_parts.append("")

        return "\n".join(output_parts)


def assemble_context(
    hit_fields: Dict[str, List[FieldSelection]],
    parent_contents: Dict[str, str],
    relation_map: Dict[str, List[Dict]]
) -> str:
    assembler = ContextAssembler(relation_map)
    table_schemas = assembler.assemble(hit_fields, parent_contents)
    return assembler.format_for_prompt(table_schemas)
