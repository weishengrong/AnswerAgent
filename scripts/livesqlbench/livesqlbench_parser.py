import json
import logging
import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FieldInfo:
    name: str
    type: str
    desc: str
    nullable: bool = True


@dataclass
class Relation:
    target_table: str
    relation_type: str
    via_field: str
    via: str


@dataclass
class TableInfo:
    table_name: str
    table_type: str = "业务表"
    table_desc: str = ""
    business_rules: List[str] = field(default_factory=list)
    fields: List[FieldInfo] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)


@dataclass
class LiveSQLBenchTask:
    instance_id: str
    selected_database: str
    query: str
    sol_sql: Optional[str] = None
    external_knowledge: Optional[List[str]] = None
    preprocess_sql: Optional[List[str]] = None
    clean_up_sql: Optional[List[str]] = None
    test_cases: Optional[List[str]] = None
    category: Optional[str] = None
    difficulty_tier: Optional[str] = None
    conditions: Optional[Dict] = None


def parse_schema_text(schema_text: str, db_name: str) -> List[TableInfo]:
    schema_text = re.sub(
        r"First\s+\d+\s+rows:.*?(?=CREATE\s+TABLE|\Z)",
        "",
        schema_text,
        flags=re.DOTALL | re.IGNORECASE
    )

    tables = []
    current_table = None
    current_fields = []
    buffer = ""

    for line in schema_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        buffer += " " + stripped if buffer else stripped

        if ";" in buffer:
            stmt = buffer.strip()
            buffer = ""

            if stmt.upper().startswith("CREATE TABLE"):
                table_match = re.search(
                    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:`?\w+`?\.)?[\"`]?(\w+)[\"`]?\s*\(", stmt, re.IGNORECASE
                )
                if table_match:
                    if current_table and current_fields:
                        current_table.fields = current_fields
                        tables.append(current_table)

                    table_name = table_match.group(1)
                    current_table = TableInfo(
                        table_name=table_name,
                        table_type="业务表",
                        table_desc=""
                    )
                    current_fields = []

                    comment_match = re.search(r"COMMENT\s*['\"](.+?)['\"]\s*\)", stmt, re.IGNORECASE)
                    if comment_match:
                        current_table.table_desc = comment_match.group(1)

                    fields_section = stmt[table_match.end():]
                    paren_depth = 1
                    fields_text = ""
                    for c in fields_section:
                        if c == "(":
                            paren_depth += 1
                        elif c == ")":
                            paren_depth -= 1
                            if paren_depth == 0:
                                break
                        fields_text += c

                    raw_fields = _split_raw_fields(fields_text)
                    for raw in raw_fields:
                        field = _parse_field(raw)
                        if field:
                            current_fields.append(field)

    if current_table and current_fields:
        current_table.fields = current_fields
        tables.append(current_table)

    _infer_relations(tables)

    return tables


def _split_raw_fields(fields_text: str) -> List[str]:
    result = []
    current = ""
    paren_depth = 0
    for c in fields_text:
        if c == "(":
            paren_depth += 1
            current += c
        elif c == ")":
            paren_depth -= 1
            current += c
        elif c == "," and paren_depth == 0:
            result.append(current.strip())
            current = ""
        else:
            current += c
    if current.strip():
        result.append(current.strip())
    return result


def _parse_field(raw: str) -> Optional[FieldInfo]:
    raw = raw.strip()
    if not raw:
        return None

    upper = raw.upper()
    if upper.startswith("PRIMARY KEY") or upper.startswith("FOREIGN KEY") or upper.startswith("KEY") or upper.startswith("INDEX") or upper.startswith("UNIQUE") or upper.startswith("CONSTRAINT"):
        return None

    name_match = re.match(r"[\"`]?(\w+)[\"`]?\s+", raw)
    if not name_match:
        return None

    field_name = name_match.group(1)
    rest = raw[name_match.end():].strip()

    type_match = re.match(r"(\w+(?:\s*\([^)]*\))?(?:\s+\w+)*)", rest)
    field_type = type_match.group(1) if type_match else "TEXT"

    nullable = True
    if re.search(r"NOT\s+NULL", rest, re.IGNORECASE):
        nullable = False

    desc = ""
    comment_match = re.search(r"COMMENT\s*['\"](.+?)['\"]", rest, re.IGNORECASE)
    if comment_match:
        desc = comment_match.group(1)

    return FieldInfo(name=field_name, type=field_type, desc=desc, nullable=nullable)


def _infer_relations(tables: List[TableInfo]):
    table_names = {t.table_name.lower(): t for t in tables}

    for table in tables:
        for field in table.fields:
            fk_match = re.match(r"(.+)_id$", field.name, re.IGNORECASE)
            if fk_match:
                target_name = fk_match.group(1)
                if target_name.lower() in table_names:
                    table.relations.append(Relation(
                        target_table=table_names[target_name.lower()].table_name,
                        relation_type="多对一",
                        via_field=field.name,
                        via=f"{table.table_name}.{field.name} = {table_names[target_name.lower()].table_name}.id"
                    ))


def parse_column_meanings(col_mean_path: str) -> Dict[str, str]:
    try:
        with open(col_mean_path, "r", encoding="utf-8") as f:
            meanings = json.load(f)
        return {k.lower(): v for k, v in meanings.items()}
    except Exception as e:
        logger.warning("读取列含义文件失败: %s, path=%s", e, col_mean_path)
        return {}


def parse_kb(kb_path: str) -> List[Dict]:
    knowledge_list = []
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        knowledge_list.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.debug("KB 文件单行 JSON 解析失败: %s, line=%s", e, line[:50])
    except Exception as e:
        logger.warning("读取 KB 文件失败: %s, path=%s", e, kb_path)
    return knowledge_list


def enrich_fields_with_meanings(tables: List[TableInfo], column_meanings: Dict[str, str], db_name: str = ""):
    for table in tables:
        for field in table.fields:
            key_dot = f"{table.table_name}.{field.name}".lower()
            key_pipe = f"{db_name}|{table.table_name}|{field.name}".lower()
            if key_pipe in column_meanings and column_meanings[key_pipe]:
                meaning = column_meanings[key_pipe]
            elif key_dot in column_meanings and column_meanings[key_dot]:
                meaning = column_meanings[key_dot]
            else:
                meaning = ""
            if meaning:
                if field.desc:
                    field.desc = f"{field.desc}；{meaning}"
                else:
                    field.desc = meaning


def load_livesqlbench_data(jsonl_path: str) -> List[Dict]:
    tasks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def load_database_schema(db_path: str, db_name: str) -> Tuple[List[TableInfo], Dict[str, str], List[Dict]]:
    schema_path = os.path.join(db_path, db_name, f"{db_name}_schema.txt")
    col_mean_path = os.path.join(db_path, db_name, f"{db_name}_column_meaning_base.json")
    kb_path = os.path.join(db_path, db_name, f"{db_name}_kb.jsonl")

    tables = []
    column_meanings = {}
    knowledge = []

    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_text = f.read()
        tables = parse_schema_text(schema_text, db_name)

    if os.path.exists(col_mean_path):
        column_meanings = parse_column_meanings(col_mean_path)
        enrich_fields_with_meanings(tables, column_meanings, db_name)

    if os.path.exists(kb_path):
        knowledge = parse_kb(kb_path)

    return tables, column_meanings, knowledge


def get_all_database_names(data_list: List[Dict]) -> List[str]:
    db_names = set()
    for task in data_list:
        db_name = task.get("selected_database", "")
        if db_name:
            db_names.add(db_name)
    return sorted(db_names)


def group_tasks_by_database(data_list: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = {}
    for task in data_list:
        db_name = task.get("selected_database", "unknown")
        if db_name not in grouped:
            grouped[db_name] = []
        grouped[db_name].append(task)
    return grouped