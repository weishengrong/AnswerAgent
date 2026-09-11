import json
import os
import re
import sqlite3
import io
import sys
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Optional, Tuple


def execute_sqlite_query(sql: str, db_path: str) -> Tuple[Optional[List[Tuple]], Optional[str]]:
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        return results, None
    except Exception as e:
        return None, str(e)
    finally:
        if conn:
            conn.close()


def execute_sqlite_queries(sqls: List[str], db_path: str) -> Tuple[Optional[List[Tuple]], Optional[str]]:
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        all_results = []
        for sql in sqls:
            sql = sql.strip()
            if not sql:
                continue
            cursor.execute(sql)
            if sql.upper().lstrip().startswith("SELECT"):
                results = cursor.fetchall()
                all_results.extend(results)
            else:
                conn.commit()
        return all_results, None
    except Exception as e:
        return None, str(e)
    finally:
        if conn:
            conn.close()


def remove_comments(sql_list: List[str]) -> List[str]:
    cleaned = []
    for sql in sql_list:
        no_block = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        no_line = re.sub(r"--.*?(\r\n|\r|\n)", r"\1", no_block)
        no_blank = re.sub(r"\n\s*\n+", "\n", no_line)
        cleaned.append(no_blank.strip())
    return cleaned


def remove_distinct(sql_list: List[str]) -> List[str]:
    cleaned = []
    for sql in sql_list:
        tokens = sql.split(" ")
        filtered = [t for t in tokens if t.lower() != "distinct"]
        cleaned.append(" ".join(filtered))
    return cleaned


def remove_round_functions(sql_string: str) -> str:
    def find_matching_paren(text, start_pos):
        paren_count = 0
        for i in range(start_pos, len(text)):
            if text[i] == "(":
                paren_count += 1
            elif text[i] == ")":
                paren_count -= 1
                if paren_count == 0:
                    return i
        return -1

    def find_first_arg_end(text, start_pos):
        paren_count = 0
        for i in range(start_pos, len(text)):
            if text[i] == "(":
                paren_count += 1
            elif text[i] == ")":
                if paren_count == 0:
                    return i
                paren_count -= 1
            elif text[i] == "," and paren_count == 0:
                return i
        return len(text)

    result = sql_string
    while True:
        pattern = re.compile(r"ROUND\s*\(", re.IGNORECASE)
        match = pattern.search(result)
        if not match:
            break
        start_pos = match.start()
        open_paren_pos = match.end() - 1
        first_arg_end = find_first_arg_end(result, open_paren_pos + 1)
        close_paren_pos = find_matching_paren(result, open_paren_pos)
        if close_paren_pos == -1:
            break
        first_arg = result[open_paren_pos + 1: first_arg_end].strip()
        result = result[:start_pos] + first_arg + result[close_paren_pos + 1:]
    return result


def remove_round(sql_list: List[str]) -> List[str]:
    return [remove_round_functions(sql) for sql in sql_list]


def preprocess_results(results, decimal_places=2):
    processed = []
    for row in results:
        processed_row = []
        for item in row:
            if isinstance(item, (date, datetime)):
                processed_row.append(item.strftime("%Y-%m-%d"))
            elif isinstance(item, Decimal):
                quantizer = Decimal(1).scaleb(-decimal_places)
                processed_row.append(float(item.quantize(quantizer, rounding=ROUND_HALF_UP)))
            elif isinstance(item, float):
                processed_row.append(round(item, decimal_places))
            elif isinstance(item, (dict, list)):
                processed_row.append(json.dumps(item, sort_keys=True))
            else:
                processed_row.append(item)
        processed.append(tuple(processed_row))
    return processed


def ex_base(pred_sqls, sol_sqls, db_path, conn, conditions=None):
    if not pred_sqls or not sol_sqls:
        return 0

    pred_results, pred_err = execute_sqlite_queries(pred_sqls, db_path)
    sol_results, sol_err = execute_sqlite_queries(sol_sqls, db_path)

    if pred_err or sol_err:
        return 0

    pred_results = preprocess_results(pred_results)
    sol_results = preprocess_results(sol_results)

    if not pred_results or not sol_results:
        return 0

    if conditions and conditions.get("order", False):
        return 1 if pred_results == sol_results else 0
    else:
        return 1 if set(pred_results) == set(sol_results) else 0


TEST_CASE_DEFAULT = """
def test_case(pred_sqls, sol_sqls, db_path, conn, conditions):
   pred_sqls = remove_comments(pred_sqls)
   sol_sqls  = remove_comments(sol_sqls)
   pred_sqls = remove_distinct(pred_sqls)
   pred_sqls = remove_round(pred_sqls)
   sol_sqls  = remove_distinct(sol_sqls)
   sol_sqls  = remove_round(sol_sqls)
   result = ex_base(pred_sqls, sol_sqls, db_path, conn, conditions)
   assert result == 1, f"ex_base returned {result} but expected 1."
   return result
"""


def run_test_case(test_code: str, pred_sqls: List[str], sol_sqls: List[str], db_path: str, db_name: str = "", conditions: Optional[Dict] = None) -> Tuple[bool, str]:
    db_root = os.path.dirname(os.path.dirname(db_path)) if os.path.dirname(db_path) else ""

    def execute_queries(sql: str, name: str, conn=None) -> tuple:
        target_db = name if name else db_name
        target_path = db_path
        if target_db and target_db != os.path.basename(os.path.dirname(db_path)):
            target_path = os.path.join(db_root, target_db, f"{target_db}_template.sqlite")
            if not os.path.exists(target_path):
                target_path = os.path.join(db_root, target_db, f"{target_db}.sqlite")
        return execute_sqlite_queries([sql] if isinstance(sql, str) else sql, target_path)

    global_env = {
        "execute_queries": execute_queries,
        "execute_sqlite_queries": execute_sqlite_queries,
        "ex_base": ex_base,
        "remove_comments": remove_comments,
        "remove_distinct": remove_distinct,
        "remove_round": remove_round,
        "preprocess_results": preprocess_results,
        "date": date,
        "datetime": datetime,
    }
    local_env = {
        "pred_sqls": pred_sqls,
        "sol_sqls": sol_sqls,
        "db_path": db_path,
        "db_name": db_name,
        "conn": None,
        "conditions": conditions or {},
    }

    full_code = "import datetime\nfrom datetime import date\n" + test_code
    call_lines = [
        "__test_case_result__ = test_case(pred_sqls, sol_sqls, db_path, conn, conditions=conditions)",
        "__test_case_result__ = test_case(pred_sqls, sol_sqls, db_name, conn)",
        "__test_case_result__ = test_case(pred_sqls, sol_sqls, db_path, conn)",
    ]
    for call_line in call_lines:
        full_code += "\n" + call_line
        break

    try:
        exec(full_code, global_env, local_env)
        return True, ""
    except AssertionError as e:
        return False, f"AssertionError: {e}"
    except TypeError as e:
        if len(call_lines) > 1:
            return _try_test_case_alternatives(test_code, pred_sqls, sol_sqls, db_path, db_name, conditions, call_lines[1:], global_env)
        return False, f"TypeError: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def _try_test_case_alternatives(test_code, pred_sqls, sol_sqls, db_path, db_name, conditions, remaining_calls, base_global_env):
    import copy
    for call_line in remaining_calls:
        local_env = {
            "pred_sqls": pred_sqls,
            "sol_sqls": sol_sqls,
            "db_path": db_path,
            "db_name": db_name,
            "conn": None,
            "conditions": conditions or {},
        }
        full_code = "import datetime\nfrom datetime import date\n" + test_code + "\n" + call_line
        try:
            exec(full_code, base_global_env, local_env)
            return True, ""
        except AssertionError as e:
            return False, f"AssertionError: {e}"
        except Exception:
            continue
    return False, "All test_case signatures failed"


def evaluate_single_task(
    pred_sql: str,
    sol_sql: str,
    db_path: str,
    db_name: str = "",
    test_cases: Optional[List[str]] = None,
    preprocess_sql: Optional[List[str]] = None,
    clean_up_sql: Optional[List[str]] = None,
    conditions: Optional[Dict] = None,
) -> Dict:
    pred_sqls = [s.strip() for s in pred_sql.split(";") if s.strip()]
    if isinstance(sol_sql, list):
        sol_sqls = [s.strip() for s in sol_sql if s and s.strip()]
    else:
        sol_sqls = [s.strip() for s in sol_sql.split(";") if s.strip()]

    if preprocess_sql:
        for sql in preprocess_sql:
            sql = sql.strip()
            if sql:
                execute_sqlite_query(sql, db_path)

    if not test_cases:
        test_cases = [TEST_CASE_DEFAULT]

    passed = 0
    failed = []
    for i, tc in enumerate(test_cases):
        ok, err = run_test_case(tc, pred_sqls, sol_sqls, db_path, db_name, conditions)
        if ok:
            passed += 1
        else:
            failed.append(f"test_{i + 1}: {err}")

    if clean_up_sql:
        for sql in clean_up_sql:
            sql = sql.strip()
            if sql:
                execute_sqlite_query(sql, db_path)

    return {
        "passed": passed,
        "total": len(test_cases),
        "failed": failed,
        "success": len(failed) == 0,
    }


def split_field(data: Dict, field: str) -> List[str]:
    value = data.get(field, "")
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [s.strip() for s in value.split(";") if s.strip()]
    return []