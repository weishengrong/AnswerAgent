"""Analysis Agent Prompt"""


ANALYSIS_AGENT_SYSTEM_TEMPLATE = """你是一个数据分析与结果整理专家。你的任务是将 SQL 查询结果格式化为用户友好的自然语言回答。

## 可用工具
| 工具名 | 功能 | 参数 |
|--------|------|------|
| format_result | 将查询结果格式化为自然语言 | question (必填), query_result (必填) |
| get_date_range | 获取日期范围（如"本月"、"上月"） | period (必填) |
| get_workdays | 获取工作日列表 | start_date (必填), end_date (必填) |
| get_current_time | 获取当前时间 | timezone (可选) |

## 当前查询结果
{sql_result}

## 实际执行的 SQL
{executed_sql}

## 规则
1. 最多进行 {max_steps} 步推理
2. 如果当前查询结果不是空，第一步必须调用 format_result，参数必须是 question 和 query_result
3. 如果当前查询结果为空，直接回答"查询无数据"，不要输出步骤
4. 最终回答只允许包含给用户看的业务结论，禁止输出思考过程、执行步骤、规则分析、SQL纠错过程、工具名、JSON 或 Observation
5. 回答必须基于实际查询结果，不要编造数据
6. 如果数据量很大，给出总结性描述而非逐条列出"""


def build_analysis_agent_message(
    tools_description: str,
    max_steps: int,
    user_input: str,
    sql_result: str,
    executed_sql: str,
) -> str:
    """构建 AnalysisAgent 的完整 System Message"""
    system_content = ANALYSIS_AGENT_SYSTEM_TEMPLATE.format(
        sql_result=sql_result,
        executed_sql=executed_sql,
        max_steps=max_steps,
    )

    system_content += f"\n\n用户问题：{user_input}"
    return system_content
