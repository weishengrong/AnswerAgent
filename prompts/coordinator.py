"""Coordinator 任务拆解 Prompt"""

from typing import List
from pydantic import BaseModel, Field


class SubTask(BaseModel):
    step: int = Field(description="步骤序号，从1开始")
    description: str = Field(description="子任务描述")
    depends_on: List[int] = Field(default=[], description="依赖的前置步骤序号")


class SubTaskList(BaseModel):
    needs_decomposition: bool = Field(description="是否需要拆解为多个子任务")
    reason: str = Field(description="判断理由")
    sub_tasks: List[SubTask] = Field(default=[], description="子任务列表，不需要拆解时为空")


COORDINATOR_SYSTEM_TEMPLATE = """你是一个查询任务分析专家。你的任务是分析用户的数据查询请求，判断是否需要拆解为多个子任务。

如果查询可以通过一条 SQL 完成，不需要拆解。
如果查询需要多步操作（如先查A再根据A的结果查B），则拆解为有序子任务。

## 用户问题
{user_input}

## 判断标准
- 简单查询（单表/多表 JOIN、聚合、筛选）→ 不拆解
- 需要中间结果的查询（先查X，再根据X查Y）→ 拆解
- 对比分析（本月vs上月）→ 拆解
- 多维度统计（按部门统计+按类型统计）→ 可不拆解（一条SQL可完成）"""


def build_coordinator_message(user_input: str) -> str:
    """构建 Coordinator 的 Prompt（用于 LLM 结构化输出）"""
    return COORDINATOR_SYSTEM_TEMPLATE.format(user_input=user_input)
