from typing import TypedDict, Dict, List, Any, Optional, Annotated
from datetime import datetime


class AgentState(TypedDict, total=False):
    """
    共享状态管理类（LangGraph TypedDict 版本）

    设计思路：
    1. 所有节点/Skill共享同一个状态字典
    2. 每个节点只修改自己负责的字段，返回 dict 更新
    3. 使用 TypedDict + total=False 使所有字段可选（LangGraph 支持部分状态更新）
    4. LangGraph 节点通过返回 dict 来更新状态，无需手动 update/snapshot 方法
    """

    # === 输入信息 ===
    user_input: str
    user_id: str
    session_id: Optional[str]

    # === 意图识别结果 ===
    intent: Optional[str]
    intent_reason: str
    need_memory: bool
    memory_reason: str
    needs_react: bool
    react_reason: str

    # === SQL 生成决策 ===
    metadata_context: str
    examples_context: str
    sql_valid: bool
    sql_error: str
    retry_count: int
    max_retries: int

    # === 记忆相关 ===
    memories: Dict[str, Any]

    # === 数据库查询相关 ===
    sql_query: Optional[str]
    query_result: List[Dict]
    query_status: str

    # === 最终响应 ===
    response: Optional[str]

    # === Skills 相关 ===
    skill_name: Optional[str]

    # === 规划相关 ===
    plan: Optional[Dict[str, Any]]
    plan_steps: List[Dict[str, Any]]
    current_step: int
    step_results: List[Dict[str, Any]]

    # === ReAct 轨迹 ===
    react_trace: List[Dict[str, Any]]

    # === 澄清相关 ===
    clarification_needed: bool
    clarification_question: Optional[str]

    # === 反思相关 ===
    reflection: Optional[Dict[str, Any]]

    # === MCP 相关 ===
    mcp_tools_used: List[str]
    chart_url: Optional[str]
    chart_type: Optional[str]

    # === 元信息 ===
    created_at: datetime
    updated_at: datetime
    node_history: List[str]

    # === LangGraph 扩展字段 ===
    messages: list[dict]
    react_step_count: int
    pending_tool_call: Optional[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]

    # === 多 Agent 协作 ===
    sub_tasks: List[Dict[str, Any]]       # Coordinator 拆解的子任务列表
    current_agent: str                     # 当前正在执行的 Agent 名称
    agent_results: Dict[str, Any]          # 各子 Agent 的输出结果
