from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime


class AgentState(BaseModel):
    """
    共享状态管理类

    设计思路：
    1. 所有节点/Skill共享同一个状态对象
    2. 每个节点只修改自己负责的字段
    3. 使用 Pydantic 进行类型校验
    4. 支持状态快照（便于调试和回滚）
    """

    # === 输入信息 ===
    user_input: str = Field(default="", description="用户原始输入")
    user_id: str = Field(default="", description="用户 ID")
    session_id: Optional[str] = Field(default=None, description="会话 ID（用于短期记忆）")

    # === 意图识别结果 ===
    intent: Optional[str] = Field(default=None, description="意图类型：database_query | daily_chat")
    intent_reason: str = Field(default="", description="意图判断理由")
    need_memory: bool = Field(default=False, description="是否需要检索长期记忆")
    memory_reason: str = Field(default="", description="是否需要检索记忆的理由")
    needs_react: bool = Field(default=False, description="是否需要 ReAct 多步推理（多表对比、趋势分析、嵌套条件等）")
    react_reason: str = Field(default="", description="是否需要 ReAct 多步推理的理由")

    # === SQL 生成决策 ===
    metadata_context: str = Field(default="", description="检索到的表结构上下文")
    examples_context: str = Field(default="", description="检索到的示例上下文")
    sql_valid: bool = Field(default=False, description="SQL 是否通过校验")
    sql_error: str = Field(default="", description="SQL 错误信息")
    retry_count: int = Field(default=0, description="SQL 生成重试次数")
    max_retries: int = Field(default=3, description="最大重试次数")

    # === 记忆相关 ===
    memories: Dict[str, Any] = Field(default_factory=dict, description="检索到的记忆")

    # === 数据库查询相关 ===
    sql_query: Optional[str] = Field(default=None, description="生成的 SQL 语句")
    query_result: List[Dict] = Field(default_factory=list, description="数据库查询结果")
    query_status: str = Field(default="pending", description="查询状态：pending | success | fail")

    # === 最终响应 ===
    response: Optional[str] = Field(default=None, description="最终返回给用户的响应")

    # === Skills 相关 ===
    skill_name: Optional[str] = Field(default=None, description="选中的技能名称")

    # === 规划相关 ===
    plan: Optional[Dict[str, Any]] = Field(default=None, description="查询规划结果")
    plan_steps: List[Dict[str, Any]] = Field(default_factory=list, description="分步执行计划")
    current_step: int = Field(default=0, description="当前执行步骤")
    step_results: List[Dict[str, Any]] = Field(default_factory=list, description="每步执行结果")

    # === ReAct 轨迹 ===
    react_trace: List[Dict[str, Any]] = Field(default_factory=list, description="Think-Act-Observe 推理轨迹")

    # === 澄清相关 ===
    clarification_needed: bool = Field(default=False, description="是否需要用户澄清")
    clarification_question: Optional[str] = Field(default=None, description="向用户提问的内容")

    # === 反思相关 ===
    reflection: Optional[Dict[str, Any]] = Field(default=None, description="错误反思结果")

    # === MCP 相关 ===
    mcp_tools_used: List[str] = Field(default_factory=list, description="本次请求使用的 MCP 工具列表")
    chart_url: Optional[str] = Field(default=None, description="MCP 图表工具生成的图表URL")
    chart_type: Optional[str] = Field(default=None, description="图表类型")

    # === 元信息 ===
    created_at: datetime = Field(default_factory=datetime.now, description="状态创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="最后更新时间")
    node_history: List[str] = Field(default_factory=list, description="已执行的节点历史")

    def update(self, **kwargs):
        """更新状态，自动更新时间戳"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()
        return self

    def add_node_history(self, node_name: str):
        """添加节点执行历史"""
        self.node_history.append(node_name)
        self.updated_at = datetime.now()
        return self

    def add_react_step(self, thought: str, action: str, parameters: Dict, observation: str):
        """添加 ReAct 推理步骤"""
        self.react_trace.append({
            "step": len(self.react_trace),
            "thought": thought,
            "action": action,
            "parameters": parameters,
            "observation": observation
        })
        self.updated_at = datetime.now()
        return self

    def snapshot(self) -> dict:
        """创建状态快照（用于调试）"""
        return self.model_dump()
