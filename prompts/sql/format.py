"""结果格式化 Prompt"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)


FORMAT_RESULT_TEMPLATE = """用户问题：{question}

查询结果数据（JSON格式）：
{query_result}

请根据用户问题和查询结果数据，用自然语言回答用户。
要求：
1. 直接回答用户的问题
2. 如果是统计类问题（如有多少人、总数），先给出具体数字
3. 如果是列表类问题，列出所有关键数据
4. 保持准确，如果数据量超过一百条，就返回总结性的信息
5. 如果数据为空或异常，也请说明情况

请直接给出回答，不要说明你是什么模型或解释过程。"""

FORMAT_RESULT_PROMPT = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(FORMAT_RESULT_TEMPLATE),
])
