
from prompts.database_schema import format_schema_for_prompt

few_shot_examples = [
    {
        "question": "系统现在一共有多少人？列出他们的名字",
        "sql": "SELECT realname FROM sys_user WHERE status = 0 AND del_flag = 0"
    },
    {
        "question": "张三今天的打卡记录有哪些？",
        "sql": "SELECT cr.check_time, cr.type, d.name as device_name, d.address FROM check_record cr JOIN user u ON cr.user_id = u.id JOIN device d ON cr.device_id = d.id WHERE u.name = '张三' AND DATE(cr.check_time) = CURDATE()"
    },
    {
        "question": "统计上个月每个部门的上班打卡总次数",
        "sql": "SELECT dept.name, COUNT(*) as check_count FROM check_record cr JOIN user u ON cr.user_id = u.id JOIN department dept ON u.department_id = dept.id WHERE cr.type = 0 AND cr.check_time >= DATE_FORMAT(CURDATE() - INTERVAL 1 MONTH, '%Y-%m-01') AND cr.check_time < DATE_FORMAT(CURDATE(), '%Y-%m-01') GROUP BY dept.name"
    },
    {
        "question": "列出所有状态正常的考勤设备及其安装位置",
        "sql": "SELECT name, address, model FROM device WHERE status = 0"
    },
    {
        "question": "查找本周尚未提交完整信息的用户名单",
        "sql": "SELECT number, name, phone FROM user WHERE info_submitted = 0 AND status = 0"
    },
    {
        "question": "昨天下午5点以后下班打卡的用户都有谁？",
        "sql": "SELECT u.name, cr.check_time FROM check_record cr JOIN user u ON cr.user_id = u.id WHERE cr.type = 1 AND cr.check_time >= DATE_SUB(CURDATE(), INTERVAL 1 DAY) + INTERVAL 17 HOUR AND cr.check_time < DATE_SUB(CURDATE(), INTERVAL 1 DAY) + INTERVAL 24 HOUR"
    }
]

def build_prompt(user_question, schema, examples):
    """构建完整的提示词"""
    prompt = """你是一个数据分析专家，需要根据用户的自然语言问题生成SQL查询语句，并利用可用的工具执行查询，拿到数据集后组织语言返回给用户。

"""
    # 添加Schema信息
    prompt += format_schema_for_prompt(schema)
    prompt += "\n"

    # 添加示例
    prompt += "以下是一些常见问题及其对应的SQL示例：\n"
    for ex in examples:
        prompt += f"问题：{ex['question']}\nSQL：{ex['sql']}\n\n"

    # 当前问题
    prompt += f"现在，请根据以上信息，为用户的问题生成SQL，然后执行查询：\n{user_question}\n"
    prompt += "注意1：生成sql后一定要分析可利用的工具，要执行查询数据库，拿到数据库的结果集。"
    prompt += "注意：最后拿到结果集后根据用户的问题组织自然语言进行回答,不要返回原始数据，你也不要返回sql，你就返回自然语言来回答用户。"

    return prompt