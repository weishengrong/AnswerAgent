


# 数据库Schema定义
DATABASE_SCHEMA = {
    "tables": [
        {
            "name": "department",
            "description": "部门信息表，记录公司各部门的基本信息和负责人",
            "columns": [
                {"name": "id", "type": "bigint", "description": "主键ID"},
                {"name": "name", "type": "varchar(64)", "description": "部门名称（研发部、市场部、行政部、财务部、运营部）"},
                {"name": "code", "type": "varchar(32)", "description": "部门编码（RD/MK/HR/FN/OP）"},
                {"name": "manager_id", "type": "bigint", "description": "部门负责人用户ID（关联 user.id）"},
                {"name": "create_time", "type": "datetime", "description": "创建时间"}
            ],
            "examples": [
                {"id": 1, "name": "研发部", "code": "RD", "manager_id": 1},
                {"id": 2, "name": "市场部", "code": "MK", "manager_id": 11}
            ]
        },
        {
            "name": "user",
            "description": "用户/员工信息表，包含员工的基本信息、账号状态及考勤设置",
            "columns": [
                {"name": "id", "type": "bigint", "description": "主键ID"},
                {"name": "number", "type": "varchar(255)", "description": "工号 (如 RD001、MK002)"},
                {"name": "name", "type": "varchar(64)", "description": "员工姓名"},
                {"name": "password", "type": "varchar(64)", "description": "密码 (md5加密)"},
                {"name": "sex", "type": "tinyint", "description": "性别 (0=女, 1=男)"},
                {"name": "birth", "type": "varchar(64)", "description": "出生年月 (示例: 2020-01-01)"},
                {"name": "user_type", "type": "tinyint", "description": "用户类型 (0=超级管理员, 1=管理员, 2=普通用户/员工)"},
                {"name": "status", "type": "tinyint", "description": "账号状态 (0=正常, 1=禁用, -1=删除)"},
                {"name": "department_id", "type": "bigint", "description": "部门ID (关联 department.id)"},
                {"name": "email", "type": "varchar(255)", "description": "邮箱地址"},
                {"name": "phone", "type": "varchar(11)", "description": "手机号"},
                {"name": "attend_type", "type": "int", "description": "是否参与考勤 (0=不参加, 1=参加)"},
                {"name": "info_submitted", "type": "int", "description": "信息是否齐全 (0=不齐全, 1=齐全)"},
                {"name": "image_url", "type": "text", "description": "人脸图片URL (多张用逗号分隔)"},
                {"name": "video_url", "type": "text", "description": "视频URL (用逗号分隔)"},
                {"name": "create_time", "type": "datetime", "description": "创建时间"},
                {"name": "update_time", "type": "datetime", "description": "更新时间"},
                {"name": "information", "type": "varchar(255)", "description": "是否接受邮件提醒"},
                {"name": "wechat_id", "type": "varchar(255)", "description": "微信ID"},
                {"name": "auto_check_count", "type": "int", "description": "每周自动打卡次数"},
                {"name": "emergency_name", "type": "varchar(255)", "description": "紧急联系人姓名"},
                {"name": "emergency_phone", "type": "varchar(11)", "description": "紧急联系人电话"}
            ],
            "examples": [
                {
                    "number": "RD001",
                    "name": "张伟",
                    "sex": 1,
                    "user_type": 1,
                    "status": 0,
                    "department_id": 1,
                    "phone": "13800000001"
                }
            ]
        },
        {
            "name": "device",
            "description": "考勤设备信息表，记录指纹机/打卡机的位置和状态",
            "columns": [
                {"name": "id", "type": "bigint", "description": "主键ID"},
                {"name": "name", "type": "varchar(255)", "description": "设备名称（如前门打卡机、研发部门禁）"},
                {"name": "model", "type": "varchar(255)", "description": "品牌型号"},
                {"name": "ip", "type": "varchar(255)", "description": "设备IP地址"},
                {"name": "address", "type": "varchar(255)", "description": "设备安装位置"},
                {"name": "department_id", "type": "bigint", "description": "所属部门ID（NULL 表示公共设备）"},
                {"name": "status", "type": "tinyint", "description": "设备状态 (0=正常, 1=关闭)"},
                {"name": "code", "type": "varchar(255)", "description": "设备识别码"},
                {"name": "type", "type": "int", "description": "设备类别 (0=指纹机器)"}
            ],
            "examples": [
                {"id": 1, "name": "前门打卡机", "model": "ZK-T5", "ip": "192.168.1.100",
                 "address": "公司前台", "department_id": None, "status": 0, "type": 0}
            ]
        },
        {
            "name": "check_record",
            "description": "员工打卡记录表，记录员工每次进出考勤的行为",
            "columns": [
                {"name": "id", "type": "bigint", "description": "主键ID"},
                {"name": "check_time", "type": "datetime", "description": "打卡具体时间"},
                {"name": "type", "type": "tinyint", "description": "打卡类型 (0=进入/上班, 1=离开/下班)"},
                {"name": "user_id", "type": "bigint", "description": "员工ID (关联 user.id)"},
                {"name": "create_time", "type": "timestamp", "description": "记录创建时间"},
                {"name": "device_id", "type": "bigint", "description": "打卡设备ID (关联 device.id)"},
                {"name": "desc", "type": "varchar(255)", "description": "备注信息（如 迟到/早退/加班）"}
            ],
            "examples": [
                {
                    "check_time": "2026-05-01 08:30:00",
                    "type": 0,
                    "user_id": 1,
                    "device_id": 1,
                    "desc": None
                },
                {
                    "check_time": "2026-05-01 17:30:00",
                    "type": 1,
                    "user_id": 1,
                    "device_id": 1,
                    "desc": None
                }
            ]
        }
    ],
    "relationships": [
        "user.department_id 关联 department.id（查询员工所属部门）",
        "department.manager_id 关联 user.id（查询部门负责人）",
        "check_record.user_id 关联 user.id（查询打卡员工姓名/部门）",
        "check_record.device_id 关联 device.id（查询打卡设备位置）",
        "device.department_id 关联 department.id（查询设备归属部门）"
    ],
    "business_notes": [
        "查询某部门员工时：必须先 JOIN department，因为 user 表只存 department_id 不存名字",
        "查询打卡情况时：通常需要 user JOIN check_record，按 check_time 聚合",
        "判断迟到：上班打卡（type=0）的 check_time 晚于 09:00 视为迟到",
        "判断早退：下班打卡（type=1）的 check_time 早于 17:00 视为早退",
        "判断加班：下班打卡（type=1）的 check_time 晚于 19:00 视为加班",
        "查询有效员工时一般加 status=0（正常账号）和 attend_type=1（参与考勤）",
        "【条件回退查询规则】当用户问题包含『有X就查X，没有就查Y』『先看X，没有用Y』『if-else / 优先A否则B』等条件回退/兜底语义时，"
        "生成的 SQL 必须满足：(1) SELECT 字段中显式包含被筛选的主体识别字段（如 u.name、u.number、d.name），"
        "让结果集能够自证最终命中的是哪个分支；(2) 优先用 UNION 写法或 JOIN 后 WHERE 过滤，避免在 WHERE 中嵌套 CASE WHEN EXISTS 这种"
        "把分支信息隐藏在子查询里、SELECT 看不出命中分支的写法。例如：用户问『查韦胜荣的打卡，没有就查邓超的』，正确写法是 "
        "JOIN user 取 u.name 后用 WHERE u.name=COALESCE(...,'邓超') 或 UNION ALL，让结果集每行都有 name 字段。"
    ]
}

def format_schema_for_prompt(schema):
    """将Schema格式化为LLM友好的文本"""
    prompt = "数据库结构如下：\n\n"
    for table in schema["tables"]:
        prompt += f"表名：{table['name']}（{table['description']}）\n"
        prompt += "字段：\n"
        for col in table["columns"]:
            prompt += f"  - {col['name']} ({col['type']})：{col['description']}\n"

        if "examples" in table:
            prompt += "示例数据：\n"
            for ex in table["examples"]:
                prompt += f"  {ex}\n"
        prompt += "\n"

    if "relationships" in schema:
        prompt += "表间关系：\n"
        for rel in schema["relationships"]:
            prompt += f"  - {rel}\n"
        prompt += "\n"

    if "business_notes" in schema:
        prompt += "业务规则：\n"
        for note in schema["business_notes"]:
            prompt += f"  - {note}\n"

    return prompt
