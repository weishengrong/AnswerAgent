-- ====================================================================
-- AnswerAgent 数据库 Schema（纯 DDL）
-- ====================================================================
-- 用途：给 scripts/rebuild_rag_index.py 解析、切块、灌入 Milvus + Redis
-- 注意：此文件只包含 CREATE TABLE，不含 INSERT 数据
-- 用法：
--   python scripts/rebuild_rag_index.py --source sql --file resource/schema.sql
--   或追加 --enrich 让 LLM 补全字段描述
-- ====================================================================


CREATE TABLE department (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '主键ID',
    name VARCHAR(64) NOT NULL COMMENT '部门名称（如：研发部、市场部、行政部、财务部、运营部）',
    code VARCHAR(32) COMMENT '部门编码（如：RD、MK、HR、FN、OP）',
    manager_id BIGINT COMMENT '部门负责人用户ID（关联 user.id）',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='部门信息表';


CREATE TABLE user (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '主键ID',
    number VARCHAR(255) COMMENT '工号（如 RD001、MK002）',
    name VARCHAR(64) NOT NULL COMMENT '员工姓名',
    password VARCHAR(64) COMMENT '密码（md5 加密）',
    sex TINYINT COMMENT '性别（0=女, 1=男）',
    birth VARCHAR(64) COMMENT '出生年月',
    user_type TINYINT DEFAULT 2 COMMENT '用户类型（0=超级管理员, 1=管理员, 2=普通用户/员工）',
    status TINYINT DEFAULT 0 COMMENT '账号状态（0=正常, 1=禁用, -1=删除）',
    department_id BIGINT COMMENT '所属部门ID（关联 department.id）',
    email VARCHAR(255) COMMENT '邮箱地址',
    phone VARCHAR(11) COMMENT '手机号',
    attend_type INT DEFAULT 1 COMMENT '是否参与考勤（0=不参与, 1=参与）',
    info_submitted INT DEFAULT 1 COMMENT '信息是否齐全（0=不齐全, 1=齐全）',
    image_url TEXT COMMENT '人脸图片URL（多张用逗号分隔）',
    video_url TEXT COMMENT '视频URL（多个用逗号分隔）',
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    information VARCHAR(255) COMMENT '是否接受邮件提醒',
    wechat_id VARCHAR(255) COMMENT '微信ID',
    auto_check_count INT DEFAULT 0 COMMENT '每周自动打卡次数',
    emergency_name VARCHAR(255) COMMENT '紧急联系人姓名',
    emergency_phone VARCHAR(11) COMMENT '紧急联系人电话'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户/员工信息表';


CREATE TABLE device (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '主键ID',
    name VARCHAR(255) COMMENT '设备名称（如前门打卡机、研发部门禁）',
    model VARCHAR(255) COMMENT '设备品牌型号（如 ZK-T5）',
    ip VARCHAR(255) COMMENT '设备IP地址',
    address VARCHAR(255) COMMENT '设备安装位置',
    department_id BIGINT COMMENT '所属部门ID（NULL 表示公共设备）',
    status TINYINT DEFAULT 0 COMMENT '设备状态（0=正常, 1=关闭/停用）',
    code VARCHAR(255) COMMENT '设备识别码',
    type INT DEFAULT 0 COMMENT '设备类别（0=指纹机器）'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='考勤设备表（指纹机/打卡机）';


CREATE TABLE check_record (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '主键ID',
    check_time DATETIME COMMENT '员工打卡的具体时间',
    type TINYINT COMMENT '打卡类型（0=进入/上班, 1=离开/下班）',
    user_id BIGINT COMMENT '打卡员工ID（关联 user.id）',
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '记录创建时间',
    device_id BIGINT COMMENT '打卡设备ID（关联 device.id）',
    `desc` VARCHAR(255) COMMENT '备注信息（如 迟到、早退、加班）'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='员工打卡记录表';
