# AnswerAgent

面向业务数据库的自然语言问答（Text-to-SQL）后端：**FastAPI** + **Pydantic AI** 工作流，集成混合 RAG（向量 + BM25）、分层记忆（Redis / Milvus）、SQL 校验与安全执行，并对复杂查询提供 **ReAct** 多步推理与结构化错误反思闭环。

## 功能概览

- **意图识别**：单次 LLM 调用输出结构化字段（查库 / 闲聊、`need_memory`、`needs_react` 等）。
- **技能路由**：`SkillSelector` 按各技能 `should_handle` 打分，分流 **管道式简单查询**、**ReAct 复杂查询**、**对话式 DataChat**。
- **ReAct**：每步约束 `ReactStep`（thought / action / parameters），降低文本解析导致的静默参数丢失。
- **反思**：`ReflectErrorTool` 输出结构化 `ReflectionResult`（含 `next_action` 枚举），主流程按分支补检索、回灌上下文或中止澄清。
- **RAG**：Milvus 子块向量检索 + Redis 父块拼装；BM25（jieba）与向量结果 **RRF 融合**；可选从 DDL 重建索引。
- **记忆**：Redis 会话滑动窗口与 Token 阈值摘要；长期记忆经 **mem0** 写入 Milvus，入库前 **MemoryFilter** 多维度加权过滤。

## 环境要求

- Python **3.11+**（推荐）
- **MySQL**（异步驱动 `aiomysql`，连接串见下方 `DATABASE_URL`）
- **Redis**
- **Milvus**（本项目默认可走 **Milvus Lite** 本地文件，见 `MILVUS_LITE_PATH`）
- 兼容 OpenAI API 的 **对话 LLM** 与 **Embedding** 服务（云端或本地 Ollama 等）

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env`，至少包含：

| 变量 | 说明 |
|------|------|
| `DATABASE_URL` | SQLAlchemy 异步 URL，例如 `mysql+aiomysql://user:pass@host:3306/dbname` |
| `LLM_API_KEY` | 对话模型 API Key |
| `LLM_BASE_URL` | 对话模型 Base URL |
| `LLM_MODEL_NAME` | 模型名 |
| `EMBEDDING_MODEL_API_KEY` | 向量模型 Key |
| `EMBEDDING_MODEL_URL` | 向量服务 Base URL |
| `EMBEDDING_MODEL_NAME` | 向量模型名 |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` / `REDIS_PASSWORD` | Redis |
| `MILVUS_LITE_PATH` | Milvus Lite 数据路径（默认 `./milvus_data.db`） |

可选：`config/settings.py` 中还有会话记忆阈值、`RAG_COLLECTION_NAME`、Agent 重试次数等，均可通过 `.env` 覆盖（参见 pydantic-settings 字段名）。

### 3. 初始化数据库（示例）

可将 `scripts/init_test_data.sql` 导入你的 MySQL 库（表结构与种子数据依项目而定）。

### 4. 构建 Schema RAG 索引（可选但推荐）

```bash
python scripts/rebuild_rag_index.py --source sql --file resource/schema.sql --enrich
```

启动应用时会尝试从 Redis 加载 BM25 索引；若未构建会降级为纯向量检索。

### 5. 启动 API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- 健康检查：`GET /health`
- 同步问答：`POST /ask`（表单字段：`question`、`user_id`、`session_id` 可选）
- 流式问答：`POST /ask/stream`（SSE，便于前端展示 Think / 工具调用 / 回答流）

## 目录结构（节选）

```
app/
  agent/          # 工作流、意图、技能（simple / complex / data_chat）、状态
  rag/            # Milvus、BM25、检索与融合
  memory/         # 会话记忆、过滤、mem0 封装
  mcp/            # MCP 工具桥接（可选）
config/           # settings、Redis、DB
prompts/          # Schema 与 Few-shot 等提示模板
scripts/          # 索引重建、意图/ReAct 评测等
resource/         # DDL 等静态资源
```

## 评测脚本（可选）

```bash
python scripts/evaluate_intent.py      # 意图与路由相关 case
python scripts/evaluate_react.py       # ReAct 健壮性与反思路径
```

## MCP

项目在部分技能中可挂载 MCP Server（时间、图表等），依赖本地配置与环境变量；未配置时相关 Server 会自动禁用，不影响核心 Text-to-SQL 路径。

## 许可证

未另行声明时，按仓库所有者约定为准。
