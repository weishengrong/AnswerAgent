# AnswerAgent

面向业务数据库的自然语言问答（Text-to-SQL）后端：**FastAPI** + **LangGraph** 多智能体工作流。系统在**意图路由**后用主图将请求分发到 **简单 SQL 查询（子图）**、**ReAct 多步推理（子图）** 或 **日常聊天** 节点，集成混合 RAG（向量 + BM25 + RRF 融合）、分层记忆（Redis 短期 / mem0 长期）、SQL 校验与安全执行，并提供结构化错误降级与澄清对话闭环。

## 功能概览

- **意图识别**：单次 LLM 结构化调用输出意图（`database_query` / `daily_chat`）、`need_memory`、`needs_react` 等字段；LLM 超时/异常时自动**降级为聊天**（保守策略）。
- **LangGraph 主图**：`START → intent_router → route_after_intent → (sql_simple | sql_react | chat) → END`，条件路由决定子图去向。
- **子图（简单 SQL）**：`sql_simple` 子图负责常规单步查询，含 SQL 生成、校验与安全执行；通过 `astream_events` 将内部 LLM token 转发给外层事件流。
- **子图（ReAct 复杂查询）**：`sql_react` 子图提供多步 `Thought / Action / Observation` 推理，记录 `react_trace` 与步数，支持澄清与反思。
- **日常聊天**：`chat` 节点复用原 DataChat 逻辑，综合短期记忆、长期记忆与表结构检索上下文，流式输出。
- **混合 RAG**：ChromaDB 子块向量检索 + 父块拼装，BM25（jieba）与向量结果 **RRF 融合**，可选从数据库/Schema 自动重建索引（含 Schema 指纹变更检测）。
- **分层记忆**：Redis 短期会话滑动窗口与 Token 阈值摘要；长期记忆经 **mem0** 写入 ChromaDB，入库前经 `MemoryFilter` 多维加权过滤与过期清理。
- **可观测性**：JSON 结构化日志（含 `trace_id`）、Prometheus `/metrics` 指标采集、会话级 `session_id`。

## 环境要求

- Python **3.11+**（推荐）
- **MySQL**（异步驱动 `aiomysql`，连接串见下方 `DATABASE_URL`）
- **Redis**
- **ChromaDB**（本地持久化向量数据库，见 `CHROMA_PERSIST_DIR`）
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
| `CHROMA_PERSIST_DIR` | ChromaDB 数据持久化目录（默认 `./chroma_data`） |

可选：`app/core/config/settings.py` 中还有会话记忆阈值、`RAG_COLLECTION_NAME`、Agent 重试次数等，均可通过 `.env` 覆盖（参见 pydantic-settings 字段名）。

### 3. 初始化数据库（示例）

可将 `scripts/db/init_test_data.sql` 导入你的 MySQL 库（表结构与种子数据依项目而定）。

### 4. 构建 Schema RAG 索引（可选但推荐）

启动时服务会自动检测并构建/重建索引（含 Schema 变更指纹检测）。也可以手动执行：

```bash
python -m scripts.index.rebuild_rag_index --source sql --file resource/schema.sql --enrich
```

或者 `python -m scripts.index.rebuild_rag_index --source db` 直接从数据库读取并重建。

### 5. 启动 API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- 健康检查：`GET /health`
- 同步问答：`POST /ask`（表单字段：`question`、`user_id`、`session_id` 可选）
- 流式问答：`POST /ask/stream`（SSE，便于前端展示 Think / 工具调用 / 回答流）
- 指标：`GET /metrics`（Prometheus）

## 用户认证

系统支持用户注册和登录功能：

- `POST /api/auth/register` - 用户注册
- `POST /api/auth/login` - 用户登录
- `GET /api/auth/me` - 获取当前用户信息

注册和登录支持滑动验证码验证。

## 目录结构（节选）

```
app/
  agent/           # LangGraph 主图(graph.py)、意图路由、sql_simple/sql_react 子图、状态
  api/v1/          # API 路由（ask / auth / session）
  auth/            # 用户认证（注册/登录/JWT）
  core/            # 配置(settings)、LLM、Redis
  rag/             # ChromaDB、BM25、检索、查询改写、重排(RRF)
  memory/          # 会话记忆、过滤、mem0 封装
  mcp/             # MCP 工具桥接（可选）
prompts/           # 意图、聊天、SQL 生成/反思等提示模板
scripts/           # db/ 数据库、index/ 索引构建、evaluation/ 评测、livesqlbench/ 评测、misc/ 工具
resource/          # DDL 等静态资源
```

## 评测脚本（可选）

```bash
python -m scripts.evaluation.evaluate_intent      # 意图与路由相关 case
python -m scripts.evaluation.evaluate_react       # ReAct 健壮性与反思路径
```

## MCP

项目在部分技能中可挂载 MCP Server（时间、图表等），依赖本地配置与环境变量；未配置时相关 Server 会自动禁用，不影响核心 Text-to-SQL 路径。

## 许可证

未另行声明时，按仓库所有者约定为准。
