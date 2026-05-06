# AnswerAgent 项目深度解析

> 基于 AnswerAgent 项目源码的完整架构与实现分析，涵盖健康检查、记忆管理、并发能力、链路追踪、Agent 架构、接口设计、事件推送等核心模块。

---

## 目录

- [一、Milvus 健康检查与优雅降级](#一milvus-健康检查与优雅降级)
  - [1.1 入口：if service_health.milvus_healthy](#11-入口if-service_healthmilvus_healthy)
  - [1.2 ServiceHealth 类完整拆解](#12-servicehealth-类完整拆解)
  - [1.3 try_recover_milvus 自动恢复机制](#13-try_recover_milvus-自动恢复机制)
  - [1.4 cleanup_expired_memories 过期记忆清理](#14-cleanup_expired_memories-过期记忆清理)
  - [1.5 完整调用链路图](#15-完整调用链路图)
  - [1.6 设计模式总结](#16-设计模式总结)
- [二、启动时记忆清理机制](#二启动时记忆清理机制)
- [三、双层记忆架构与清理范围](#三双层记忆架构与清理范围)
  - [3.1 短期记忆 vs 长期记忆](#31-短期记忆-vs-长期记忆)
  - [3.2 只清理 user_id="default" 的问题](#32-只清理-user_iddefault-的问题)
  - [3.3 短期到长期的记忆流转](#33-短期到长期的记忆流转)
- [四、高并发能力分析](#四高并发能力分析)
  - [4.1 致命问题：同步阻塞事件循环](#41-致命问题同步阻塞事件循环)
  - [4.2 严重问题：资源浪费与泄漏](#42-严重问题资源浪费与泄漏)
  - [4.3 中等问题：缺少并发控制](#43-中等问题缺少并发控制)
  - [4.4 量化估算](#44-量化估算)
  - [4.5 改造优先级](#45-改造优先级)
- [五、链路追踪：TraceIdFilter](#五链路追踪traceidfilter)
  - [5.1 代码逐行解读](#51-代码逐行解读)
  - [5.2 ContextVar 协程隔离](#52-contextvar-协程隔离)
  - [5.3 日志格式引用](#53-日志格式引用)
  - [5.4 实际用途](#54-实际用途)
- [六、整体架构分析](#六整体架构分析)
  - [6.1 架构全景图](#61-架构全景图)
  - [6.2 逐层拆解](#62-逐层拆解)
  - [6.3 架构风格总结](#63-架构风格总结)
- [七、/ask 接口详解](#七ask-接口详解)
  - [7.1 完整代码逐行解读](#71-完整代码逐行解读)
  - [7.2 /ask 接口完整时序](#72-ask-接口完整时序)
- [八、/ask/stream 接口详解](#七askstream-接口详解)
  - [8.1 完整代码逐行解读](#81-完整代码逐行解读)
  - [8.2 /ask/stream 接口完整时序](#82-askstream-接口完整时序)
- [九、Emitter 事件推送机制](#九emitter-事件推送机制)
  - [9.1 EventEmitter 与 NullEmitter](#91-eventemitter-与-nullemitter)
  - [9.2 为什么 /ask 不传 emitter](#92-为什么-ask-不传-emitter)
  - [9.3 为什么设计 event_stream()](#93-为什么设计-event_stream)
  - [9.4 StreamingResponse 的精确时序](#94-streamingresponse-的精确时序)
- [十、两个接口核心区别](#十两个接口核心区别)

---

## 一、Milvus 健康检查与优雅降级

### 1.1 入口：if service_health.milvus_healthy

`if service_health.milvus_healthy:` 出现在 `app/main.py` 第31行，位于 FastAPI 的 `lifespan` 生命周期管理函数中。它是一个**条件守卫（guard clause）**，用于判断 Milvus 向量数据库当前是否可用。只有当 Milvus 健康时，才执行依赖 Milvus 的操作（如清理过期记忆）；否则跳过，进入降级模式。

完整上下文：

```python
# 第一步：尝试连接 Milvus
try:
    from pymilvus import connections
    connections.connect(
        host=milvus_settings.MILVUS_HOST,
        port=milvus_settings.MILVUS_PORT
    )
    service_health.mark_milvus_up()       # 连接成功 → 标记健康
    logging.info("✅ Milvus 连接成功")
except Exception as e:
    service_health.mark_milvus_down()     # 连接失败 → 标记不健康
    logging.warning(f"⚠️ Milvus 连接失败（降级运行）：{e}")

# 第二步：根据健康状态决定是否清理过期记忆
if service_health.milvus_healthy:         # ← 条件守卫
    # Milvus 可用 → 执行清理
else:
    # Milvus 不可用 → 跳过清理
```

### 1.2 ServiceHealth 类完整拆解

`service_health` 是 `app/middleware.py` 第198行创建的 `ServiceHealth` 类全局单例：

```python
service_health = ServiceHealth()
```

#### 构造函数 `__init__`

```python
def __init__(self):
    self._milvus_healthy = True    # 乐观初始化：默认健康
    self._redis_healthy = True
    self._llm_healthy = True
```

设计意图：默认假设所有服务都是健康的。这是一种"乐观策略"——先假设一切正常，遇到问题再标记为异常。

#### `milvus_healthy` 属性（@property）

```python
@property
def milvus_healthy(self) -> bool:
    return self._milvus_healthy
```

`@property` 把方法变成属性访问方式。外部写 `service_health.milvus_healthy` 看起来像访问一个变量，但实际是调用了一个方法。用 `@property` + 私有变量 `_milvus_healthy`，外部只能**读取**，不能直接修改，必须通过 `mark_milvus_down()` / `mark_milvus_up()` 来变更状态。

#### `mark_milvus_down()` —— 标记 Milvus 不可用

```python
def mark_milvus_down(self):
    self._milvus_healthy = False
    logger.warning("🔴 Milvus 标记为不可用，降级为无 RAG 模式")
```

纯状态标记方法，直接将 `_milvus_healthy` 设为 `False`，不尝试重连，不抛异常。

#### `mark_milvus_up()` —— 标记 Milvus 恢复可用

```python
def mark_milvus_up(self):
    if not self._milvus_healthy:       # 只有当前是不健康状态才操作
        self._milvus_healthy = True
        logger.info("🟢 Milvus 恢复可用")
```

幂等性保护：`if not self._milvus_healthy` 确保只有从 `False` → `True` 的转换才会打印日志，避免刷屏。

### 1.3 try_recover_milvus 自动恢复机制

```python
def try_recover_milvus(self) -> bool:
    # 快速路径：如果已经健康，直接返回 True
    if self._milvus_healthy:
        return True

    # 慢路径：尝试重新连接
    try:
        from pymilvus import connections
        from config.settings import milvus_settings
        connections.connect(
            host=milvus_settings.MILVUS_HOST,
            port=milvus_settings.MILVUS_PORT,
            alias=f"health_check_{uuid.uuid4().hex[:8]}"
        )
        self.mark_milvus_up()    # 连接成功 → 标记恢复
        return True
    except Exception as e:
        logger.debug(f"Milvus 健康探测失败：{e}")
        return False
```

逐行解析：

1. **`if self._milvus_healthy: return True`** —— 快速返回优化。如果 Milvus 已经是健康的，不需要浪费时间重连。
2. **`from pymilvus import connections`** —— 延迟导入。不在模块顶部导入，即使 `pymilvus` 包没安装，其他功能仍可正常工作。
3. **`alias=f"health_check_{uuid.uuid4().hex[:8]}"`** —— 随机别名避免冲突。pymilvus 的 `connections.connect()` 如果使用已存在的 alias 会报错。
4. **`logger.debug(...)`** —— 恢复探测失败用 `debug` 级别而非 `warning`，因为这是预期内的情况。

调用时机：
- `main.py:38-39`：启动清理记忆失败后
- `main.py:108-109`：每次 `/ask` 请求进来时
- `main.py:155-156`：每次 `/ask/stream` 请求进来时

### 1.4 cleanup_expired_memories 过期记忆清理

定义在 `app/memory/filter.py` 第597-666行，`memory_filter` 是 `MemoryFilter` 类的全局单例。

**衰减模型**：`score = e^(-λ × t) × 10`

四种记忆类型的衰减系数：

| 类型 | λ | 半衰期（约） | 含义 |
|------|---|-------------|------|
| `state` | 10.0 | ~4分钟 | 状态信息，极短命 |
| `chat` | 2.0 | ~21分钟 | 对话内容，短命 |
| `event` | 0.5 | ~1.4小时 | 事件记录，中等寿命 |
| `fact` | 0.01 | ~69小时 | 事实知识，长命 |

过期判定条件：
- `state` 类型：score < 0.1 且 > 1小时 → 删除
- `chat` 类型：score < 0.1 且 > 6小时 → 删除
- `event` 类型：score < 0.1 且 > 48小时 → 删除
- `fact` 类型：衰减极慢，不会被此方法清理

### 1.5 完整调用链路图

```
应用启动 (lifespan)
  │
  ├─ pymilvus.connections.connect(host, port)
  │    ├─ 成功 → service_health.mark_milvus_up()
  │    └─ 失败 → service_health.mark_milvus_down()
  │
  ├─ if service_health.milvus_healthy:
  │    ├─ True → memory_filter.cleanup_expired_memories("default")
  │    │         ├─ memory_manager.get_all() → 查询 Milvus
  │    │         ├─ 遍历记忆，计算 exp(-λ×t)×10 衰减分
  │    │         ├─ 过期判定 + memory_manager.memory.delete()
  │    │         └─ 返回 {"deleted": n, "kept": m, "details": [...]}
  │    │    如果清理过程中异常：
  │    │    → if not service_health.milvus_healthy:
  │    │         → try_recover_milvus()
  │    └─ False → 打印 "Milvus 不可用，跳过过期记忆清理"
  │
  └─ 继续加载 BM25 索引...
```

### 1.6 设计模式总结

| 模式 | 体现 |
|------|------|
| **单例模式** | `service_health = ServiceHealth()` 全局唯一实例 |
| **优雅降级** | Milvus 挂了不崩溃，跳过依赖它的操作继续运行 |
| **自动恢复** | 每次请求检查 + `try_recover_milvus()` 尝试重连 |
| **幂等性** | `mark_milvus_up()` 重复调用不会重复打日志 |
| **快速路径优化** | `try_recover_milvus()` 已健康时直接返回 |
| **延迟导入** | `from pymilvus import connections` 在方法内导入 |
| **属性封装** | `@property` 保护状态变量，外部只读不可写 |
| **乐观初始化** | 默认 `True`，先假设健康，出问题再标记 |

---

## 二、启动时记忆清理机制

每次服务启动都会尝试清理过期记忆，但有一个前提条件：Milvus 必须健康。

`lifespan` 是 FastAPI 的生命周期钩子，在应用启动时执行 `yield` 之前的代码。所以每次服务启动，清理逻辑都会跑一遍。

三种情况：

| 启动时 Milvus 状态 | 行为 |
|---|---|
| 连接成功 | ✅ 执行 `cleanup_expired_memories("default")` |
| 连接失败 | ❌ 跳过清理，打印 "Milvus 不可用，跳过过期记忆清理" |
| 连接成功但清理过程中 Milvus 断了 | ⚠️ 捕获异常，尝试 `try_recover_milvus()` 恢复 |

---

## 三、双层记忆架构与清理范围

### 3.1 短期记忆 vs 长期记忆

| | 短期记忆 | 长期记忆 |
|---|---|---|
| **管理器** | `smart_session_memory` | `memory_manager` |
| **存储后端** | Redis | Milvus |
| **数据形态** | JSON 消息列表 | 向量嵌入 + 元数据 |
| **生命周期** | TTL 1小时自动过期 | 永久存储，半衰期衰减 |
| **清理方式** | Redis TTL 自动淘汰 | `cleanup_expired_memories()` 主动清理 |
| **检索方式** | 按范围读取（lrange） | 语义搜索（embedding 相似度） |
| **容量限制** | 最多30条消息 | 无硬性限制 |
| **跨会话** | 否（按 session_id 隔离） | 是（按 user_id 全局共享） |

短期记忆存在 Redis 里，有 TTL（1小时），到期 Redis 自动删除，不需要手动清理。长期记忆存在 Milvus 里，没有 TTL 机制，如果不主动清理就会一直堆积。所以 `cleanup_expired_memories()` 清理的就是 Milvus 中的长期记忆。

### 3.2 只清理 user_id="default" 的问题

```python
result = memory_filter.cleanup_expired_memories(user_id="default")
```

硬编码了 `"default"`，意味着启动时只清理 default 用户的过期记忆，其他用户的过期记忆不会被清理。

原因分析：
1. **当前系统以单用户为主** —— `/ask` 接口的 `user_id: str = "default"`，大部分使用场景可能就一个用户
2. **避免启动时间过长** —— 遍历所有用户清理会导致启动时间随用户数量线性增长
3. **清理不是紧急操作** —— 过期记忆多存一会儿不会导致系统崩溃

这是一个**设计不足**，如果系统有多用户场景，其他用户的过期记忆永远不会被清理。

### 3.3 短期到长期的记忆流转

短期记忆在消息数量达到 `HARD_LIMIT`（30条）时，触发批量评估：

1. `memory_filter.evaluate_batch()` 通过 LLM 多维度评分（持久性、独特性、可操作性等），筛选出值得长期保存的内容
2. 筛选通过的记忆通过 `memory_manager.add_memory()` 写入 Milvus，成为长期记忆
3. 会话结束（`clear()`）时也会触发一次最终评估

---

## 四、高并发能力分析

**结论：当前项目扛不住高并发。**

### 4.1 致命问题：同步阻塞事件循环

FastAPI 基于 `asyncio` 事件循环，一个同步阻塞调用会卡住整个服务。

#### 问题1：memory_manager 所有方法都是同步的

`mem0.Memory` 的 `add/search/get_all/delete` 全是同步 HTTP 调用，但在 async 函数中被直接调用：

```python
# intent.py — 异步函数中同步调用
async def retrieve_long_term_memory(self, user_input, user_id):
    memories = memory_manager.search_memory(query=..., user_id=user_id)  # 同步阻塞！

# session.py — 异步函数中同步调用
async def _trigger_batch_evaluate(self, messages, user_id, session_id):
    memory_manager.add_memory([...], user_id=user_id)  # 同步阻塞！
```

#### 问题2：Milvus client.search() 是同步的

```python
async def retrieve_child_chunks_async(question, limit=10):
    vector = await _get_embedding_vector_async(question)
    client = _get_milvus_client()
    result = client.search(...)  # 同步阻塞！
```

#### 问题3：Embedding 调用也是同步的

`filter.py` 中在 async 上下文中直接调用同步 OpenAI 客户端计算 embedding，连 `run_in_executor` 都没用。

一个典型请求至少有 **300ms+ 的同步阻塞**，10个并发就是 3 秒的串行等待。

### 4.2 严重问题：资源浪费与泄漏

- **每次调用都新建客户端**：`retrieval.py` 中每次获取 embedding 都新建 `OpenAI` 客户端；`mcp_tools.py` 每次调用都新建 `httpx.AsyncClient`；`sql_tool.py` 每次执行都新建 LLM Agent
- **所有单例初始化无锁保护**：`MilvusClientFactory.__new__`、`get_redis_client()`、`_get_client()` 等都存在竞态条件
- **`_eval_counter` 字典非线程安全**

### 4.3 中等问题：缺少并发控制

整个项目没有：
- ❌ `asyncio.Semaphore` 限制并发数
- ❌ LLM API 速率限制
- ❌ 请求排队/背压机制
- ❌ 全局超时控制
- ❌ 数据库连接池调优

### 4.4 量化估算

| 并发数 | 理想吞吐量 (req/s) | 实际吞吐量 (估算) | 瓶颈 |
|--------|-------------------|------------------|------|
| 1 | ~0.3 | ~0.3 | LLM 延迟 |
| 5 | ~1.5 | ~0.5 | 同步阻塞串行化 |
| 10 | ~3.0 | ~0.6 | 事件循环严重阻塞 |
| 50 | ~15 | ~0.8 | 全面崩溃 |

### 4.5 改造优先级

| 优先级 | 改造项 | 效果 |
|--------|--------|------|
| **P0** | `memory_manager` 同步调用改 `run_in_executor` | 消除最大阻塞点 |
| **P0** | Milvus `client.search()` 改 `run_in_executor` | 消除第二大阻塞点 |
| **P1** | 复用 OpenAI/httpx 客户端 | 减少连接开销 |
| **P1** | 单例初始化加 `asyncio.Lock` | 消除竞态条件 |
| **P2** | 加 `asyncio.Semaphore(10)` 限制工作流并发 | 防止 LLM API 被打满 |
| **P2** | 数据库/Redis 连接池调优 | 提升连接复用率 |
| **P3** | `sql_tool.py` 中 Agent 实例复用 | 减少对象创建开销 |

---

## 五、链路追踪：TraceIdFilter

### 5.1 代码逐行解读

```python
class TraceIdFilter(logging.Filter):
    def filter(self, record):
        record.trace_id = get_trace_id()   # 从上下文中取出 trace_id，挂到日志记录上
        return True                         # 返回 True 表示这条日志要输出

for handler in logging.root.handlers:
    handler.addFilter(TraceIdFilter())      # 把过滤器加到所有根日志处理器上
```

`logging.Filter` 的 `filter()` 方法每条日志输出前都会经过。返回 `True` 日志正常输出，返回 `False` 日志被丢弃。这里永远返回 `True`，所以它不过滤任何日志，而是利用 Filter 的"每条日志必经"特性，往日志记录上动态注入 `trace_id` 字段。

### 5.2 ContextVar 协程隔离

```python
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

def get_trace_id() -> str:
    return trace_id_var.get()

def new_trace_id() -> str:
    return str(uuid.uuid4())[:12]
```

`ContextVar` 是 Python 3.7+ 提供的**协程安全的上下文变量**：
- 每个 asyncio Task 有自己独立的副本，互不干扰
- 请求 A 设置的 trace_id，请求 B 读不到
- 比线程局部变量（`threading.local`）更适合 async 环境

trace_id 在 `workflow.execute()` 中设置：

```python
async def execute(self, user_input, user_id, session_id, emitter):
    tid = new_trace_id()     # 生成新的 trace_id
    set_trace_id(tid)        # 写入 ContextVar
```

### 5.3 日志格式引用

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s] %(message)s'
)
```

`[%(trace_id)s]` 不是 logging 的内置字段，而是 `TraceIdFilter` 动态注入的。

### 5.4 实际用途

当多个请求并发时，日志是交错输出的。有了 trace_id，一行命令就能过滤：

```bash
grep "a3f2b1c8-4d5e" app.log
```

立刻拿到某个请求的完整链路日志。这是分布式系统中经典的**链路追踪（Tracing）**思想的简化版。

---

## 六、整体架构分析

### 6.1 架构全景图

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI (main.py)                        │
│  /ask  /ask/stream  /session  /health                          │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  lifespan    │  │  TraceIdFilter   │  │  ServiceHealth   │  │
│  │  启动检查    │  │  日志链路追踪    │  │  降级容错        │  │
│  └──────────────┘  └──────────────────┘  └──────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Agent Workflow (workflow.py)                    │
│                    线性流水线编排器                               │
│                                                                 │
│   用户输入 ──→ IntentNode ──→ SkillSelector ──→ Skill.execute   │
│                  │                │                  │           │
│              意图识别          技能路由            技能执行       │
└─────────────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ SimpleQuery  │  │ ComplexQuery │  │   DataChat       │
│  单步SQL     │  │  ReAct多步   │  │   日常对话        │
│              │  │  推理循环    │  │                   │
└──────┬───────┘  └──────┬───────┘  └────────┬──────────┘
       │                 │                    │
       ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Tool Layer (tools/)                         │
│  SearchSchema  GenerateSQL  ExecuteSQL  FormatResult            │
│  SearchExamples ValidateSQL ReflectError AskUser                │
│  SearchMemory  BuildContext  MCPToolBridge                      │
└─────────────────────────────────────────────────────────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                     ▼
┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  RAG Layer  │  │  Memory Layer   │  │   MCP Layer      │
│  (rag/)     │  │  (memory/)      │  │   (mcp/)         │
│             │  │                 │  │                  │
│ Milvus向量  │  │ 短期:Redis+TTL  │  │  Registry        │
│ BM25关键词  │  │ 长期:Milvus+mem0│  │  Manager         │
│ RRF融合     │  │ Filter评分+衰减 │  │  Bridge          │
└─────────────┘  └─────────────────┘  └──────────────────┘
```

### 6.2 逐层拆解

#### 第一层：API 网关层（main.py）

FastAPI 异步 Web 框架 + 生命周期管理。负责路由、启动初始化、链路追踪、降级容错、SSE 流式。没有认证、限流、请求校验中间件。

#### 第二层：Agent 编排层（workflow.py）

**线性流水线（Pipeline）**模式。固定三步：意图识别 → 技能路由 → 技能执行。`AgentState` 是核心数据结构，所有节点读写同一个 state。

#### 第三层：意图识别层（intent.py）

**LLM 结构化输出**模式。用 `pydantic_ai` 的 `Agent` 让 LLM 输出严格符合 `IntentResult` schema 的结构化结果。

#### 第四层：技能路由层（selector.py）

**LLM 优先 + 规则兜底的双层路由**。先让 LLM 选择技能，失败则降级到每个 Skill 的 `should_handle(state) -> float` 规则匹配。

#### 第五层：技能执行层（skills/）

**策略模式 + 两种推理范式**：

| Skill | 推理范式 | 说明 |
|-------|---------|------|
| `SimpleQuerySkill` | 单步生成 | 一次 LLM 调用生成 SQL → 执行 → 格式化 |
| `ComplexQuerySkill` | ReAct 循环 | Think→Act→Observe 最多8轮迭代 |
| `DataChatSkill` | 直接对话 | 无工具调用，纯 LLM 对话 |

#### 第六层：工具层（tools/）

**工具抽象 + 适配器模式**。`BaseTool` 统一接口，`MCPToolBridge` 把 MCP Server 上的远程工具适配成本地接口。

#### 第七层：RAG 检索层（rag/）

**混合检索 + RRF 融合**。向量检索（Milvus）+ 关键词检索（BM25）+ RRF 排序融合。

#### 第八层：记忆层（memory/）

**双层记忆 + 自动沉淀**。短期记忆（Redis，滑动窗口+TTL）→ 批量评估 → 长期记忆（Milvus，向量+衰减）。

#### 第九层：MCP 扩展层（mcp/）

**插件化工具注册 + 懒加载连接**。Registry-Manager-Bridge 三层结构，新增 MCP Server 只需改配置文件。

### 6.3 架构风格总结

| 维度 | 架构风格 | 说明 |
|------|---------|------|
| **整体** | LLM-driven Agent Architecture | LLM 是决策核心 |
| **编排** | Linear Pipeline + ReAct Loop | 外层线性流水线，ComplexQuery 内部 ReAct 循环 |
| **路由** | AI-First + Rule Fallback | LLM 优先选择，规则兜底 |
| **状态** | Shared State Object | `AgentState` 贯穿全流程 |
| **工具** | Strategy + Adapter | 统一 BaseTool 接口，MCP 用适配器桥接 |
| **检索** | Hybrid RAG (Vector + BM25 + RRF) | 混合检索融合 |
| **记忆** | Dual-Layer + Auto-Consolidation | 双层记忆 + 自动沉淀 |
| **容错** | Graceful Degradation | 服务降级而非崩溃 |
| **扩展** | Plugin Registry | MCP Server 可插拔 |

---

## 七、/ask 接口详解

### 7.1 完整代码逐行解读

```python
@app.post("/ask")                              # POST 路由
async def handle_ask(
    question: str,                             # 用户问题，必填
    user_id: str = "default",                  # 用户ID，默认 "default"
    session_id: str = None                     # 会话ID，可选
):
    if not session_id:                         # 如果前端没传 session_id
        session_id = f"session-{str(uuid.uuid4())[:8]}"   # 生成新的

    if not service_health.milvus_healthy:       # Milvus 不可用时尝试恢复
        service_health.try_recover_milvus()

    # /ask 不传 emitter，workflow 内部会使用 NullEmitter（空实现），
    # 工作流执行过程中的所有中间事件都被 NullEmitter 静默吞掉，最终只返回完整结果。
    final_state = await workflow.execute(question, user_id, session_id=session_id)

    if final_state.response and not final_state.clarification_needed:
        # 有实际回答且不是澄清问题 → 保存到短期记忆
        try:
            await smart_session_memory.add_conversation(
                session_id=session_id,
                user_message=question,
                assistant_message=final_state.response,
                user_id=user_id
            )
        except Exception as e:
            logging.warning(f"⚠️ Redis 保存失败（降级）：{e}")
            service_health.mark_redis_down()    # 标记 Redis 不可用

    if final_state.clarification_needed:
        # 需要用户澄清 → 返回澄清类型响应
        return {
            "response_type": "clarification",
            "clarification_question": final_state.clarification_question,
            "session_id": session_id,
            "intent": final_state.intent,
            "skill": final_state.skill_name,
            "trace_id": get_trace_id()
        }

    # 正常回答
    return {
        "response_type": "answer",
        "response": final_state.response,
        "intent": final_state.intent,
        "session_id": session_id,
        "skill": final_state.skill_name,
        "trace_id": get_trace_id(),
        "react_trace": final_state.react_trace if final_state.react_trace else None
    }
```

### 7.2 /ask 接口完整时序

```
前端 POST /ask {"question": "研发部有多少人", "session_id": "xxx"}
  │
  ├─ 生成/复用 session_id
  ├─ Milvus 健康检查 + 恢复尝试
  │
  ├─ await workflow.execute()  ← 阻塞等待，可能 3-10 秒
  │    ├─ 意图识别 (LLM, ~2s)
  │    ├─ 技能路由 (LLM, ~1s)
  │    └─ 技能执行 (LLM+工具, ~2-7s)
  │
  ├─ 保存对话到 Redis 短期记忆
  │
  └─ 返回 JSON 响应
```

关键特征：前端发请求后**一直等待**，直到整个工作流执行完毕才收到响应。

---

## 八、/ask/stream 接口详解

### 8.1 完整代码逐行解读

```python
@app.post("/ask/stream")
async def handle_ask_stream(
    question: str,
    user_id: str = "default",
    session_id: str = None
):
    if not session_id:
        session_id = f"session-{str(uuid.uuid4())[:8]}"

    if not service_health.milvus_healthy:
        service_health.try_recover_milvus()

    # 创建 EventEmitter 实例，作为工作流中间事件的推送通道。
    # 工作流执行中各节点调用 emitter.emit() 将事件推入 asyncio.Queue，
    # 下方 event_stream() 通过 emitter.stream() 消费队列，实时推送给前端。
    emitter = EventEmitter()

    async def run_and_save():
        # 将 emitter 传入工作流，使执行过程中的中间事件推入队列。
        # 这是 /ask/stream 与 /ask 的核心区别：/ask 不传 emitter，中间事件被 NullEmitter 吞掉。
        final_state = await workflow.execute(
            question, user_id, session_id=session_id, emitter=emitter
        )

        if final_state.response and not final_state.clarification_needed:
            try:
                await smart_session_memory.add_conversation(
                    session_id=session_id,
                    user_message=question,
                    assistant_message=final_state.response,
                    user_id=user_id
                )
            except Exception as e:
                logging.warning(f"⚠️ Redis 保存失败（降级）：{e}")
                service_health.mark_redis_down()

    async def event_stream():
        # 启动后台 Task 执行工作流（生产者：emit() 往队列放事件）
        task = asyncio.create_task(run_and_save())
        # 异步迭代消费队列（消费者：stream() 从队列取事件，格式化为 SSE 推送）
        async for chunk in emitter.stream():
            yield chunk
        # 确保工作流 Task 完成（如保存对话到 Redis）后再结束，避免资源泄漏
        await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",       # SSE 标准 MIME 类型
        headers={
            "Cache-Control": "no-cache",       # 禁止缓存
            "Connection": "keep-alive",        # 保持 TCP 连接
            "X-Accel-Buffering": "no"          # 禁止 Nginx 缓冲
        }
    )
```

### 8.2 /ask/stream 接口完整时序

```
时间 ──────────────────────────────────────────────────────→

Task (run_and_save):
  │─ workflow.execute() ──────────────────────────────────────│
  │  ├─ emit(THINKING, "正在识别意图")  ─→ Queue               │
  │  ├─ intent_node.execute() ──┐                              │
  │  │                          │ (LLM 调用)                   │
  │  ├─ emit(THINKING, "选择技能")  ─→ Queue                    │
  │  ├─ emit(TOOL_CALL, "search_schema") ─→ Queue              │
  │  ├─ emit(ANSWER_STREAM, delta="研发部") ─→ Queue            │
  │  ├─ emit(ANSWER_STREAM, delta="共有") ─→ Queue              │
  │  ├─ emit(ANSWER_STREAM, delta="25人") ─→ Queue              │
  │  ├─ emit(ANSWER, {...}) ─→ Queue                            │
  │  ├─ emit_done() ─→ Queue                                    │
  │  └─ add_conversation() ──────────────────────────────────│

event_stream (消费者):
  │─ Queue.get() → "data: {type:thinking}" ──→ 前端            │
  │─ Queue.get() → "data: {type:tool_call}" ──→ 前端           │
  │─ Queue.get() → "data: {type:answer_stream, delta:研发部}" ──→ 前端 │
  │─ Queue.get() → "data: {type:answer_stream, delta:共有}" ──→ 前端   │
  │─ Queue.get() → "data: {type:answer_stream, delta:25人}" ──→ 前端   │
  │─ Queue.get() → "data: {type:answer}" ──→ 前端              │
  │─ Queue.get() → "data: {type:done}" ──→ 前端                │
  │─ await task (确保 run_and_save 完成)                        │
```

---

## 九、Emitter 事件推送机制

### 9.1 EventEmitter 与 NullEmitter

```python
class EventEmitter:
    """SSE 事件发射器 —— 生产者-消费者模式中的生产者"""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._closed = False

    def emit(self, event_type, data=None):
        """将事件推入队列，供 stream() 消费。"""
        if self._closed:
            return
        event = {"type": event_type, "data": data, "timestamp": time.time()}
        self._queue.put_nowait(event)

    def emit_done(self, **kwargs):
        """发送 DONE 事件并关闭发射器，stream() 收到后终止迭代。"""
        self.emit(EventType.DONE, kwargs)
        self._closed = True

    async def stream(self) -> AsyncGenerator[str, None]:
        """从队列消费事件，格式化为 SSE 文本流。300秒无事件发 ping 心跳。"""
        while True:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=300.0)
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                continue
            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            if event.get("type") == EventType.DONE:
                break


class NullEmitter:
    """空事件发射器 —— 空对象模式（Null Object Pattern）"""

    def emit(self, event_type, data=None):
        pass

    def emit_done(self, **kwargs):
        pass
```

### 9.2 为什么 /ask 不传 emitter

因为 `/ask` 是同步等待完整响应的接口，它不需要中间事件。

如果 `/ask` 也传了 `EventEmitter`，这些事件会进入队列，但没有任何消费者从队列取事件（因为没有 `StreamingResponse`），队列会无限增长，最终内存泄漏。

NullEmitter 的设计让同一套工作流代码无需任何 `if` 判断就能同时服务两种接口，这是**空对象模式**的经典应用。

### 9.3 为什么设计 event_stream()

`event_stream()` 解决的核心问题是：**让生产者和消费者并行运行，且生命周期可控**。

**方案A（错误）**：先跑完工作流，再流式输出

```python
async def event_stream_wrong():
    final_state = await run_and_save()              # 等工作流跑完（3-10秒）
    async for chunk in emitter.stream():            # 然后才开始输出
        yield chunk
```

这和 `/ask` 没有任何区别——前端还是要等 3-10 秒才能看到第一个字。

**方案B（错误）**：不启动 Task，直接消费队列

```python
async def event_stream_wrong():
    async for chunk in emitter.stream():            # 从队列取事件
        yield chunk
```

没有人往队列里放事件，`emitter.stream()` 会永远等下去。

**方案C（错误）**：在 StreamingResponse 之前启动

```python
task = asyncio.create_task(run_and_save())          # 在外面启动
return StreamingResponse(emitter.stream(), ...)      # 直接传 emitter.stream()
```

`await task` 没有地方执行，工作流如果抛异常会被静默吞掉。

**正确设计**：event_stream() 内部同时管理生产者和消费者

```python
async def event_stream():
    task = asyncio.create_task(run_and_save())       # ① 后台启动工作流
    async for chunk in emitter.stream():             # ② 同时消费队列
        yield chunk
    await task                                       # ③ 确保收尾完成
```

三个关键设计点：
1. **`create_task` 而非 `await`** —— 让工作流在后台执行，不阻塞事件消费
2. **`async for chunk in emitter.stream()`** —— 每次 yield 一个 SSE chunk 给前端
3. **`await task`** —— 流结束后确保后台 Task 也完成，避免资源泄漏

### 9.4 StreamingResponse 的精确时序

```
① 请求进来
② 前置检查（session_id、milvus 健康检查）
③ 创建 EventEmitter
④ return StreamingResponse(event_stream(), ...)
   │
   │  ⚠️ 此时 event_stream() 还没有执行！
   │  return 只是"注册"了这个异步生成器，告诉 FastAPI：
   │  "响应体从这个生成器里取"
   │
⑤ FastAPI 拿到 StreamingResponse 后，开始迭代 event_stream()
   │
   │  这时候才真正进入 event_stream() 内部：
   │  ├─ create_task(run_and_save())   ← 工作流开始跑
   │  ├─ async for chunk in emitter.stream()  ← 开始消费
   │  │    yield chunk ──→ FastAPI 写入 HTTP ──→ 前端收到
   │  └─ await task
```

`return` 不是"启动"，而是"注册"。真正启动是在 FastAPI 开始消费 `StreamingResponse` 的时候。

---

## 十、两个接口核心区别

| 维度 | `/ask` | `/ask/stream` |
|------|--------|---------------|
| **响应方式** | 一次性返回完整 JSON | SSE 流式推送事件 |
| **emitter** | `NullEmitter()`（空实现） | `EventEmitter()`（真实推送） |
| **前端体验** | 转圈等待 → 一次性看到回答 | 实时看到"正在思考"、"正在调用工具"、逐字输出 |
| **并发模型** | 串行：请求 → 等待 → 响应 | 并行：Task 执行工作流 + 协程消费事件 |
| **超时风险** | 整个请求超时才报错 | 300秒无事件才发心跳，连接更持久 |
| **保存对话** | 工作流完成后同步保存 | 工作流完成后在后台 Task 中保存 |
| **适用场景** | 简单问答、API 对接 | 前端交互、需要展示过程 |

**一句话总结**：`/ask` 是"等菜上桌"，`/ask/stream` 是"看厨师做菜"。
