from contextlib import asynccontextmanager

import sys
# Windows 兼容性：prometheus-client 在 Windows 上因 resource.getpagesize() 报错
if sys.platform == 'win32':
    import resource as _resource
    if not hasattr(_resource, 'getpagesize'):
        _resource.getpagesize = lambda: 4096

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from app.memory.filter import memory_filter
from app.middleware import get_trace_id, service_health
from app.api.v1 import ask_router, auth_router, session_router
from app.auth.models import init_auth_db
from app.core.config.settings import chroma_settings
import asyncio
import logging

# ============================================================
# JSON 结构化日志格式化器
# ============================================================

class JsonFormatter(logging.Formatter):
    """将日志输出为单行 JSON 格式，便于 Loki/ELK 消费"""

    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "trace_id": getattr(record, 'trace_id', ''),
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_entry["stack_info"] = self.formatStack(record.stack_info)
        import json
        return json.dumps(log_entry, ensure_ascii=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_auth_db()
    logging.info("✅ 用户认证数据库初始化完成")

    service_health.mark_chroma_up()
    logging.info("✅ Chroma 已初始化")

    if service_health.chroma_healthy:
        try:
            result = memory_filter.cleanup_expired_memories(user_id="default")
            if result["deleted"] > 0:
                logging.info(f"🧹 启动清理过期记忆：删除 {result['deleted']} 条，保留 {result['kept']} 条")
        except Exception as e:
            logging.warning(f"⚠️ 启动清理过期记忆失败（非致命）：{e}")
            if not service_health.chroma_healthy:
                service_health.try_recover_chroma()
    else:
        logging.info("⏭️ Chroma 不可用，跳过过期记忆清理")

    try:
        from app.rag.bm25_index import bm25_index
        from utils.db.connector import DBConnector

        loaded = await bm25_index.load_from_redis()

        need_rebuild = False
        if not loaded:
            logging.info("📦 BM25 索引不存在，需要构建")
            need_rebuild = True
        else:
            # BM25 存在，检查 schema 是否有变更
            connector = DBConnector()
            try:
                # 🎯 使用 asyncio.wait_for 添加总超时保护（15秒）
                changed = await asyncio.wait_for(
                    connector.schema_changed(),
                    timeout=15.0
                )
                if changed:
                    logging.info("🔄 检测到数据库 Schema 变更，正在重建索引...")
                    need_rebuild = True
                else:
                    logging.info("✅ Schema 未变，BM25 索引直接使用")
            except asyncio.TimeoutError:
                # ⚠️ Schema 检查超时，沿用现有索引（不影响启动）
                logging.warning("⚠️ Schema 指纹检测超时（>15s），沿用现有索引")
            except Exception as e:
                # 指纹检测失败不影响启动，沿用现有索引
                logging.warning(f"⚠️ Schema 指纹检测异常，沿用现有索引：{e}")

        if need_rebuild:
            try:
                from scripts.index.rebuild_rag_index import rebuild_from_db, rebuild_index

                logging.info("🔄 RAG 索引需要重建，开始自动构建...")
                tables = await rebuild_from_db(enrich=False)
                if not tables:
                    logging.warning("⚠️ 数据库未读取到任何表，索引未构建")
                else:
                    await rebuild_index(tables)
                    logging.info("🎉 RAG 索引自动构建完成")
            except Exception as e:
                logging.warning(f"⚠️ 自动建索引失败（降级为纯向量检索）：{e}")
    except Exception as e:
        logging.warning(f"⚠️ RAG 索引初始化异常（降级）：{e}")

    yield

    try:
        from app.rag.chroma import ChromaSessionLocal
        ChromaSessionLocal().close()
        logging.info("✅ Chroma 连接已关闭")
    except Exception:
        pass

    try:
        from app.core.llm import close_llm
        await close_llm()
        logging.info("✅ LLM httpx 连接池已关闭")
    except Exception:
        pass


app = FastAPI(title="AnswerAgent", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "http://124.223.93.65", "http://124.223.93.65:3000", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(ask_router)
app.include_router(session_router)

# Prometheus 指标采集中间件（自动采集 HTTP QPS/延迟/错误率等标准指标）
instrumentator = Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s] %(message)s'
)

# 使用 JSON 格式化器替换默认 formatter
json_formatter = JsonFormatter(datefmt='%Y-%m-%dT%H:%M:%S')
for handler in logging.root.handlers:
    handler.setFormatter(json_formatter)


class TraceIdFilter(logging.Filter):
    def filter(self, record):
        record.trace_id = get_trace_id()
        return True


for handler in logging.root.handlers:
    handler.addFilter(TraceIdFilter())

@app.get("/health")
async def health_check():
    """服务健康检查"""
    return {
        "status": "healthy",
        "trace_id": get_trace_id(),
        "services": service_health.status()
    }


