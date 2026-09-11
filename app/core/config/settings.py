from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class LlmSettings(BaseSettings):
    LLM_API_KEY: str
    LLM_BASE_URL: str
    LLM_MODEL_NAME: str
    DEBUG: bool = False

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"),
        extra="ignore")


llm_settings = LlmSettings()


class EmbeddingSettings(BaseSettings):
    EMBEDDING_MODEL_API_KEY: str
    EMBEDDING_MODEL_NAME: str
    EMBEDDING_MODEL_URL: str

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"),
        extra="ignore")


embedding = EmbeddingSettings()


class ChromaSettings(BaseSettings):
    CHROMA_PERSIST_DIR: str = "./chroma_data"

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"),
        extra="ignore")


chroma_settings = ChromaSettings()


class RedisSettings(BaseSettings):
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int
    REDIS_PASSWORD: str

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"),
        extra="ignore")


redis_settings = RedisSettings()


class AgentSettings(BaseSettings):
    MAX_RETRIES: int = 3
    REACT_MAX_STEPS: int = 8
    INTENT_LOW_CONFIDENCE: float = 0.5

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"),
        extra="ignore")


agent_settings = AgentSettings()


class MemorySettings(BaseSettings):
    SESSION_HARD_LIMIT: int = 30
    SESSION_SOFT_LIMIT: int = 20
    SESSION_KEEP_RAW: int = 10
    SESSION_SUMMARIZE: int = 10
    SESSION_TOKEN_THRESHOLD: int = 5000
    SESSION_TTL_SECONDS: int = 3600

    MEMORY_HIGH_WATERMARK: float = 0.75
    MEMORY_LOW_WATERMARK: float = 0.60
    MEMORY_CONTENT_WEIGHT: float = 0.6
    MEMORY_RECENCY_WEIGHT: float = 0.25
    MEMORY_RELEVANCE_WEIGHT: float = 0.15
    MEMORY_DEDUP_THRESHOLD: float = 0.9

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"),
        extra="ignore")


memory_settings = MemorySettings()


class RAGSettings(BaseSettings):
    RAG_CHILD_LIMIT: int = 10
    RAG_COLLECTION_NAME: str = "try"
    RAG_MEMORY_COLLECTION: str = "answer_agent_memory"
    RAG_EMBEDDING_DIMS: int = 1024
    RAG_CHROMA_PERSIST_DIR: str = "./rag_chroma_data"  # RAG schema 独立目录（与 mem0 的 chroma_data 隔离）

    RAG_TOP_K_VECTOR: int = 30
    RAG_PARENT_TOP_K: int = 5  # Parent 粗筛只取 top 5 张表
    RAG_TOP_K_BM25: int = 30
    RAG_TOP_K_TABLE_VECTOR: int = 3
    RAG_TOP_K_TABLE_BM25: int = 5
    RAG_RERANK_TOP_K: int = 15
    RAG_RERANK_THRESHOLD: float = 0.3
    RAG_TABLE_RERANK_MODE: str = "dynamic_fields"
    RAG_TABLE_DYNAMIC_MIN: int = 1
    RAG_TABLE_DYNAMIC_MAX: int = 6
    RAG_TABLE_SCORE_RATIO: float = 0.25
    RAG_TABLE_PREV_SCORE_RATIO: float = 0.65
    RAG_FIELD_SCORE_RATIO: float = 0.3
    RAG_FIELD_MIN_PER_TABLE: int = 3
    RAG_FIELD_SOFT_MAX_PER_TABLE: int = 8
    RAG_FIELD_HARD_MAX_PER_TABLE: int = 12
    RAG_FUSION_MODE: str = "anchor"  # "anchor" (模式A/Parent锚定) 或 "union" (模式C/并集去重)

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"),
        extra="ignore")


rag_settings = RAGSettings()


class MonitoringSettings(BaseSettings):
    METRICS_ENABLED: bool = True
    LOG_FORMAT: str = "json"  # json | text

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"),
        extra="ignore")


monitoring_settings = MonitoringSettings()
