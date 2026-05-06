from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class LlmSettings(BaseSettings):
    LLM_API_KEY: str
    LLM_BASE_URL: str
    LLM_MODEL_NAME: str
    DEBUG: bool = False

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
        extra="ignore")


llm_settings = LlmSettings()


class EmbeddingSettings(BaseSettings):
    EMBEDDING_MODEL_API_KEY: str
    EMBEDDING_MODEL_NAME: str
    EMBEDDING_MODEL_URL: str

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
        extra="ignore")


embedding = EmbeddingSettings()


class MilvusSettings(BaseSettings):
    MILVUS_HOST: str
    MILVUS_PORT: str
    MILVUS_USER: str
    MILVUS_PWD: str

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
        extra="ignore")


milvus_settings = MilvusSettings()


class RedisSettings(BaseSettings):
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int
    REDIS_PASSWORD: str

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
        extra="ignore")


redis_settings = RedisSettings()


class AgentSettings(BaseSettings):
    MAX_RETRIES: int = 3
    REACT_MAX_STEPS: int = 8
    INTENT_LOW_CONFIDENCE: float = 0.5

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
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
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
        extra="ignore")


memory_settings = MemorySettings()


class RAGSettings(BaseSettings):
    RAG_CHILD_LIMIT: int = 10
    RAG_COLLECTION_NAME: str = "try"
    RAG_MEMORY_COLLECTION: str = "answer_agent_memory"
    RAG_EMBEDDING_DIMS: int = 1024

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
        extra="ignore")


rag_settings = RAGSettings()
