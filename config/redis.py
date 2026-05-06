import redis.asyncio as redis
from config.settings import redis_settings

redis_client = None
raw_redis_client = None


async def get_redis_client():
    """获取 Redis 客户端（单例，自动 utf-8 解码，用于 JSON/字符串场景）"""
    global redis_client
    if redis_client is None:
        redis_client = redis.Redis(
            host=redis_settings.REDIS_HOST,
            port=redis_settings.REDIS_PORT,
            db=redis_settings.REDIS_DB,
            password=redis_settings.REDIS_PASSWORD,
            decode_responses=True
        )

    return redis_client


async def get_raw_redis_client():
    """获取 Redis 客户端（单例，保持 bytes 不解码，用于 pickle 等二进制场景）

    BM25 索引等用 pickle 序列化的数据必须走这个客户端读写，
    否则 get 时会被自动 utf-8 解码触发 'invalid start byte' 错误。
    """
    global raw_redis_client
    if raw_redis_client is None:
        raw_redis_client = redis.Redis(
            host=redis_settings.REDIS_HOST,
            port=redis_settings.REDIS_PORT,
            db=redis_settings.REDIS_DB,
            password=redis_settings.REDIS_PASSWORD,
            decode_responses=False
        )

    return raw_redis_client


async def close_redis_client():
    global redis_client, raw_redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
    if raw_redis_client:
        await raw_redis_client.close()
        raw_redis_client = None