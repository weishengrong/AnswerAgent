import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker,AsyncSession
from app.core.config.settings import llm_settings
from dotenv import load_dotenv

load_dotenv()

database_url = os.getenv("DATABASE_URL")

engine = create_async_engine(
    database_url,
    echo=llm_settings.DEBUG,
    # 🎯 数据库连接超时配置（防止启动卡住）
    pool_timeout=10,           # 连接池获取连接超时：10 秒
    pool_recycle=3600,         # 连接回收时间：1 小时（避免使用过期连接）
    pool_pre_ping=True,        # 每次取连接时先检测是否可用
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
)