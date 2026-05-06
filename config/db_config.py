import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker,AsyncSession
from config.settings import llm_settings
from dotenv import load_dotenv

load_dotenv()

database_url = os.getenv("DATABASE_URL")

engine = create_async_engine(
    database_url,
    echo=llm_settings.DEBUG,

)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
)