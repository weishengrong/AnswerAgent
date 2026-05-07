from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from config.db_config import engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    nickname = Column(String(100), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


async def init_auth_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
