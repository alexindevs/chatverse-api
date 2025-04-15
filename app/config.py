from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from pydantic_settings import BaseSettings

# Database URL (SQLite)
DATABASE_URL = "sqlite+aiosqlite:///./database.db"

# Pydantic settings
class Settings:
    secret_key: str = "your-secret-key"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    class Config:
        env_file = ".env"

# Create database engine
engine = create_async_engine(DATABASE_URL, future=True, echo=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Base for models
Base = declarative_base()

# Load settings
settings=Settings()

