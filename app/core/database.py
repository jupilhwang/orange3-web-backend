"""
Database configuration for Orange3 Web Backend.

Supports multiple databases via DATABASE_URL environment variable:
    - SQLite (default): sqlite+aiosqlite:///path/to/db.db
    - PostgreSQL: postgresql+asyncpg://user:pass@host:5432/dbname
    - MySQL/MariaDB: mysql+aiomysql://user:pass@host:3306/dbname
    - Oracle: oracle+oracledb://user:pass@host:1521/dbname

Required packages for each database:
    - SQLite: aiosqlite (included by default)
    - PostgreSQL: asyncpg
    - MySQL: aiomysql
    - Oracle: oracledb
"""
import os
from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool

from .paths import get_database_url

# Database URL - SQLite with aiosqlite driver
# Uses DATABASE_URL env var, or DATABASE_DIR env var, or defaults to app folder
DATABASE_URL = get_database_url()

# Create async engine
# For SQLite, we use StaticPool to share connections in async context
engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
    # SQLite specific settings
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    poolclass=StaticPool if "sqlite" in DATABASE_URL else None,
)

# Async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# Base class for ORM models
class Base(DeclarativeBase):
    pass


async def init_db():
    """Initialize database - create all tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        # Enable WAL mode for SQLite (better concurrency)
        if "sqlite" in DATABASE_URL:
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA busy_timeout=5000"))


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_db():
    """Close database connections"""
    await engine.dispose()


