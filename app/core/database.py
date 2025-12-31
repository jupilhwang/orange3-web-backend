"""
Database configuration for Orange3 Web Backend.

Supports multiple databases via configuration:
    - SQLite (default): sqlite+aiosqlite:///path/to/db.db
    - PostgreSQL: postgresql+asyncpg://user:pass@host:5432/dbname
    - MySQL/MariaDB: mysql+aiomysql://user:pass@host:3306/dbname
    - Oracle: oracle+oracledb://user:pass@host:1521/dbname

Configuration priority:
    1. Config file (database.url in orange3-web.properties)
    2. Environment variable (DATABASE_URL)
    3. Default (SQLite in app directory)

Required packages for each database:
    - SQLite: aiosqlite (included by default)
    - PostgreSQL: asyncpg
    - MySQL: aiomysql
    - Oracle: oracledb
"""
from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool

from .config import get_config, get_database_url

# Get configuration
_config = get_config()

# Database URL
# Priority: config file > env var > default SQLite
DATABASE_URL = get_database_url()

# Create async engine with database-specific settings
_engine_kwargs = {
    "echo": _config.database.echo or _config.log.database_echo,
}

# SQLite specific settings
if "sqlite" in DATABASE_URL:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    _engine_kwargs["poolclass"] = StaticPool

engine = create_async_engine(DATABASE_URL, **_engine_kwargs)

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


