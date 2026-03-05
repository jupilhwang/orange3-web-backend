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

from typing import Any, AsyncGenerator, Dict, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool

from .config import get_config, get_database_url


# Base class for ORM models
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Lazy singletons — engine and session maker are created on first access
# rather than at import time, so that get_config() is not called eagerly.
# ---------------------------------------------------------------------------

_engine = None
_async_session_maker = None


def _build_engine():
    """Build the async engine using the current configuration."""
    config = get_config()
    db_url = get_database_url()

    engine_kwargs: Dict[str, Any] = {
        "echo": config.database.echo or config.log.database_echo,
    }

    if "sqlite" in db_url:
        engine_kwargs["connect_args"] = {"check_same_thread": False}
        engine_kwargs["poolclass"] = StaticPool
    elif "postgresql" in db_url or "postgres" in db_url:
        engine_kwargs["pool_size"] = getattr(config.database, "pool_size", 5)
        engine_kwargs["max_overflow"] = getattr(config.database, "max_overflow", 10)
        engine_kwargs["pool_timeout"] = getattr(config.database, "pool_timeout", 30)
        engine_kwargs["pool_recycle"] = getattr(config.database, "pool_recycle", 1800)
        engine_kwargs["pool_pre_ping"] = True
    elif "mysql" in db_url:
        engine_kwargs["pool_size"] = getattr(config.database, "pool_size", 5)
        engine_kwargs["max_overflow"] = getattr(config.database, "max_overflow", 10)
        engine_kwargs["pool_pre_ping"] = True

    return create_async_engine(db_url, **engine_kwargs)


def get_engine():
    """Return the async engine, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_session_maker():
    """Return the async session maker, creating it on first call."""
    global _async_session_maker
    if _async_session_maker is None:
        _async_session_maker = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _async_session_maker


# ---------------------------------------------------------------------------
# Public aliases — kept for backward compatibility with the rest of the codebase.
# Code that does ``from .database import async_session_maker`` will receive
# a callable proxy that always delegates to the lazy singleton.
# ---------------------------------------------------------------------------


class _SessionMakerProxy:
    """Thin proxy so ``async_session_maker(...)`` keeps working after refactor."""

    def __call__(self, *args, **kwargs):
        return get_session_maker()(*args, **kwargs)

    def __repr__(self):
        return repr(get_session_maker())


async_session_maker = _SessionMakerProxy()

# Expose engine as a module-level attribute for code that imports it directly.
# (engine is resolved lazily on first attribute access via __getattr__ fallback,
# but most usages access it at call-time so the proxy is fine.)


def _get_engine_proxy():
    return get_engine()


# Keep DATABASE_URL accessible for callers that import it as a constant.
# Evaluated lazily on first access.
class _DatabaseURLProxy(str):
    """Lazy string proxy for DATABASE_URL — avoids get_config() at import time."""

    _value: Optional[str] = None

    def __new__(cls):
        # Use an empty string as placeholder; actual value resolved on demand.
        return str.__new__(cls, "")

    def _resolve(self) -> str:
        if self._value is None:
            type(self)._value = get_database_url()
        return self._value

    def __contains__(self, item):  # type: ignore[override]
        return item in self._resolve()

    def __str__(self):
        return self._resolve()

    def __repr__(self):
        return repr(self._resolve())


DATABASE_URL = _DatabaseURLProxy()

# engine module-level alias (resolved lazily)
engine = None  # set to None; callers should use get_engine() for reliability


async def init_db():
    """Initialize database — create all tables."""
    _engine = get_engine()
    db_url = get_database_url()
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        if "sqlite" in db_url:
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA busy_timeout=5000"))


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with get_session_maker()() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_db():
    """Close database connections."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
