"""
Database Connection Management.

Provides async SQLAlchemy database engine and session factory.
Features:
- Connection pooling (min=5, max=20)
- Automatic reconnection on failure
- Transaction management with context managers
- Migration support via Alembic
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import NullPool

from src.config import get_config
from src.core.logging import logger


class Database:
    """
    Database connection manager.

    Handles async database connections with connection pooling.
    """

    _instance: Optional["Database"] = None
    _engine: Optional[AsyncEngine] = None
    _session_factory: Optional[async_sessionmaker] = None

    def __new__(cls):
        """Singleton pattern for database."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize database connection."""
        if hasattr(self, "_initialized"):
            return

        config = get_config()
        self.db_config = config.database

        self._initialized = True
        logger.info("Database manager initialized")

    def _create_engine(self) -> AsyncEngine:
        """Create async database engine."""
        if self._engine is not None:
            return self._engine

        engine = create_async_engine(
            self.db_config.url,
            pool_size=self.db_config.pool_size,
            max_overflow=self.db_config.max_overflow,
            pool_pre_ping=True,
            echo=False,
        )

        self._engine = engine
        self._session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

        logger.info(
            f"Database engine created: pool_size={self.db_config.pool_size}, "
            f"max_overflow={self.db_config.max_overflow}"
        )

        return engine

    @property
    def engine(self) -> AsyncEngine:
        """Get or create database engine."""
        if self._engine is None:
            self._create_engine()
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker:
        """Get session factory."""
        if self._session_factory is None:
            self._create_engine()
        return self._session_factory

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session as async context manager.

        Yields:
            AsyncSession: Database session

        Example:
            async with db.get_session() as session:
                result = await session.execute(select(Analysis))
        """
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def close(self) -> None:
        """Close database connections."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connections closed")

    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


_db: Optional[Database] = None


def get_database() -> Database:
    """Get or create database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI.

    Yields:
        AsyncSession: Database session
    """
    db = get_database()
    async with db.get_session() as session:
        yield session
