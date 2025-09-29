"""
Async database session management for PDF to Markdown MCP Server.

This module provides SQLAlchemy async session configuration for high-performance
async operations with PostgreSQL and PGVector extension.
"""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..core.performance import get_performance_monitor

logger = logging.getLogger(__name__)

# Database configuration from environment variables - NO DEFAULT CREDENTIALS
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable is required. "
        "Example: postgresql://username:password@host:port/database"
    )

# Convert sync PostgreSQL URL to async (asyncpg driver)
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Optimized async connection pool configuration
ASYNC_POOL_SIZE = int(os.environ.get("ASYNC_DB_POOL_SIZE", "20"))
ASYNC_MAX_OVERFLOW = int(os.environ.get("ASYNC_DB_MAX_OVERFLOW", "10"))
ASYNC_POOL_PRE_PING = os.environ.get("ASYNC_DB_POOL_PRE_PING", "true").lower() == "true"
ASYNC_POOL_RECYCLE = int(os.environ.get("ASYNC_DB_POOL_RECYCLE", "1800"))  # 30 minutes
ASYNC_POOL_TIMEOUT = int(os.environ.get("ASYNC_DB_POOL_TIMEOUT", "20"))
ASYNC_CONNECT_TIMEOUT = int(os.environ.get("ASYNC_DB_CONNECT_TIMEOUT", "10"))
ASYNC_SQL_ECHO = os.environ.get("ASYNC_SQL_ECHO", "false").lower() == "true"

# Connection retry configuration
ASYNC_MAX_RETRIES = int(os.environ.get("ASYNC_DB_MAX_RETRIES", "3"))
ASYNC_RETRY_DELAY = float(os.environ.get("ASYNC_DB_RETRY_DELAY", "1.0"))
ASYNC_RETRY_BACKOFF_FACTOR = float(
    os.environ.get("ASYNC_DB_RETRY_BACKOFF_FACTOR", "2.0")
)

# Create async engine with optimized configuration
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    # poolclass removed - async engines use AsyncPool by default
    pool_size=ASYNC_POOL_SIZE,
    max_overflow=ASYNC_MAX_OVERFLOW,
    pool_pre_ping=ASYNC_POOL_PRE_PING,
    pool_recycle=ASYNC_POOL_RECYCLE,
    pool_timeout=ASYNC_POOL_TIMEOUT,
    echo=ASYNC_SQL_ECHO,
    # Async connection arguments for PostgreSQL optimization
    connect_args={
        "command_timeout": ASYNC_CONNECT_TIMEOUT,
        "server_settings": {
            "application_name": "pdf_to_markdown_mcp_async",
            "statement_timeout": "300000",  # 5 minutes
        },
    },
    # Additional async engine options
    isolation_level="READ_COMMITTED",
    future=True,  # Use SQLAlchemy 2.0 style
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,  # Manual flush control for better performance
)


class AsyncDatabaseManager:
    """Async database connection and session manager with performance monitoring."""

    def __init__(self):
        self.engine = async_engine
        self.session_factory = AsyncSessionLocal
        self.performance_monitor = get_performance_monitor()
        self._setup_event_listeners()

    def _setup_event_listeners(self):
        """Set up async event listeners for performance monitoring."""

        @event.listens_for(self.engine.sync_engine, "before_cursor_execute")
        def before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            context._query_start_time = asyncio.get_event_loop().time()
            context._statement = statement

        @event.listens_for(self.engine.sync_engine, "after_cursor_execute")
        def after_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            if hasattr(context, "_query_start_time"):
                execution_time = (
                    asyncio.get_event_loop().time() - context._query_start_time
                ) * 1000

                if execution_time > 500:  # Log slow queries (> 500ms)
                    logger.warning(
                        f"Slow async query detected: {execution_time:.2f}ms - "
                        f"{statement[:100]}..."
                    )

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session with automatic resource management.

        Yields:
            AsyncSession: Configured async database session

        Example:
            async with db_manager.get_async_session() as db:
                result = await db.execute(select(User))
        """
        async with self.performance_monitor.measure_performance("async_db_session"):
            session = self.session_factory()
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()

    async def execute_async_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        fetch_results: bool = True,
    ) -> Any:
        """
        Execute an async query with performance monitoring.

        Args:
            query: SQL query string
            parameters: Query parameters
            fetch_results: Whether to fetch and return results

        Returns:
            Query results if fetch_results=True, else None
        """
        async with self.get_async_session() as session:
            async with self.performance_monitor.measure_performance(
                "async_query_execution"
            ):
                try:
                    result = await session.execute(text(query), parameters or {})

                    if fetch_results:
                        if query.strip().upper().startswith("SELECT"):
                            return result.fetchall()
                        else:
                            return result

                    return None

                except SQLAlchemyError as e:
                    logger.error(f"Async query execution failed: {e}")
                    raise

    async def check_async_connection(self) -> bool:
        """
        Check async database connection health.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        try:
            async with self.get_async_session() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Async connection check failed: {e}")
            return False

    async def ensure_pgvector_extension_async(self) -> bool:
        """
        Ensure PGVector extension is available (async version).

        Returns:
            bool: True if PGVector is available, False otherwise
        """
        try:
            async with self.get_async_session() as session:
                # Check if PGVector extension is available
                result = await session.execute(
                    text(
                        """
                    SELECT EXISTS(
                        SELECT 1 FROM pg_extension WHERE extname = 'vector'
                    )
                """
                    )
                )

                pgvector_exists = result.scalar()

                if not pgvector_exists:
                    logger.warning("PGVector extension not installed")
                    return False

                # Verify vector operations work
                await session.execute(
                    text(
                        """
                    SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector as distance
                """
                    )
                )

                logger.info("PGVector extension verified and working (async)")
                return True

        except Exception as e:
            logger.error(f"Async PGVector extension check failed: {e}")
            return False

    async def get_async_connection_stats(self) -> dict[str, Any]:
        """
        Get async connection pool statistics.

        Returns:
            Dict containing connection pool statistics
        """
        pool = self.engine.pool

        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalidated": pool.invalidated(),
            "pool_timeout": ASYNC_POOL_TIMEOUT,
            "max_overflow": ASYNC_MAX_OVERFLOW,
            "total_capacity": ASYNC_POOL_SIZE + ASYNC_MAX_OVERFLOW,
            "utilization_percent": (
                pool.checkedout() / (ASYNC_POOL_SIZE + ASYNC_MAX_OVERFLOW)
            )
            * 100,
        }

    async def close_async_engine(self):
        """Close the async engine and all connections."""
        await self.async_engine.dispose()
        logger.info("Async database engine closed")


# Global async database manager instance
async_db_manager = AsyncDatabaseManager()


@asynccontextmanager
async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function for FastAPI to get async database sessions.

    Yields:
        AsyncSession: Configured async database session
    """
    async with async_db_manager.get_async_session() as session:
        yield session


class AsyncVectorSearchResult:
    """Container for async vector search results."""

    def __init__(
        self,
        document_id: int,
        chunk_id: int | None,
        filename: str,
        source_path: str,
        content: str,
        similarity_score: float,
        page_number: int | None = None,
        chunk_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.filename = filename
        self.source_path = source_path
        self.content = content
        self.similarity_score = similarity_score
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "filename": self.filename,
            "source_path": self.source_path,
            "content": self.content,
            "similarity_score": self.similarity_score,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


async def health_check_async() -> dict[str, Any]:
    """
    Perform async database health check.

    Returns:
        Dict containing health status and statistics
    """
    health_status = {
        "database": "unknown",
        "pgvector": "unknown",
        "connection_pool": {},
    }

    try:
        # Check basic connection
        connection_healthy = await async_db_manager.check_async_connection()
        health_status["database"] = "healthy" if connection_healthy else "unhealthy"

        # Check PGVector extension
        if connection_healthy:
            pgvector_available = (
                await async_db_manager.ensure_pgvector_extension_async()
            )
            health_status["pgvector"] = (
                "available" if pgvector_available else "unavailable"
            )

        # Get connection pool stats
        health_status[
            "connection_pool"
        ] = await async_db_manager.get_async_connection_stats()

    except Exception as e:
        logger.error(f"Async health check failed: {e}")
        health_status["error"] = str(e)

    return health_status


# Convenience functions for common operations
async def async_vector_similarity_search(
    query_embedding: list,
    limit: int = 10,
    similarity_threshold: float = 0.7,
    distance_metric: str = "cosine",
) -> list:
    """
    Perform async vector similarity search with optimized performance.

    Args:
        query_embedding: Query vector for similarity search
        limit: Maximum number of results to return
        similarity_threshold: Minimum similarity threshold
        distance_metric: Distance metric ('cosine', 'euclidean', 'inner_product')

    Returns:
        List of AsyncVectorSearchResult objects
    """
    # Distance operators for different metrics
    distance_ops = {"cosine": "<=>", "euclidean": "<->", "inner_product": "<#>"}

    distance_op = distance_ops.get(distance_metric, "<=>")

    async with async_db_manager.get_async_session() as db:
        # Optimized async vector search query
        query = text(
            f"""
            SELECT
                de.document_id,
                de.id as embedding_id,
                d.filename,
                d.source_path,
                de.chunk_text,
                de.page_number,
                de.chunk_index,
                de.metadata as chunk_metadata,
                d.metadata as doc_metadata,
                CASE
                    WHEN :distance_metric = 'inner_product' THEN
                        (de.embedding <#> :query_embedding::vector)
                    ELSE
                        1 - (de.embedding {distance_op} :query_embedding::vector)
                END as similarity_score
            FROM document_embeddings de
            JOIN documents d ON de.document_id = d.id
            WHERE
                CASE
                    WHEN :distance_metric = 'inner_product' THEN
                        (de.embedding <#> :query_embedding::vector) >= :threshold
                    ELSE
                        1 - (de.embedding {distance_op} :query_embedding::vector) >= :threshold
                END
            ORDER BY
                CASE
                    WHEN :distance_metric = 'inner_product' THEN
                        (de.embedding <#> :query_embedding::vector)
                    ELSE
                        de.embedding {distance_op} :query_embedding::vector
                END
                {"DESC" if distance_metric == "inner_product" else "ASC"}
            LIMIT :limit
        """
        )

        result = await db.execute(
            query,
            {
                "query_embedding": query_embedding,
                "threshold": similarity_threshold,
                "limit": limit,
                "distance_metric": distance_metric,
            },
        )

        results = []
        for row in result:
            metadata = {}
            if row.chunk_metadata:
                metadata.update(row.chunk_metadata)
            if row.doc_metadata:
                metadata.update(row.doc_metadata)

            results.append(
                AsyncVectorSearchResult(
                    document_id=row.document_id,
                    chunk_id=row.embedding_id,
                    filename=row.filename,
                    source_path=row.source_path,
                    content=row.chunk_text,
                    similarity_score=float(row.similarity_score),
                    page_number=row.page_number,
                    chunk_index=row.chunk_index,
                    metadata=metadata,
                )
            )

        return results
